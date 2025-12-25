"""
F7: Judge評価（3観点評価）
納得感 / 根拠の妥当性 / 誇張抑制を0-100で採点
"""
import os
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from models import (
    RequirementWithEvidence,
    Gap,
    Improvements,
    InterviewQAs,
    JudgeEvaluation,
    F7Output
)

# 環境変数読み込み
load_dotenv()


def evaluate_with_judge(
    job_text: str,
    resume_text: str,
    matched: List[RequirementWithEvidence],
    gaps: List[Gap],
    improvements: Improvements,
    interview_qas: Optional[InterviewQAs] = None,
    options: Optional[Dict[str, Any]] = None
) -> JudgeEvaluation:
    """
    Judge評価を実行する（F7）
    
    Args:
        job_text: 求人票のテキスト
        resume_text: 職務経歴書のテキスト
        matched: マッチした要件と根拠のペア
        gaps: ギャップのある要件
        improvements: 改善案
        interview_qas: 面接Q&A（オプション）
        options: オプション辞書
            - llm_provider: "openai" or "anthropic"（デフォルト "openai"）
            - model_name: モデル名（デフォルト gpt-4o-mini）
    
    Returns:
        JudgeEvaluation: Judge評価結果
    """
    # オプションのデフォルト値
    if options is None:
        options = {}
    
    llm_provider = options.get("llm_provider", "openai")
    model_name = options.get("model_name", None)
    
    # コスト重視のため、miniモデルをデフォルトに
    if not model_name:
        if llm_provider == "anthropic":
            model_name = "claude-3-haiku-20240307"
        else:
            model_name = "gpt-4o-mini"
    
    # LLMの初期化
    try:
        if llm_provider == "anthropic":
            llm = ChatAnthropic(
                model=model_name,
                temperature=0.0,  # 評価なので温度は0
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
        else:  # openai
            llm = ChatOpenAI(
                model=model_name,
                temperature=0.0,
                api_key=os.getenv("OPENAI_API_KEY")
            )
        
        # パーサー設定
        parser = PydanticOutputParser(pydantic_object=F7Output)
        
        # 評価対象データを準備
        matched_str = "\n".join([
            f"- {m.requirement.description}: {m.evidence.reason[:150]}..."
            for m in matched[:5]
        ]) if matched else "マッチした要件なし"
        
        gaps_str = "\n".join([
            f"- {g.requirement.description}: {g.evidence.reason[:150]}..."
            for g in gaps[:5]
        ]) if gaps else "ギャップなし"
        
        improvements_str = ""
        if improvements.resume_edits:
            improvements_str += "職務経歴書編集案:\n"
            for edit in improvements.resume_edits[:3]:
                improvements_str += f"- {edit.template[:150]}...\n"
        if improvements.action_items:
            improvements_str += "\n行動計画:\n"
            for item in improvements.action_items[:3]:
                improvements_str += f"- [{item.priority}] {item.action[:150]}...\n"
        
        interview_qa_str = ""
        if interview_qas and interview_qas.qa_list:
            interview_qa_str = "\n".join([
                f"Q: {qa.question[:150]}..."
                for qa in interview_qas.qa_list[:3]
            ])
        
        # プロンプト作成
        prompt_template = PromptTemplate(
            template="""あなたはAI応募適合度チェッカーのJudge評価専門家です。以下の分析結果を3つの観点で評価してください。

【求人票（抜粋）】
{job_text}

【職務経歴書（抜粋）】
{resume_text}

【マッチした要件（サンプル）】
{matched_str}

【ギャップのある要件（サンプル）】
{gaps_str}

【改善案（サンプル）】
{improvements_str}

【面接Q&A（サンプル）】
{interview_qa_str}

評価観点（各0-100点で評価）：

1. **納得感（Convincing）**
   - ユーザーが判断しやすい構造・説明になっているか
   - 分析結果が分かりやすく提示されているか
   - 根拠と判定理由が明確か

2. **根拠の妥当性（Grounding）**
   - 引用が要件に適切に紐づいているか
   - 根拠（evidence）が妥当か
   - 判定理由が説得力があるか

3. **誇張抑制（No Exaggeration）**
   - 職務経歴にないことを断定していないか
   - 改善案やQ&Aで捏造を推奨していないか
   - 「学習中/計画中」など現実的な表現を使っているか

評価ルール：
- 各観点を0-100点で評価
- 問題点を3つまで指摘（優先度順）
- 改善提案を3つまで提示（優先度順）
- 問題がない場合は「問題なし」と記載

{format_instructions}
""",
            input_variables=["job_text", "resume_text", "matched_str", "gaps_str", "improvements_str", "interview_qa_str"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        # テキストをカット（長すぎる場合）
        job_text_trimmed = job_text[:1500] + "..." if len(job_text) > 1500 else job_text
        resume_text_trimmed = resume_text[:1500] + "..." if len(resume_text) > 1500 else resume_text
        
        # LLM実行とパース（最大3回リトライ）
        max_retries = 3
        for attempt in range(max_retries):
            try:
                prompt = prompt_template.format(
                    job_text=job_text_trimmed,
                    resume_text=resume_text_trimmed,
                    matched_str=matched_str,
                    gaps_str=gaps_str,
                    improvements_str=improvements_str,
                    interview_qa_str=interview_qa_str or "面接Q&Aなし"
                )
                output = llm.invoke(prompt)
                result = parser.parse(output.content)
                judge_evaluation = result.judge_evaluation
                break
            except Exception as parse_error:
                if attempt == max_retries - 1:
                    # 最後の試行でも失敗した場合は例外を投げる
                    raise parse_error
                # リトライ
        
    except Exception as e:
        print(f"⚠️  Judge評価に失敗、fallbackを使用: {e}")
        # Fallback: ルールベース評価
        judge_evaluation = _fallback_judge_evaluate(matched, gaps, improvements)
    
    return judge_evaluation


def _fallback_judge_evaluate(
    matched: List[RequirementWithEvidence],
    gaps: List[Gap],
    improvements: Improvements
) -> JudgeEvaluation:
    """
    Fallback: ルールベースで簡易評価
    
    Args:
        matched: マッチした要件
        gaps: ギャップのある要件
        improvements: 改善案
    
    Returns:
        JudgeEvaluation: Judge評価結果
    """
    from models import JudgeScore
    
    # 簡易的な評価（全て75点）
    scores = JudgeScore(
        convincing=75.0,
        grounding=75.0,
        no_exaggeration=75.0
    )
    
    issues = [
        "LLM-as-Judgeの詳細評価を推奨します",
        "引用の正確性を再確認してください",
        "改善案の具体性をさらに高めることを検討してください"
    ]
    
    fix_suggestions = [
        "Judge評価を再実行して詳細な評価を取得してください",
        "引用が職務経歴書に実際に存在するか確認してください",
        "改善案をより具体的に記述してください"
    ]
    
    return JudgeEvaluation(
        scores=scores,
        issues=issues,
        fix_suggestions=fix_suggestions
    )









