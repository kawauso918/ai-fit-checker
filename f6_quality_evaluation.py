"""
F6: 最終出力の品質評価
LLM-as-Judgeで分析結果の品質を自己評価
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
    InterviewQAs
)

# 環境変数読み込み
load_dotenv()


# ==================== 品質評価モデル ====================
class QualityScore(BaseModel):
    """観点別スコア"""
    criterion: str = Field(..., description="評価観点名")
    score: float = Field(..., ge=0.0, le=100.0, description="スコア（0-100）")
    reason: str = Field(..., description="評価理由")


class QualityEvaluation(BaseModel):
    """品質評価結果"""
    overall_score: float = Field(..., ge=0.0, le=100.0, description="総合スコア（0-100）")
    criterion_scores: List[QualityScore] = Field(..., description="観点別スコア")
    improvement_points: List[str] = Field(..., description="改善ポイント（3つ）")


class F6Output(BaseModel):
    """F6の出力形式"""
    quality_evaluation: QualityEvaluation = Field(..., description="品質評価結果")


def evaluate_quality(
    job_text: str,
    resume_text: str,
    matched: List[RequirementWithEvidence],
    gaps: List[Gap],
    improvements: Improvements,
    interview_qas: Optional[InterviewQAs] = None,
    options: Optional[Dict[str, Any]] = None
) -> QualityEvaluation:
    """
    最終出力の品質を評価する（F6）

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
        QualityEvaluation: 品質評価結果
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
        parser = PydanticOutputParser(pydantic_object=F6Output)

        # 評価対象データを準備
        matched_str = "\n".join([
            f"- {m.requirement.description}: {m.evidence.reason[:100]}..."
            for m in matched[:5]
        ]) if matched else "マッチした要件なし"

        gaps_str = "\n".join([
            f"- {g.requirement.description}: {g.evidence.reason[:100]}..."
            for g in gaps[:5]
        ]) if gaps else "ギャップなし"

        improvements_str = ""
        if improvements.resume_edits:
            improvements_str += "職務経歴書編集案:\n"
            for edit in improvements.resume_edits[:3]:
                improvements_str += f"- {edit.template[:100]}...\n"
        if improvements.action_items:
            improvements_str += "\n行動計画:\n"
            for item in improvements.action_items[:3]:
                improvements_str += f"- [{item.priority}] {item.action[:100]}...\n"

        interview_qa_str = ""
        if interview_qas and interview_qas.qa_list:
            interview_qa_str = "\n".join([
                f"Q: {qa.question[:100]}..."
                for qa in interview_qas.qa_list[:3]
            ])

        # プロンプト作成
        prompt_template = PromptTemplate(
            template="""あなたはAI応募適合度チェッカーの品質評価専門家です。以下の分析結果の品質を評価してください。

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

1. **根拠の妥当性（Evidence Validity）**
   - マッチした要件の根拠（evidence.reason）が妥当か
   - 引用（resume_quotes）が適切か
   - ギャップ判定の理由が妥当か

2. **捏造リスクの低さ（No Fabrication）**
   - 職務経歴にない経験を「経験がある」としていないか
   - 改善案やQ&Aで捏造を推奨していないか
   - 「学習中/計画中」など現実的な表現を使っているか

3. **具体性（Specificity）**
   - 改善案が「何をどう書く/何をやる」まで具体的か
   - Q&Aの回答骨子が具体的か
   - 抽象的な表現が少ないか

4. **実行可能性（Feasibility）**
   - 改善案が現実的に実行可能か
   - 時間的・技術的に無理な提案がないか
   - 優先度付けが適切か

5. **一貫性（Consistency）**
   - 分析結果全体が一貫しているか
   - 矛盾する記述がないか
   - スコアと判定が整合しているか

6. **読みやすさ（Readability）**
   - 出力が読みやすいか
   - 専門用語の説明が適切か
   - 構造が分かりやすいか

評価ルール：
- 各観点を0-100点で評価
- 総合スコアは各観点の平均値（0-100点）
- 改善ポイントを3つ提示（優先度順）
- 問題があれば具体的に指摘

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
                quality_evaluation = result.quality_evaluation
                break
            except Exception as parse_error:
                if attempt == max_retries - 1:
                    # 最後の試行でも失敗した場合は例外を投げる
                    raise parse_error
                # リトライ

    except Exception as e:
        print(f"⚠️  品質評価に失敗、fallbackを使用: {e}")
        # Fallback: ルールベース評価
        quality_evaluation = _fallback_evaluate(matched, gaps, improvements)

    return quality_evaluation


def _fallback_evaluate(
    matched: List[RequirementWithEvidence],
    gaps: List[Gap],
    improvements: Improvements
) -> QualityEvaluation:
    """
    Fallback: ルールベースで簡易評価

    Args:
        matched: マッチした要件
        gaps: ギャップのある要件
        improvements: 改善案

    Returns:
        QualityEvaluation: 品質評価結果
    """
    # 簡易的な評価（全て80点）
    criterion_scores = [
        QualityScore(
            criterion="根拠の妥当性",
            score=80.0,
            reason="簡易評価のため詳細確認が必要"
        ),
        QualityScore(
            criterion="捏造リスクの低さ",
            score=80.0,
            reason="簡易評価のため詳細確認が必要"
        ),
        QualityScore(
            criterion="具体性",
            score=80.0,
            reason="簡易評価のため詳細確認が必要"
        ),
        QualityScore(
            criterion="実行可能性",
            score=80.0,
            reason="簡易評価のため詳細確認が必要"
        ),
        QualityScore(
            criterion="一貫性",
            score=80.0,
            reason="簡易評価のため詳細確認が必要"
        ),
        QualityScore(
            criterion="読みやすさ",
            score=80.0,
            reason="簡易評価のため詳細確認が必要"
        ),
    ]

    improvement_points = [
        "LLM-as-Judgeの詳細評価を推奨します",
        "引用の正確性を再確認してください",
        "改善案の具体性をさらに高めることを検討してください"
    ]

    return QualityEvaluation(
        overall_score=80.0,
        criterion_scores=criterion_scores,
        improvement_points=improvement_points
    )





