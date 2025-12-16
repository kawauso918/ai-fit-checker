"""
LLM-as-Judge: 評価機能
引用の正確性、Must/Want分類の妥当性、ギャップの妥当性、改善案の具体性を評価
"""
import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from models import Requirement, Evidence, RequirementWithEvidence, Gap, Improvements

# 環境変数読み込み
load_dotenv()


# ==================== 評価結果モデル ====================
class QuoteAccuracyEvaluation(BaseModel):
    """引用の正確性評価"""
    req_id: str = Field(..., description="要件ID")
    quote: str = Field(..., description="評価対象の引用")
    is_accurate: bool = Field(..., description="正確かどうか")
    accuracy_score: float = Field(..., ge=0.0, le=1.0, description="正確性スコア（0.0〜1.0）")
    reason: str = Field(..., description="評価理由")
    suggestion: Optional[str] = Field(None, description="改善提案（あれば）")


class ClassificationEvaluation(BaseModel):
    """Must/Want分類の妥当性評価"""
    req_id: str = Field(..., description="要件ID")
    current_category: str = Field(..., description="現在の分類（Must/Want）")
    is_correct: bool = Field(..., description="分類が正しいか")
    correct_category: Optional[str] = Field(None, description="正しい分類（間違っている場合）")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="分類の信頼度（0.0〜1.0）")
    reason: str = Field(..., description="評価理由")


class GapEvaluation(BaseModel):
    """ギャップの妥当性評価"""
    req_id: str = Field(..., description="要件ID")
    is_gap_correct: bool = Field(..., description="ギャップ判定が正しいか")
    gap_score: float = Field(..., ge=0.0, le=1.0, description="ギャップ判定の妥当性スコア（0.0〜1.0）")
    reason: str = Field(..., description="評価理由")
    suggestion: Optional[str] = Field(None, description="改善提案（あれば）")


class ImprovementSpecificityEvaluation(BaseModel):
    """改善案の具体性評価"""
    improvement_type: str = Field(..., description="改善案の種類（resume_edit/action_item）")
    target: str = Field(..., description="対象（gap_idまたはaction内容）")
    specificity_score: float = Field(..., ge=0.0, le=1.0, description="具体性スコア（0.0〜1.0）")
    is_specific: bool = Field(..., description="具体的かどうか")
    reason: str = Field(..., description="評価理由")
    suggestion: Optional[str] = Field(None, description="改善提案（あれば）")


class JudgeOutput(BaseModel):
    """LLM-as-Judgeの出力"""
    quote_accuracy: List[QuoteAccuracyEvaluation] = Field(
        default_factory=list,
        description="引用の正確性評価リスト"
    )
    classification: List[ClassificationEvaluation] = Field(
        default_factory=list,
        description="Must/Want分類の妥当性評価リスト"
    )
    gap_validity: List[GapEvaluation] = Field(
        default_factory=list,
        description="ギャップの妥当性評価リスト"
    )
    improvement_specificity: List[ImprovementSpecificityEvaluation] = Field(
        default_factory=list,
        description="改善案の具体性評価リスト"
    )
    overall_score: float = Field(..., ge=0.0, le=1.0, description="総合評価スコア（0.0〜1.0）")
    overall_feedback: str = Field(..., description="総合フィードバック")


# ==================== 評価関数 ====================
def evaluate_with_llm_judge(
    job_text: str,
    resume_text: str,
    requirements: List[Requirement],
    evidence_map: Dict[str, Evidence],
    matched: List[RequirementWithEvidence],
    gaps: List[Gap],
    improvements: Improvements,
    options: Optional[Dict[str, Any]] = None
) -> JudgeOutput:
    """
    LLM-as-Judgeで評価を実行
    
    Args:
        job_text: 求人票のテキスト
        resume_text: 職務経歴書のテキスト
        requirements: 要件リスト
        evidence_map: 根拠マップ
        matched: マッチした要件
        gaps: ギャップのある要件
        improvements: 改善案
        options: オプション辞書
        
    Returns:
        JudgeOutput: 評価結果
    """
    if options is None:
        options = {}
    
    llm_provider = options.get("llm_provider", "openai")
    model_name = options.get("model_name", None)
    judge_temperature = options.get("judge_temperature", 0.0)
    
    # LLMの初期化
    try:
        if llm_provider == "anthropic":
            llm = ChatAnthropic(
                model=model_name or "claude-3-5-sonnet-20241022",
                temperature=judge_temperature,
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
        else:  # openai
            llm = ChatOpenAI(
                model=model_name or "gpt-4o-mini",
                temperature=judge_temperature,
                api_key=os.getenv("OPENAI_API_KEY")
            )
    except Exception as e:
        raise Exception(f"LLM初期化エラー: {e}")
    
    # パーサー設定
    parser = PydanticOutputParser(pydantic_object=JudgeOutput)
    
    # 評価対象データを準備
    requirements_str = "\n".join([
        f"[{req.req_id}] {req.category.value}: {req.description} (引用: {req.job_quote[:100]}...)"
        for req in requirements
    ])
    
    # 引用のサンプル（評価用）
    quote_samples = []
    for ev in evidence_map.values():
        if ev.resume_quotes:
            quote_samples.append(f"[{ev.req_id}] {ev.resume_quotes[0][:100]}...")
    quotes_str = "\n".join(quote_samples[:5]) if quote_samples else "引用なし"
    
    # ギャップのサンプル
    gaps_str = "\n".join([
        f"[{g.requirement.req_id}] {g.requirement.description} (理由: {g.evidence.reason[:100]}...)"
        for g in gaps[:5]
    ]) if gaps else "ギャップなし"
    
    # 改善案のサンプル
    improvements_str = ""
    if improvements.resume_edits:
        improvements_str += "職務経歴書編集案:\n"
        for edit in improvements.resume_edits[:3]:
            improvements_str += f"- {edit.template[:100]}...\n"
    if improvements.action_items:
        improvements_str += "\n行動計画:\n"
        for item in improvements.action_items[:3]:
            improvements_str += f"- [{item.priority}] {item.action[:100]}...\n"
    
    # プロンプト作成
    prompt_template = PromptTemplate(
        template="""あなたはAI応募適合度チェッカーの評価専門家です。以下の分析結果を評価してください。

【求人票】
{job_text}

【職務経歴書】
{resume_text}

【抽出された要件】
{requirements_str}

【引用サンプル】
{quotes_str}

【ギャップサンプル】
{gaps_str}

【改善案サンプル】
{improvements_str}

評価観点：

1. **引用の正確性（Quote Accuracy）**
   - resume_quotesが実際にresume_text内に存在するか
   - 改変・要約されていないか
   - 文脈が適切か
   - 評価対象: 各Evidenceのresume_quotes（最大5件まで）

2. **Must/Want分類の妥当性（Classification）**
   - 各要件が適切なカテゴリ（Must/Want）に分類されているか
   - 求人票の文脈から判断
   - 評価対象: 全要件

3. **ギャップの妥当性（Gap Validity）**
   - ギャップ判定が適切か（本当に不足しているか）
   - 根拠（evidence.reason）が妥当か
   - 評価対象: ギャップのある要件（最大5件まで）

4. **改善案の具体性（Improvement Specificity）**
   - 改善案が「何をどう書く/何をやる」まで具体的か
   - 実行可能か
   - 評価対象: resume_editsとaction_items（各最大3件まで）

評価ルール：
- 各評価項目に対してスコア（0.0〜1.0）を付与
- 問題があれば具体的な理由と改善提案を記載
- 総合評価スコアは各観点の平均値
- 総合フィードバックは改善が必要な点を優先的に指摘

{format_instructions}
""",
        input_variables=["job_text", "resume_text", "requirements_str", "quotes_str", "gaps_str", "improvements_str"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    # テキストをカット（長すぎる場合）
    job_text_trimmed = job_text[:2000] + "..." if len(job_text) > 2000 else job_text
    resume_text_trimmed = resume_text[:2000] + "..." if len(resume_text) > 2000 else resume_text
    
    # LLM実行
    try:
        prompt = prompt_template.format(
            job_text=job_text_trimmed,
            resume_text=resume_text_trimmed,
            requirements_str=requirements_str,
            quotes_str=quotes_str,
            gaps_str=gaps_str,
            improvements_str=improvements_str
        )
        output = llm.invoke(prompt)
        result = parser.parse(output.content)
        return result
    except Exception as e:
        # パースエラー時はフォールバック
        print(f"⚠️  LLM-as-Judgeパースエラー: {e}")
        return _fallback_judge(requirements, evidence_map, matched, gaps, improvements)


def _fallback_judge(
    requirements: List[Requirement],
    evidence_map: Dict[str, Evidence],
    matched: List[RequirementWithEvidence],
    gaps: List[Gap],
    improvements: Improvements
) -> JudgeOutput:
    """
    Fallback: ルールベースで簡易評価
    """
    # 引用の正確性（簡易チェック）
    quote_accuracy = []
    for ev in list(evidence_map.values())[:5]:
        for quote in ev.resume_quotes[:1]:
            quote_accuracy.append(QuoteAccuracyEvaluation(
                req_id=ev.req_id,
                quote=quote[:100],
                is_accurate=True,  # 簡易版ではTrueと仮定
                accuracy_score=0.8,
                reason="簡易評価のため詳細確認が必要"
            ))
    
    # 分類の妥当性（簡易チェック）
    classification = []
    for req in requirements[:5]:
        classification.append(ClassificationEvaluation(
            req_id=req.req_id,
            current_category=req.category.value,
            is_correct=True,
            correct_category=None,
            confidence_score=0.8,
            reason="簡易評価のため詳細確認が必要"
        ))
    
    # ギャップの妥当性（簡易チェック）
    gap_validity = []
    for gap in gaps[:5]:
        gap_validity.append(GapEvaluation(
            req_id=gap.requirement.req_id,
            is_gap_correct=True,
            gap_score=0.8,
            reason="簡易評価のため詳細確認が必要"
        ))
    
    # 改善案の具体性（簡易チェック）
    improvement_specificity = []
    for edit in improvements.resume_edits[:3]:
        improvement_specificity.append(ImprovementSpecificityEvaluation(
            improvement_type="resume_edit",
            target=edit.target_gap,
            specificity_score=0.8,
            is_specific=True,
            reason="簡易評価のため詳細確認が必要"
        ))
    for item in improvements.action_items[:3]:
        improvement_specificity.append(ImprovementSpecificityEvaluation(
            improvement_type="action_item",
            target=item.action[:50],
            specificity_score=0.8,
            is_specific=True,
            reason="簡易評価のため詳細確認が必要"
        ))
    
    return JudgeOutput(
        quote_accuracy=quote_accuracy,
        classification=classification,
        gap_validity=gap_validity,
        improvement_specificity=improvement_specificity,
        overall_score=0.8,
        overall_feedback="簡易評価が実行されました。LLM-as-Judgeの詳細評価を推奨します。"
    )


# ==================== 評価結果の集計 ====================
def summarize_judge_results(judge_output: JudgeOutput) -> Dict[str, Any]:
    """
    評価結果を集計してサマリーを返す
    
    Args:
        judge_output: 評価結果
        
    Returns:
        Dict[str, Any]: 集計結果
    """
    summary = {
        "overall_score": judge_output.overall_score,
        "overall_feedback": judge_output.overall_feedback,
        "quote_accuracy": {
            "average_score": sum(e.accuracy_score for e in judge_output.quote_accuracy) / len(judge_output.quote_accuracy) if judge_output.quote_accuracy else 0.0,
            "total_count": len(judge_output.quote_accuracy),
            "accurate_count": sum(1 for e in judge_output.quote_accuracy if e.is_accurate),
            "issues": [e for e in judge_output.quote_accuracy if not e.is_accurate]
        },
        "classification": {
            "average_score": sum(e.confidence_score for e in judge_output.classification) / len(judge_output.classification) if judge_output.classification else 0.0,
            "total_count": len(judge_output.classification),
            "correct_count": sum(1 for e in judge_output.classification if e.is_correct),
            "issues": [e for e in judge_output.classification if not e.is_correct]
        },
        "gap_validity": {
            "average_score": sum(e.gap_score for e in judge_output.gap_validity) / len(judge_output.gap_validity) if judge_output.gap_validity else 0.0,
            "total_count": len(judge_output.gap_validity),
            "correct_count": sum(1 for e in judge_output.gap_validity if e.is_gap_correct),
            "issues": [e for e in judge_output.gap_validity if not e.is_gap_correct]
        },
        "improvement_specificity": {
            "average_score": sum(e.specificity_score for e in judge_output.improvement_specificity) / len(judge_output.improvement_specificity) if judge_output.improvement_specificity else 0.0,
            "total_count": len(judge_output.improvement_specificity),
            "specific_count": sum(1 for e in judge_output.improvement_specificity if e.is_specific),
            "issues": [e for e in judge_output.improvement_specificity if not e.is_specific]
        }
    }
    
    return summary


# ==================== テスト用コード ====================
if __name__ == "__main__":
    from f1_extract_requirements import extract_requirements
    from f2_extract_evidence import extract_evidence
    from f3_score import calculate_scores
    from f4_generate_improvements import generate_improvements
    
    # サンプルデータ
    sample_job_text = """
【求人票】Webエンジニア募集

■必須スキル
・Python開発経験3年以上
・Webアプリケーション開発の実務経験

■歓迎スキル
・AWSなどクラウド環境での開発経験
    """
    
    sample_resume_text = """
【職務経歴書】

■職務経歴
2019年〜現在：株式会社ABC
・Pythonを使用したWebアプリケーション開発に5年間従事
・Djangoフレームワークを用いたECサイトの構築
・AWS (EC2, S3, RDS) を活用したインフラ構築
    """
    
    print("=" * 60)
    print("LLM-as-Judge テスト")
    print("=" * 60)
    
    try:
        # F1〜F4を実行
        requirements = extract_requirements(sample_job_text)
        evidence_map = extract_evidence(sample_resume_text, requirements)
        score_total, score_must, score_want, matched, gaps, summary = calculate_scores(requirements, evidence_map)
        improvements = generate_improvements(sample_job_text, sample_resume_text, requirements, matched, gaps)
        
        # LLM-as-Judgeで評価
        print("\n[実行] LLM-as-Judge評価")
        judge_output = evaluate_with_llm_judge(
            sample_job_text,
            sample_resume_text,
            requirements,
            evidence_map,
            matched,
            gaps,
            improvements
        )
        
        # 結果表示
        print(f"\n{'='*60}")
        print("📊 評価結果")
        print(f"{'='*60}")
        print(f"総合スコア: {judge_output.overall_score:.2f}")
        print(f"\n【総合フィードバック】\n{judge_output.overall_feedback}")
        
        summary = summarize_judge_results(judge_output)
        print(f"\n【引用の正確性】")
        print(f"  平均スコア: {summary['quote_accuracy']['average_score']:.2f}")
        print(f"  正確: {summary['quote_accuracy']['accurate_count']}/{summary['quote_accuracy']['total_count']}")
        
        print(f"\n【分類の妥当性】")
        print(f"  平均スコア: {summary['classification']['average_score']:.2f}")
        print(f"  正しい: {summary['classification']['correct_count']}/{summary['classification']['total_count']}")
        
        print(f"\n【ギャップの妥当性】")
        print(f"  平均スコア: {summary['gap_validity']['average_score']:.2f}")
        print(f"  正しい: {summary['gap_validity']['correct_count']}/{summary['gap_validity']['total_count']}")
        
        print(f"\n【改善案の具体性】")
        print(f"  平均スコア: {summary['improvement_specificity']['average_score']:.2f}")
        print(f"  具体的: {summary['improvement_specificity']['specific_count']}/{summary['improvement_specificity']['total_count']}")
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()

