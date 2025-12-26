"""
F8: 応募メール文面生成
求人票、職務経歴書、分析結果から応募メール文面を生成
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


# ==================== 応募メールモデル ====================
class ApplicationEmail(BaseModel):
    """応募メール文面"""
    subject: str = Field(..., description="件名")
    body: str = Field(..., description="本文")
    attachment_suggestions: List[str] = Field(
        default_factory=list,
        description="添付資料の提案（例: 職務経歴書、ポートフォリオなど）"
    )
    tips: List[str] = Field(
        default_factory=list,
        description="送信時の注意点・ヒント"
    )


class F8Output(BaseModel):
    """F8の出力形式"""
    application_email: ApplicationEmail = Field(..., description="応募メール文面")


def generate_application_email(
    job_text: str,
    resume_text: str,
    company_info: Optional[str],
    matched: List[RequirementWithEvidence],
    gaps: List[Gap],
    improvements: Improvements,
    summary: str,
    options: Optional[Dict[str, Any]] = None
) -> ApplicationEmail:
    """
    応募メール文面を生成する（F8）
    
    Args:
        job_text: 求人票のテキスト
        resume_text: 職務経歴書のテキスト
        company_info: 企業情報（オプション）
        matched: マッチした要件と根拠のペア
        gaps: ギャップのある要件
        improvements: 改善案
        summary: スコアの総評
        options: オプション辞書
            - llm_provider: "openai" or "anthropic"（デフォルト "openai"）
            - model_name: モデル名（デフォルト gpt-4o-mini）
    
    Returns:
        ApplicationEmail: 応募メール文面
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
                temperature=0.7,  # 文面生成なので創造性を少し高める
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
        else:  # openai
            llm = ChatOpenAI(
                model=model_name,
                temperature=0.7,
                api_key=os.getenv("OPENAI_API_KEY")
            )
        
        # パーサー設定
        parser = PydanticOutputParser(pydantic_object=F8Output)
        
        # 評価対象データを準備
        matched_summary = "\n".join([
            f"- {m.requirement.description} (一致度: {m.evidence.confidence:.0%})"
            for m in matched[:5]
        ]) if matched else "マッチした要件なし"
        
        gaps_summary = "\n".join([
            f"- {g.requirement.description}"
            for g in gaps[:3]
        ]) if gaps else "ギャップなし"
        
        company_info_str = company_info if company_info and company_info.strip() else "企業情報なし"
        
        # プロンプト作成
        prompt_template = PromptTemplate(
            template="""あなたは応募メール文面作成の専門家です。以下の情報から、効果的な応募メール文面を生成してください。

【求人票（抜粋）】
{job_text}

【企業情報】
{company_info}

【職務経歴書（抜粋）】
{resume_text}

【適合度分析結果】
総評: {summary}

マッチした要件（強み）:
{matched_summary}

ギャップのある要件:
{gaps_summary}

【改善案の方向性】
{improvements_str}

応募メール文面の要件：
1. **件名**: 簡潔で目を引く件名（30文字以内推奨）
2. **本文**: 
   - 冒頭: 応募の動機・志望動機（企業情報がある場合は企業文化に合わせる）
   - 中盤: 職務経歴書の要点と求人要件との適合点を簡潔に
   - 終盤: 今後の意欲・連絡先の案内
3. **添付資料の提案**: 職務経歴書、ポートフォリオなど
4. **送信時の注意点**: 誤字脱字チェック、送信前の確認事項など

注意事項：
- 職務経歴にないことを断定しない（「学習中」「計画中」など現実的な表現）
- 企業情報がある場合は、企業文化や価値観に合わせた文面にする
- 簡潔で読みやすい文面（本文は300-500文字程度）
- 丁寧で誠実なトーン

{format_instructions}
""",
            input_variables=["job_text", "company_info", "resume_text", "summary", "matched_summary", "gaps_summary", "improvements_str"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        # テキストをカット（長すぎる場合）
        job_text_trimmed = job_text[:2000] + "..." if len(job_text) > 2000 else job_text
        resume_text_trimmed = resume_text[:2000] + "..." if len(resume_text) > 2000 else resume_text
        company_info_trimmed = company_info_str[:1000] + "..." if len(company_info_str) > 1000 else company_info_str
        
        improvements_str = improvements.overall_strategy[:500] if improvements.overall_strategy else "改善案なし"
        
        # LLM実行とパース（最大3回リトライ）
        max_retries = 3
        for attempt in range(max_retries):
            try:
                prompt = prompt_template.format(
                    job_text=job_text_trimmed,
                    company_info=company_info_trimmed,
                    resume_text=resume_text_trimmed,
                    summary=summary[:500] if summary else "分析結果なし",
                    matched_summary=matched_summary,
                    gaps_summary=gaps_summary,
                    improvements_str=improvements_str
                )
                output = llm.invoke(prompt)
                result = parser.parse(output.content)
                application_email = result.application_email
                break
            except Exception as parse_error:
                if attempt == max_retries - 1:
                    # 最後の試行でも失敗した場合は例外を投げる
                    raise parse_error
                # リトライ
        
    except Exception as e:
        print(f"⚠️  応募メール文面生成に失敗、fallbackを使用: {e}")
        # Fallback: 簡易的な文面生成
        application_email = _fallback_generate_email(job_text, resume_text, summary)
    
    return application_email


def _fallback_generate_email(
    job_text: str,
    resume_text: str,
    summary: str
) -> ApplicationEmail:
    """
    Fallback: 簡易的な応募メール文面生成
    
    Args:
        job_text: 求人票のテキスト
        resume_text: 職務経歴書のテキスト
        summary: スコアの総評
    
    Returns:
        ApplicationEmail: 応募メール文面
    """
    subject = "【応募】エンジニア職への応募"
    
    body = f"""お世話になっております。

この度、貴社の求人に応募させていただきたく、ご連絡いたしました。

【志望動機】
貴社の事業内容に興味を持ち、応募させていただきました。

【自己PR】
職務経歴書に記載の通り、{resume_text[:200]}...の経験があります。

【今後の意欲】
貴社で貢献できるよう、精進してまいります。

ご検討のほど、よろしくお願いいたします。

【添付資料】
- 職務経歴書

以上、よろしくお願いいたします。
"""
    
    attachment_suggestions = [
        "職務経歴書",
        "ポートフォリオ（該当する場合）"
    ]
    
    tips = [
        "誤字脱字をチェックしてください",
        "送信前に内容を再確認してください",
        "企業名・役職名が正しいか確認してください"
    ]
    
    return ApplicationEmail(
        subject=subject,
        body=body,
        attachment_suggestions=attachment_suggestions,
        tips=tips
    )









