"""
応募メール下書き生成
求人票、企業情報、職務経歴書、分析結果から応募メール下書きを生成
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
    Requirement,
    RequirementWithEvidence,
    Gap,
    Improvements,
    EmailDraft,
    EmailEvidence
)

# 環境変数読み込み
load_dotenv()


# ==================== 応募メール下書きモデル ====================
class EmailEvidence(BaseModel):
    """本文中の主張に対する根拠"""
    claim: str = Field(..., description="主張（例: 「Python開発経験5年」）")
    evidence_type: str = Field(..., description="根拠タイプ（requirement/resume/achievement）")
    evidence_text: str = Field(..., description="根拠テキスト（引用または箇条書き）")
    requirement_id: Optional[str] = Field(None, description="対応する要件ID（該当する場合）")


class EmailDraft(BaseModel):
    """応募メール下書き"""
    subject_options: List[str] = Field(..., min_length=2, max_length=3, description="件名案（2〜3件）")
    body: str = Field(..., description="本文テンプレート（丁寧、短め、300-500文字程度）")
    evidence_list: List[EmailEvidence] = Field(
        default_factory=list,
        description="本文中の主張に対する根拠リスト（どの実績・どの要件に紐づくか）"
    )
    notes: List[str] = Field(
        default_factory=list,
        description="注意事項・ヒント（捏造禁止、未経験の表現方法など）"
    )


class EmailDraftOutput(BaseModel):
    """応募メール下書きの出力形式"""
    email_draft: EmailDraft = Field(..., description="応募メール下書き")


def generate_email_draft(
    job_text: str,
    resume_text: str,
    company_text: Optional[str],
    requirements: List[Requirement],
    matched: List[RequirementWithEvidence],
    gaps: List[Gap],
    improvements: Improvements,
    options: Optional[Dict[str, Any]] = None
) -> EmailDraft:
    """
    応募メール下書きを生成する
    
    Args:
        job_text: 求人票のテキスト
        resume_text: 職務経歴書のテキスト
        company_text: 企業情報（オプション）
        requirements: 全要件リスト
        matched: マッチした要件と根拠のペア
        gaps: ギャップのある要件
        improvements: 改善案
        options: オプション辞書
            - llm_provider: "openai" or "anthropic"（デフォルト "openai"）
            - model_name: モデル名（デフォルト gpt-4o-mini）
    
    Returns:
        EmailDraft: 応募メール下書き
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
        parser = PydanticOutputParser(pydantic_object=EmailDraftOutput)
        
        # 分析結果を準備
        matched_summary = "\n".join([
            f"- [{m.requirement.req_id}] {m.requirement.description} (一致度: {m.evidence.confidence:.0%})"
            for m in matched[:5]
        ]) if matched else "マッチした要件なし"
        
        # マッチした要件の根拠（引用）を準備
        matched_evidence = []
        for m in matched[:3]:  # 上位3件
            if m.evidence.quotes:
                quotes_text = "\n".join([f"  - {q.text[:100]}..." for q in m.evidence.quotes[:2]])
                matched_evidence.append(f"[{m.requirement.req_id}] {m.requirement.description}\n根拠:\n{quotes_text}")
        
        matched_evidence_str = "\n\n".join(matched_evidence) if matched_evidence else "根拠なし"
        
        company_info_str = company_text if company_text and company_text.strip() else "企業情報なし"
        
        # プロンプト作成
        prompt_template = PromptTemplate(
            template="""あなたは応募メール下書き作成の専門家です。以下の情報から、効果的な応募メール下書きを生成してください。

【求人票（抜粋）】
{job_text}

【企業情報】
{company_info}

【職務経歴書（抜粋）】
{resume_text}

【マッチした要件（強み）】
{matched_summary}

【マッチした要件の根拠（引用）】
{matched_evidence_str}

【改善案の方向性】
{improvements_str}

応募メール下書きの要件：

1. **件名案（2〜3件）**
   - 簡潔で目を引く件名（30文字以内推奨）
   - 応募職種や志望動機が分かる内容

2. **本文テンプレート**
   - 丁寧で短め（300-500文字程度）
   - 冒頭: 応募の動機・志望動機（企業情報がある場合は企業文化に合わせる）
   - 中盤: 職務経歴の要点と求人要件との適合点を簡潔に
   - 終盤: 今後の意欲・連絡先の案内

3. **根拠リスト（evidence_list）**
   - 本文中の各主張に対して、どの実績・どの要件に紐づくかを明記
   - evidence_type: "requirement"（要件に紐づく）、"resume"（職務経歴書に記載）、"achievement"（実績）
   - evidence_text: 引用または箇条書きで根拠を提示
   - requirement_id: 対応する要件ID（該当する場合）

4. **注意事項（notes）**
   - 捏造禁止: 職務経歴にないことを断定しない
   - 未経験の表現: 「学習中」「経験を活かして挑戦」「計画中」など現実的な表現
   - 送信前の確認事項

重要ルール：
- 職務経歴にないことを断定しない（「学習中」「計画中」など現実的な表現）
- 企業情報がある場合は、企業文化や価値観に合わせた文面にする
- 簡潔で読みやすい文面
- 丁寧で誠実なトーン
- 根拠は必ず職務経歴書または分析結果に基づく

{format_instructions}
""",
            input_variables=["job_text", "company_info", "resume_text", "matched_summary", "matched_evidence_str", "improvements_str"],
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
                    matched_summary=matched_summary,
                    matched_evidence_str=matched_evidence_str,
                    improvements_str=improvements_str
                )
                output = llm.invoke(prompt)
                result = parser.parse(output.content)
                email_draft = result.email_draft
                break
            except Exception as parse_error:
                if attempt == max_retries - 1:
                    # 最後の試行でも失敗した場合は例外を投げる
                    raise parse_error
                # リトライ
        
    except Exception as e:
        print(f"⚠️  応募メール下書き生成に失敗、fallbackを使用: {e}")
        # Fallback: 簡易的な下書き生成
        email_draft = _fallback_generate_draft(job_text, resume_text, matched)
    
    return email_draft


def _fallback_generate_draft(
    job_text: str,
    resume_text: str,
    matched: List[RequirementWithEvidence]
) -> EmailDraft:
    """
    Fallback: 簡易的な応募メール下書き生成
    
    Args:
        job_text: 求人票のテキスト
        resume_text: 職務経歴書のテキスト
        matched: マッチした要件と根拠のペア
    
    Returns:
        EmailDraft: 応募メール下書き
    """
    subject_options = [
        "【応募】エンジニア職への応募",
        "【応募】ご応募させていただきます",
        "【応募】求人への応募"
    ]
    
    body = f"""お世話になっております。

この度、貴社の求人に応募させていただきたく、ご連絡いたしました。

【志望動機】
貴社の事業内容に興味を持ち、応募させていただきました。

【自己PR】
職務経歴書に記載の通り、{resume_text[:200]}...の経験があります。

【今後の意欲】
貴社で貢献できるよう、精進してまいります。

ご検討のほど、よろしくお願いいたします。

以上、よろしくお願いいたします。
"""
    
    evidence_list = []
    for m in matched[:2]:
        if m.evidence.quotes:
            quote_text = m.evidence.quotes[0].text[:100] + "..." if len(m.evidence.quotes[0].text) > 100 else m.evidence.quotes[0].text
            evidence_list.append(EmailEvidence(
                claim=f"{m.requirement.description}の経験",
                evidence_type="requirement",
                evidence_text=quote_text,
                requirement_id=m.requirement.req_id
            ))
    
    notes = [
        "誤字脱字をチェックしてください",
        "送信前に内容を再確認してください",
        "企業名・役職名が正しいか確認してください",
        "職務経歴にない経験を断定しないよう注意してください"
    ]
    
    return EmailDraft(
        subject_options=subject_options,
        body=body,
        evidence_list=evidence_list,
        notes=notes
    )

