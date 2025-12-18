"""
F2: 職務経歴書から根拠を抽出
PydanticOutputParser + 原文引用検証で安定化 + セクション分解で精度向上
"""
import os
import re
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from models import Requirement, Evidence, F2Output, ConfidenceLevel

# 環境変数読み込み
load_dotenv()


# ==================== セクション分解モデル ====================
class ResumeSection(BaseModel):
    """職務経歴書のセクション"""
    section_type: str = Field(..., description="セクションタイプ（summary/skills/projects/achievements/roles）")
    content: str = Field(..., description="セクションの内容")


class StructuredResume(BaseModel):
    """構造化された職務経歴書"""
    sections: List[ResumeSection] = Field(..., description="セクションリスト")


def _structure_resume_text(
    resume_text: str,
    llm_provider: str = "openai",
    model_name: Optional[str] = None
) -> Optional[Dict[str, str]]:
    """
    職務経歴書をセクションに分解
    
    Args:
        resume_text: 職務経歴書のテキスト
        llm_provider: LLMプロバイダー
        model_name: モデル名
    
    Returns:
        Optional[Dict[str, str]]: セクションタイプ -> 内容の辞書。失敗時はNone
    """
    if not resume_text or len(resume_text.strip()) < 50:
        return None
    
    try:
        # LLMの初期化
        if llm_provider == "anthropic":
            llm = ChatAnthropic(
                model=model_name or "claude-3-5-sonnet-20241022",
                temperature=0.0,
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
        else:  # openai
            llm = ChatOpenAI(
                model=model_name or "gpt-4o-mini",
                temperature=0.0,
                api_key=os.getenv("OPENAI_API_KEY")
            )
        
        # パーサー設定
        parser = PydanticOutputParser(pydantic_object=StructuredResume)
        
        # プロンプト作成
        prompt_template = PromptTemplate(
            template="""あなたは職務経歴書の構造化専門家です。以下の職務経歴書をセクションに分類してください。

職務経歴書：
{resume_text}

セクション分類：
以下のセクションタイプに分類してください（該当するもののみ）：
- summary: 要約・自己PR・プロフィール
- skills: スキル・技術スタック
- projects: プロジェクト・開発実績
- achievements: 成果・実績・受賞歴
- roles: 職務経歴・職務内容・担当業務

分類ルール：
1. 各セクションは元のテキストをそのまま保持（改変禁止）
2. 1つのテキストが複数のセクションタイプに該当する場合は、最も適切な1つに分類
3. 分類できない部分は無視してOK
4. セクションタイプは小文字で指定（summary/skills/projects/achievements/roles）

{format_instructions}
""",
            input_variables=["resume_text"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        # テキストをカット（長すぎる場合）
        resume_text_trimmed = resume_text[:3000] + "..." if len(resume_text) > 3000 else resume_text
        
        # LLM実行
        prompt = prompt_template.format(resume_text=resume_text_trimmed)
        output = llm.invoke(prompt)
        result = parser.parse(output.content)
        
        # 辞書形式に変換
        sections_dict = {}
        for section in result.sections:
            if section.section_type in sections_dict:
                # 既存のセクションに追記
                sections_dict[section.section_type] += "\n\n" + section.content
            else:
                sections_dict[section.section_type] = section.content
        
        return sections_dict if sections_dict else None
        
    except Exception as e:
        print(f"⚠️  セクション分解に失敗、フォールバック: {e}")
        return None


def extract_evidence(
    resume_text: str,
    requirements: List[Requirement],
    options: Optional[dict] = None
) -> Dict[str, Evidence]:
    """
    職務経歴書から要件に対する根拠を抽出する（F2）

    Args:
        resume_text: 職務経歴書のテキスト
        requirements: 抽出された要件リスト（F1の出力）
        options: オプション辞書
            - llm_provider: "openai" or "anthropic"（デフォルト "openai"）
            - model_name: モデル名
            - verify_quotes: 引用検証を行うか（デフォルト True）

    Returns:
        Dict[str, Evidence]: req_id -> Evidence の辞書（全req_idが必ず存在）
    """
    # オプションのデフォルト値
    if options is None:
        options = {}

    llm_provider = options.get("llm_provider", "openai")
    model_name = options.get("model_name", None)
    verify_quotes = options.get("verify_quotes", True)

    # 職務経歴書をセクション分解（失敗時は従来通り）
    structured_resume = None
    try:
        structured_resume = _structure_resume_text(resume_text, llm_provider, model_name)
    except Exception as e:
        print(f"⚠️  セクション分解をスキップ: {e}")
        structured_resume = None

    # LLMの初期化
    try:
        if llm_provider == "anthropic":
            llm = ChatAnthropic(
                model=model_name or "claude-3-5-sonnet-20241022",
                temperature=0.0,
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
        else:  # openai
            llm = ChatOpenAI(
                model=model_name or "gpt-4o-mini",
                temperature=0.0,
                api_key=os.getenv("OPENAI_API_KEY")
            )

        # パーサー設定
        parser = PydanticOutputParser(pydantic_object=F2Output)

        # 要件リストを文字列化
        requirements_str = "\n".join([
            f"[{req.req_id}] {req.category.value}: {req.description}"
            for req in requirements
        ])

        # プロンプト作成（セクション情報を含む）
        if structured_resume:
            # セクション情報を整形
            sections_str = ""
            for section_type, content in structured_resume.items():
                section_label = {
                    "summary": "【要約・自己PR】",
                    "skills": "【スキル・技術スタック】",
                    "projects": "【プロジェクト・開発実績】",
                    "achievements": "【成果・実績】",
                    "roles": "【職務経歴・担当業務】"
                }.get(section_type, f"【{section_type}】")
                sections_str += f"{section_label}\n{content}\n\n"
            
            resume_text_for_prompt = sections_str.strip()
            prompt_note = "\n注意: 職務経歴書はセクションごとに分類されています。各セクションの内容を参照して根拠を抽出してください。"
        else:
            resume_text_for_prompt = resume_text
            prompt_note = ""
        
        prompt_template = PromptTemplate(
            template="""あなたは職務経歴書の分析専門家です。以下の職務経歴書から、各要件に対する根拠を抽出してください。

職務経歴書：
{resume_text}{prompt_note}

要件リスト：
{requirements_str}

抽出ルール：
1. **重要**: resume_quotesには職務経歴書からの原文をそのまま引用すること（改変・要約は絶対に禁止）
2. 各要件に対して、マッチする経験やスキルがあれば原文を引用
3. マッチしない場合はresume_quotesを空リスト[]にし、confidenceを0.0に設定
4. confidenceは0.0〜1.0で設定（1.0=完全一致、0.7以上=高、0.4〜0.7=中、0.4未満=低、0.0=マッチなし）
5. confidence_levelは自動設定されるため、confidenceの値だけ設定すればOK
6. reasonには判定理由を簡潔に記載（なぜマッチする/しないか）
7. **全要件に対して必ず根拠を抽出**（マッチしない場合も含める）

{format_instructions}
""",
            input_variables=["resume_text", "requirements_str", "prompt_note"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        # LLM実行とパース（最大3回リトライ）
        max_retries = 3
        for attempt in range(max_retries):
            try:
                prompt = prompt_template.format(
                    resume_text=resume_text_for_prompt,
                    requirements_str=requirements_str,
                    prompt_note=prompt_note
                )
                output = llm.invoke(prompt)
                result = parser.parse(output.content)
                evidence_list = result.evidence_list
                break
            except Exception as parse_error:
                if attempt == max_retries - 1:
                    # 最後の試行でも失敗した場合は例外を投げる
                    raise parse_error
                # リトライ

    except Exception as e:
        print(f"⚠️  LLM抽出に失敗、fallbackを使用: {e}")
        # Fallback: ルールベース抽出
        evidence_list = _fallback_extract(resume_text, requirements)

    # 引用検証
    if verify_quotes:
        evidence_list = _verify_quotes(evidence_list, resume_text)
        
        # 無効な引用がある場合、1回だけ再抽出を試みる（コスト増えすぎない範囲で）
        invalid_evidences = [ev for ev in evidence_list if len(ev.resume_quotes) == 0 and ev.confidence == 0.0]
        if invalid_evidences and len(invalid_evidences) <= 3:  # 最大3件まで再抽出
            try:
                # 無効な要件のみを再抽出
                invalid_req_ids = [ev.req_id for ev in invalid_evidences]
                invalid_requirements = [req for req in requirements if req.req_id in invalid_req_ids]
                
                if invalid_requirements:
                    retry_evidence_list = _retry_extract_for_requirements(
                        resume_text, invalid_requirements, llm, parser, prompt_template
                    )
                    
                    # 再抽出結果で置き換え
                    for retry_ev in retry_evidence_list:
                        # 既存のリストから該当するEvidenceを探して置き換え
                        for i, ev in enumerate(evidence_list):
                            if ev.req_id == retry_ev.req_id:
                                # 再抽出結果を検証
                                verified_retry = _verify_single_evidence(retry_ev, resume_text)
                                evidence_list[i] = verified_retry
                                break
            except Exception as retry_error:
                # 再抽出失敗は無視（元の結果を使用）
                print(f"⚠️  再抽出に失敗（無視）: {retry_error}")

    # dict[req_id -> Evidence] に変換
    evidence_map = {ev.req_id: ev for ev in evidence_list}

    # 全req_idが存在するか確認し、無ければ補完
    evidence_map = _ensure_all_requirements(evidence_map, requirements)

    return evidence_map


def _verify_quotes(
    evidence_list: List[Evidence],
    resume_text: str
) -> List[Evidence]:
    """
    resume_quotesが実際にresume_text内に存在するか検証（強化版）
    存在しない引用はconfidenceを下げ、reasonに追記
    """
    verified_list = []

    for ev in evidence_list:
        invalid_quotes = []
        valid_quotes = []

        for quote in ev.resume_quotes:
            quote_clean = quote.strip()
            if not quote_clean:
                continue
            
            # 原文に存在するか確認（部分一致 + 柔軟なマッチング）
            found = False
            
            # 1. 完全一致
            if quote_clean in resume_text:
                found = True
            # 2. 空白を無視した一致
            elif quote_clean.replace(" ", "").replace("　", "") in resume_text.replace(" ", "").replace("　", ""):
                found = True
            # 3. 主要キーワードが含まれているか（最低3単語以上の場合）
            else:
                words = [w for w in quote_clean.split() if len(w) >= 2]
                if len(words) >= 3:
                    matched_words = sum(1 for word in words if word in resume_text)
                    if matched_words >= len(words) * 0.7:  # 70%以上の単語が一致
                        found = True

            if found:
                valid_quotes.append(quote)
            else:
                invalid_quotes.append(quote)

        # 無効な引用がある場合
        if invalid_quotes:
            # confidenceを大幅に下げる（partial/noneに調整）
            # 全て無効な場合は0.0（NONE）、一部無効な場合は0.3以下（LOW）
            if len(valid_quotes) == 0:
                new_confidence = 0.0
                new_level = ConfidenceLevel.NONE
            else:
                # 一部有効な場合は、有効な引用の割合に応じて調整
                valid_ratio = len(valid_quotes) / (len(valid_quotes) + len(invalid_quotes))
                new_confidence = max(0.0, ev.confidence * valid_ratio - 0.2)
                if new_confidence >= 0.4:
                    new_level = ConfidenceLevel.MEDIUM
                elif new_confidence > 0.0:
                    new_level = ConfidenceLevel.LOW
                else:
                    new_level = ConfidenceLevel.NONE

            # reasonに追記（「原文に存在しないため要確認/再抽出」）
            new_reason = ev.reason + f"\n⚠️ 引用検証失敗: {len(invalid_quotes)}件の引用が原文に存在しないため要確認/再抽出"

            # 新しいEvidenceを作成（有効な引用のみ）
            verified_ev = Evidence(
                req_id=ev.req_id,
                resume_quotes=valid_quotes,
                confidence=new_confidence,
                confidence_level=new_level,
                reason=new_reason
            )
            verified_list.append(verified_ev)
        else:
            # 全て有効な引用
            verified_list.append(ev)

    return verified_list


def _retry_extract_for_requirements(
    resume_text: str,
    requirements: List[Requirement],
    llm,
    parser,
    prompt_template
) -> List[Evidence]:
    """
    特定の要件に対して再抽出を試みる（1回のみ）
    
    Args:
        resume_text: 職務経歴書のテキスト
        requirements: 再抽出対象の要件リスト
        llm: LLMインスタンス
        parser: パーサー
        prompt_template: プロンプトテンプレート
        
    Returns:
        List[Evidence]: 再抽出されたEvidenceリスト
    """
    if not requirements:
        return []
    
    requirements_str = "\n".join([
        f"[{req.req_id}] {req.category.value}: {req.description}"
        for req in requirements
    ])
    
    try:
        prompt = prompt_template.format(
            resume_text=resume_text,
            requirements_str=requirements_str
        )
        output = llm.invoke(prompt)
        result = parser.parse(output.content)
        return result.evidence_list
    except Exception:
        # 再抽出失敗時は空のリストを返す
        return []


def _verify_single_evidence(ev: Evidence, resume_text: str) -> Evidence:
    """
    単一のEvidenceの引用を検証
    
    Args:
        ev: 検証対象のEvidence
        resume_text: 職務経歴書のテキスト
        
    Returns:
        Evidence: 検証済みのEvidence
    """
    invalid_quotes = []
    valid_quotes = []

    for quote in ev.resume_quotes:
        quote_clean = quote.strip()
        if not quote_clean:
            continue
        
        found = False
        if quote_clean in resume_text:
            found = True
        elif quote_clean.replace(" ", "").replace("　", "") in resume_text.replace(" ", "").replace("　", ""):
            found = True
        else:
            words = [w for w in quote_clean.split() if len(w) >= 2]
            if len(words) >= 3:
                matched_words = sum(1 for word in words if word in resume_text)
                if matched_words >= len(words) * 0.7:
                    found = True

        if found:
            valid_quotes.append(quote)
        else:
            invalid_quotes.append(quote)

    if invalid_quotes:
        if len(valid_quotes) == 0:
            new_confidence = 0.0
            new_level = ConfidenceLevel.NONE
        else:
            valid_ratio = len(valid_quotes) / (len(valid_quotes) + len(invalid_quotes))
            new_confidence = max(0.0, ev.confidence * valid_ratio - 0.2)
            if new_confidence >= 0.4:
                new_level = ConfidenceLevel.MEDIUM
            elif new_confidence > 0.0:
                new_level = ConfidenceLevel.LOW
            else:
                new_level = ConfidenceLevel.NONE

        new_reason = ev.reason + f"\n⚠️ 引用検証失敗: {len(invalid_quotes)}件の引用が原文に存在しないため要確認/再抽出"
        
        return Evidence(
            req_id=ev.req_id,
            resume_quotes=valid_quotes,
            confidence=new_confidence,
            confidence_level=new_level,
            reason=new_reason
        )
    else:
        return ev


def _ensure_all_requirements(
    evidence_map: Dict[str, Evidence],
    requirements: List[Requirement]
) -> Dict[str, Evidence]:
    """
    全RequirementのIDが辞書に存在することを保証
    存在しない場合は空のEvidenceを追加
    """
    for req in requirements:
        if req.req_id not in evidence_map:
            # 空のEvidenceを作成
            evidence_map[req.req_id] = Evidence(
                req_id=req.req_id,
                resume_quotes=[],
                confidence=0.0,
                confidence_level=ConfidenceLevel.NONE,
                reason="職務経歴書から該当する経験・スキルが見つかりませんでした"
            )

    return evidence_map


def _fallback_extract(
    resume_text: str,
    requirements: List[Requirement]
) -> List[Evidence]:
    """
    Fallback: キーワードベースで簡易抽出
    要件の説明からキーワードを抽出し、resume_text内で検索
    """
    evidence_list = []

    for req in requirements:
        # 要件からキーワードを抽出（簡易版：スペースで分割）
        keywords = []

        # 数字+年などのパターン
        year_match = re.search(r'(\d+)年', req.description)
        if year_match:
            keywords.append(year_match.group(0))

        # 技術キーワード（カタカナ、英語、一般的な技術用語）
        tech_keywords = re.findall(r'[A-Za-z]+|[ァ-ヶー]+', req.description)
        keywords.extend([k for k in tech_keywords if len(k) >= 2])

        # resume_text内でキーワードを検索
        found_quotes = []
        confidence = 0.0

        for keyword in keywords[:3]:  # 最大3つのキーワード
            # キーワードを含む文を抽出
            lines = resume_text.split('\n')
            for line in lines:
                if keyword in line and line.strip():
                    found_quotes.append(line.strip())
                    confidence += 0.2

        # confidenceを0.0〜1.0に正規化
        confidence = min(1.0, confidence)

        # confidence_levelを設定
        if confidence >= 0.7:
            level = ConfidenceLevel.HIGH
        elif confidence >= 0.4:
            level = ConfidenceLevel.MEDIUM
        elif confidence > 0.0:
            level = ConfidenceLevel.LOW
        else:
            level = ConfidenceLevel.NONE

        reason = f"キーワードマッチング（{len(found_quotes)}件の関連記述を発見）" if found_quotes else "該当する記述が見つかりませんでした"

        evidence_list.append(Evidence(
            req_id=req.req_id,
            resume_quotes=found_quotes[:3],  # 最大3件
            confidence=confidence,
            confidence_level=level,
            reason=reason
        ))

    return evidence_list


# ==================== テスト用コード ====================
if __name__ == "__main__":
    from f1_extract_requirements import extract_requirements
    from models import RequirementType

    # サンプル求人票（F1用）
    sample_job_text = """
【求人票】Webエンジニア募集

■必須スキル
・Python開発経験3年以上
・Webアプリケーション開発の実務経験

■歓迎スキル
・AWSなどクラウド環境での開発経験
・機械学習・データ分析の知識
    """

    # サンプル職務経歴書
    sample_resume_text = """
【職務経歴書】

■職務経歴
2019年〜現在：株式会社ABC
・Pythonを使用したWebアプリケーション開発に5年間従事
・Djangoフレームワークを用いたECサイトの構築
・AWS (EC2, S3, RDS) を活用したインフラ構築

■スキル
・Python（5年）、JavaScript（3年）
・Django, Flask, FastAPI
・AWS, Docker, Git
    """

    print("=" * 60)
    print("F1→F2 統合テスト")
    print("=" * 60)

    try:
        # F1: 要件抽出
        print("\n[Step 1] F1: 求人要件抽出")
        requirements = extract_requirements(
            job_text=sample_job_text,
            options={"max_must": 3, "max_want": 3}
        )
        print(f"✅ 抽出成功：{len(requirements)}件\n")

        for req in requirements:
            print(f"  [{req.req_id}] {req.category.value}: {req.description}")

        # F2: 根拠抽出
        print("\n[Step 2] F2: 根拠抽出")
        evidence_map = extract_evidence(
            resume_text=sample_resume_text,
            requirements=requirements,
            options={"verify_quotes": True}
        )
        print(f"✅ 抽出成功：{len(evidence_map)}件\n")

        for req_id, ev in evidence_map.items():
            print(f"[{req_id}] Confidence: {ev.confidence:.2f} ({ev.confidence_level.value})")
            print(f"  引用数: {len(ev.resume_quotes)}")
            if ev.resume_quotes:
                for i, quote in enumerate(ev.resume_quotes[:2], 1):
                    print(f"    {i}. {quote[:60]}...")
            print(f"  理由: {ev.reason}")
            print()

    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
