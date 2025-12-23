"""
F1: 求人票から要件を抽出
PydanticOutputParser + RetryWithErrorOutputParser で安定化
"""
import os
import re
from typing import List, Optional
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from models import Requirement, F1Output, RequirementType

# 環境変数読み込み
load_dotenv()


def extract_requirements(
    job_text: str,
    options: Optional[dict] = None
) -> List[Requirement]:
    """
    求人票から要件を抽出する（F1）

    Args:
        job_text: 求人票のテキスト
        options: オプション辞書
            - max_must: Must要件の最大件数（デフォルト10）
            - max_want: Want要件の最大件数（デフォルト10）
            - strict_mode: 曖昧一致を認めない（デフォルトFalse）
            - llm_provider: "openai" or "anthropic"（デフォルト "openai"）
            - model_name: モデル名（デフォルト gpt-4o-mini / claude-3-5-sonnet-20241022）
            - company_text: 企業情報（オプション、背景文脈として補助利用）

    Returns:
        List[Requirement]: 抽出された要件リスト
    """
    # オプションのデフォルト値
    if options is None:
        options = {}

    max_must = options.get("max_must", 10)
    max_want = options.get("max_want", 10)
    strict_mode = options.get("strict_mode", False)
    llm_provider = options.get("llm_provider", "openai")
    model_name = options.get("model_name", None)
    company_text = options.get("company_text", None)

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
        parser = PydanticOutputParser(pydantic_object=F1Output)

        # プロンプト作成
        strict_instruction = ""
        if strict_mode:
            strict_instruction = "\n注意：曖昧な表現や推測は避け、求人票に明示的に書かれている要件のみを抽出してください。"

        # 企業情報の扱いルール
        company_info_section = ""
        company_info_rules = ""
        if company_text and company_text.strip():
            # 企業情報を要約（長すぎる場合は先頭1000文字）
            company_text_trimmed = company_text.strip()[:1000] + "..." if len(company_text.strip()) > 1000 else company_text.strip()
            company_info_section = f"""
【企業情報（参考・背景文脈）】
{company_text_trimmed}
"""
            company_info_rules = """
会社情報の扱いルール（重要）：
- 企業情報は「背景文脈」として補助的に利用してください（求人票の理解を深めるため）
- **会社紹介・沿革・所在地などの情報は要件として抽出しない**
- 企業文化・価値観はWant要件に含めてもよいが、技術スキル・経験要件を優先
- 例：「フラットな組織」→ Want要件として抽出可（ただし技術要件を優先）
- 例：「設立2010年」「本社：東京都」→ 要件として抽出しない
- 求人票に明示されていない要件を企業情報から推測して抽出しない
"""

        prompt_template = PromptTemplate(
            template="""あなたは求人票分析の専門家です。以下の求人票から、Must要件とWant要件を抽出してください。

求人票：
{job_text}
{company_info_section}
抽出ルール：
1. Must要件：「必須」「〜以上」「経験必須」など、必ず満たすべき条件
2. Want要件：「歓迎」「尚可」「あれば尚良し」など、あると望ましい条件
3. 各要件には必ず求人票からの原文引用（job_quote）を含めること
4. Must要件は最大{max_must}件、Want要件は最大{max_want}件まで
5. importance（重要度）は1〜5で設定（5が最重要）
6. req_idは仮でM1,M2...、W1,W2...のように連番を振る（後で採番し直す）

重複・粒度に関する重要ルール：
7. **同じ内容の重複は絶対に禁止**：MustとWantで同じ技術・経験を重複させない
   例：「Python経験3年以上」がMustにあれば、Wantに「Python経験」を入れない
8. **粒度は適切に**：細かすぎず粗すぎず、1要件につき1つの明確なスキル/経験
   良い例：「Python開発経験3年以上」
   悪い例：「Python」「3年以上」を別々にする（細かすぎ）
   悪い例：「PythonとJavaとRubyの経験」（粗すぎ、分割すべき）
9. **MustとWantの使い分け**：同じ技術でもレベルが違う場合は明示
   例：Must「Python基本経験」、Want「Python上級（フレームワーク開発経験）」{company_info_rules}{strict_instruction}

{format_instructions}
""",
            input_variables=["job_text", "max_must", "max_want", "strict_instruction", "company_info_section", "company_info_rules"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        # LLM実行
        prompt = prompt_template.format(
            job_text=job_text,
            max_must=max_must,
            max_want=max_want,
            strict_instruction=strict_instruction,
            company_info_section=company_info_section,
            company_info_rules=company_info_rules
        )

        # LLM実行とパース（最大3回リトライ）
        max_retries = 3
        for attempt in range(max_retries):
            try:
                output = llm.invoke(prompt)
                result = parser.parse(output.content)
                requirements = result.requirements
                break
            except Exception as parse_error:
                if attempt == max_retries - 1:
                    # 最後の試行でも失敗した場合は例外を投げる
                    raise parse_error
                # リトライ

    except Exception as e:
        print(f"⚠️  LLM抽出に失敗、fallbackを使用: {e}")
        # Fallback: ルールベース抽出
        requirements = _fallback_extract(job_text, max_must, max_want)

    # 後処理：重複統合、ID採番とweight設定
    requirements = _merge_duplicate_requirements(requirements)
    requirements = _post_process_requirements(requirements)

    return requirements


def _merge_duplicate_requirements(requirements: List[Requirement]) -> List[Requirement]:
    """
    同じ意味の要件が複数出たら統合（Must/Wantの件数が無駄に増えるのを防ぐ）

    重複パターン:
    1. 同じカテゴリ内での重複 → 統合
    2. MustとWantをまたぐ重複 → Mustを優先して残す

    Args:
        requirements: 抽出された要件リスト

    Returns:
        List[Requirement]: 統合後の要件リスト
    """
    if len(requirements) <= 1:
        return requirements

    # ステップ1: MustとWantをまたぐ重複を検出し、Wantを除外
    must_reqs = [r for r in requirements if r.category == RequirementType.MUST]
    want_reqs = [r for r in requirements if r.category == RequirementType.WANT]

    filtered_want_reqs = []
    for want_req in want_reqs:
        # Mustに同じ内容がないかチェック
        has_duplicate_in_must = False
        for must_req in must_reqs:
            if _are_similar_requirements(want_req, must_req):
                # Mustに同じ内容があるので、このWantは除外
                has_duplicate_in_must = True
                break

        if not has_duplicate_in_must:
            filtered_want_reqs.append(want_req)

    # Mustとフィルタリング後のWantを結合
    requirements = must_reqs + filtered_want_reqs

    # ステップ2: 同じカテゴリ内での重複統合（既存ロジック）
    merged = []
    used_indices = set()

    for i, req1 in enumerate(requirements):
        if i in used_indices:
            continue

        # 同じカテゴリの要件を探す
        similar_reqs = [req1]
        for j, req2 in enumerate(requirements[i+1:], start=i+1):
            if j in used_indices:
                continue

            # 同じカテゴリで、意味が似ているかチェック
            if req1.category == req2.category and _are_similar_requirements(req1, req2):
                similar_reqs.append(req2)
                used_indices.add(j)

        # 統合
        if len(similar_reqs) > 1:
            # 最も詳細な説明を選ぶ、または統合
            merged_req = _merge_requirements(similar_reqs)
            merged.append(merged_req)
        else:
            merged.append(req1)

        used_indices.add(i)

    return merged


def _are_similar_requirements(req1: Requirement, req2: Requirement) -> bool:
    """
    2つの要件が同じ意味かどうかを判定

    Args:
        req1: 要件1
        req2: 要件2

    Returns:
        bool: 同じ意味ならTrue
    """
    desc1 = req1.description.lower()
    desc2 = req2.description.lower()

    # 1. 完全一致
    if desc1 == desc2:
        return True

    # 2. 一方が他方を含む（例: "Python経験" と "Python開発経験3年以上"）
    if desc1 in desc2 or desc2 in desc1:
        return True

    # 3. 技術キーワードが一致（カタカナ、英語）
    tech_keywords1 = set(re.findall(r'[A-Za-z]+|[ァ-ヶー]+', desc1))
    tech_keywords2 = set(re.findall(r'[A-Za-z]+|[ァ-ヶー]+', desc2))

    # 技術キーワードが2つ以上一致し、それが主要キーワードの場合
    common_tech = tech_keywords1 & tech_keywords2
    if len(common_tech) >= 2:
        # 長いキーワード（3文字以上）が含まれている場合
        long_common = {k for k in common_tech if len(k) >= 3}
        if long_common:
            return True

    # 4. 同じ技術の異なるレベル表現を検出
    # 例: "Python経験" と "Python上級経験" は似ているが、レベルが違うので別物として扱う
    # ただし、単に "Python" だけが一致する場合は重複とみなす
    if len(tech_keywords1) == 1 and len(tech_keywords2) == 1:
        if tech_keywords1 == tech_keywords2:
            return True

    # 5. キーワードの重複率が高い（80%以上に引き上げ、より厳格に）
    words1 = set(desc1.split())
    words2 = set(desc2.split())

    if len(words1) == 0 or len(words2) == 0:
        return False

    common_words = words1 & words2
    overlap_ratio = len(common_words) / max(len(words1), len(words2))

    # 80%以上のキーワードが一致し、かつ主要キーワードが一致
    if overlap_ratio >= 0.8:
        # 主要キーワード（3文字以上の単語）が一致するか
        important_words1 = {w for w in words1 if len(w) >= 3}
        important_words2 = {w for w in words2 if len(w) >= 3}
        if important_words1 & important_words2:
            return True

    return False


def _merge_requirements(requirements: List[Requirement]) -> Requirement:
    """
    複数の要件を1つに統合
    
    Args:
        requirements: 統合対象の要件リスト
        
    Returns:
        Requirement: 統合後の要件
    """
    if len(requirements) == 1:
        return requirements[0]
    
    # 最も詳細な説明を選ぶ（長い説明を優先）
    merged_desc = max(requirements, key=lambda r: len(r.description)).description
    
    # 最高の重要度を採用
    merged_importance = max(req.importance for req in requirements)
    
    # 最も詳細な引用を選ぶ
    merged_quote = max(requirements, key=lambda r: len(r.job_quote)).job_quote
    
    # 最初の要件のカテゴリとweightを使用
    category = requirements[0].category
    weight = requirements[0].weight
    
    return Requirement(
        req_id=requirements[0].req_id,  # 仮ID（後で採番し直す）
        category=category,
        description=merged_desc,
        importance=merged_importance,
        job_quote=merged_quote,
        weight=weight
    )


def _post_process_requirements(requirements: List[Requirement]) -> List[Requirement]:
    """
    後処理：IDをREQ_001...に採番、weightをmust=1.0/want=0.5に設定
    会社紹介っぽい要件をフィルタ
    """
    processed = []
    req_counter = 1

    for req in requirements:
        # 会社紹介っぽい要件をフィルタ
        if _is_company_intro_requirement(req):
            # 会社紹介っぽい要件は除外（ログ出力）
            print(f"⚠️  会社紹介っぽい要件を除外: {req.description}")
            continue
        
        # 新しいIDを採番
        new_id = f"REQ_{req_counter:03d}"

        # weightを設定
        weight = 1.0 if req.category == RequirementType.MUST else 0.5

        # 新しいRequirementオブジェクトを作成
        new_req = Requirement(
            req_id=new_id,
            category=req.category,
            description=req.description,
            importance=req.importance,
            job_quote=req.job_quote,
            weight=weight
        )

        processed.append(new_req)
        req_counter += 1

    return processed


def _is_company_intro_requirement(req: Requirement) -> bool:
    """
    会社紹介っぽい要件かどうかを判定
    
    Args:
        req: 要件
    
    Returns:
        bool: 会社紹介っぽい要件ならTrue
    """
    desc_lower = req.description.lower()
    quote_lower = req.job_quote.lower()
    
    # 会社紹介キーワード
    company_intro_keywords = [
        "設立", "創業", "本社", "所在地", "住所", "資本金", "従業員数",
        "沿革", "歴史", "事業内容", "事業領域", "サービス", "製品",
        "東京都", "大阪府", "神奈川県", "愛知県",  # 所在地
        "年", "月", "日",  # 日付（設立年月日など）
        "株式会社", "有限会社", "合同会社",  # 会社形態
    ]
    
    # 説明文または引用文に会社紹介キーワードが含まれているか
    for keyword in company_intro_keywords:
        if keyword in desc_lower or keyword in quote_lower:
            # ただし、技術要件として使われている場合は除外
            # 例：「設立10年の会社での経験」は技術要件として有効
            tech_keywords = ["経験", "スキル", "開発", "実装", "設計", "運用"]
            has_tech_context = any(tech in desc_lower for tech in tech_keywords)
            if not has_tech_context:
                return True
    
    # 所在地パターン（都道府県名のみ、または「〜県〜市」など）
    location_patterns = [
        r'[都道府県]',
        r'[市区町村]',
        r'[0-9]+-[0-9]+',  # 郵便番号
    ]
    for pattern in location_patterns:
        if re.search(pattern, desc_lower) or re.search(pattern, quote_lower):
            # 技術要件の文脈がない場合は除外
            tech_keywords = ["経験", "スキル", "開発", "実装", "設計", "運用"]
            has_tech_context = any(tech in desc_lower for tech in tech_keywords)
            if not has_tech_context:
                return True
    
    return False


def _fallback_extract(job_text: str, max_must: int, max_want: int) -> List[Requirement]:
    """
    Fallback: ルールベースで簡易抽出
    「必須」「歓迎」などの見出しや箇条書きを抽出
    """
    requirements = []
    lines = job_text.split('\n')

    current_category = None
    req_counter_must = 1
    req_counter_want = 1

    # 必須/歓迎のキーワードパターン
    must_patterns = [
        r'必須',
        r'required',
        r'必要な経験',
        r'応募資格',
        r'〜以上',
        r'経験.*年'
    ]

    want_patterns = [
        r'歓迎',
        r'preferred',
        r'尚可',
        r'あれば.*良し',
        r'望ましい'
    ]

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # カテゴリー判定
        if any(re.search(p, line, re.IGNORECASE) for p in must_patterns):
            # Must要件として抽出
            if req_counter_must <= max_must:
                requirements.append(Requirement(
                    req_id=f"M{req_counter_must}",
                    category=RequirementType.MUST,
                    description=line,
                    importance=3,
                    job_quote=line,
                    weight=1.0
                ))
                req_counter_must += 1

        elif any(re.search(p, line, re.IGNORECASE) for p in want_patterns):
            # Want要件として抽出
            if req_counter_want <= max_want:
                requirements.append(Requirement(
                    req_id=f"W{req_counter_want}",
                    category=RequirementType.WANT,
                    description=line,
                    importance=2,
                    job_quote=line,
                    weight=0.5
                ))
                req_counter_want += 1

    # 最低1件は抽出する
    if not requirements:
        requirements.append(Requirement(
            req_id="M1",
            category=RequirementType.MUST,
            description="求人票から要件を抽出できませんでした",
            importance=1,
            job_quote=job_text[:100],
            weight=1.0
        ))

    return requirements


# ==================== テスト用コード ====================
if __name__ == "__main__":
    # サンプル求人票
    sample_job_text = """
【求人票】Webエンジニア募集

■必須スキル
・Python開発経験3年以上
・Webアプリケーション開発の実務経験
・GitHubを使ったチーム開発経験

■歓迎スキル
・AWSなどクラウド環境での開発経験
・機械学習・データ分析の知識
・英語でのコミュニケーション能力

■業務内容
自社プロダクトのバックエンド開発をお任せします。
    """

    print("=" * 60)
    print("F1: 求人要件抽出テスト")
    print("=" * 60)

    # テスト実行
    try:
        requirements = extract_requirements(
            job_text=sample_job_text,
            options={
                "max_must": 5,
                "max_want": 5,
                "strict_mode": False,
                "llm_provider": "openai"  # or "anthropic"
            }
        )

        print(f"\n✅ 抽出成功：{len(requirements)}件\n")

        for req in requirements:
            print(f"[{req.req_id}] {req.category.value} (重要度:{req.importance}, 重み:{req.weight})")
            print(f"  説明: {req.description}")
            print(f"  引用: {req.job_quote[:50]}...")
            print()

    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
