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

        prompt_template = PromptTemplate(
            template="""あなたは求人票分析の専門家です。以下の求人票から、Must要件とWant要件を抽出してください。

求人票：
{job_text}

抽出ルール：
1. Must要件：「必須」「〜以上」「経験必須」など、必ず満たすべき条件
2. Want要件：「歓迎」「尚可」「あれば尚良し」など、あると望ましい条件
3. 各要件には必ず求人票からの原文引用（job_quote）を含めること
4. Must要件は最大{max_must}件、Want要件は最大{max_want}件まで
5. importance（重要度）は1〜5で設定（5が最重要）
6. req_idは仮でM1,M2...、W1,W2...のように連番を振る（後で採番し直す）{strict_instruction}

{format_instructions}
""",
            input_variables=["job_text", "max_must", "max_want", "strict_instruction"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        # LLM実行
        prompt = prompt_template.format(
            job_text=job_text,
            max_must=max_must,
            max_want=max_want,
            strict_instruction=strict_instruction
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
    
    Args:
        requirements: 抽出された要件リスト
        
    Returns:
        List[Requirement]: 統合後の要件リスト
    """
    if len(requirements) <= 1:
        return requirements
    
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
    
    # 完全一致
    if desc1 == desc2:
        return True
    
    # 一方が他方を含む（例: "Python経験" と "Python開発経験3年以上"）
    if desc1 in desc2 or desc2 in desc1:
        return True
    
    # キーワードの重複率が高い（70%以上）
    words1 = set(desc1.split())
    words2 = set(desc2.split())
    
    if len(words1) == 0 or len(words2) == 0:
        return False
    
    common_words = words1 & words2
    overlap_ratio = len(common_words) / max(len(words1), len(words2))
    
    # 70%以上のキーワードが一致し、かつ主要キーワードが一致
    if overlap_ratio >= 0.7:
        # 主要キーワード（技術名、年数など）が一致するか
        important_words1 = {w for w in words1 if len(w) >= 3 and not w.isdigit()}
        important_words2 = {w for w in words2 if len(w) >= 3 and not w.isdigit()}
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
    """
    processed = []
    req_counter = 1

    for req in requirements:
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
