"""
F4: 改善案を生成
PydanticOutputParser + Must優先ギャップ絞り込み
"""
import os
from typing import List, Optional, Dict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from models import (
    Requirement,
    RequirementWithEvidence,
    Gap,
    Improvements,
    ResumeEdit,
    ActionItem,
    F4Output,
    RequirementType
)

# 環境変数読み込み
load_dotenv()


def generate_improvements(
    job_text: str,
    resume_text: str,
    requirements: List[Requirement],
    matched: List[RequirementWithEvidence],
    gaps: List[Gap],
    options: Optional[dict] = None
) -> Improvements:
    """
    ギャップ分析から改善案を生成する（F4）

    Args:
        job_text: 求人票のテキスト
        resume_text: 職務経歴書のテキスト
        requirements: 全要件リスト
        matched: マッチした要件と根拠のペア
        gaps: ギャップのある要件
        options: オプション辞書
            - llm_provider: "openai" or "anthropic"（デフォルト "openai"）
            - model_name: モデル名
            - max_gaps: 最大ギャップ件数（デフォルト5）

    Returns:
        Improvements: 改善案
    """
    # オプションのデフォルト値
    if options is None:
        options = {}

    llm_provider = options.get("llm_provider", "openai")
    model_name = options.get("model_name", None)
    max_gaps = options.get("max_gaps", 5)

    # ギャップをMust優先でソート、上位N件を選択
    sorted_gaps = _prioritize_gaps(gaps, max_count=max_gaps)

    # LLMの初期化
    try:
        if llm_provider == "anthropic":
            llm = ChatAnthropic(
                model=model_name or "claude-3-5-sonnet-20241022",
                temperature=0.2,  # 少し創造性を持たせる
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
        else:  # openai
            llm = ChatOpenAI(
                model=model_name or "gpt-4o-mini",
                temperature=0.2,
                api_key=os.getenv("OPENAI_API_KEY")
            )

        # パーサー設定
        parser = PydanticOutputParser(pydantic_object=F4Output)

        # ギャップ情報を文字列化
        gaps_str = "\n".join([
            f"[{g.requirement.req_id}] {g.requirement.category.value}: {g.requirement.description}\n  理由: {g.evidence.reason}"
            for g in sorted_gaps
        ])

        # マッチ情報を文字列化（参考情報）
        matched_str = "\n".join([
            f"[{m.requirement.req_id}] {m.requirement.description}"
            for m in matched[:3]  # 最大3件
        ])

        # プロンプト作成
        prompt_template = PromptTemplate(
            template="""あなたはキャリアアドバイザーです。以下の情報をもとに、求職者が求人票の要件を満たすための改善案を提案してください。

求人票（抜粋）：
{job_text}

職務経歴書（抜粋）：
{resume_text}

現在マッチしている要件（参考）：
{matched_str}

ギャップのある要件（改善対象）：
{gaps_str}

改善案作成ルール：

1. **resume_edits（職務経歴書の編集・追記案）**
   - 既に持っている経験・スキルを強調する場合：edit_type="emphasize"
   - 新たに追記すべき内容がある場合：edit_type="add"
   - 書き換えが必要な場合：edit_type="rewrite"
   - templateには「何を書くべきか」の項目テンプレート
   - exampleには具体的な記述例を提示

2. **action_items（行動計画）**
   - Must要件の不足は優先度A（最優先・短期）
   - Want要件の不足は優先度B（中期）またはC（長期）
   - 学習、資格取得、実績作りなど具体的な行動を提案
   - estimated_impactは効果の高さ（High/Medium/Low）

3. **overall_strategy（全体戦略）**
   - 改善の方向性を1〜2文で要約
   - 「まず〜、次に〜」のような優先順位を示す

**重要**: 経験がないものは捏造せず、「学習」「実績作り」などの現実的な行動計画を提案すること。

{format_instructions}
""",
            input_variables=["job_text", "resume_text", "matched_str", "gaps_str"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        # job/resumeを必要最小限にカット（長文で壊れやすい場合）
        job_text_trimmed = _trim_job_text(job_text, sorted_gaps)
        resume_text_trimmed = _trim_resume_text(resume_text, sorted_gaps)
        
        # LLM実行とパース（最大3回リトライ）
        max_retries = 3
        for attempt in range(max_retries):
            try:
                prompt = prompt_template.format(
                    job_text=job_text_trimmed,
                    resume_text=resume_text_trimmed,
                    matched_str=matched_str,
                    gaps_str=gaps_str
                )
                output = llm.invoke(prompt)
                result = parser.parse(output.content)
                improvements = result.improvements
                break
            except Exception as parse_error:
                if attempt == max_retries - 1:
                    # 最後の試行でも失敗した場合は例外を投げる
                    raise parse_error
                # リトライ

    except Exception as e:
        print(f"⚠️  LLM生成に失敗、fallbackを使用: {e}")
        # Fallback: ルールベース生成
        improvements = _fallback_generate(sorted_gaps)

    return improvements


def _trim_job_text(job_text: str, gaps: List[Gap], max_length: int = 800) -> str:
    """
    job_textを必要最小限にカット（ギャップに関連する部分を優先）
    
    Args:
        job_text: 求人票のテキスト
        gaps: ギャップリスト
        max_length: 最大文字数
        
    Returns:
        str: カット後のテキスト
    """
    if len(job_text) <= max_length:
        return job_text
    
    # ギャップに関連する部分を抽出
    relevant_parts = []
    for gap in gaps:
        quote = gap.requirement.job_quote
        if quote and quote in job_text:
            # 引用の前後100文字を取得
            idx = job_text.find(quote)
            start = max(0, idx - 100)
            end = min(len(job_text), idx + len(quote) + 100)
            relevant_parts.append((start, end))
    
    # 重複を除去してソート
    relevant_parts = sorted(set(relevant_parts))
    
    # 関連部分を結合
    if relevant_parts:
        trimmed = ""
        for start, end in relevant_parts:
            trimmed += job_text[start:end] + "\n\n"
        
        # 長すぎる場合はさらにカット
        if len(trimmed) > max_length:
            trimmed = trimmed[:max_length] + "..."
        
        return trimmed
    else:
        # 関連部分が見つからない場合は先頭を返す
        return job_text[:max_length] + "..."


def _trim_resume_text(resume_text: str, gaps: List[Gap], max_length: int = 800) -> str:
    """
    resume_textを必要最小限にカット（ギャップに関連する部分を優先）
    
    Args:
        resume_text: 職務経歴書のテキスト
        gaps: ギャップリスト
        max_length: 最大文字数
        
    Returns:
        str: カット後のテキスト
    """
    if len(resume_text) <= max_length:
        return resume_text
    
    # ギャップの要件に関連するキーワードを抽出
    keywords = []
    for gap in gaps:
        desc_words = gap.requirement.description.split()
        keywords.extend([w for w in desc_words if len(w) >= 2])
    
    # キーワードを含む行を優先的に抽出
    lines = resume_text.split('\n')
    relevant_lines = []
    other_lines = []
    
    for line in lines:
        if any(kw.lower() in line.lower() for kw in keywords):
            relevant_lines.append(line)
        else:
            other_lines.append(line)
    
    # 関連行を優先して結合
    trimmed = "\n".join(relevant_lines)
    
    # まだ余裕がある場合は他の行も追加
    if len(trimmed) < max_length:
        remaining = max_length - len(trimmed)
        trimmed += "\n" + "\n".join(other_lines)[:remaining]
    
    # 長すぎる場合はカット
    if len(trimmed) > max_length:
        trimmed = trimmed[:max_length] + "..."
    
    return trimmed


def _prioritize_gaps(gaps: List[Gap], max_count: int = 5) -> List[Gap]:
    """
    ギャップをMust優先でソート、上位N件を選択

    Args:
        gaps: ギャップリスト
        max_count: 最大件数

    Returns:
        List[Gap]: ソート済みギャップ（上位N件）
    """
    # Must要件を優先（category, importanceの降順）
    sorted_gaps = sorted(
        gaps,
        key=lambda g: (
            0 if g.requirement.category == RequirementType.MUST else 1,
            -g.requirement.importance
        )
    )

    return sorted_gaps[:max_count]


def _fallback_generate(gaps: List[Gap]) -> Improvements:
    """
    Fallback: ルールベースで簡易的な改善案を生成

    Args:
        gaps: ギャップリスト

    Returns:
        Improvements: 改善案
    """
    resume_edits = []
    action_items = []

    for i, gap in enumerate(gaps[:5]):
        req = gap.requirement
        is_must = req.category == RequirementType.MUST

        # ResumeEdit生成
        resume_edits.append(ResumeEdit(
            target_gap=req.req_id,
            edit_type="add",
            template=f"【{req.description}に関する経験】",
            example=f"{req.description}に関するプロジェクトや学習経験を記載してください。"
        ))

        # ActionItem生成
        priority = "A" if is_must else ("B" if i < 2 else "C")
        estimated_impact = "High" if is_must else "Medium"

        action_items.append(ActionItem(
            priority=priority,
            action=f"{req.description}に関するスキルを習得する",
            rationale=f"求人票で{'必須' if is_must else '歓迎'}とされているため",
            estimated_impact=estimated_impact
        ))

    # Overall strategy
    must_gaps = [g for g in gaps if g.requirement.category == RequirementType.MUST]
    if must_gaps:
        strategy = f"まずMust要件の不足（{len(must_gaps)}件）を埋めることを最優先に、学習や実績作りに取り組んでください。"
    else:
        strategy = "Want要件を強化することで、さらに適合度を高めることができます。"

    return Improvements(
        resume_edits=resume_edits,
        action_items=action_items,
        overall_strategy=strategy
    )


# ==================== テスト用コード ====================
if __name__ == "__main__":
    from models import Requirement, Evidence, RequirementType, ConfidenceLevel

    # ダミーデータ作成
    print("=" * 60)
    print("F4: 改善案生成テスト（ダミーデータ使用）")
    print("=" * 60)

    # サンプル求人票
    sample_job_text = """
【求人票】Webエンジニア募集

■必須スキル
・Python開発経験3年以上
・Webアプリケーション開発の実務経験
・Docker/Kubernetesの実務経験

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

    # ダミー要件（F1の出力相当）
    requirements = [
        Requirement(
            req_id="REQ_001",
            category=RequirementType.MUST,
            description="Python開発経験3年以上",
            importance=5,
            job_quote="Python開発経験3年以上",
            weight=1.0
        ),
        Requirement(
            req_id="REQ_002",
            category=RequirementType.MUST,
            description="Docker/Kubernetesの実務経験",
            importance=4,
            job_quote="Docker/Kubernetesの実務経験",
            weight=1.0
        ),
        Requirement(
            req_id="REQ_003",
            category=RequirementType.WANT,
            description="機械学習・データ分析の知識",
            importance=3,
            job_quote="機械学習・データ分析の知識",
            weight=0.5
        ),
    ]

    # ダミーマッチ（F3の出力相当）
    matched = [
        RequirementWithEvidence(
            requirement=requirements[0],
            evidence=Evidence(
                req_id="REQ_001",
                resume_quotes=["Pythonを使用したWebアプリケーション開発に5年間従事"],
                confidence=1.0,
                confidence_level=ConfidenceLevel.HIGH,
                reason="5年間のPython経験が明記されている"
            )
        )
    ]

    # ダミーギャップ（F3の出力相当）
    gaps = [
        Gap(
            requirement=requirements[1],
            evidence=Evidence(
                req_id="REQ_002",
                resume_quotes=["Docker"],
                confidence=0.3,
                confidence_level=ConfidenceLevel.LOW,
                reason="Dockerの記載はあるがKubernetesの実務経験が不明"
            )
        ),
        Gap(
            requirement=requirements[2],
            evidence=Evidence(
                req_id="REQ_003",
                resume_quotes=[],
                confidence=0.0,
                confidence_level=ConfidenceLevel.NONE,
                reason="機械学習・データ分析に関する記載がない"
            )
        ),
    ]

    try:
        # F4: 改善案生成
        print("\n[実行] F4: 改善案生成")
        improvements = generate_improvements(
            job_text=sample_job_text,
            resume_text=sample_resume_text,
            requirements=requirements,
            matched=matched,
            gaps=gaps,
            options={"max_gaps": 5}
        )

        print(f"\n{'='*60}")
        print("📋 改善案")
        print(f"{'='*60}")

        print(f"\n【全体戦略】\n{improvements.overall_strategy}\n")

        # ResumeEdits
        if improvements.resume_edits:
            print(f"\n✏️  職務経歴書の編集・追記案（{len(improvements.resume_edits)}件）")
            for i, edit in enumerate(improvements.resume_edits, 1):
                print(f"\n  {i}. 対象: {edit.target_gap} ({edit.edit_type})")
                print(f"     テンプレート: {edit.template}")
                print(f"     例: {edit.example[:100]}...")

        # ActionItems
        if improvements.action_items:
            print(f"\n🎯 行動計画（{len(improvements.action_items)}件）")
            for i, item in enumerate(improvements.action_items, 1):
                print(f"\n  {i}. [優先度{item.priority}] {item.action}")
                print(f"     根拠: {item.rationale}")
                print(f"     効果: {item.estimated_impact}")

    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
