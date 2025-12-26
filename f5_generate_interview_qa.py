"""
F5: 面接想定Q&Aを生成
分析結果から面接で聞かれそうな質問と回答の骨子を生成
"""
import os
from typing import List, Optional
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from models import (
    RequirementWithEvidence,
    Gap,
    InterviewQA,
    InterviewQAs,
    F5Output
)

# 環境変数読み込み
load_dotenv()


def generate_interview_qa(
    job_text: str,
    resume_text: str,
    matched: List[RequirementWithEvidence],
    gaps: List[Gap],
    summary: str,
    options: Optional[dict] = None
) -> InterviewQAs:
    """
    分析結果から面接想定Q&Aを生成する（F5）

    Args:
        job_text: 求人票のテキスト
        resume_text: 職務経歴書のテキスト
        matched: マッチした要件と根拠のペア
        gaps: ギャップのある要件
        summary: スコアの総評
        options: オプション辞書
            - llm_provider: "openai" or "anthropic"（デフォルト "openai"）
            - model_name: モデル名（デフォルト gpt-4o-mini）

    Returns:
        InterviewQAs: 面接Q&A
    """
    # オプションのデフォルト値
    if options is None:
        options = {}

    llm_provider = options.get("llm_provider", "openai")
    model_name = options.get("model_name", None)
    
    # コスト重視のため、miniモデルをデフォルトに
    if not model_name:
        if llm_provider == "anthropic":
            model_name = "claude-3-haiku-20240307"  # より安価なモデル
        else:
            model_name = "gpt-4o-mini"  # デフォルト

    # LLMの初期化
    try:
        if llm_provider == "anthropic":
            llm = ChatAnthropic(
                model=model_name,
                temperature=0.3,  # 少し創造性を持たせる
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
        else:  # openai
            llm = ChatOpenAI(
                model=model_name,
                temperature=0.3,
                api_key=os.getenv("OPENAI_API_KEY")
            )

        # パーサー設定
        parser = PydanticOutputParser(pydantic_object=F5Output)

        # マッチ情報を文字列化
        matched_str = "\n".join([
            f"- {m.requirement.description} (一致度: {m.evidence.confidence:.0%})"
            for m in matched[:5]  # 最大5件
        ])

        # ギャップ情報を文字列化
        gaps_str = "\n".join([
            f"- {g.requirement.description} ({g.requirement.category.value}): {g.evidence.reason}"
            for g in gaps[:5]  # 最大5件
        ])

        # プロンプト作成
        prompt_template = PromptTemplate(
            template="""あなたは面接官です。以下の分析結果をもとに、面接で聞かれそうな質問を10問程度生成してください。

求人票（抜粋）：
{job_text}

職務経歴書（抜粋）：
{resume_text}

分析結果サマリー：
{summary}

マッチした要件（強み）：
{matched_str}

ギャップのある要件（不足点）：
{gaps_str}

質問生成ルール：
1. **質問の種類**：
   - 強み深掘り（3-4問）：マッチした要件について、具体的な経験や成果を聞く質問
   - ギャップ突っ込み（3-4問）：不足している要件について、どう対応するか聞く質問
   - 志望動機寄せ（2-3問）：求人票と職務経歴の関連性、志望動機を聞く質問

2. **回答の骨子（answer_outline）**：
   - 職務経歴に記載がある内容：具体的な経験・成果・数値を含める
   - 職務経歴に記載がない内容：「学習中」「計画中」「今後取り組みたい」など現実的な表現を使う
   - 捏造は絶対に禁止：職務経歴にない経験を「経験がある」と書かない
   - 箇条書きで3-5項目程度

3. **質問の具体性**：
   - 抽象的すぎず、具体的な経験や行動を聞く質問
   - 「なぜ」「どのように」「どのような成果」など深掘りする質問

**重要**: 職務経歴にない内容は「学習中/計画」として回答骨子を作ること。捏造は絶対に禁止。

{format_instructions}
""",
            input_variables=["job_text", "resume_text", "summary", "matched_str", "gaps_str"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        # テキストを必要最小限にカット
        job_text_trimmed = job_text[:1000] if len(job_text) > 1000 else job_text
        resume_text_trimmed = resume_text[:1000] if len(resume_text) > 1000 else resume_text

        # LLM実行とパース（最大3回リトライ）
        max_retries = 3
        for attempt in range(max_retries):
            try:
                prompt = prompt_template.format(
                    job_text=job_text_trimmed,
                    resume_text=resume_text_trimmed,
                    summary=summary,
                    matched_str=matched_str,
                    gaps_str=gaps_str
                )
                output = llm.invoke(prompt)
                result = parser.parse(output.content)
                interview_qas = result.interview_qas
                break
            except Exception as parse_error:
                if attempt == max_retries - 1:
                    # 最後の試行でも失敗した場合は例外を投げる
                    raise parse_error
                # リトライ

    except Exception as e:
        print(f"⚠️  LLM生成に失敗、fallbackを使用: {e}")
        # Fallback: ルールベース生成
        interview_qas = _fallback_generate(matched, gaps)

    return interview_qas


def _fallback_generate(
    matched: List[RequirementWithEvidence],
    gaps: List[Gap]
) -> InterviewQAs:
    """
    Fallback: ルールベースで簡易的なQ&Aを生成

    Args:
        matched: マッチした要件
        gaps: ギャップのある要件

    Returns:
        InterviewQAs: 面接Q&A
    """
    qa_list = []

    # 強み深掘り（最大3問）
    for i, m in enumerate(matched[:3], 1):
        qa_list.append(InterviewQA(
            question=f"{m.requirement.description}について、具体的な経験を教えてください。",
            answer_outline=[
                "職務経歴書に記載した経験を具体的に説明",
                "使用した技術やツール",
                "達成した成果や数値"
            ]
        ))

    # ギャップ突っ込み（最大3問）
    for i, g in enumerate(gaps[:3], 1):
        qa_list.append(InterviewQA(
            question=f"{g.requirement.description}について、どのように対応しますか？",
            answer_outline=[
                "現状の理解",
                "学習計画や取り組み方針",
                "今後の目標"
            ]
        ))

    # 志望動機寄せ（2問）
    qa_list.append(InterviewQA(
        question="この求人に応募した理由を教えてください。",
        answer_outline=[
            "求人票のどの点に魅力を感じたか",
            "自分の経験やスキルとの関連性",
            "今後のキャリアプラン"
        ]
    ))

    qa_list.append(InterviewQA(
        question="当社でどのような貢献ができますか？",
        answer_outline=[
            "強みを活かせる領域",
            "具体的な貢献内容",
            "チームへの価値提供"
        ]
    ))

    return InterviewQAs(qa_list=qa_list[:10])  # 最大10問


# ==================== テスト用コード ====================
if __name__ == "__main__":
    from models import Requirement, Evidence, RequirementType, ConfidenceLevel

    # ダミーデータ
    matched = [
        RequirementWithEvidence(
            requirement=Requirement(
                req_id="REQ_001",
                category=RequirementType.MUST,
                description="Python開発経験3年以上",
                importance=5,
                job_quote="Python開発経験3年以上",
                weight=1.0
            ),
            evidence=Evidence(
                req_id="REQ_001",
                resume_quotes=["Pythonを使用したWebアプリケーション開発に5年間従事"],
                confidence=1.0,
                confidence_level=ConfidenceLevel.HIGH,
                reason="5年間のPython経験が明記されている"
            )
        )
    ]

    gaps = [
        Gap(
            requirement=Requirement(
                req_id="REQ_002",
                category=RequirementType.MUST,
                description="Docker/Kubernetesの実務経験",
                importance=4,
                job_quote="Docker/Kubernetesの実務経験",
                weight=1.0
            ),
            evidence=Evidence(
                req_id="REQ_002",
                resume_quotes=[],
                confidence=0.0,
                confidence_level=ConfidenceLevel.NONE,
                reason="Docker/Kubernetesに関する記載がない"
            )
        )
    ]

    sample_job_text = "Python開発経験3年以上、Docker/Kubernetesの実務経験"
    sample_resume_text = "Pythonを使用したWebアプリケーション開発に5年間従事"
    sample_summary = "総合適合度は中程度です（50点）。Must要件のうち1件が不足しています。"

    try:
        interview_qas = generate_interview_qa(
            job_text=sample_job_text,
            resume_text=sample_resume_text,
            matched=matched,
            gaps=gaps,
            summary=sample_summary,
            options={"llm_provider": "openai"}
        )

        print(f"\n{'='*60}")
        print("📋 面接想定Q&A")
        print(f"{'='*60}")

        for i, qa in enumerate(interview_qas.qa_list, 1):
            print(f"\n{i}. {qa.question}")
            print("   回答の骨子:")
            for outline in qa.answer_outline:
                print(f"   - {outline}")

    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()













