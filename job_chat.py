"""
求人深掘りチャット
求人の「この要件は何を意味する？」を解釈、応募戦略を提案、応募メールの改善案を提示
"""
import os
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# 環境変数読み込み
load_dotenv()


def ask_job_chat(
    user_message: str,
    job_text: str,
    resume_text: str,
    company_text: Optional[str],
    requirements: List,
    matched: List,
    gaps: List,
    summary: str,
    chat_history: List[tuple],
    mode: str = "job_understanding",
    options: Optional[Dict[str, Any]] = None
) -> str:
    """
    求人深掘りチャットに質問する
    
    Args:
        user_message: ユーザーの質問
        job_text: 求人票のテキスト
        resume_text: 職務経歴書のテキスト（要約して渡す）
        company_text: 企業情報（オプション）
        requirements: 全要件リスト
        matched: マッチした要件と根拠のペア
        gaps: ギャップのある要件
        summary: スコアの総評
        chat_history: チャット履歴 [(user, assistant), ...]
        mode: チャットモード
            - "job_understanding": 求人理解
            - "email_improvement": 応募メール改善
            - "interview_questions": 面接質問作成
        options: オプション辞書
            - llm_provider: "openai" or "anthropic"（デフォルト "openai"）
            - model_name: モデル名（デフォルト gpt-4o-mini）
    
    Returns:
        str: アシスタントの応答
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
    if llm_provider == "anthropic":
        llm = ChatAnthropic(
            model=model_name,
            temperature=0.7,
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
    else:  # openai
        llm = ChatOpenAI(
            model=model_name,
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    
    # 入力値の検証とデフォルト値設定
    # None チェックと文字列型への変換
    job_text = str(job_text) if job_text is not None and job_text != "" else ""
    resume_text = str(resume_text) if resume_text is not None and resume_text != "" else ""
    company_text = str(company_text) if company_text is not None and company_text != "" else None
    summary = str(summary) if summary is not None and summary != "" else ""
    
    # 職務経歴書を要約（長い場合は先頭500文字）
    resume_summary = resume_text[:500] + "..." if len(resume_text) > 500 else resume_text
    
    # 分析結果を要約
    requirements_summary = "\n".join([
        f"- [{r.req_id}] {r.category.value}: {r.description}"
        for r in requirements[:10]  # 最大10件
    ]) if requirements else "要件なし"
    
    matched_summary = "\n".join([
        f"- [{m.requirement.req_id}] {m.requirement.description} (一致度: {m.evidence.confidence:.0%})"
        for m in matched[:5]  # 最大5件
    ]) if matched else "マッチした要件なし"
    
    gaps_summary = "\n".join([
        f"- [{g.requirement.req_id}] {g.requirement.description}"
        for g in gaps[:5]  # 最大5件
    ]) if gaps else "ギャップなし"
    
    # 企業情報の処理
    if company_text and isinstance(company_text, str) and company_text.strip():
        company_info_str = company_text
    else:
        company_info_str = "企業情報なし"
    
    # モードに応じたシステムプロンプトと出力フォーマット
    mode_configs = {
        "job_understanding": {
            "role": "求人理解の専門家",
            "description": "求人票の要件や業務内容を解釈し、応募者が理解すべきポイントを明確にします。",
            "output_format": """以下の構造で必ず回答してください：

## 解釈
[質問された要件や内容の具体的な解釈]

## 重要ポイント
[この要件が重要な理由、期待されるスキルや経験のレベル]

## 確認すべき点（質問案）
[面接や応募前に確認すべき質問を3〜5個、箇条書きで提示]

## 次のアクション
[この要件を満たすために取るべき行動（学習、経験の整理など）]"""
        },
        "email_improvement": {
            "role": "応募メール改善の専門家",
            "description": "応募メールの文面を改善し、より効果的な表現を提案します。",
            "output_format": """以下の構造で必ず回答してください：

## 改善方針
[現在の文面の問題点と改善の方向性]

## 改善後の文案（短い）
[改善後の文面を具体的に提示（100-200文字程度）]

## 修正理由
[なぜこの改善が効果的なのか、根拠を説明]

## 追加で必要な情報
[より良い文面を作るために確認すべき点や追加すべき情報]"""
        },
        "interview_questions": {
            "role": "面接質問作成の専門家",
            "description": "面接で想定される質問を作成し、回答の骨子を提示します。",
            "output_format": """以下の構造で必ず回答してください：

## 想定質問10個
[面接で想定される質問を10個、番号付きで提示]

## 狙い
[各質問の意図や評価ポイント]

## 回答の骨子（箇条書き）
[各質問に対する回答の骨子を箇条書きで提示（職務経歴に基づく）]

## 逆質問案5個
[面接官に質問すべき内容を5個、箇条書きで提示]"""
        }
    }
    
    # モードのデフォルト値
    if mode not in mode_configs:
        mode = "job_understanding"
    
    mode_config = mode_configs[mode]
    
    # システムプロンプト
    system_prompt = f"""あなたは求人深掘りチャットのアシスタントです。
モード: {mode_config["role"]}
{mode_config["description"]}

【あなたの役割】
- ユーザーの質問に対して、モードに応じた専門的なアドバイスを提供

【重要なルール】
- **不確実な推測は推測と明記**: 求人票に明示されていない情報は「推測ですが...」と明記
- **捏造禁止**: 職務経歴にないことを断定しない（「学習中」「計画中」など現実的な表現を推奨）
- **具体的に**: 例文、質問例、確認項目を提示
- **出力フォーマット**: 必ず以下の構造で回答してください

{mode_config["output_format"]}

【コンテキスト情報】
- 求人票、企業情報、職務経歴書（要約）、分析結果を参照
- 分析結果に基づいて具体的なアドバイスを提供
"""
    
    # コンテキスト情報を準備
    # job_textが空でない場合のみトリミング
    if job_text and len(job_text) > 1500:
        job_text_trimmed = job_text[:1500] + "..."
    else:
        job_text_trimmed = job_text if job_text else "求人票情報なし"
    
    context_info = f"""【求人票】
{job_text_trimmed}

【企業情報】
{company_info_str[:1000] + "..." if isinstance(company_info_str, str) and len(company_info_str) > 1000 else company_info_str}

【職務経歴書（要約）】
{resume_summary}

【分析結果】
総評: {summary[:300] if summary else '分析結果なし'}

抽出された要件:
{requirements_summary}

マッチした要件（強み）:
{matched_summary}

ギャップのある要件:
{gaps_summary}
"""
    
    # チャット履歴をメッセージ形式に変換
    messages = [SystemMessage(content=system_prompt)]
    
    # コンテキスト情報を最初のメッセージとして追加
    messages.append(HumanMessage(content=f"{context_info}\n\n上記の情報を参考に、ユーザーの質問に答えてください。"))
    
    # チャット履歴を追加（直近5件のみ）
    for user_msg, assistant_msg in chat_history[-5:]:
        messages.append(HumanMessage(content=user_msg))
        messages.append(AIMessage(content=assistant_msg))
    
    # 現在のユーザーメッセージを追加
    messages.append(HumanMessage(content=user_message))
    
    # LLM実行
    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        return f"エラーが発生しました: {str(e)}"








