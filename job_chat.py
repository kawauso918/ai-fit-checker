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
    
    company_info_str = company_text if company_text and company_text.strip() else "企業情報なし"
    
    # システムプロンプト
    system_prompt = """あなたは求人深掘りチャットのアシスタントです。
ユーザーが求人内容を深掘りしたり、応募戦略や応募メールの改善案を求めたりする際に、適切なアドバイスを提供してください。

【あなたの役割】
1. **求人の解釈**: 「この要件は何を意味する？」を具体的に説明
2. **応募戦略の提案**: 強調すべき点/避けるべき点を提案
3. **応募メールの改善案**: 言い回し・構成の改善案を提示

【重要なルール】
- **不確実な推測は推測と明記**: 求人票に明示されていない情報は「推測ですが...」と明記
- **捏造禁止**: 職務経歴にないことを断定しない（「学習中」「計画中」など現実的な表現を推奨）
- **具体的に**: 例文、質問例、確認項目を提示
- **返信フォーマット**: 以下の構造で回答
  1. 解釈（質問に対する回答）
  2. 確認すべき点（面接/応募前に聞く質問）
  3. 応募文面の改善案（短い例文、該当する場合）

【コンテキスト情報】
- 求人票、企業情報、職務経歴書（要約）、分析結果を参照
- 分析結果に基づいて具体的なアドバイスを提供
"""
    
    # コンテキスト情報を準備
    context_info = f"""【求人票】
{job_text[:1500] if len(job_text) > 1500 else job_text}

【企業情報】
{company_info_str[:1000] if len(company_info_str) > 1000 else company_info_str}

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

