"""
AI応募適合度チェッカー - チャット機能
求人内容の深掘り考察、応募文面改善の提案
"""
import os
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 環境変数読み込み
load_dotenv()


def get_chat_response(
    user_message: str,
    job_text: str,
    resume_text: str,
    company_info: Optional[str],
    analysis_result: Optional[Dict[str, Any]],
    chat_history: List[tuple],
    options: Optional[Dict[str, Any]] = None
) -> str:
    """
    チャット応答を生成
    
    Args:
        user_message: ユーザーのメッセージ
        job_text: 求人票のテキスト
        resume_text: 職務経歴書のテキスト
        company_info: 企業情報（オプション）
        analysis_result: 分析結果（オプション）
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
    
    # システムプロンプト
    system_prompt = """あなたはAI応募適合度チェッカーのチャットアシスタントです。
ユーザーが求人内容を深掘り考察したり、応募文面改善の提案を求めたりする際に、適切なアドバイスを提供してください。

【あなたの役割】
1. 求人内容の深掘り考察
   - 募集要件の詳細な解釈
   - 仕事内容の理解促進
   - 企業文化や価値観の分析（企業情報がある場合）

2. 応募文面改善の提案
   - 応募メール文面の改善点
   - 職務経歴書の書き方アドバイス
   - 面接対策の提案

【重要な注意事項】
- 職務経歴にないことを断定しない（「学習中」「計画中」など現実的な表現を推奨）
- 捏造を推奨しない
- 具体的で実践的なアドバイスを提供
- 簡潔で分かりやすい説明を心がける
"""
    
    # コンテキスト情報を準備
    context_info = f"""【求人票】
{job_text[:1500] if len(job_text) > 1500 else job_text}

【職務経歴書】
{resume_text[:1500] if len(resume_text) > 1500 else resume_text}
"""
    
    if company_info and company_info.strip():
        context_info += f"""
【企業情報】
{company_info[:1000] if len(company_info) > 1000 else company_info}
"""
    
    if analysis_result:
        summary = analysis_result.get('summary', '')
        score_total = analysis_result.get('score_total', 0)
        context_info += f"""
【分析結果サマリ】
総合スコア: {score_total}点
{summary[:500] if summary else '分析結果なし'}
"""
    
    # チャット履歴をメッセージ形式に変換
    messages = [SystemMessage(content=system_prompt)]
    
    # コンテキスト情報を最初のメッセージとして追加
    messages.append(HumanMessage(content=f"{context_info}\n\n上記の情報を参考に、ユーザーの質問に答えてください。"))
    
    # チャット履歴を追加
    for user_msg, assistant_msg in chat_history[-5:]:  # 直近5件のみ
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

