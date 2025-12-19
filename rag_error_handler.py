"""
RAG関連のエラーハンドリング共通関数
"""
import os
import logging
from typing import Tuple, Optional, Dict, List

# ロガー設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# コンソールハンドラーを追加（既に設定されていない場合）
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def validate_rag_inputs(
    achievement_notes: Optional[str],
    require_api_key: bool = False
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    RAG入力の検証
    
    Args:
        achievement_notes: 実績メモのテキスト
        require_api_key: APIキーが必須かどうか（Trueの場合、キーがないとエラー）
    
    Returns:
        Tuple[bool, Optional[str], Optional[str]]: 
            (is_valid, error_message, warning_message)
            is_valid: TrueならRAGを使用可能、Falseなら使用不可
            error_message: エラーメッセージ（処理を停止すべき場合）
            warning_message: 警告メッセージ（処理は継続可能）
    """
    # 実績メモが空の場合
    if not achievement_notes or not achievement_notes.strip():
        logger.info("実績メモが空のため、RAG検索をスキップします")
        return False, None, None
    
    # テキスト長チェック（最大15000文字）
    MAX_TEXT_LENGTH = 15000
    if len(achievement_notes) > MAX_TEXT_LENGTH:
        warning_msg = f"実績メモが長すぎます（{len(achievement_notes)}文字）。最初の{MAX_TEXT_LENGTH}文字のみを使用します。"
        logger.warning(warning_msg)
        return True, None, warning_msg
    
    # OpenAI APIキーの確認
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        if require_api_key:
            error_msg = (
                "OPENAI_API_KEYが設定されていません。\n\n"
                "RAG検索を使用するには、環境変数OPENAI_API_KEYを設定してください。\n"
                "設定方法:\n"
                "- macOS/Linux: `export OPENAI_API_KEY=your_api_key`\n"
                "- Windows: `set OPENAI_API_KEY=your_api_key`\n"
                "- .envファイル: `OPENAI_API_KEY=your_api_key`"
            )
            logger.error("OPENAI_API_KEYが設定されていません（RAG検索必須）")
            return False, error_msg, None
        else:
            warning_msg = "OPENAI_API_KEYが設定されていません。RAG検索をスキップします。"
            logger.warning(warning_msg)
            return False, None, warning_msg
    
    return True, None, None


def handle_rag_initialization_error(
    error: Exception,
    error_type: str = "unknown"
) -> Tuple[Dict, str]:
    """
    RAG初期化エラーのハンドリング
    
    Args:
        error: 発生した例外
        error_type: エラーの種類（"embeddings", "vectorstore", "dependency", "io"など）
    
    Returns:
        Tuple[Dict, str]: (空のrag_evidence辞書, エラーメッセージ)
    """
    error_messages = {
        "embeddings": "OpenAI Embeddingsの初期化に失敗しました",
        "vectorstore": "ベクトルストアの初期化に失敗しました",
        "dependency": "必要な依存パッケージが不足しています",
        "io": "ファイル/IO操作に失敗しました",
        "unknown": "RAG検索の初期化に失敗しました"
    }
    
    base_message = error_messages.get(error_type, error_messages["unknown"])
    error_msg = f"{base_message}: {str(error)}"
    
    logger.error(f"RAG初期化エラー ({error_type}): {error}", exc_info=True)
    
    return {}, error_msg


def handle_rag_search_error(
    req_id: str,
    error: Exception
) -> List:
    """
    RAG検索エラーのハンドリング（個別の要件に対する検索失敗）
    
    Args:
        req_id: 要件ID
        error: 発生した例外
    
    Returns:
        List: 空のリスト（エラー時は結果なし）
    """
    logger.warning(f"RAG検索エラー（req_id={req_id}）: {error}")
    return []


def get_rag_status(
    achievement_notes: Optional[str],
    rag_error_message: Optional[str],
    rag_evidence_count: int = 0
) -> Tuple[str, str]:
    """
    RAG状態を取得（UI表示用）
    
    Args:
        achievement_notes: 実績メモのテキスト
        rag_error_message: RAGエラーメッセージ
        rag_evidence_count: RAG検索で取得した根拠候補数
    
    Returns:
        Tuple[str, str]: (status, message)
            status: "enabled", "disabled", "error", "empty"
            message: 表示メッセージ
    """
    if not achievement_notes or not achievement_notes.strip():
        return "empty", "実績メモが入力されていません（RAG検索は無効）"
    
    if rag_error_message:
        return "error", f"RAG検索エラー: {rag_error_message}"
    
    if rag_evidence_count > 0:
        return "enabled", f"RAG検索が有効です（{rag_evidence_count}件の根拠候補を取得）"
    else:
        return "disabled", "RAG検索は実行されましたが、根拠候補が見つかりませんでした"

