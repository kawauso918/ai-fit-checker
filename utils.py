"""
AI応募適合度チェッカー - ユーティリティ関数
引用検証などの共通処理
"""
import re


def normalize_text(text: str) -> str:
    """
    テキストを正規化（改行/連続空白/全角半角の差を吸収）
    
    Args:
        text: 正規化対象のテキスト
    
    Returns:
        str: 正規化後のテキスト
    """
    if not text:
        return ""
    
    # 1. 全角空白を半角空白に統一
    normalized = text.replace("　", " ")
    
    # 2. 改行を空白に統一（引用検証のため）
    normalized = normalized.replace("\n", " ").replace("\r", " ")
    
    # 3. 連続空白を1つに統一
    normalized = re.sub(r'\s+', ' ', normalized)
    
    # 4. 前後の空白を削除
    normalized = normalized.strip()
    
    return normalized


def verify_quote_in_text(quote: str, text: str) -> bool:
    """
    引用がテキスト内に存在するか検証（正規化後で比較）
    
    Args:
        quote: 検証対象の引用
        text: 検索対象のテキスト
    
    Returns:
        bool: 引用が見つかった場合True、見つからない場合False
    """
    if not quote or not text:
        return False
    
    # 正規化
    normalized_quote = normalize_text(quote)
    normalized_text = normalize_text(text)
    
    if not normalized_quote or not normalized_text:
        return False
    
    # 完全一致で検索
    if normalized_quote in normalized_text:
        return True
    
    # 部分一致（引用が長い場合、主要部分が含まれているか）
    # 引用が3単語以上の場合、70%以上の単語が一致すればOK
    quote_words = [w for w in normalized_quote.split() if len(w) >= 2]
    if len(quote_words) >= 3:
        matched_words = sum(1 for word in quote_words if word in normalized_text)
        if matched_words >= len(quote_words) * 0.7:
            return True
    
    return False





