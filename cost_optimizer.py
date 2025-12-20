"""
AI応募適合度チェッカー - コスト最適化
プロンプト圧縮、キャッシュ管理
"""
import hashlib
from typing import List, Tuple
import re


def compress_text(text: str, max_length: int = 200) -> str:
    """
    テキストを圧縮（長い場合は先頭末尾の要点を残す）
    
    Args:
        text: 圧縮対象のテキスト
        max_length: 最大文字数
    
    Returns:
        str: 圧縮後のテキスト
    """
    if not text:
        return ""
    
    if len(text) <= max_length:
        return text
    
    # 先頭と末尾から要点を残す
    head_length = max_length // 2
    tail_length = max_length - head_length
    
    compressed = text[:head_length] + "..." + text[-tail_length:]
    return compressed


def limit_candidate_sentences(
    sentences: List[str],
    max_count: int = 30,
    max_length_per_sentence: int = 200
) -> List[str]:
    """
    候補文を上限N件に絞り、長い文は圧縮
    
    Args:
        sentences: 候補文のリスト
        max_count: 最大件数
        max_length_per_sentence: 1文あたりの最大文字数
    
    Returns:
        List[str]: 絞り込まれた候補文リスト
    """
    if not sentences:
        return []
    
    # 長い文を圧縮
    compressed_sentences = [
        compress_text(s, max_length_per_sentence) for s in sentences
    ]
    
    # 上限件数に絞る
    if len(compressed_sentences) > max_count:
        # 先頭からmax_count件を取得（重要度順に並んでいる想定）
        return compressed_sentences[:max_count]
    
    return compressed_sentences


def get_cache_key(
    job_text: str,
    resume_text: str,
    achievement_notes: str = None,
    options: dict = None
) -> str:
    """
    キャッシュキーを生成（入力テキストのハッシュ）
    
    Args:
        job_text: 求人票のテキスト
        resume_text: 職務経歴書のテキスト
        achievement_notes: 実績メモ（オプション）
        options: オプション辞書
    
    Returns:
        str: キャッシュキー（ハッシュ値）
    """
    # キャッシュキーに含める情報を文字列化
    cache_data = {
        "job_text": job_text.strip() if job_text else "",
        "resume_text": resume_text.strip() if resume_text else "",
        "achievement_notes": achievement_notes.strip() if achievement_notes else "",
        "options": {
            "max_must": options.get("max_must", 10) if options else 10,
            "max_want": options.get("max_want", 10) if options else 10,
            "strict_mode": options.get("strict_mode", False) if options else False,
            "model_name": options.get("model_name", None) if options else None,
        }
    }
    
    # 辞書をソートして文字列化（順序を固定）
    cache_str = str(sorted(cache_data.items()))
    
    # SHA256ハッシュを生成
    hash_obj = hashlib.sha256(cache_str.encode('utf-8'))
    return hash_obj.hexdigest()


def compress_resume_text(resume_text: str, max_length: int = 5000) -> str:
    """
    職務経歴書のテキストを圧縮（長すぎる場合）
    
    Args:
        resume_text: 職務経歴書のテキスト
        max_length: 最大文字数
    
    Returns:
        str: 圧縮後のテキスト
    """
    if not resume_text:
        return ""
    
    if len(resume_text) <= max_length:
        return resume_text
    
    # 長い場合は先頭と末尾から要点を残す
    head_length = max_length // 2
    tail_length = max_length - head_length
    
    compressed = resume_text[:head_length] + "\n\n[... 中略 ...]\n\n" + resume_text[-tail_length:]
    return compressed


def split_into_sentences(text: str) -> List[str]:
    """
    テキストを文単位で分割
    
    Args:
        text: 分割対象のテキスト
    
    Returns:
        List[str]: 文のリスト
    """
    if not text:
        return []
    
    # 改行で分割
    lines = text.split('\n')
    
    # 各ラインをさらに文単位で分割（。！？で区切る）
    sentences = []
    for line in lines:
        if not line.strip():
            continue
        
        # 文末記号で分割
        line_sentences = re.split(r'([。！？])', line)
        
        # 文末記号を含めて結合
        current_sentence = ""
        for i, part in enumerate(line_sentences):
            if part in ['。', '！', '？']:
                current_sentence += part
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                current_sentence = ""
            else:
                current_sentence += part
        
        # 残りの部分
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
    
    return sentences

