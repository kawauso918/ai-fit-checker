"""
スモークテスト: 分析処理の基本動作確認
sample_inputs.mdからサンプルを読み込み、RAGなし/ありでテスト
"""
import os
import sys
import re
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app import run_analysis_core


def load_sample_inputs():
    """
    sample_inputs.mdからサンプルデータを読み込む
    
    Returns:
        tuple: (job_text, resume_text)
    """
    sample_file = project_root / "sample_inputs.md"
    
    if not sample_file.exists():
        raise FileNotFoundError(f"sample_inputs.mdが見つかりません: {sample_file}")
    
    with open(sample_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 求人票を抽出（```で囲まれた部分）
    job_match = re.search(r"## サンプル1: 求人票\s+```(.*?)```", content, re.DOTALL)
    if not job_match:
        raise ValueError("求人票のサンプルが見つかりません")
    job_text = job_match.group(1).strip()
    
    # 職務経歴を抽出（```で囲まれた部分）
    resume_match = re.search(r"## サンプル1: 職務経歴\s+```(.*?)```", content, re.DOTALL)
    if not resume_match:
        raise ValueError("職務経歴のサンプルが見つかりません")
    resume_text = resume_match.group(1).strip()
    
    return job_text, resume_text


def test_case_1_no_rag():
    """
    ケース1: RAGなし（実績メモ空）で分析実行
    必須キーが揃うことをassert
    """
    print("=" * 60)
    print("ケース1: RAGなし（実績メモ空）")
    print("=" * 60)
    
    job_text, resume_text = load_sample_inputs()
    
    # 分析実行
    result = run_analysis_core(
        job_text=job_text,
        resume_text=resume_text,
        achievement_notes=None,  # RAGなし
        emphasis_axes=[],
        options={}
    )
    
    # 必須キーの存在を確認
    required_keys = [
        "timestamp",
        "execution_time",
        "requirements",
        "evidence_map",
        "score_total",
        "score_must",
        "score_want",
        "matched",
        "gaps",
        "summary",
        "improvements",
        "interview_qas",
    ]
    
    for key in required_keys:
        assert key in result, f"必須キー '{key}' が存在しません"
        print(f"✅ {key}: 存在確認")
    
    # 型チェック
    assert isinstance(result["score_total"], (int, float)), "score_totalは数値である必要があります"
    assert isinstance(result["score_must"], (int, float)), "score_mustは数値である必要があります"
    assert isinstance(result["score_want"], (int, float)), "score_wantは数値である必要があります"
    assert isinstance(result["matched"], list), "matchedはリストである必要があります"
    assert isinstance(result["gaps"], list), "gapsはリストである必要があります"
    assert len(result["requirements"]) > 0, "要件が抽出されていません"
    
    print(f"\n✅ ケース1完了:")
    print(f"  - 総合スコア: {result['score_total']}点")
    print(f"  - Mustスコア: {result['score_must']}点")
    print(f"  - Wantスコア: {result['score_want']}点")
    print(f"  - マッチ数: {len(result['matched'])}件")
    print(f"  - ギャップ数: {len(result['gaps'])}件")
    print(f"  - 実行時間: {result['execution_time']:.2f}秒")
    
    return True


def test_case_2_with_rag():
    """
    ケース2: RAGあり（短い実績メモ）で分析実行
    結果が返ることをassert
    """
    print("\n" + "=" * 60)
    print("ケース2: RAGあり（短い実績メモ）")
    print("=" * 60)
    
    # OpenAI APIキーの確認
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEYが設定されていません。ケース2をスキップします。")
        print("   実行方法: export OPENAI_API_KEY=your_api_key")
        return None
    
    job_text, resume_text = load_sample_inputs()
    
    # 短い実績メモ
    achievement_notes = """
【プロジェクトA】ECサイトリニューアル
- リードエンジニアとして5名のチームをマネジメント
- マイクロサービスアーキテクチャへの移行を主導
- レスポンスタイムを50%改善、売上20%向上

【プロジェクトB】機械学習モデル開発
- レコメンデーションシステムの開発
- 精度90%を達成、A/Bテストで効果を検証
"""
    
    # 分析実行
    result = run_analysis_core(
        job_text=job_text,
        resume_text=resume_text,
        achievement_notes=achievement_notes.strip(),
        emphasis_axes=[],
        options={}
    )
    
    # 結果が返ることを確認
    assert result is not None, "結果が返りませんでした"
    assert "score_total" in result, "score_totalが存在しません"
    assert isinstance(result["score_total"], (int, float)), "score_totalは数値である必要があります"
    
    # RAGエラーメッセージの確認（エラーがあれば表示）
    if result.get("rag_error_message"):
        print(f"⚠️  RAG検索エラー: {result['rag_error_message']}")
    else:
        print("ℹ️  RAG検索が正常に実行されました")
    
    print(f"\n✅ ケース2完了:")
    print(f"  - 総合スコア: {result['score_total']}点")
    print(f"  - Mustスコア: {result['score_must']}点")
    print(f"  - Wantスコア: {result['score_want']}点")
    print(f"  - マッチ数: {len(result['matched'])}件")
    print(f"  - ギャップ数: {len(result['gaps'])}件")
    print(f"  - 実行時間: {result['execution_time']:.2f}秒")
    
    return True


def main():
    """メイン実行"""
    print("\n" + "=" * 60)
    print("AI応募適合度チェッカー - スモークテスト")
    print("=" * 60)
    
    results = []
    
    # ケース1: RAGなし
    try:
        result1 = test_case_1_no_rag()
        results.append(("ケース1 (RAGなし)", result1))
    except Exception as e:
        print(f"\n❌ ケース1でエラー: {e}")
        import traceback
        traceback.print_exc()
        results.append(("ケース1 (RAGなし)", False))
    
    # ケース2: RAGあり
    try:
        result2 = test_case_2_with_rag()
        if result2 is not None:
            results.append(("ケース2 (RAGあり)", result2))
        else:
            results.append(("ケース2 (RAGあり)", "スキップ"))
    except Exception as e:
        print(f"\n❌ ケース2でエラー: {e}")
        import traceback
        traceback.print_exc()
        results.append(("ケース2 (RAGあり)", False))
    
    # 結果サマリー
    print("\n" + "=" * 60)
    print("テスト結果サマリー")
    print("=" * 60)
    for test_name, result in results:
        if result is True:
            print(f"{test_name}: ✅ 成功")
        elif result == "スキップ":
            print(f"{test_name}: ⏭️  スキップ（OPENAI_API_KEY未設定）")
        else:
            print(f"{test_name}: ❌ 失敗")
    
    # 終了コード
    all_passed = all(r is True for _, r in results)
    if all_passed:
        print("\n✅ 全テスト成功")
        return 0
    else:
        print("\n❌ 一部のテストが失敗しました")
        return 1


if __name__ == "__main__":
    exit(main())



