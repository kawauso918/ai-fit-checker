"""
AI応募適合度チェッカー - メインアプリケーション
Streamlitを使用した1ページ完結型Webアプリケーション
"""
import streamlit as st
import time
from datetime import datetime

from f1_extract_requirements import extract_requirements
from f2_extract_evidence import extract_evidence
from f3_score import calculate_scores
from f4_generate_improvements import generate_improvements
from f5_generate_interview_qa import generate_interview_qa
from f6_quality_evaluation import evaluate_quality
from models import RequirementType, ConfidenceLevel, QuoteSource
from utils import verify_quote_in_text
from pdf_export import generate_pdf


def run_analysis_core(
    job_text: str,
    resume_text: str,
    achievement_notes: str = None,
    emphasis_axes: list = None,
    options: dict = None
) -> dict:
    """
    分析処理のコア関数（Streamlit UIに依存しない）
    
    Args:
        job_text: 求人票のテキスト
        resume_text: 職務経歴書のテキスト
        achievement_notes: 実績メモ（オプション）
        emphasis_axes: 強調軸のリスト（オプション）
        options: オプション辞書（llm_provider, model_name, temperature等）
    
    Returns:
        dict: 分析結果の辞書
            - timestamp: 実行日時
            - execution_time: 実行時間（秒）
            - requirements: 抽出された要件リスト
            - evidence_map: 根拠マップ
            - score_total: 総合スコア
            - score_must: Mustスコア
            - score_want: Wantスコア
            - matched: マッチした要件リスト
            - gaps: ギャップのある要件リスト
            - summary: サマリ
            - improvements: 改善案
            - interview_qas: 面接Q&A
            - quality_evaluation: 品質評価（Noneの可能性あり）
            - rag_error_message: RAGエラーメッセージ（Noneの可能性あり）
    """
    import time
    from datetime import datetime
    
    # デフォルト値の設定
    if options is None:
        options = {}
    if emphasis_axes is None:
        emphasis_axes = []
    
    # 実行時間計測開始
    start_time = time.time()
    
    # F1: 求人要件抽出
    requirements = extract_requirements(job_text, options)
    
    # F2: 根拠抽出
    options_with_notes = options.copy()
    options_with_notes["achievement_notes"] = achievement_notes if achievement_notes else None
    evidence_map = extract_evidence(resume_text, requirements, options_with_notes)
    
    # RAGエラーメッセージを取得
    rag_error_message = options_with_notes.get("rag_error_message")
    
    # F3: スコア計算
    score_total, score_must, score_want, matched, gaps, summary = calculate_scores(
        requirements, evidence_map, emphasis_axes=emphasis_axes
    )
    
    # F4: 改善案生成
    improvements = generate_improvements(
        job_text, resume_text, requirements, matched, gaps, options
    )
    
    # F5: 面接想定Q&A生成
    interview_qas = generate_interview_qa(
        job_text, resume_text, matched, gaps, summary, options
    )
    
    # F6: 品質評価（失敗時はスキップ）
    quality_evaluation = None
    try:
        quality_evaluation = evaluate_quality(
            job_text, resume_text, matched, gaps, improvements, interview_qas, options
        )
    except Exception:
        # エラー時はスキップ（Noneのまま）
        pass
    
    # 実行時間計測終了
    end_time = time.time()
    execution_time = end_time - start_time
    
    # 結果を辞書にまとめる
    result = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "execution_time": execution_time,
        "resume_text": resume_text,
        "requirements": requirements,
        "evidence_map": evidence_map,
        "score_total": score_total,
        "score_must": score_must,
        "score_want": score_want,
        "matched": matched,
        "gaps": gaps,
        "summary": summary,
        "improvements": improvements,
        "interview_qas": interview_qas,
        "quality_evaluation": quality_evaluation,
        "rag_error_message": rag_error_message,
    }
    
    return result


def main():
    # ページ設定
    st.set_page_config(
        page_title="AI応募適合度チェッカー",
        page_icon="📊",
        layout="wide"
    )

    # タイトル
    st.title("📊 AI応募適合度チェッカー")
    st.markdown("**求人票と職務経歴書を比較分析し、適合度を自動評価します**")

    # 注意書き
    st.info(
        "⚠️ **個人情報の取り扱いについて**\n\n"
        "本アプリはLLM（大規模言語モデル）を使用します。個人情報（氏名、住所、電話番号など）は"
        "入力前にマスクすることを強く推奨します。"
    )

    st.divider()

    # ==================== 入力フォーム ====================
    st.header("📝 入力情報")

    # 比較モードの切り替え
    compare_mode = st.checkbox(
        "🔀 比較モード（最大3つの求人票を比較）",
        value=False,
        key="compare_mode"
    )

    # 2カラムレイアウト
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("求人票")
        
        if compare_mode:
            # 比較モード：タブで複数の求人票を入力
            job_tabs = st.tabs(["求人1", "求人2", "求人3"])
            job_texts = []
            
            for i, tab in enumerate(job_tabs, 1):
                with tab:
                    job_text_input = st.text_area(
                        f"求人{i}のテキストを貼り付けてください",
                        height=250,
                        placeholder=f"【求人票{i}】\n\n■必須スキル\n・Python開発経験3年以上\n・Webアプリケーション開発の実務経験\n\n■歓迎スキル\n・AWSなどクラウド環境での開発経験",
                        key=f"job_text_{i}"
                    )
                    job_texts.append(job_text_input)
            
            # 空でない求人票のみを有効とする
            job_texts = [jt for jt in job_texts if jt.strip()]
            
            if not job_texts:
                job_text = None  # 1件モードとの互換性のため
            else:
                job_text = job_texts[0]  # デフォルトは最初の求人票
        else:
            # 通常モード：1つの求人票
            job_text = st.text_area(
                "求人票のテキストを貼り付けてください",
                height=300,
                placeholder="【求人票】\n\n■必須スキル\n・Python開発経験3年以上\n・Webアプリケーション開発の実務経験\n\n■歓迎スキル\n・AWSなどクラウド環境での開発経験",
                key="job_text"
            )
            job_texts = [job_text] if job_text else []

    with col2:
        st.subheader("職務経歴書")
        resume_text = st.text_area(
            "職務経歴書のテキストを貼り付けてください",
            height=300,
            placeholder="【職務経歴書】\n\n■職務経歴\n2019年〜現在：株式会社ABC\n・Pythonを使用したWebアプリケーション開発に5年間従事\n・Djangoフレームワークを用いたECサイトの構築",
            key="resume_text"
        )

    # 任意項目
    st.subheader("任意情報（オプション）")
    col_opt1, col_opt2 = st.columns(2)

    with col_opt1:
        desired_position = st.text_input(
            "志望職種（分析の参考情報として使用）",
            placeholder="例: Webエンジニア、データサイエンティスト",
            key="desired_position"
        )

    with col_opt2:
        emphasis_axis = st.text_input(
            "強調したい軸（分析時に重視する観点）",
            placeholder="例: 技術力、リーダーシップ、グローバル経験",
            key="emphasis_axis"
        )

    # 実績メモ（オプション）
    with st.expander("📝 実績メモ（オプション）", expanded=False):
        st.markdown("**追加の実績・経験を記載してください**")
        st.markdown("複数の実績を記載することで、根拠抽出の精度が向上します。")
        achievement_notes = st.text_area(
            "実績メモを貼り付けてください（複数の実績を改行区切りで記載可能）",
            height=200,
            placeholder="例：\n\n【プロジェクトA】\n・ECサイトのリニューアルをリード\n・レスポンスタイムを50%改善\n・チーム5名をマネジメント\n\n【プロジェクトB】\n・機械学習モデルの開発\n・精度90%を達成",
            key="achievement_notes"
        )

    # 詳細設定（expander）
    with st.expander("⚙️ 詳細設定（上級者向け）"):
        st.markdown("**LLMモデル設定**")
        col_adv1, col_adv2 = st.columns(2)

        with col_adv1:
            llm_provider = st.selectbox(
                "LLMプロバイダー",
                options=["openai", "anthropic"],
                index=0,
                key="llm_provider"
            )

            model_name = st.text_input(
                "モデル名（空欄でデフォルト）",
                placeholder="gpt-4o-mini / claude-3-5-sonnet-20241022",
                key="model_name"
            )

        with col_adv2:
            temperature = st.slider(
                "Temperature（創造性）",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.1,
                key="temperature"
            )

        st.markdown("**抽出設定**")
        col_adv3, col_adv4, col_adv5 = st.columns(3)

        with col_adv3:
            max_must = st.number_input(
                "Must要件の最大件数",
                min_value=1,
                max_value=20,
                value=10,
                key="max_must"
            )

        with col_adv4:
            max_want = st.number_input(
                "Want要件の最大件数",
                min_value=1,
                max_value=20,
                value=10,
                key="max_want"
            )

        with col_adv5:
            strict_mode = st.checkbox(
                "Strictモード（曖昧一致を防止）",
                value=False,
                key="strict_mode"
            )

    st.divider()

    # ==================== 実行ボタン ====================
    col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 2])

    with col_btn2:
        analyze_button = st.button(
            "🚀 分析を実行",
            type="primary",
            use_container_width=True
        )

    # ==================== 分析実行 ====================
    if analyze_button:
        # 入力チェック
        if not resume_text:
            st.error("❌ 職務経歴書を入力してください。")
            return
        
        if compare_mode:
            # 比較モード：複数の求人票をチェック
            if not job_texts or len(job_texts) == 0:
                st.error("❌ 比較モードでは、少なくとも1つの求人票を入力してください。")
                return
            if len(job_texts) > 3:
                st.error("❌ 比較モードでは、最大3つの求人票まで入力できます。")
                return
        else:
            # 通常モード：1つの求人票をチェック
            if not job_text:
                st.error("❌ 求人票を入力してください。")
                return
            job_texts = [job_text]

        # 強調軸をリストに変換（カンマ区切り対応）
        emphasis_axes_list = []
        if emphasis_axis:
            # カンマ区切りで分割し、空白を削除
            emphasis_axes_list = [axis.strip() for axis in emphasis_axis.split(",") if axis.strip()]

        # オプション辞書を作成
        options = {
            "llm_provider": llm_provider,
            "model_name": model_name if model_name else None,
            "temperature": temperature,
            "max_must": max_must,
            "max_want": max_want,
            "strict_mode": strict_mode,
        }

        # 実行時間計測開始
        start_time = time.time()

        try:
            if compare_mode:
                # 比較モード：複数の求人票に対して順番に実行
                all_results = []
                
                for idx, job_text_item in enumerate(job_texts, 1):
                    st.markdown(f"### 📋 求人{idx}の分析中...")
                    
                    # F1: 求人要件抽出
                    with st.spinner(f"⏳ 求人{idx} - F1: 求人要件を抽出中..."):
                        requirements = extract_requirements(job_text_item, options)
                    
                    # F2: 根拠抽出
                    with st.spinner(f"⏳ 求人{idx} - F2: 職務経歴から根拠を抽出中..."):
                        # 実績メモをoptionsに追加
                        options_with_notes = options.copy()
                        options_with_notes["achievement_notes"] = achievement_notes if achievement_notes else None
                        evidence_map = extract_evidence(resume_text, requirements, options_with_notes)
                        
                        # RAG状態を表示
                        rag_error = options_with_notes.get("rag_error_message")
                        if rag_error:
                            st.warning(f"⚠️ RAG検索: {rag_error}")
                        elif achievement_notes and achievement_notes.strip():
                            st.info("ℹ️ RAG検索が有効です（実績メモから根拠候補を取得）")
                    
                    # F3: スコア計算
                    with st.spinner(f"⏳ 求人{idx} - F3: スコアを計算中..."):
                        score_total, score_must, score_want, matched, gaps, summary = calculate_scores(
                            requirements, evidence_map, emphasis_axes=emphasis_axes_list
                        )
                    
                    # F4: 改善案生成
                    with st.spinner(f"⏳ 求人{idx} - F4: 改善案を生成中..."):
                        improvements = generate_improvements(
                            job_text_item, resume_text, requirements, matched, gaps, options
                        )
                    
                    # F5: 面接想定Q&A生成
                    with st.spinner(f"⏳ 求人{idx} - F5: 面接想定Q&Aを生成中..."):
                        interview_qas = generate_interview_qa(
                            job_text_item, resume_text, matched, gaps, summary, options
                        )
                    
                    # F6: 品質評価（失敗時はスキップ）
                    quality_evaluation = None
                    try:
                        with st.spinner(f"⏳ 求人{idx} - F6: 品質評価を実行中..."):
                            quality_evaluation = evaluate_quality(
                                job_text_item, resume_text, matched, gaps, improvements, interview_qas, options
                            )
                    except Exception as e:
                        # エラー時はスキップ（警告は出さない、比較モードでは簡潔に）
                        pass
                    
                    # 結果を保存
                    all_results.append({
                        "job_index": idx,
                        "job_text": job_text_item,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "requirements": requirements,
                        "evidence_map": evidence_map,
                        "score_total": score_total,
                        "score_must": score_must,
                        "score_want": score_want,
                        "matched": matched,
                        "gaps": gaps,
                        "summary": summary,
                        "improvements": improvements,
                        "interview_qas": interview_qas,
                        "quality_evaluation": quality_evaluation,  # Noneの可能性あり
                    })
                    
                    st.success(f"✅ 求人{idx}の分析完了: 総合スコア {score_total}点")
                
                # 実行時間計測終了
                end_time = time.time()
                execution_time = end_time - start_time
                
                # 結果をsession_stateに保存（比較モード用）
                st.session_state.compare_results = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "execution_time": execution_time,
                    "resume_text": resume_text,
                    "results": all_results,
                }
                
                st.balloons()
            else:
                # 通常モード：1つの求人票に対して実行
                with st.spinner("⏳ 分析を実行中..."):
                    # コア関数を呼び出し
                    result = run_analysis_core(
                        job_text=job_text,
                        resume_text=resume_text,
                        achievement_notes=achievement_notes,
                        emphasis_axes=emphasis_axes_list,
                        options=options
                    )
                    
                    # RAG状態を表示
                    if result.get("rag_error_message"):
                        st.warning(f"⚠️ RAG検索: {result['rag_error_message']}")
                    elif achievement_notes and achievement_notes.strip():
                        st.info("ℹ️ RAG検索が有効です（実績メモから根拠候補を取得）")
                    
                    # 各ステップの成功メッセージを表示
                    st.success(f"✅ F1完了: {len(result['requirements'])}件の要件を抽出")
                    st.success(f"✅ F2完了: {len(result['evidence_map'])}件の根拠を分析")
                    st.success(f"✅ F3完了: 総合スコア {result['score_total']}点")
                    st.success(f"✅ F4完了: {len(result['improvements'].action_items)}件の行動計画を生成")
                    st.success(f"✅ F5完了: {len(result['interview_qas'].qa_list)}件のQ&Aを生成")
                    if result.get('quality_evaluation'):
                        st.success(f"✅ F6完了: 総合品質スコア {result['quality_evaluation'].overall_score:.1f}点")
                    else:
                        st.info("ℹ️ F6（品質評価）をスキップしました")

                # 結果をsession_stateに保存
                st.session_state.result = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "execution_time": execution_time,
                    "requirements": requirements,
                    "evidence_map": evidence_map,
                    "score_total": score_total,
                    "score_must": score_must,
                    "score_want": score_want,
                    "matched": matched,
                    "gaps": gaps,
                    "summary": summary,
                    "improvements": improvements,
                    "interview_qas": interview_qas,
                    "quality_evaluation": quality_evaluation,  # Noneの可能性あり
                    "resume_text": resume_text,  # 引用検証用に保存
                }

                st.balloons()

        except Exception as e:
            st.error(f"❌ エラーが発生しました: {e}")
            import traceback
            with st.expander("詳細なエラー情報"):
                st.code(traceback.format_exc())
            return

    # ==================== 結果表示 ====================
    # 比較モードの結果表示
    if "compare_results" in st.session_state:
        compare_results = st.session_state.compare_results
        
        st.divider()
        st.header("📊 比較結果")
        
        # スコアランキング表示
        st.subheader("🏆 スコアランキング")
        results = compare_results["results"]
        
        # スコア順にソート
        sorted_results = sorted(results, key=lambda x: x["score_total"], reverse=True)
        
        # ランキング表示
        col_rank1, col_rank2, col_rank3 = st.columns(3)
        rank_cols = [col_rank1, col_rank2, col_rank3]
        
        for i, result_item in enumerate(sorted_results[:3], 1):
            with rank_cols[i-1]:
                st.metric(
                    label=f"🏅 {i}位: 求人{result_item['job_index']}",
                    value=f"{result_item['score_total']}点",
                    delta=f"Must: {result_item['score_must']} / Want: {result_item['score_want']}"
                )
        
        st.divider()
        
        # 各求人の詳細（折りたたみ表示）
        st.subheader("📋 各求人の詳細")
        
        for result_item in sorted_results:
            with st.expander(
                f"求人{result_item['job_index']}: 総合スコア {result_item['score_total']}点 "
                f"(Must: {result_item['score_must']}点 / Want: {result_item['score_want']}点)",
                expanded=False
            ):
                # 通常モードと同じ表示ロジックを使用
                _render_single_result(result_item, compare_results["resume_text"])
        
        st.divider()
    
    # 通常モードの結果表示
    if "result" in st.session_state:
        result = st.session_state.result

        st.divider()
        st.header("📊 分析結果")

        # PDFダウンロードボタン
        try:
            pdf_bytes = generate_pdf(result)
            st.download_button(
                label="📥 PDFレポートをダウンロード",
                data=pdf_bytes,
                file_name=f"ai-fit-checker-report-{result.get('timestamp', 'report').replace(' ', '_').replace(':', '-')}.pdf",
                mime="application/pdf",
                use_container_width=False
            )
        except Exception as e:
            st.warning(f"⚠️ PDF生成に失敗しました: {e}")

        st.divider()

        # 通常モードの結果表示（関数化したロジックを使用）
        _render_single_result(result, result.get("resume_text", ""))

        # 実行ログ
        with st.expander("📋 実行ログ"):
            st.markdown(f"**実行日時**: {result['timestamp']}")
            st.markdown(f"**実行時間**: {result['execution_time']:.2f}秒")
            st.markdown(f"**抽出要件数**: {len(result['requirements'])}件")
            st.markdown(f"**マッチ数**: {len(result['matched'])}件")
            st.markdown(f"**ギャップ数**: {len(result['gaps'])}件")


def _render_single_result(result_dict: dict, resume_text: str):
    """
    単一の分析結果を表示（通常モードと比較モードで共通使用）
    
    Args:
        result_dict: 分析結果の辞書（result または compare_results["results"][i]）
        resume_text: 職務経歴書のテキスト（引用検証用）
    """
    # メトリクス表示
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)

    with col_m1:
        st.metric(
            label="総合スコア",
            value=f"{result_dict['score_total']}点",
            delta=None
        )

    with col_m2:
        st.metric(
            label="Mustスコア",
            value=f"{result_dict['score_must']}点",
            delta=None
        )

    with col_m3:
        st.metric(
            label="Wantスコア",
            value=f"{result_dict['score_want']}点",
            delta=None
        )

    with col_m4:
        st.metric(
            label="マッチ数/ギャップ数",
            value=f"{len(result_dict['matched'])}/{len(result_dict['gaps'])}",
            delta=None
        )

    # 差分サマリ（強みTop3 + 致命的ギャップTop3）
    st.subheader("⚡ 差分サマリ")
    col_summary1, col_summary2 = st.columns(2)

    with col_summary1:
        # 強みTop3を抽出
        top_strengths = _get_top_strengths(result_dict['matched'], top_n=3)
        if top_strengths:
            st.markdown("**✅ 強みTop3**")
            for i, m in enumerate(top_strengths, 1):
                category_label = "Must" if m.requirement.category == RequirementType.MUST else "Want"
                confidence_label = f"{m.evidence.confidence:.0%}"
                st.markdown(f"{i}. **{m.requirement.description}** ({category_label}, 一致度: {confidence_label})")
        else:
            st.markdown("**✅ 強みTop3**")
            st.markdown("*強みが見つかりませんでした*")

    with col_summary2:
        # 致命的ギャップTop3を抽出
        top_gaps = _get_top_critical_gaps(result_dict['gaps'], top_n=3)
        if top_gaps:
            st.markdown("**⚠️ 致命的ギャップTop3**")
            for i, g in enumerate(top_gaps, 1):
                category_label = "Must" if g.requirement.category == RequirementType.MUST else "Want"
                st.markdown(f"{i}. **{g.requirement.description}** ({category_label})")
        else:
            st.markdown("**⚠️ 致命的ギャップTop3**")
            st.markdown("*致命的なギャップはありません*")

    st.divider()

    # サマリー
    st.subheader("📝 総評")
    st.info(result_dict['summary'])

    st.divider()

    # マッチした要件
    st.subheader(f"✅ マッチした要件（{len(result_dict['matched'])}件）")

    if result_dict['matched']:
        for i, m in enumerate(result_dict['matched'], 1):
            with st.expander(
                f"**[{m.requirement.req_id}]** {m.requirement.description} "
                f"（一致度: {m.evidence.confidence:.0%}）"
            ):
                st.markdown(f"**カテゴリ**: {m.requirement.category.value}")
                st.markdown(f"**重要度**: {'⭐' * m.requirement.importance}")
                st.markdown(f"**一致度**: {m.evidence.confidence:.2f} ({m.evidence.confidence_level.value})")

                st.markdown("**判定理由**:")
                st.write(m.evidence.reason)

                # 引用を表示（quotesを使用、後方互換性でresume_quotesも対応）
                quotes_to_display = m.evidence.quotes if m.evidence.quotes else [
                    type('Quote', (), {'text': q, 'source': QuoteSource.RESUME, 'source_id': None})()
                    for q in (m.evidence.resume_quotes or [])
                ]
                
                if quotes_to_display:
                    st.markdown("**職務経歴からの引用**:")
                    
                    for quote_obj in quotes_to_display:
                        # Quote構造体から情報を取得
                        quote_text = quote_obj.text if hasattr(quote_obj, 'text') else quote_obj
                        source = quote_obj.source if hasattr(quote_obj, 'source') else QuoteSource.RESUME
                        source_id = getattr(quote_obj, 'source_id', None)
                        
                        # 引用の出どころを表示
                        if source == QuoteSource.RESUME:
                            source_label = "📄 [職務経歴書]"
                        else:
                            if source_id is not None:
                                source_label = f"🔍 [実績DB #{source_id + 1}]"
                            else:
                                source_label = "🔍 [実績DB]"
                        
                        # 引用が実際に存在するか検証
                        is_valid = verify_quote_in_text(quote_text, resume_text)
                        if is_valid:
                            st.markdown(f"> **{source_label}** {quote_text}")
                        else:
                            # 警告表示：引用が見つからない場合
                            st.markdown(f"> **{source_label}** ⚠️ **引用要確認**")
                            st.markdown(f"> {quote_text}")

                st.markdown("**求人票からの引用**:")
                st.markdown(f"> {m.requirement.job_quote}")
    else:
        st.write("マッチした要件はありません。")

    st.divider()

    # ギャップのある要件
    st.subheader(f"⚠️ ギャップのある要件（{len(result_dict['gaps'])}件）")

    if result_dict['gaps']:
        for i, g in enumerate(result_dict['gaps'], 1):
            with st.expander(
                f"**[{g.requirement.req_id}]** {g.requirement.description} "
                f"（{g.requirement.category.value}）",
                expanded=(i <= 3)  # 最初の3件は展開
            ):
                st.markdown(f"**カテゴリ**: {g.requirement.category.value}")
                st.markdown(f"**重要度**: {'⭐' * g.requirement.importance}")

                st.markdown("**不足理由**:")
                st.warning(g.evidence.reason)

                st.markdown("**埋め方のヒント**:")
                st.markdown(
                    f"- 該当する経験があれば職務経歴書に**明示的に記載**してください\n"
                    f"- 経験がない場合は、下記の「改善案」を参考に**学習・実績作り**を検討してください"
                )
    else:
        st.write("ギャップはありません。全ての要件を満たしています！")

    st.divider()

    # 改善案
    improvements = result_dict.get('improvements')
    if improvements:
        st.subheader("💡 改善案")

        st.markdown(f"**【全体戦略】**")
        st.success(improvements.overall_strategy)

        # 職務経歴書の編集・追記案
        if improvements.resume_edits:
            st.markdown(f"### ✏️ 職務経歴書の編集・追記案（{len(improvements.resume_edits)}件）")

            for i, edit in enumerate(improvements.resume_edits, 1):
                st.markdown(f"**{i}. 対象要件**: {edit.target_gap} ({edit.edit_type})")
                
                st.markdown("**追記テンプレート**:")
                st.code(edit.template, language="text")
                
                st.markdown("**具体例**:")
                st.code(edit.example, language="text")
                st.markdown("---")

        # 行動計画
        if improvements.action_items:
            st.markdown(f"### 🎯 行動計画（{len(improvements.action_items)}件）")

            # 優先度別にグループ化
            priority_a = [a for a in improvements.action_items if a.priority == "A"]
            priority_b = [a for a in improvements.action_items if a.priority == "B"]
            priority_c = [a for a in improvements.action_items if a.priority == "C"]

            if priority_a:
                st.markdown("#### 🔴 優先度A（最優先・短期）")
                for a in priority_a:
                    st.markdown(f"- **{a.action}**")
                    st.markdown(f"  - 根拠: {a.rationale}")
                    st.markdown(f"  - 期待効果: {a.estimated_impact}")

            if priority_b:
                st.markdown("#### 🟡 優先度B（中期）")
                for a in priority_b:
                    st.markdown(f"- **{a.action}**")
                    st.markdown(f"  - 根拠: {a.rationale}")
                    st.markdown(f"  - 期待効果: {a.estimated_impact}")

            if priority_c:
                st.markdown("#### 🟢 優先度C（長期・余裕があれば）")
                for a in priority_c:
                    st.markdown(f"- **{a.action}**")
                    st.markdown(f"  - 根拠: {a.rationale}")
                    st.markdown(f"  - 期待効果: {a.estimated_impact}")

    st.divider()

    # 面接想定Q&A
    interview_qas = result_dict.get('interview_qas')
    if interview_qas and interview_qas.qa_list:
        st.subheader("🎤 面接想定Q&A")
        st.markdown(f"**{len(interview_qas.qa_list)}件の質問と回答の骨子**")

        for i, qa in enumerate(interview_qas.qa_list, 1):
            with st.expander(
                f"**Q{i}:** {qa.question}",
                expanded=(i <= 3)  # 最初の3件は展開
            ):
                st.markdown("**回答の骨子:**")
                for outline in qa.answer_outline:
                    st.markdown(f"- {outline}")

    st.divider()

    # 品質評価
    quality_evaluation = result_dict.get('quality_evaluation')
    if quality_evaluation:
        st.subheader("📊 品質評価")
        
        # 総合スコア
        st.markdown(f"**総合品質スコア: {quality_evaluation.overall_score:.1f}点**")
        
        # 観点別スコア
        st.markdown("### 観点別スコア")
        col_q1, col_q2 = st.columns(2)
        
        for i, criterion_score in enumerate(quality_evaluation.criterion_scores):
            col = col_q1 if i % 2 == 0 else col_q2
            with col:
                st.metric(
                    label=criterion_score.criterion,
                    value=f"{criterion_score.score:.1f}点",
                    delta=None
                )
                with st.expander(f"{criterion_score.criterion}の詳細", expanded=False):
                    st.markdown(f"**評価理由:** {criterion_score.reason}")
        
        st.divider()
        
        # 改善ポイント
        st.markdown("### 💡 改善ポイント")
        for i, point in enumerate(quality_evaluation.improvement_points, 1):
            st.markdown(f"{i}. {point}")


def _get_top_strengths(matched, top_n=3):
    """
    強みTop3を抽出（confidence strong > partial、Must > Want を優先）
    
    Args:
        matched: マッチした要件と根拠のペアリスト
        top_n: 取得件数（デフォルト3）
    
    Returns:
        List[RequirementWithEvidence]: ソート済み強みリスト（上位N件）
    """
    if not matched:
        return []
    
    # ソートキー（降順にするため負の値を使用）：
    # 1. confidenceが高い順（0.7以上=HIGH > 0.4-0.7=MEDIUM）
    # 2. Must優先（MUST=0, WANT=1）
    # 3. importance降順
    sorted_matched = sorted(
        matched,
        key=lambda m: (
            -m.evidence.confidence,  # confidence降順（負の値で大きい値が前に来る）
            0 if m.requirement.category == RequirementType.MUST else 1,  # Must優先
            -m.requirement.importance  # importance降順（負の値で大きい値が前に来る）
        )
    )
    
    return sorted_matched[:top_n]


def _get_top_critical_gaps(gaps, top_n=3):
    """
    致命的ギャップTop3を抽出（Must優先）
    
    Args:
        gaps: ギャップリスト
        top_n: 取得件数（デフォルト3）
    
    Returns:
        List[Gap]: ソート済みギャップリスト（上位N件）
    """
    if not gaps:
        return []
    
    # ソートキー：
    # 1. Must優先（MUST=0, WANT=1）
    # 2. importance降順
    sorted_gaps = sorted(
        gaps,
        key=lambda g: (
            0 if g.requirement.category == RequirementType.MUST else 1,  # Must優先
            -g.requirement.importance  # importance降順
        )
    )
    
    return sorted_gaps[:top_n]


if __name__ == "__main__":
    main()
