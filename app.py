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
from f7_judge_evaluation import evaluate_with_judge
from f8_generate_application_email import generate_application_email
from models import RequirementType, ConfidenceLevel, QuoteSource
from utils import verify_quote_in_text
from pdf_export import generate_pdf
from rag_error_handler import validate_rag_inputs, get_rag_status
from input_validator import validate_inputs, validate_requirements_extracted
from ui_components import render_requirements_by_category
from chat_interface import get_chat_response
import os


def run_analysis_core(
    job_text: str,
    resume_text: str,
    achievement_notes: str = None,
    company_info: str = None,
    emphasis_axes: list = None,
    options: dict = None
) -> dict:
    """
    分析処理のコア関数（Streamlit UIに依存しない）
    
    Args:
        job_text: 求人票のテキスト
        resume_text: 職務経歴書のテキスト
        achievement_notes: 実績メモ（オプション）
        company_info: 企業情報（オプション）
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
            - judge_evaluation: Judge評価（Noneの可能性あり）
            - rag_error_message: RAGエラーメッセージ（Noneの可能性あり）
            - rag_warning_message: RAG警告メッセージ（Noneの可能性あり）
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
    
    try:
        # F1: 求人要件抽出
        requirements = extract_requirements(job_text, options)
        
        # 要件抽出結果の検証
        is_valid, error_message = validate_requirements_extracted(requirements)
        if not is_valid:
            # エラーを返すために例外を発生
            raise ValueError(f"要件抽出に失敗しました: {error_message}")
        
        # F2: 根拠抽出
        options_with_notes = options.copy()
        options_with_notes["achievement_notes"] = achievement_notes if achievement_notes else None
        evidence_map = extract_evidence(resume_text, requirements, options_with_notes)
        
        # RAGエラーメッセージを取得
        rag_error_message = options_with_notes.get("rag_error_message")
        rag_warning_message = options_with_notes.get("rag_warning_message")
        
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
        
        # F7: Judge評価（失敗時はスキップ）
        judge_evaluation = None
        try:
            judge_evaluation = evaluate_with_judge(
                job_text, resume_text, matched, gaps, improvements, interview_qas, options
            )
        except Exception:
            # エラー時はスキップ（Noneのまま）
            pass
        
        # F8: 応募メール文面生成（失敗時はスキップ）
        application_email = None
        try:
            application_email = generate_application_email(
                job_text, resume_text, company_info, matched, gaps, improvements, summary, options
            )
        except Exception:
            # エラー時はスキップ（Noneのまま）
            pass
        
        # 実行時間計測終了
        end_time = time.time()
        execution_time = end_time - start_time
        
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
            "judge_evaluation": judge_evaluation,
            "application_email": application_email,
            "rag_error_message": rag_error_message,
            "rag_warning_message": rag_warning_message,
        }
        
        return result
    except Exception:
        # エラーが発生した場合でも、execution_timeを計算してから例外を再発生
        end_time = time.time()
        execution_time = end_time - start_time
        # 例外を再発生（呼び出し元でキャッチされる）
        raise


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

    # 企業情報（オプション）
    with st.expander("🏢 企業情報（オプション）", expanded=False):
        st.markdown("**会社概要や採用ページ全体などの情報を記載してください**")
        st.markdown("企業情報を追加すると、より詳細な分析や応募文面の生成が可能になります。")
        company_info = st.text_area(
            "企業情報を貼り付けてください（会社概要、採用ページ、企業文化など）",
            height=200,
            placeholder="例：\n\n【会社概要】\n・設立：2010年\n・従業員数：100名\n・事業内容：SaaS開発・提供\n\n【企業文化】\n・フラットな組織体制\n・リモートワーク推奨\n・技術力重視",
            key="company_info"
        )
    
    # 実績メモ（オプション）
    with st.expander("📝 実績メモ（オプション）", expanded=False):
        st.markdown("**追加の実績・経験を記載してください**")
        st.markdown("複数の実績を記載することで、根拠抽出の精度が向上します。")
        achievement_notes = st.text_area(
            "実績メモを貼り付けてください（複数の実績を改行区切りで記載可能、最大15000文字）",
            height=200,
            placeholder="例：\n\n【プロジェクトA】\n・ECサイトのリニューアルをリード\n・レスポンスタイムを50%改善\n・チーム5名をマネジメント\n\n【プロジェクトB】\n・機械学習モデルの開発\n・精度90%を達成",
            key="achievement_notes"
        )
        
        # RAG使用時のAPIキーチェック（実績メモが入力されている場合のみ）
        if achievement_notes and achievement_notes.strip():
            is_valid, error_msg, warning_msg = validate_rag_inputs(achievement_notes, require_api_key=True)
            if error_msg:
                st.error(error_msg)
                st.stop()
            elif warning_msg:
                st.warning(warning_msg)

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
        
        # 入力検証（求人票/職務経歴書の長さチェック）
        for idx, job_text_item in enumerate(job_texts, 1):
            is_valid, error_message, warning_message = validate_inputs(job_text_item, resume_text)
            if not is_valid:
                st.error(f"❌ 入力検証エラー（求人{idx if compare_mode else ''}）:\n\n{error_message}")
                st.stop()
                return
            # 警告メッセージがある場合は表示（処理は続行）
            if warning_message:
                st.warning(f"⚠️ 警告（求人{idx if compare_mode else ''}）:\n\n{warning_message}")

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
        execution_time = 0.0  # エラー時でも確実に定義されるように初期化

        try:
            if compare_mode:
                # 比較モード：複数の求人票に対して順番に実行
                all_results = []
                
                for idx, job_text_item in enumerate(job_texts, 1):
                    st.markdown(f"### 📋 求人{idx}の分析中...")
                    
                    # F1: 求人要件抽出
                    with st.spinner(f"⏳ 求人{idx} - F1: 求人要件を抽出中..."):
                        requirements = extract_requirements(job_text_item, options)
                    
                    # 要件抽出結果の検証
                    is_valid, error_message = validate_requirements_extracted(requirements)
                    if not is_valid:
                        st.error(f"❌ 求人{idx}の要件抽出に失敗しました:\n\n{error_message}")
                        st.stop()
                        return
                    
                    # F2: 根拠抽出
                    with st.spinner(f"⏳ 求人{idx} - F2: 職務経歴から根拠を抽出中..."):
                        # 実績メモをoptionsに追加
                        options_with_notes = options.copy()
                        options_with_notes["achievement_notes"] = achievement_notes if achievement_notes else None
                        evidence_map = extract_evidence(resume_text, requirements, options_with_notes)
                        
                        # RAG状態を表示（最初の求人のみ表示）
                        if idx == 1:
                            rag_error = options_with_notes.get("rag_error_message")
                            rag_warning = options_with_notes.get("rag_warning_message")
                            # RAG検索で取得した根拠候補数を計算（各EvidenceのquotesからRAG由来をカウント）
                            rag_evidence_count = 0
                            for ev in evidence_map.values():
                                if hasattr(ev, 'quotes') and ev.quotes:
                                    rag_evidence_count += sum(1 for q in ev.quotes if q.source.value == "rag")
                            
                            # RAG状態表示（expander内）
                            with st.expander("🔍 RAG検索状態", expanded=False):
                                status, status_msg = get_rag_status(
                                    achievement_notes,
                                    rag_error,
                                    rag_evidence_count
                                )
                                if status == "enabled":
                                    st.success(f"✅ {status_msg}")
                                elif status == "error":
                                    st.error(f"❌ {status_msg}")
                                elif status == "disabled":
                                    st.info(f"ℹ️ {status_msg}")
                                else:
                                    st.info(f"ℹ️ {status_msg}")
                                
                                if rag_warning:
                                    st.warning(f"⚠️ {rag_warning}")
                    
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
                        company_info=company_info if 'company_info' in locals() else None,
                        emphasis_axes=emphasis_axes_list,
                        options=options
                    )
                    
                    # RAG状態を表示
                    rag_error = result.get("rag_error_message")
                    rag_warning = result.get("rag_warning_message")
                    # RAG検索で取得した根拠候補数を計算（各EvidenceのquotesからRAG由来をカウント）
                    rag_evidence_count = 0
                    for ev in result.get("evidence_map", {}).values():
                        if hasattr(ev, 'quotes') and ev.quotes:
                            rag_evidence_count += sum(1 for q in ev.quotes if q.source.value == "rag")
                    
                    # RAG状態表示（expander内）
                    with st.expander("🔍 RAG検索状態", expanded=False):
                        status, status_msg = get_rag_status(
                            achievement_notes,
                            rag_error,
                            rag_evidence_count
                        )
                        if status == "enabled":
                            st.success(f"✅ {status_msg}")
                        elif status == "error":
                            st.error(f"❌ {status_msg}")
                        elif status == "disabled":
                            st.info(f"ℹ️ {status_msg}")
                        else:
                            st.info(f"ℹ️ {status_msg}")
                        
                        if rag_warning:
                            st.warning(f"⚠️ {rag_warning}")
                    
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
                    "timestamp": result["timestamp"],
                    "execution_time": result["execution_time"],
                    "requirements": result["requirements"],
                    "evidence_map": result["evidence_map"],
                    "score_total": result["score_total"],
                    "score_must": result["score_must"],
                    "score_want": result["score_want"],
                    "matched": result["matched"],
                    "gaps": result["gaps"],
                    "summary": result["summary"],
                    "improvements": result["improvements"],
                    "interview_qas": result["interview_qas"],
                    "quality_evaluation": result.get("quality_evaluation"),  # Noneの可能性あり
                    "judge_evaluation": result.get("judge_evaluation"),  # Noneの可能性あり
                    "application_email": result.get("application_email"),  # Noneの可能性あり
                    "resume_text": result["resume_text"],  # 引用検証用に保存
                    "job_text": result.get("job_text"),  # チャット機能用
                    "company_info": result.get("company_info"),  # チャット機能用
                    "options": options,  # チャット機能用
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
        _render_single_result(
            result, 
            result.get("resume_text", ""),
            job_text=result.get("job_text"),
            company_info=result.get("company_info")
        )

        # 実行ログ
        with st.expander("📋 実行ログ"):
            st.markdown(f"**実行日時**: {result.get('timestamp', 'N/A')}")
            execution_time = result.get('execution_time', 0.0)
            st.markdown(f"**実行時間**: {execution_time:.2f}秒")
            st.markdown(f"**抽出要件数**: {len(result.get('requirements', []))}件")
            st.markdown(f"**マッチ数**: {len(result.get('matched', []))}件")
            st.markdown(f"**ギャップ数**: {len(result.get('gaps', []))}件")


def _render_single_result(result_dict: dict, resume_text: str, job_text: str = None, company_info: str = None):
    """
    単一の分析結果を表示（通常モードと比較モードで共通使用）
    
    Args:
        result_dict: 分析結果の辞書（result または compare_results["results"][i]）
        resume_text: 職務経歴書のテキスト（引用検証用）
        job_text: 求人票のテキスト（チャット機能用、オプション）
        company_info: 企業情報（チャット機能用、オプション）
    """
    # チャット機能用に情報を追加
    if job_text:
        result_dict['job_text'] = job_text
    if company_info:
        result_dict['company_info'] = company_info
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

    # 要件と根拠をMust/Wantでセクション分けして表示（改善版）
    render_requirements_by_category(
        result_dict['matched'],
        result_dict['gaps'],
        resume_text
    )

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
    
    st.divider()
    
    # Judge評価（F7）
    judge_evaluation = result_dict.get('judge_evaluation')
    if judge_evaluation:
        st.subheader("⚖️ Judge評価結果（3観点評価）")
        
        # 3観点のスコア表示
        col_j1, col_j2, col_j3 = st.columns(3)
        
        with col_j1:
            st.metric(
                label="納得感",
                value=f"{judge_evaluation.scores.convincing:.1f}点",
                delta=None
            )
            st.caption("ユーザーが判断しやすい構造・説明か")
        
        with col_j2:
            st.metric(
                label="根拠の妥当性",
                value=f"{judge_evaluation.scores.grounding:.1f}点",
                delta=None
            )
            st.caption("引用が要件に適切に紐づいているか")
        
        with col_j3:
            st.metric(
                label="誇張抑制",
                value=f"{judge_evaluation.scores.no_exaggeration:.1f}点",
                delta=None
            )
            st.caption("職務経歴にないことを断定していないか")
        
        st.divider()
        
        # 問題点
        if judge_evaluation.issues:
            st.markdown("### ⚠️ 問題点")
            for i, issue in enumerate(judge_evaluation.issues, 1):
                st.markdown(f"{i}. {issue}")
        
        st.divider()
        
        # 改善提案
        if judge_evaluation.fix_suggestions:
            st.markdown("### 💡 改善提案")
            for i, suggestion in enumerate(judge_evaluation.fix_suggestions, 1):
                st.markdown(f"{i}. {suggestion}")
    
    st.divider()
    
    # 応募メール文面（F8）
    application_email = result_dict.get('application_email')
    if application_email:
        st.subheader("📧 応募メール文面")
        
        # 件名
        st.markdown("### 件名")
        st.code(application_email.subject, language="text")
        
        # 本文
        st.markdown("### 本文")
        st.text_area(
            "本文（コピー用）",
            value=application_email.body,
            height=300,
            key="email_body_copy"
        )
        
        # 添付資料の提案
        if application_email.attachment_suggestions:
            st.markdown("### 📎 添付資料の提案")
            for attachment in application_email.attachment_suggestions:
                st.markdown(f"- {attachment}")
        
        # 送信時の注意点
        if application_email.tips:
            st.markdown("### 💡 送信時の注意点")
            for i, tip in enumerate(application_email.tips, 1):
                st.markdown(f"{i}. {tip}")
    
    st.divider()
    
    # チャット機能
    with st.expander("💬 チャットで求人内容を深掘り考察", expanded=False):
        st.markdown("**求人内容の深掘り考察や応募文面改善の提案ができます**")
        
        # チャット履歴を初期化（session_state）
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # チャット履歴を表示
        if st.session_state.chat_history:
            st.markdown("### チャット履歴")
            for i, (user_msg, assistant_msg) in enumerate(st.session_state.chat_history):
                with st.expander(f"💬 会話 {i+1}", expanded=False):
                    st.markdown(f"**あなた**: {user_msg}")
                    st.markdown(f"**アシスタント**: {assistant_msg}")
        
        # チャット入力
        user_input = st.text_input(
            "質問を入力してください",
            placeholder="例: この求人の必須スキルについて詳しく教えてください / 応募メールの改善点を教えてください",
            key="chat_input"
        )
        
        col_chat1, col_chat2 = st.columns([1, 4])
        with col_chat1:
            send_button = st.button("送信", type="primary", key="chat_send")
        
        # チャット送信
        if send_button and user_input:
            with st.spinner("考え中..."):
                # 分析結果を取得（result_dictから）
                analysis_result = {
                    'summary': result_dict.get('summary', ''),
                    'score_total': result_dict.get('score_total', 0),
                    'matched': result_dict.get('matched', []),
                    'gaps': result_dict.get('gaps', [])
                }
                
                # チャット応答を生成
                assistant_response = get_chat_response(
                    user_message=user_input,
                    job_text=result_dict.get('job_text', '') if 'job_text' in result_dict else '',
                    resume_text=result_dict.get('resume_text', ''),
                    company_info=result_dict.get('company_info', None),
                    analysis_result=analysis_result,
                    chat_history=st.session_state.chat_history,
                    options=result_dict.get('options', {}) if 'options' in result_dict else {}
                )
                
                # チャット履歴に追加
                st.session_state.chat_history.append((user_input, assistant_response))
                
                # ページをリロードして履歴を表示
                st.rerun()


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
