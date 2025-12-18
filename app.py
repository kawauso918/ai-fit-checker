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
from models import RequirementType, ConfidenceLevel
from utils import verify_quote_in_text


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

    # 2カラムレイアウト
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("求人票")
        job_text = st.text_area(
            "求人票のテキストを貼り付けてください",
            height=300,
            placeholder="【求人票】\n\n■必須スキル\n・Python開発経験3年以上\n・Webアプリケーション開発の実務経験\n\n■歓迎スキル\n・AWSなどクラウド環境での開発経験",
            key="job_text"
        )

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
        if not job_text or not resume_text:
            st.error("❌ 求人票と職務経歴書の両方を入力してください。")
            return

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
            # F1: 求人要件抽出
            with st.spinner("⏳ F1: 求人要件を抽出中..."):
                requirements = extract_requirements(job_text, options)
                st.success(f"✅ F1完了: {len(requirements)}件の要件を抽出")

            # F2: 根拠抽出
            with st.spinner("⏳ F2: 職務経歴から根拠を抽出中..."):
                evidence_map = extract_evidence(resume_text, requirements, options)
                st.success(f"✅ F2完了: {len(evidence_map)}件の根拠を分析")

            # F3: スコア計算
            with st.spinner("⏳ F3: スコアを計算中..."):
                score_total, score_must, score_want, matched, gaps, summary = calculate_scores(
                    requirements, evidence_map
                )
                st.success(f"✅ F3完了: 総合スコア {score_total}点")

            # F4: 改善案生成
            with st.spinner("⏳ F4: 改善案を生成中..."):
                improvements = generate_improvements(
                    job_text, resume_text, requirements, matched, gaps, options
                )
                st.success(f"✅ F4完了: {len(improvements.action_items)}件の行動計画を生成")

            # 実行時間計測終了
            end_time = time.time()
            execution_time = end_time - start_time

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
    if "result" in st.session_state:
        result = st.session_state.result

        st.divider()
        st.header("📊 分析結果")

        # メトリクス表示
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)

        with col_m1:
            st.metric(
                label="総合スコア",
                value=f"{result['score_total']}点",
                delta=None
            )

        with col_m2:
            st.metric(
                label="Mustスコア",
                value=f"{result['score_must']}点",
                delta=None
            )

        with col_m3:
            st.metric(
                label="Wantスコア",
                value=f"{result['score_want']}点",
                delta=None
            )

        with col_m4:
            st.metric(
                label="マッチ数/ギャップ数",
                value=f"{len(result['matched'])}/{len(result['gaps'])}",
                delta=None
            )

        # 差分サマリ（強みTop3 + 致命的ギャップTop3）
        st.subheader("⚡ 差分サマリ")
        col_summary1, col_summary2 = st.columns(2)

        with col_summary1:
            # 強みTop3を抽出
            top_strengths = _get_top_strengths(result['matched'], top_n=3)
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
            top_gaps = _get_top_critical_gaps(result['gaps'], top_n=3)
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
        st.info(result['summary'])

        st.divider()

        # マッチした要件
        st.subheader(f"✅ マッチした要件（{len(result['matched'])}件）")

        if result['matched']:
            for i, m in enumerate(result['matched'], 1):
                with st.expander(
                    f"**[{m.requirement.req_id}]** {m.requirement.description} "
                    f"（一致度: {m.evidence.confidence:.0%}）"
                ):
                    st.markdown(f"**カテゴリ**: {m.requirement.category.value}")
                    st.markdown(f"**重要度**: {'⭐' * m.requirement.importance}")
                    st.markdown(f"**一致度**: {m.evidence.confidence:.2f} ({m.evidence.confidence_level.value})")

                    st.markdown("**判定理由**:")
                    st.write(m.evidence.reason)

                    if m.evidence.resume_quotes:
                        st.markdown("**職務経歴からの引用**:")
                        resume_text_for_verification = result.get("resume_text", "")
                        for quote in m.evidence.resume_quotes:
                            # 引用が実際に存在するか検証
                            is_valid = verify_quote_in_text(quote, resume_text_for_verification)
                            if is_valid:
                                st.markdown(f"> {quote}")
                            else:
                                # 警告表示：引用が見つからない場合
                                st.markdown("> ⚠️ **引用要確認**")
                                st.markdown(f"> {quote}")

                    st.markdown("**求人票からの引用**:")
                    st.markdown(f"> {m.requirement.job_quote}")
        else:
            st.write("マッチした要件はありません。")

        st.divider()

        # ギャップのある要件
        st.subheader(f"⚠️ ギャップのある要件（{len(result['gaps'])}件）")

        if result['gaps']:
            for i, g in enumerate(result['gaps'], 1):
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
        st.subheader("💡 改善案")

        st.markdown(f"**【全体戦略】**")
        st.success(result['improvements'].overall_strategy)

        # 職務経歴書の編集・追記案
        if result['improvements'].resume_edits:
            st.markdown(f"### ✏️ 職務経歴書の編集・追記案（{len(result['improvements'].resume_edits)}件）")

            for i, edit in enumerate(result['improvements'].resume_edits, 1):
                st.markdown(f"**{i}. 対象要件**: {edit.target_gap} ({edit.edit_type})")
                
                st.markdown("**追記テンプレート**:")
                st.code(edit.template, language="text")
                
                st.markdown("**具体例**:")
                st.code(edit.example, language="text")
                st.markdown("---")

        # 行動計画
        if result['improvements'].action_items:
            st.markdown(f"### 🎯 行動計画（{len(result['improvements'].action_items)}件）")

            # 優先度別にグループ化
            priority_a = [a for a in result['improvements'].action_items if a.priority == "A"]
            priority_b = [a for a in result['improvements'].action_items if a.priority == "B"]
            priority_c = [a for a in result['improvements'].action_items if a.priority == "C"]

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

        # 実行ログ
        with st.expander("📋 実行ログ"):
            st.markdown(f"**実行日時**: {result['timestamp']}")
            st.markdown(f"**実行時間**: {result['execution_time']:.2f}秒")
            st.markdown(f"**抽出要件数**: {len(result['requirements'])}件")
            st.markdown(f"**マッチ数**: {len(result['matched'])}件")
            st.markdown(f"**ギャップ数**: {len(result['gaps'])}件")


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
