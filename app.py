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


# Streamlitメニュー項目の日本語翻訳マッピング
STREAMLIT_MENU_TRANSLATIONS = {
    'Rerun': '再実行',
    'Settings': '設定',
    'Print': '印刷',
    'Record a screencast': 'スクリーンキャストを録画',
    'Developer options': '開発者オプション',
    'Clear cache': 'キャッシュをクリア'
}


def inject_menu_translations():
    """Streamlitメニュー項目を日本語化するJavaScriptを注入"""
    import json
    
    # Pythonの翻訳マッピングをJSONに変換
    translations_json = json.dumps(STREAMLIT_MENU_TRANSLATIONS, ensure_ascii=False)
    
    return f"""
    <style>
    /* Deployボタンを非表示 */
    button[kind="header"][class*="deploy"],
    button[kind="header"][class*="Deploy"],
    a[href*="deploy.streamlit"],
    [data-testid*="stToolbarDeployButton"],
    [data-testid*="Deploy"],
    button[title*="Deploy"],
    button[aria-label*="Deploy"] {{
        display: none !important;
        visibility: hidden !important;
    }}
    </style>
    <script>
    (function() {{
        'use strict';
        
        // Pythonから渡された翻訳マッピング
        const translations = {translations_json};
        
        // すべてのテキストノードを再帰的に検索して置き換え
        function replaceTextInElement(element) {{
            if (!element) return;
            
            // すべてのテキストノードを検索
            const walker = document.createTreeWalker(
                element,
                NodeFilter.SHOW_TEXT,
                {{
                    acceptNode: function(node) {{
                        // 親要素がscriptやstyleタグの場合はスキップ
                        let parent = node.parentElement;
                        while (parent) {{
                            if (parent.tagName === 'SCRIPT' || parent.tagName === 'STYLE') {{
                                return NodeFilter.FILTER_REJECT;
                            }}
                            parent = parent.parentElement;
                        }}
                        return NodeFilter.FILTER_ACCEPT;
                    }}
                }},
                false
            );
            
            const textNodes = [];
            let node;
            while (node = walker.nextNode()) {{
                textNodes.push(node);
            }}
            
            // テキストノードを置き換え
            textNodes.forEach(textNode => {{
                const originalText = textNode.textContent;
                const trimmedText = originalText.trim();
                
                // 完全一致するテキストを置き換え
                if (translations[trimmedText]) {{
                    textNode.textContent = originalText.replace(trimmedText, translations[trimmedText]);
                }}
            }});
            
            // 要素内の直接のテキストも確認（子要素がない場合）
            const allElements = element.querySelectorAll('*');
            allElements.forEach(el => {{
                // 子要素がない、または子要素がSVGのみの場合
                const hasOnlySvg = el.children.length === 1 && el.querySelector('svg');
                if (el.children.length === 0 || hasOnlySvg) {{
                    const text = el.textContent.trim();
                    if (translations[text]) {{
                        // SVGを保持
                        const svg = el.querySelector('svg');
                        if (svg) {{
                            const svgClone = svg.cloneNode(true);
                            el.innerHTML = '';
                            el.appendChild(svgClone);
                            el.appendChild(document.createTextNode(' ' + translations[text]));
                        }} else {{
                            el.textContent = translations[text];
                        }}
                    }}
                }}
            }});
        }}
        
        // メニュー項目を日本語化する関数
        function translateMenuItems() {{
            // メニューコンテナを検索（複数のパターンに対応）
            const menuContainers = [
                '[role="menu"]',
                '[data-baseweb="popover"]',
                '[data-baseweb="menu"]',
                'ul[role="menu"]',
                '[data-testid="stHeader"] [role="menu"]'
            ];
            
            menuContainers.forEach(selector => {{
                try {{
                    const containers = document.querySelectorAll(selector);
                    containers.forEach(container => {{
                        replaceTextInElement(container);
                    }});
                }} catch (e) {{
                    // セレクタが無効な場合は無視
                }}
            }});
            
            // メニュー項目を直接検索
            const menuItemSelectors = [
                '[role="menuitem"]',
                '[data-baseweb="menu-item"]',
                'li[role="menuitem"]'
            ];
            
            menuItemSelectors.forEach(selector => {{
                try {{
                    const items = document.querySelectorAll(selector);
                    items.forEach(item => {{
                        replaceTextInElement(item);
                    }});
                }} catch (e) {{
                    // セレクタが無効な場合は無視
                }}
            }});
            
            // ヘッダー内のすべての要素も確認
            const header = document.querySelector('[data-testid="stHeader"]');
            if (header) {{
                replaceTextInElement(header);
            }}
        }}
        
        // Deployボタンを非表示
        function hideDeployButton() {{
            const allElements = document.querySelectorAll('*');
            allElements.forEach(el => {{
                const text = el.textContent.trim();
                if (text === 'Deploy') {{
                    if (el.tagName === 'BUTTON' || 
                        el.getAttribute('role') === 'button' ||
                        el.closest('button')) {{
                        el.style.display = 'none';
                        el.style.visibility = 'hidden';
                    }}
                }}
            }});
        }}
        
        // 実行関数
        function executeTranslation() {{
            translateMenuItems();
            hideDeployButton();
        }}
        
        // 初期実行
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', function() {{
                executeTranslation();
                // 少し遅延して再実行（DOMが完全に構築されるまで待つ）
                setTimeout(executeTranslation, 100);
                setTimeout(executeTranslation, 500);
            }});
        }} else {{
            executeTranslation();
            setTimeout(executeTranslation, 100);
            setTimeout(executeTranslation, 500);
        }}
        
        // MutationObserverで監視（より積極的に）
        const observer = new MutationObserver(function(mutations) {{
            let shouldTranslate = false;
            mutations.forEach(mutation => {{
                if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {{
                    mutation.addedNodes.forEach(node => {{
                        if (node.nodeType === Node.ELEMENT_NODE) {{
                            const el = node;
                            if (el.getAttribute('role') === 'menu' ||
                                el.getAttribute('role') === 'menuitem' ||
                                el.querySelector('[role="menu"]') ||
                                el.querySelector('[role="menuitem"]')) {{
                                shouldTranslate = true;
                            }}
                        }}
                    }});
                }}
            }});
            if (shouldTranslate) {{
                setTimeout(executeTranslation, 10);
                setTimeout(executeTranslation, 100);
            }}
        }});
        observer.observe(document.body, {{
            childList: true,
            subtree: true,
            characterData: true
        }});
        
        // クリックイベントで実行（メニューが開いた時）
        document.addEventListener('click', function(e) {{
            // メニューボタンがクリックされた可能性がある
            const target = e.target;
            if (target.closest('[data-testid="stHeader"]') || 
                target.closest('button[kind="header"]')) {{
                setTimeout(executeTranslation, 10);
                setTimeout(executeTranslation, 50);
                setTimeout(executeTranslation, 150);
                setTimeout(executeTranslation, 300);
            }}
        }}, true);
        
        // フォーカスイベントでも実行（メニューが開く可能性がある）
        document.addEventListener('focusin', function(e) {{
            if (e.target.closest('[data-testid="stHeader"]')) {{
                setTimeout(executeTranslation, 50);
            }}
        }}, true);
        
        // 定期的に実行（念のため・パフォーマンスを考慮して間隔を延長）
        setInterval(executeTranslation, 2000);
    }})();
    </script>
    """


def main():
    # ページ設定
    st.set_page_config(
        page_title="AI応募適合度チェッカー",
        page_icon="📊",
        layout="wide"
    )

    # StreamlitのUIボタンを日本語化（Pythonで翻訳マッピングを管理）
    st.markdown(inject_menu_translations(), unsafe_allow_html=True)

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
                        for quote in m.evidence.resume_quotes:
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
                st.info(edit.example)
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


if __name__ == "__main__":
    main()
