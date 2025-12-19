# コミット分割案

## 1. chore/test: smoke test追加

**変更ファイル:**
- `scripts/smoke_test.py` (新規)
- `app.py` (`run_analysis_core()`関数追加)
- `README.md` (スモークテスト実行方法を追記)

**コミットメッセージ:**
```
chore(test): add smoke test script for analysis core

- Add scripts/smoke_test.py that reads from sample_inputs.md
- Extract analysis logic to run_analysis_core() function in app.py
- Test case 1: RAGなし (assert required keys)
- Test case 2: RAGあり (skip if OPENAI_API_KEY not set)
- Update README.md with execution instructions
```

## 2. feat/ui: 引用出どころ表示

**変更ファイル:**
- `models.py` (`QuoteSource` Enum, `Quote`構造体追加, `Evidence`を`quotes`に変更)
- `f2_extract_evidence.py` (`Quote`構造体を使用、RAG由来引用に`source_id`を付与)
- `app.py` (引用表示時にラベルを付与)
- `README.md` (引用の表示ルールを追記)

**コミットメッセージ:**
```
feat(ui): add quote source tracking with Quote structure

- Add QuoteSource enum and Quote structure to models.py
- Change Evidence.resume_quotes to Evidence.quotes (List[Quote])
- Add source_id to RAG-derived quotes (chunk index)
- Update UI to display quote labels: [職務経歴書] / [実績DB #N]
- Maintain backward compatibility with resume_quotes/quote_sources
- Update README.md with quote display rules
```

## 3. fix: 異常系ハンドリング

**変更ファイル:**
- `rag_error_handler.py` (新規、エラーハンドリング共通関数)
- `f2_extract_evidence.py` (RAGエラーハンドリング改善)
- `app.py` (OPENAI_API_KEYチェック、RAG状態表示追加)
- `README.md` (想定外ケースの挙動を追記)

**コミットメッセージ:**
```
fix: enhance error handling for RAG feature

- Add rag_error_handler.py with common error handling functions
- Improve RAG error handling (text length 15k chars, initialization failures)
- Add OPENAI_API_KEY check in app.py (st.error + st.stop)
- Add RAG status display in expander (enabled/disabled/error/empty)
- Update README.md with unexpected case behaviors
- Use logger for error logging
```

