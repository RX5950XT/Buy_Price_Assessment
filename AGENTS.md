# 專案代理規範

- 一律使用繁體中文（臺灣用語），回覆精簡且以結果為主。
- CLI 優先；能直接完成就不要把手動操作交回使用者。
- 非簡單任務先維護 `tasks/todo.md`，實作後持續勾選並補 Review。
- 新功能與修正遵循 TDD，測試覆蓋率至少 80%；修改後必須執行 lint、type check 與測試。
- 先找根因，採最小且清楚的改動；外部資料必須驗證，錯誤不可靜默吞掉。
- 研究不得有 look-ahead bias；所有特徵必須記錄其實際可取得時間，績效只以樣本外結果判定。
- 價格序列必須明確區分原始成交價與權息／分割還原價。
- 不可硬編碼 secrets；新增依賴前檢查維護狀態與漏洞並鎖定版本。
- Python 固定 3.12 並使用 `uv`；完成驗證至少執行 `ruff format --check`、`ruff check`、`mypy`、`pytest`。
- TWSE 全歷史月資料只能單線節流並使用逐月快取，不可平行轟炸官方端點。
- 每次任務完成同步精簡維護 `AGENTS.md`、`CLAUDE.md`、`CONTEXT.md`。
- Commit 格式為 `<type>: <description>`，type 限 `feat|fix|refactor|docs|test|chore|perf|ci`。
