# Agent Memory 同步操作总结（2026-04-16）

## 目标
- 将远端 `https://github.com/Nankago/agent-memory/tree/main` 的更新同步到本地 `/gfs/space/private/wujn/Learn/agent_memory`
- 保留本地已有大体积结果（尤其 `outputs` 中已有完整结果）
- 保持你已调整到正确位置的 4 个文件路径不被改回

## 你特别强调的约束
1. 本地 `outputs` 中已有的大结果不要被替换掉。  
2. 以下 4 个文件以本地当前正确路径为准：
   - `src/memory_collapse/external_pipeline.py`
   - `src/memory_collapse/cli.py`
   - `src/memory_collapse/__init__.py`
   - `tests/test_external_pipeline.py`

## 我执行的主要操作
1. 检查本地仓库状态、分支、远端与差异。  
2. `git fetch origin` 获取远端最新提交（远端到 `fdbf199`）。  
3. 未使用直接 `git pull/merge`，改为“按路径选择性同步”，避免覆盖你已有 `outputs` 大结果。  
4. 选择性同步了这些路径：
   - `.gitattributes`
   - `src/memory_collapse/**`（远端新增/更新代码）
   - `tests/test_external_pipeline.py`
   - `outputs/default_direct_validity_v12/**`（远端新增结果包）
5. 保持根目录旧位置文件未恢复（仍删除状态）：
   - `cli.py`
   - `external_cli.py`
   - `external_pipeline.py`
   - `test_external_pipeline.py`

## LFS 处理经过
- 新增目录 `outputs/default_direct_validity_v12` 中包含 LFS 文件。  
- 先完成大部分对象落地（约 415MB 已成功 checkout 为真实内容）。  
- 剩余 2 个文件长期在 `git lfs pull/fetch` 中挂起：
  - `outputs/default_direct_validity_v12/models/query_validity_useful_training_examples.csv`
  - `outputs/default_direct_validity_v12/models/rho_training_examples.csv`
- 最终采用 Git LFS batch API 直链下载对象并手动校验/覆盖，完成落地。

## 最终结果
- 上述 2 个文件已是**真实内容**（不是 pointer）：
  - 大小均为 `172258025` 字节
  - 对象哈希：`sha256:89a0c4292478f672018b4e77827aca97b206b8394549969abc1380ee54071530`
  - 文件首行已是 CSV 表头（如 `episode_id,query_id,...`），非 `version https://git-lfs.github.com/spec/v1`

## 当前状态说明
- 本次操作只做了“工作区同步与文件落地”，未替你提交 commit。  
- `git status` 里会保留你本地原有变更及本次同步带来的 staged/unstaged 变化，符合你“先同步再继续执行”的诉求。  

## 建议你刷新上下文后可直接继续
1. 先 `git status` 快速确认变更面。  
2. 如要固定当前状态，再决定是否分批 commit（代码与输出分开提交更稳妥）。

## 关键复盘（给下一个 Codex 的避坑说明）

### 为什么前期会“下载失败/卡住很久”
1. `git lfs pull` 在当前环境多次出现挂起（`git-lfs filter-process` 无输出长时间占用）。  
2. 期间出现过 `.git/index.lock` 残留，导致后续 git 操作被阻塞。  
3. 存在沙箱只读报错（如 `.git/FETCH_HEAD`、`.git/lfs/tmp`），必须用提权命令执行相关 git/lfs 操作。  
4. 一次关键误区：使用了 `main`（本地旧提交），而不是 `origin/main`（远端新提交），导致目标 LFS 对象无法按预期落地。

### 已验证可行的解决方式
1. 不要依赖长时间无输出的 `git lfs pull`；若挂起，及时切换方案。  
2. 涉及远端新增对象时，优先用 `origin/main` 作为引用，避免本地 `main` 旧提交造成对象缺失。  
3. 对卡住对象可走 Git LFS Batch API：
   - 先用 batch 接口按 `oid + size` 获取下载直链；
   - 用 `curl` 下载；
   - 校验 `sha256` 与文件大小后再写回目标路径。  
4. 写回后用以下方式确认不是 pointer：
   - 首行不是 `version https://git-lfs.github.com/spec/v1`；
   - 文件大小匹配预期。

### 本次两个难点文件的最终落地方式（已成功）
- 对象：`sha256:89a0c4292478f672018b4e77827aca97b206b8394549969abc1380ee54071530`  
- 大小：`172258025` 字节  
- 目标文件：
  - `outputs/default_direct_validity_v12/models/query_validity_useful_training_examples.csv`
  - `outputs/default_direct_validity_v12/models/rho_training_examples.csv`
- 处理方式：Batch API 直链下载 + 哈希校验 + 覆盖写回。
