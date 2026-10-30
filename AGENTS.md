# VoxCPM2 Voice Cloner（專案藍圖）

> 本檔為跨 Agent 通用的專案藍圖（AGENTS.md 開放標準）。任何 Agent 的每個 session 都應先讀本檔＋`handoff.md`。
> Claude Code 預設只在沒有 `CLAUDE.md` 時才讀 `AGENTS.md`，故由 `CLAUDE.md` 的 `@AGENTS.md` import 本檔（`CLAUDE.md` 只有這一行）；Claude 專屬規範寫在本檔〈Claude Code 專屬〉一節。

## 專案簡介

用 VoxCPM2 克隆聲音並生成任意語音的 **AI Agent 工具包**。設計原則是「人類只負責錄音，其餘全部用自然語言交給 Agent」——錄音走網頁 UI，生成、對話合成等操作由 Agent 直接呼叫腳本完成。模型為 `openbmb/VoxCPM2`（Apache-2.0，可商用）。

## 關鍵時程

<!-- 目前無外部期程 -->

## 目標與路線圖

- [x] 階段一：環境安裝（uv + Python 3.12 venv + torch cu128 + voxcpm）
- [x] 階段二：確認 GPU 路徑可用（RTX 5060 Ti / CUDA，已實測生成成功）
- [x] 階段三：錄音 UI 可用（`record_ui.py`，繞過 Smart App Control 封鎖）
- [x] 階段四：評估 `optimize=True` — **結案：維持 `False`**。本機沒有 triton／`cl.exe`，`torch.compile` 會被靜默降級（`Warning: torch.compile disabled - triton is not installed`），暖機後 RTF 兩者同為 1.17、VRAM 同為 6.46GB，但 `optimize=True` 會多跑一次暖機生成、載入從 9.7s 變 13.0s。要真正生效得裝非官方的 `triton-windows`＋MSVC，有 Smart App Control 封鎖風險，不划算
- [ ] 階段五：評估常駐生成服務（現在每次執行 `clone.py` 都要重載模型約 10 秒）
- [x] 階段六：RTX Spark（Windows on ARM）用 GPU 生成——`install.ps1` 建原生 ARM64＋CUDA 環境（NVIDIA 預覽版 PyTorch），**RTF 1.7**，失敗退回 x64 CPU（見〈裝置偵測〉）

## 資料夾結構

```
media-voice-clone/
├── AGENTS.md                 # 本檔：跨 Agent 專案藍圖
├── CLAUDE.md                 # 橋接檔（@AGENTS.md）
├── handoff.md                # 交接檔
├── README.md                 # 給人看的完整說明
├── record_ui.py              # 錄音 UI（唯一介面，僅用標準庫）
├── clone.py                  # Agent 工具：用聲音生成語音
├── dialogue.py               # Agent 工具：多聲音對話
├── tts_lexicon.py            # 讀音替換表載入（clone.py／dialogue.py 共用）
├── tts-lexicon.md            # 讀音替換表（四個 TTS repo 各一份、內容相同）
├── voice_prep.py             # 生成前後處理：繁轉簡、去靜音、濾波、−15 LUFS（clone／dialogue／audition 共用）
├── audition.py               # 候選串成試聽檔、挑一支（搭配 clone.py --takes）
├── record.py                 # 命令列錄音（備案）
├── app.py                    # 舊版錄音 UI（gradio，本機無法執行）
├── webui_record.py           # 舊版網頁錄音（gradio，本機無法執行）
├── start.bat                 # 雙擊啟動錄音 UI
├── install.bat / install.ps1 # 自動偵測 GPU + 安裝依賴（環境建在 ~/.local/share/media-voice-clone/.venv）
├── sac-pins.txt              # Smart App Control 放行的套件版本（install.ps1 被擋時才用）
├── requirements-win-arm64.txt # Windows on ARM＋NVIDIA 的原生 ARM64＋CUDA 環境套件（含固定的預覽版 torch）
├── shims/sklearn-unavailable/ # scikit-learn 被 SAC 擋時的 import 空殼（只給上面那個環境用）
├── texts/sample_text.txt     # 錄音時朗讀的文字
├── skills/voice-clone/      # 全域技能原始檔（不進版控，見下方「全域技能」）
├── voices/                   # 已錄製的聲音（不進版控）
└── output/                   # 生成的語音（不進版控）
```

## 同步層級（本專案初始化至第 3 層級）

| 層級 | 平台 | 位置 | 讀取時機 |
|------|------|------|---------|
| L1 | 本地（GDrive） | `AGENTS.md`＋`handoff.md`＋`CLAUDE.md`（橋接） | 每個 session |
| L2 | GitHub | [changyiwu/media-voice-clone](https://github.com/changyiwu/media-voice-clone)（**公開**） | 指定時 |
| L3 | Obsidian | `media-voice-clone/專案工作流程.md` | 有需要時 |

## 環境

| 項目 | 值 |
|------|-----|
| 專案目錄 | 雲端硬碟的 `agents/media-voice-clone`（實際絕對路徑每台不同，一律以 cwd 為準） |
| Python | **本機共用位置，不在專案裡**：Windows `~\.local\share\media-voice-clone\.venv\Scripts\python.exe`／macOS `~/.local/share/media-voice-clone/.venv/bin/python`（由 `install.ps1` 建立）。下文指令裡的 `$py` 就是它，每個新 shell 先設：`$py = "$HOME\.local\share\media-voice-clone\.venv\Scripts\python.exe"` |
| 模型 | `openbmb/VoxCPM2`（Apache-2.0；權重已在 HuggingFace 快取） |
| 裝置 | 三種模式自動切換：NVIDIA CUDA／Apple Silicon MPS／CPU。`clone.py` 與 `dialogue.py` 直接問 torch（見〈裝置偵測〉） |
| 效能 | 模型載入 ~9.7 秒（每次執行都要重載），尖峰 VRAM 6.46GB。RTF 隨稿長變化：719 字長稿 **1.0**、中等句 1.1~1.3、10 字以內短句 2.5~2.9（固定開銷攤不掉）。**長稿一次念完比拆成多次划算**。Windows on ARM（LT-YI-RS）：原生 ARM64＋RTX Spark GPU **RTF 1.7**、載入約 25 秒；退回 x64 模擬＋CPU 時極慢（**RTF ~42**，5.9 秒語音要 4 分鐘） |
| 輸出 | `output/cloned_voice.wav`（`--output` 可改） |

> 所有指令都以專案目錄為工作目錄執行。若尚未安裝（`$py` 不存在）：Windows 請使用者雙擊 `install.bat`，macOS 跑 `pwsh -File install.ps1`（`.bat` 是 Windows 專屬啟動器，mac 沒有）。

### ⚠️ 環境限制：Smart App Control（PC-YI-SL）

本機 Smart App Control 處於**強制執行模式**，Code Integrity 會封鎖 PyPI wheel 中未簽章的原生程式庫。實測被擋：**pandas、pyarrow、datasets、gradio**，錯誤訊息為
`ImportError: DLL load failed ...: 應用程式控制原則已封鎖此檔案`。

- 這**不是**防毒問題。在 Defender 加排除項目無效——攔截來自核心模式的 Code Integrity，不是防毒引擎，且 Smart App Control 設計上不提供白名單。
- 因此 `app.py` 與 `webui_record.py`（皆依賴 gradio）**在本機無法啟動**，保留僅為相容其他電腦。
- 錄音一律用 **`record_ui.py`**：只用 Python 標準庫的 `http.server`，完全避開 pandas/pyarrow。
- torch、torchaudio、voxcpm、transformers、soundfile、sounddevice、numpy、resampy 均不受影響；`clone.py` 的生成路徑已實測不受影響。

> Agent 若看到上述 ImportError，**不要重裝同一版套件**，那不會有幫助；改走不依賴該套件的路徑，或換成 SAC 已放行的舊版。

**LT-YI-RS（Windows on ARM）的 SAC 已於 2026-10-30 關閉**（`VerifiedAndReputablePolicyState = 0`），下面的擋法只在開著 SAC 的 Windows on ARM 才會遇到。開著時連生成路徑都會被擋：太新的 wheel 信譽不夠（torch 2.14.1 的 `torch_python.dll`、torchaudio 2.10 的 `_torchaudio`、numba 0.68.1 的 `_dynfunc`）。`install.ps1` 的處理：先裝最新版，驗證碰到 `WinError 4551` 就改用 `sac-pins.txt` 的版本重裝。新的套件被擋時，在 `sac-pins.txt` 加一行已放行的版本（越舊越容易過），不要整包降版。

> 另一個相關坑：本機只有 Windows PowerShell 5.1（無 pwsh），會用 big5 讀取無 BOM 的 `.ps1`，導致中文腳本 parse 失敗。專案內的 `.ps1` 已加上 UTF-8 BOM，新增 `.ps1` 時請比照辦理。

## 工具清單

| 工具 | 用途 | 呼叫方式 |
|------|------|---------|
| `clone.py` | 用已錄聲音生成語音 | `clone.py "文字" --voice <名稱>` |
| `dialogue.py` | 多聲音對話 | 編輯腳本內對話清單後執行 |
| `record.py` | 命令列錄音（備案） | `record.py --voice <名稱>` |
| `record_ui.py` | 錄音 UI（給人類用） | `start.bat` → http://127.0.0.1:7860 |

### clone.py — 用已錄製的聲音生成語音

```powershell
& $py clone.py "同學們早安，今天我們來上數學課。" --voice 小吳
```

常用參數：

- `--voice <名稱>`：對應 `voices/<名稱>/` 目錄（預設 `小吳`）
- `--file <路徑>`：從文字檔讀取要生成的內容
- `--output <路徑>`：輸出路徑（預設 `output/cloned_voice.wav`）
- `--device cuda|cpu`：強制指定裝置（預設自動偵測）
- `--takes N`：同一句生 N 個候選（seed 1…N）到 `<輸出檔名>_takes/`，並串好 `audition.wav`；用 `audition.py <資料夾> --pick sK -o <正式檔>` 挑一支。使用者說「念得怪怪的」「換一個」時用這個，不要只重跑一次碰運氣
- `--raw`：跳過後處理（預設：去頭尾靜音、highpass 70 Hz＋lowpass 11 kHz、響度 −15 LUFS、取樣峰值 ≤ −1 dBFS）
- `--dry-run`：只印送進模型的文字（讀音替換表＋繁轉簡），不載模型

送進模型前一律經 `voice_prep.to_model_text()`：讀音替換表 → 繁轉簡（VoxCPM2 念簡體比較穩，media-short-film 實測；沒裝 opencc 就跳過）。生成後經 `voice_prep.finish()`。響度 −15 LUFS 跟 media-video-specs、media-short-film 的成片目標一致。嘶聲（4 kHz 以上能量比）要在濾波**前**量，`--takes` 會記在每支候選旁的 `.json`，`audition.py` 優先用它。

### dialogue.py — 用多個聲音生成對話

```powershell
& $py dialogue.py
```

- 換聲音：改腳本頂部的 `SPEAKER_A` / `SPEAKER_B` 兩個常數（預設 `小吳` / `老柯`），對話清單與輸出檔名都會跟著走
- 輸出：`output/dialogue_<A>_<B>.wav`
- 自訂對話內容：編輯 `dialogue.py` 中的 `dialogue` 清單
- 每句各自經 `voice_prep.finish()`（去頭尾靜音、濾波、−15 LUFS）再拼接，句間固定 0.4 秒，兩個聲音不會一大一小
- 模型只載入一次、逐句切換聲音，攤下來每句只多約 10 秒。實測 8 句 35.8 秒對話共 59 秒（**第一句是暖機、會特別慢**，之後穩定在 RTF ~1.1）

### tts_lexicon.py — 讀音替換表

`clone.py`、`dialogue.py` 送進模型前會先套 `tts-lexicon.md`（`| 原文 | 改寫 | 說明 |`），把念錯的多音字、英文詞改字（奇數→基數、十一少一→十一少了一、GPT-4→GPT four），並印出「改寫後」的文字。參考音逐字稿（`prompt.txt`）不替換。

- **表在本專案根目錄**（`tts-lexicon.md`）。這份表有**四份、內容逐位元組相同**（另外三份在 media-video-specs、media-short-film、agent-speak/speak/；四個 repo 都公開，各自單獨下載就要能用）。`tts_lexicon.py` 以自己的位置為起點（不靠 cwd）往上找另外三份，有就一起讀，本專案這份優先；設了 `TTS_LEXICON` 就改讀它指的表來取代本專案這份
- **覆蓋**：輸出資料夾往上每一層的同名 `tts-lexicon.md`，近的優先（只對某批輸出有效的條目放這裡）
- 找不到任何表只印一行提示、照原文念，不報錯
- **使用者說某個詞念錯了** → 條目**四份一起加**（說明欄寫正確讀音、出處、是否已人耳確認），用 `python ../media-short-film/tools/tts_lexicon.py --check`／`--sync` 檢查與合併。加完重生那句給使用者聽
- 試表不必載模型：`& $py tts_lexicon.py "句子"`
- 本專案是獨立公開 repo，**不要 import 別的 repo 的程式**；解析規則改了四個 repo 各自改

### record_ui.py — 錄音 UI

```powershell
& $py record_ui.py --open
```

瀏覽器端用 MediaRecorder 收音，經 Web Audio 重採樣成 16kHz 單聲道後在前端編成 WAV；伺服器端只用標準庫 `http.server`，存檔用 `soundfile`。只綁 `127.0.0.1`，不對外開放。

## 聲音管理

- 查看已錄製的聲音：`ls voices\`（目錄名即聲音名稱）
- 每個聲音目錄包含：
  - `ref_voice.wav` — 參考音檔（16kHz 單聲道 PCM_16，峰值正規化至 0.95）
  - `prompt.txt` — 逐字稿（錄音時念的文字）
- `voices/` **不進版控**（屬個人資料）
- `voices/` 位於 Google 雲端硬碟同步範圍內，聲音會跨電腦同步；本機目前有 `小吳`、`老柯`

## 裝置偵測

`clone.py`／`dialogue.py` 的 `detect_device()` **直接問這台的 torch**：cuda → mps → cpu；`--device` 可強制指定。

- **不再用 `.gpu_type`**：它跟著雲端硬碟同步，是最後一台跑安裝的電腦寫的，兩個方向都會錯（NB-Yi 寫 `cpu` 會讓有 GPU 的 PC-YI-SL 跑 CPU）。`install.ps1` 不再寫它，還會刪掉舊檔
- 用不用得到 GPU 取決於**這台的環境裝的是哪種 torch**，那是 `install.ps1` 依硬體決定的：
  - NVIDIA GPU（x64 Windows）→ cu128 版 torch
  - Apple Silicon → PyPI 預設版（含 MPS）
  - 無獨顯 → CPU 版
  - **Windows on ARM＋NVIDIA（RTX Spark）→ 原生 ARM64＋CUDA**，失敗才退回 x64 CPU（見下）
  - Windows on ARM、沒有 NVIDIA → 先試原生 ARM64 CPU，不行改建 x64（Prism 模擬）。原生 CPU 目前不行：torchaudio 官方沒有 win_arm64 wheel
- 不支援 Intel Arc（XPU）

### Windows on ARM＋RTX Spark 的 GPU 環境

2026-10-28 LT-YI-RS 實測：**RTF 1.7**（暖機後；第一句 2.9），比 x64 CPU 環境（RTF ~45）快約 27 倍，接近 RTX 5060 Ti。`install.ps1` 的做法：

- **原生 ARM64、Python 3.13**（NVIDIA 只出 cp313）。套件全在 `requirements-win-arm64.txt`，voxcpm 另外 `--no-deps` 裝
- **torch／torchaudio 固定在 [NVIDIA/pytorch-windows-ci releases](https://github.com/NVIDIA/pytorch-windows-ci/releases) 的一批 `woa-nightly-*`**（現為 `2.16.0.dev20261027+cu134`），需求檔裡是直接 URL＋`#sha256=`，由 uv 驗證。nightly 每天一批、torch 與 torchaudio 同批建置；換版本時兩行一起換、sha256 取 release 頁的 digest，換完要重跑生成驗證。**不要用 `pypi.nvidia.com/nvtorch_oot_nightly`**：同樣的檔只給 62 KB/s，torchaudio 也跟 torch 不同批
- 只裝生成路徑用得到的套件：pyarrow、tiktoken、cryptography、funasr、modelscope 只有訓練／ASR／降噪才用（而且沒有 win_arm64 wheel）；wetext 只有 `normalize=True` 才用
- 幾個非最新的固定版本，理由都寫在需求檔：librosa **0.9.2**（最後一版不需要 soxr，soxr 沒有任何 win_arm64 wheel）＋`setuptools<81`（給它 `pkg_resources`）。numba **0.68.1**、scipy **1.18.1** 是關 SAC 後實測的版本；開著 SAC 的機器要改回放行的 numba **0.68.0rc1**、scipy **1.16.3**
- **scikit-learn**：librosa 0.9.2 一 import 就要它，開著 SAC 時 cp313 win_arm64 每一版都被擋；LT-YI-RS 關 SAC 後已是真的 1.9.1。`install.ps1` 先裝真的、試 import，被擋才換成 `shims/sklearn-unavailable`：import 得到，真的用到時丟清楚的 `ImportError`。生成路徑（`librosa.load`、`effects.trim`）不會用到。SAC 放行後重跑 `install.ps1` 會自動換回真的
- **關 SAC 也裝不到的**：soxr、pyarrow、tiktoken、funasr 沒有 cp313 win_arm64 wheel（2026-10-30 重查）；datasets 因為相依 pyarrow 也裝不到。它們都不在生成路徑上
- 驗證除了一般的 smoke test，還要 `torch.cuda.is_available()` 且 GPU 上 bf16 矩陣乘法跑得動
- **原本的環境不會被弄壞**：x64 環境先改名成 `.venv-backup`，GPU 環境驗證通過才刪備份，失敗就改回原名再退回 x64；已經是 GPU 環境時，更新失敗（例如沒網路）但原環境還能用就留著。三條路徑都實測過（`-ArmRequirements` 指向一份會失敗的需求檔）
- `uv venv` 一律 `--python-preference only-managed`：這台設了 `UV_PYTHON_PREFERENCE=system`，不指定的話 uv 會挑使用者自己裝的 Python，哪天移除或升級它環境就壞。不能用 `--managed-python`，跟那個環境變數衝突

## 使用者自然語言對照

| 使用者說的 | Agent 動作 |
|-----------|-----------|
| 「用OOO的聲音念出／說出／講出XXX」 | 走 `voice-clone` 全域技能（見下節）；沒裝技能就直接 `clone.py "XXX" --voice OOO` |
| 「讓A和B對話」 | 編輯 `dialogue.py` 對話清單 → 執行 |
| 「有哪些聲音」 | `ls voices\` |
| 「XX 念錯了」「XX 要念成 OO」 | 四個 repo 的讀音替換表一起加一列 → 重生那句（見〈tts_lexicon.py〉） |
| 「念得怪怪的」「換一個版本」「聲調不對但字沒錯」 | `clone.py ... --takes 4` → 把 `audition.wav` 交給使用者聽 → `audition.py --pick` |
| 「我要錄聲音」 | 引導使用者雙擊 `start.bat`，或開啟 http://127.0.0.1:7860 |

## 全域技能 voice-clone

讓四家 Agent 在**任何工作目錄**都能用「用小吳的聲音念出……」這句話直接生成語音。

- **原始檔**：`skills/voice-clone/SKILL.md`（本專案內）。`skills/` 已加進 `.gitignore`——技能含本機絕對路徑與個人聲音名稱，**不進這個公開 repo**，跨電腦只靠 Google 雲端硬碟同步
- **安裝副本四份**：Claude Code `~/.claude/skills/`、Codex `~/.agents/skills/`、OpenCode `~/.config/opencode/skills/`、Antigravity `~/.gemini/config/skills/`
- **改完原始檔**說一句「同步技能」，交給全域技能 `skill-sync` 覆蓋四份副本
- **換電腦的第一次**：`skill-sync` 只覆蓋已裝過的副本，所以會回報 `voice-clone` 未安裝並徵求同意首裝——答應即可
- 技能行為：預設聲音 `小吳`、文字走暫存檔＋`--file`（避開 PowerShell 標點問題）、輸出帶時間戳（不蓋掉上一次）、把音檔本身交給使用者

## 三個檔案的職責（依「時效性」分家，不是依「詳細程度」）

| 檔案 | 時效 | 寫入方式 | 放什麼 |
|------|------|---------|--------|
| `handoff.md` | **只對下一個 session 有效**，過期即丟 | 每次收工整份重寫 | 做到哪、下一步、**這次**的暫時 workaround |
| `AGENTS.md`（本檔） | **長期有效**，每個 session 都適用 | 只有規則本身變了才改 | 目標、路線圖、常設規則、結構 |
| Obsidian／`git log` | **歷史**：發生過什麼、為什麼 | 只增不刪 | 決策紀錄、踩坑完整版、逐次進度 |

驗收標準：**`handoff.md` 整份刪掉，不應損失任何長期資訊**——會的話代表該升級進本檔卻沒升級。

**本檔不要出現的東西**：❌ `## 最近進度`／逐次工作紀錄、❌ 決策理由與踩坑完整版。歷史寫 L3 筆記的〈🗓️ 最近更動紀錄〉〈🧠 決策紀錄〉〈🕳️ 踩坑筆記〉；踩過的坑只把**結論**收斂成一條祈使句寫進〈工作約定〉，原因留 L3。

## 專案專屬規則

- **`skills/` 不進版控**。`voice-clone` 的原始檔在 `skills/voice-clone/`，含本機絕對路徑與個人聲音名稱，而 repo 是公開的，所以已 gitignore。**跨電腦只靠 GDrive 同步**（跟 `handoff.md` 同一套做法）——換電腦前確認雲端同步跑完。`voices/`、`output/` 同樣不進版控
- **不要在專案裡建 `.venv`**。專案在雲端硬碟同步範圍內，venv 寫死 base Python 的絕對路徑，換電腦或 uv 搬動 Python 就壞（2026-10-28 LT-YI-RS 的 `.venv` 指向已不存在的 `AppData\Roaming\uv\python\...`）。環境一律在本機共用位置，`install.ps1` 會順手刪掉專案裡的舊 `.venv`
- **改 `requirements-win-arm64.txt` 的版本前，先用另一個 venv 驗證生成**：`install.ps1` 就地更新後若驗證失敗，會刪掉 GPU 環境再退回 x64，沒有備份
- **不要開 `optimize=True`**。本機沒有 triton／`cl.exe`，`torch.compile` 會被 `try/except` 靜默降級（只印 `Warning: torch.compile disabled - triton is not installed`），速度零改善卻多花 3.3 秒載入
- **模型載入偶爾變 60～90 秒是正常的**：HuggingFace 快取在重新驗證（Windows 無 symlink），不是壞掉，等它跑完即可。正常約 9.7 秒
- **全域技能不能用相對路徑**：venv 直譯器的相對路徑在非專案目錄會 `CommandNotFoundException`，一律用絕對路徑（兩平台的檔名不同，先偵測再組）
- **沙箱會擋下 `Remove-Item -Recurse -Force`**（回報 `path '*' is blocked`），改用 `[System.IO.Directory]::Delete($p,$true)`
- **Smart App Control 在 PC-YI-SL、LT-YI-RS 驗證過**：PC-YI-SL 生成路徑不受影響；LT-YI-RS 開著時要靠 `sac-pins.txt`，現已關閉（見〈環境限制〉）
- **從 Claude 桌面版跑的程式寫不到真正的 `%LOCALAPPDATA%`**：它是 MSIX 封裝，新資料夾會被虛擬化到 `Packages\Claude_*\LocalCache\Local\`，檔案總管、`start.bat`、其他 Agent 都看不到。要跨程式共用的東西放 `~\.local\share\` 之類的使用者資料夾
- **所有檔案一律 UTF-8 無 BOM**（`.ps1` 也是）。舊的「`.ps1` 要存成 UTF-8 with BOM」規則前提是 PowerShell 5.1，已隨 pwsh 7 定案而廢止；`SKILL.md` 有 BOM 會讓 frontmatter 解析失敗、技能觸發不了。詳見 `agent-project-flow/platform.md`
- repo 為**公開**，新增檔案前先確認不含個資

## 工作約定

- 任何 Agent、任何電腦：**開工先讀 `handoff.md`，收工必更新 `handoff.md`**
- 修改共用檔案前先讀最新內容，避免覆蓋其他 Agent 的變更
- 所有回應與文件使用繁體中文
- 修改前先確認計畫，優先保留原有資料結構
- **產出一律放 `output/`**：本專案產生的檔案或資料夾（成品、匯出、報表、生成圖、渲染等）預設放專案根目錄的 `output/`，不另開 `outputs/`、`exports/`、`generated/`、`out/` 之類的資料夾，方便找；`output/` 不進 git。部署工具指定的建置目錄（`dist/`、`build/`）不在此限

## Claude Code 專屬

- 執行 `record_ui.py` 或任何伺服器時用背景執行，用完記得停掉並確認 port 已釋放。
- 生成語音後用 SendUserFile 把音檔交給使用者試聽，不要只回報路徑。
