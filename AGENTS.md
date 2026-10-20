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

## 資料夾結構

```
voxcpm2-voice-cloner/
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
├── install.bat / install.ps1 # 自動偵測 GPU + 安裝依賴
├── .gpu_type                 # 安裝時寫入的 GPU 模式標記
├── texts/sample_text.txt     # 錄音時朗讀的文字
├── skills/voice-cloner/      # 全域技能原始檔（不進版控，見下方「全域技能」）
├── voices/                   # 已錄製的聲音（不進版控）
└── output/                   # 生成的語音（不進版控）
```

## 同步層級（本專案初始化至第 3 層級）

| 層級 | 平台 | 位置 | 讀取時機 |
|------|------|------|---------|
| L1 | 本地（GDrive） | `AGENTS.md`＋`handoff.md`＋`CLAUDE.md`（橋接） | 每個 session |
| L2 | GitHub | [changyiwu/voxcpm2-voice-cloner](https://github.com/changyiwu/voxcpm2-voice-cloner)（**公開**） | 指定時 |
| L3 | Obsidian | `voxcpm2-voice-cloner/專案工作流程.md` | 有需要時 |

## 環境

| 項目 | 值 |
|------|-----|
| 專案目錄 | 雲端硬碟的 `agents/voxcpm2-voice-cloner`（實際絕對路徑每台不同，一律以 cwd 為準） |
| Python | Windows `.venv\Scripts\python.exe`／macOS `.venv/bin/python`（專案目錄下，由 `install.ps1` 建立） |
| 模型 | `openbmb/VoxCPM2`（Apache-2.0；權重已在 HuggingFace 快取） |
| 裝置 | 三種模式自動切換：NVIDIA CUDA／Apple Silicon MPS／CPU。安裝時寫進 `.gpu_type`，`clone.py` 與 `dialogue.py` 讀它 |
| 效能 | 模型載入 ~9.7 秒（每次執行都要重載），尖峰 VRAM 6.46GB。RTF 隨稿長變化：719 字長稿 **1.0**、中等句 1.1~1.3、10 字以內短句 2.5~2.9（固定開銷攤不掉）。**長稿一次念完比拆成多次划算** |
| 輸出 | `output/cloned_voice.wav`（`--output` 可改） |

> 所有指令都以專案目錄為工作目錄執行。若尚未安裝（`.venv` 不存在）：Windows 請使用者雙擊 `install.bat`，macOS 跑 `pwsh -File install.ps1`（`.bat` 是 Windows 專屬啟動器，mac 沒有）。

### ⚠️ 環境限制：Smart App Control（PC-YI-SL）

本機 Smart App Control 處於**強制執行模式**，Code Integrity 會封鎖 PyPI wheel 中未簽章的原生程式庫。實測被擋：**pandas、pyarrow、datasets、gradio**，錯誤訊息為
`ImportError: DLL load failed ...: 應用程式控制原則已封鎖此檔案`。

- 這**不是**防毒問題。在 Defender 加排除項目無效——攔截來自核心模式的 Code Integrity，不是防毒引擎，且 Smart App Control 設計上不提供白名單。
- 因此 `app.py` 與 `webui_record.py`（皆依賴 gradio）**在本機無法啟動**，保留僅為相容其他電腦。
- 錄音一律用 **`record_ui.py`**：只用 Python 標準庫的 `http.server`，完全避開 pandas/pyarrow。
- torch、torchaudio、voxcpm、transformers、soundfile、sounddevice、numpy、resampy 均不受影響；`clone.py` 的生成路徑已實測不受影響。

> Agent 若看到上述 ImportError，**不要重裝套件**，那不會有幫助；改走不依賴該套件的路徑。

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
.\.venv\Scripts\python.exe clone.py "同學們早安，今天我們來上數學課。" --voice 小吳
```

常用參數：

- `--voice <名稱>`：對應 `voices/<名稱>/` 目錄（預設 `小吳`）
- `--file <路徑>`：從文字檔讀取要生成的內容
- `--output <路徑>`：輸出路徑（預設 `output/cloned_voice.wav`）
- `--device cuda|cpu`：強制指定裝置（預設自動偵測）
- `--takes N`：同一句生 N 個候選（seed 1…N）到 `<輸出檔名>_takes/`，並串好 `audition.wav`；用 `audition.py <資料夾> --pick sK -o <正式檔>` 挑一支。使用者說「念得怪怪的」「換一個」時用這個，不要只重跑一次碰運氣
- `--raw`：跳過後處理（預設：去頭尾靜音、highpass 70 Hz＋lowpass 11 kHz、響度 −15 LUFS、取樣峰值 ≤ −1 dBFS）
- `--dry-run`：只印送進模型的文字（讀音替換表＋繁轉簡），不載模型

送進模型前一律經 `voice_prep.to_model_text()`：讀音替換表 → 繁轉簡（VoxCPM2 念簡體比較穩，ai-video-workflow 實測；沒裝 opencc 就跳過）。生成後經 `voice_prep.finish()`。響度 −15 LUFS 跟 agent-video-specs、ai-video-workflow 的成片目標一致。嘶聲（4 kHz 以上能量比）要在濾波**前**量，`--takes` 會記在每支候選旁的 `.json`，`audition.py` 優先用它。

### dialogue.py — 用多個聲音生成對話

```powershell
.\.venv\Scripts\python.exe dialogue.py
```

- 換聲音：改腳本頂部的 `SPEAKER_A` / `SPEAKER_B` 兩個常數（預設 `小吳` / `老柯`），對話清單與輸出檔名都會跟著走
- 輸出：`output/dialogue_<A>_<B>.wav`
- 自訂對話內容：編輯 `dialogue.py` 中的 `dialogue` 清單
- 每句各自經 `voice_prep.finish()`（去頭尾靜音、濾波、−15 LUFS）再拼接，句間固定 0.4 秒，兩個聲音不會一大一小
- 模型只載入一次、逐句切換聲音，攤下來每句只多約 10 秒。實測 8 句 35.8 秒對話共 59 秒（**第一句是暖機、會特別慢**，之後穩定在 RTF ~1.1）

### tts_lexicon.py — 讀音替換表

`clone.py`、`dialogue.py` 送進模型前會先套 `tts-lexicon.md`（`| 原文 | 改寫 | 說明 |`），把念錯的多音字、英文詞改字（奇數→基數、十一少一→十一少了一、GPT-4→GPT four），並印出「改寫後」的文字。參考音逐字稿（`prompt.txt`）不替換。

- **表在本專案根目錄**（`tts-lexicon.md`）。這份表有**四份、內容逐位元組相同**（另外三份在 agent-video-specs、ai-video-workflow、agent-speak-skill/speak/；四個 repo 都公開，各自單獨下載就要能用）。`tts_lexicon.py` 以自己的位置為起點（不靠 cwd）往上找另外三份，有就一起讀，本專案這份優先；設了 `TTS_LEXICON` 就改讀它指的表來取代本專案這份
- **覆蓋**：輸出資料夾往上每一層的同名 `tts-lexicon.md`，近的優先（只對某批輸出有效的條目放這裡）
- 找不到任何表只印一行提示、照原文念，不報錯
- **使用者說某個詞念錯了** → 條目**四份一起加**（說明欄寫正確讀音、出處、是否已人耳確認），用 `python ../ai-video-workflow/tools/tts_lexicon.py --check`／`--sync` 檢查與合併。加完重生那句給使用者聽
- 試表不必載模型：`.\.venv\Scripts\python.exe tts_lexicon.py "句子"`
- 本專案是獨立公開 repo，**不要 import 別的 repo 的程式**；解析規則改了四個 repo 各自改

### record_ui.py — 錄音 UI

```powershell
.\.venv\Scripts\python.exe record_ui.py --open
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

執行 `clone.py` / `dialogue.py` 時自動偵測，順序為：讀取 `.gpu_type`（`install.ps1` 產生）→ 若無則用 torch 偵測。

- NVIDIA GPU → CUDA
- 無獨顯 → CPU（較慢）
- Windows on ARM（含 RTX Spark 的 NVIDIA GPU）→ 一律 CPU，`.venv` 必須是 **x64** 環境（`install.ps1` 會指定 `cpython-3.12-windows-x86_64-none`）。原生 ARM64 環境裝不起 voxcpm（`torchaudio`、`pyarrow`、`tiktoken`、`soxr` 都沒有 win_arm64 wheel），不要嘗試
- `.gpu_type` 跟著雲端硬碟同步、可能是別台寫的，所以 `detect_device()` 讀到後會再確認一次這台真的可用，不可用就退回 CPU

> 只支援 CUDA 與 CPU 兩種模式，不支援 Intel Arc（XPU）。`.gpu_type` 若出現 `xpu` 等無效值，一律退回 CPU。

## 使用者自然語言對照

| 使用者說的 | Agent 動作 |
|-----------|-----------|
| 「用OOO的聲音念出／說出／講出XXX」 | 走 `voice-cloner` 全域技能（見下節）；沒裝技能就直接 `clone.py "XXX" --voice OOO` |
| 「讓A和B對話」 | 編輯 `dialogue.py` 對話清單 → 執行 |
| 「有哪些聲音」 | `ls voices\` |
| 「XX 念錯了」「XX 要念成 OO」 | 四個 repo 的讀音替換表一起加一列 → 重生那句（見〈tts_lexicon.py〉） |
| 「念得怪怪的」「換一個版本」「聲調不對但字沒錯」 | `clone.py ... --takes 4` → 把 `audition.wav` 交給使用者聽 → `audition.py --pick` |
| 「我要錄聲音」 | 引導使用者雙擊 `start.bat`，或開啟 http://127.0.0.1:7860 |

## 全域技能 voice-cloner

讓四家 Agent 在**任何工作目錄**都能用「用小吳的聲音念出……」這句話直接生成語音。

- **原始檔**：`skills/voice-cloner/SKILL.md`（本專案內）。`skills/` 已加進 `.gitignore`——技能含本機絕對路徑與個人聲音名稱，**不進這個公開 repo**，跨電腦只靠 Google 雲端硬碟同步
- **安裝副本四份**：Claude Code `~/.claude/skills/`、Codex `~/.agents/skills/`、OpenCode `~/.config/opencode/skills/`、Antigravity `~/.gemini/config/skills/`
- **改完原始檔**說一句「同步技能」，交給全域技能 `sync-skills` 覆蓋四份副本
- **換電腦的第一次**：`sync-skills` 只覆蓋已裝過的副本，所以會回報 `voice-cloner` 未安裝並徵求同意首裝——答應即可
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

- **`skills/` 不進版控**。`voice-cloner` 的原始檔在 `skills/voice-cloner/`，含本機絕對路徑與個人聲音名稱，而 repo 是公開的，所以已 gitignore。**跨電腦只靠 GDrive 同步**（跟 `handoff.md` 同一套做法）——換電腦前確認雲端同步跑完。`voices/`、`output/` 同樣不進版控
- **不要開 `optimize=True`**。本機沒有 triton／`cl.exe`，`torch.compile` 會被 `try/except` 靜默降級（只印 `Warning: torch.compile disabled - triton is not installed`），速度零改善卻多花 3.3 秒載入
- **模型載入偶爾變 60～90 秒是正常的**：HuggingFace 快取在重新驗證（Windows 無 symlink），不是壞掉，等它跑完即可。正常約 9.7 秒
- **全域技能不能用相對路徑**：venv 直譯器的相對路徑在非專案目錄會 `CommandNotFoundException`，一律用絕對路徑（兩平台的檔名不同，先偵測再組）
- **沙箱會擋下 `Remove-Item -Recurse -Force`**（回報 `path '*' is blocked`），改用 `[System.IO.Directory]::Delete($p,$true)`
- **Smart App Control 的限制目前只在 PC-YI-SL 驗證過**，其他電腦未實測；生成路徑不受影響
- **所有檔案一律 UTF-8 無 BOM**（`.ps1` 也是）。舊的「`.ps1` 要存成 UTF-8 with BOM」規則前提是 PowerShell 5.1，已隨 pwsh 7 定案而廢止；`SKILL.md` 有 BOM 會讓 frontmatter 解析失敗、技能觸發不了。詳見 `cross-device-agent-skills/platform.md`
- repo 為**公開**，新增檔案前先確認不含個資

## 工作約定

- 任何 Agent、任何電腦：**開工先讀 `handoff.md`，收工必更新 `handoff.md`**
- 修改共用檔案前先讀最新內容，避免覆蓋其他 Agent 的變更
- 所有回應與文件使用繁體中文
- 修改前先確認計畫，優先保留原有資料結構

## Claude Code 專屬

- 執行 `record_ui.py` 或任何伺服器時用背景執行，用完記得停掉並確認 port 已釋放。
- 生成語音後用 SendUserFile 把音檔交給使用者試聽，不要只回報路徑。
