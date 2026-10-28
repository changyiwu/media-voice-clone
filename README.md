# VoxCPM2 Voice Cloner

用 VoxCPM2 克隆你的聲音，生成任意語音。全自動安裝，自動偵測 GPU。

**錄音走 UI，其他全部透過 AI Agent 自然語言操作。**

## 特色

- **自動偵測加速裝置**：NVIDIA CUDA / Apple Silicon MPS / CPU 三種模式自動切換
- **Ultimate Cloning**：同時使用參考音 + 逐字稿，連語氣節奏都一起複製
- **網頁錄音**：`record_ui.py` 提供簡潔錄音介面（取名 → 看稿 → 錄音 → 儲存），零額外依賴
- **自然語言操作**：錄完後，直接對 AI Agent 說「用王老師的聲音說一段話」，Agent 自動呼叫工具
- **Apache-2.0 授權**：VoxCPM2 模型可商用

## 系統需求

- Windows 10/11 或 macOS
- PowerShell 7（`pwsh`）；macOS 用 `brew install --cask powershell` 安裝
- Python 3.10–3.12（安裝腳本會用 uv 自動建立 3.12 環境）
- 加速裝置（擇一）：
  - NVIDIA GPU（CUDA 12+，約 8GB VRAM）
  - Apple Silicon（M 系列，走 Metal／MPS）
  - 無獨顯也可用 CPU（較慢，RTF 約 8x）
- 約 5GB 硬碟空間（模型權重）
- 麥克風

## 快速開始（雙擊即可）

### 1. 安裝

Windows 雙擊 `install.bat`（或 `install.ps1`）；macOS 跑 `pwsh -File install.ps1`。腳本會自動完成：

1. 檢查／安裝 uv 套件管理器
2. 在本機建立 Python 虛擬環境 `~/.local/share/media-voice-clone/.venv`（Windows 的 `~` 是使用者資料夾）。不建在專案裡，因為專案常放在雲端同步資料夾，venv 換電腦就壞。Windows on ARM＋NVIDIA 會建原生 ARM64＋CUDA 環境（Python 3.13），不行就改建 x64 CPU 環境
3. 偵測加速裝置（CUDA／MPS／CPU）
4. 安裝對應版本的 PyTorch + voxcpm

### 2. 錄音

Windows 雙擊 `start.bat`；macOS 跑 `~/.local/share/media-voice-clone/.venv/bin/python record_ui.py --open`。
瀏覽器打開 → 取名 → 對著麥克風念稿 → 儲存。

> `install.bat` 與 `start.bat` 是 Windows 專屬的雙擊啟動器，macOS 沒有對應檔案，照上面的指令直接跑即可。

### 3. 使用（透過 AI Agent）

錄完後，直接對 AI Agent 說：

```
用王老師的聲音說「同學們早安，今天我們來上數學課」
```

Agent 會自動找到對應聲音、生成語音、回傳音檔。

> 💡 本專案設計為 **AI Agent 工具包**，人類只做錄音，其他交給 Agent。

## 命令列工具（替代方案，不需 GUI 時可用）

> 以下範例都在專案資料夾執行，並先把 `$py` 設成這台的直譯器（每開一個新終端機都要設一次）：
>
> ```powershell
> $py = "$HOME\.local\share\media-voice-clone\.venv\Scripts\python.exe"   # Windows
> $py = "$HOME/.local/share/media-voice-clone/.venv/bin/python"         # macOS
> ```
>
> 安裝腳本跑完也會把這台的完整路徑印在「下一步」。

### 錄製參考音

**方式 A：網頁介面**

```powershell
& $py record_ui.py --open
```

瀏覽器開啟後，有錄音按鈕、逐字稿顯示、試聽，滿意再存檔。

**方式 B：命令列**

```powershell
& $py record.py --voice 我的聲音
```

螢幕會顯示一段文字，對著麥克風自然地朗讀，念完按 Enter 停止。

### 生成克隆語音

```powershell
& $py clone.py "你好，這是我的克隆聲音。" --voice 我的聲音
```

或從文字檔生成：

```powershell
& $py clone.py --file my_script.txt
```

輸出檔案預設在 `output/cloned_voice.wav`。

生成後會自動做**後處理**（`voice_prep.py`）：去頭尾靜音、濾掉低頻轟聲與 11 kHz 以上的嘶聲、響度調到 **−15 LUFS**（取樣峰值不超過 −1 dBFS）。不要後處理就加 `--raw`。送進模型前的文字會先套讀音替換表、再**繁轉簡**（VoxCPM2 念簡體比較穩；需要 `opencc-python-reimplemented`，`install.ps1` 會裝，沒裝就跳過）。

**同一句多生幾個挑一個**：VoxCPM2 每次生成都有隨機性，同一句換個 seed，聲調、停頓、有沒有嘶聲都不一樣。

```powershell
& $py clone.py "這句話念了好幾次都怪怪的" --takes 4 -o output/line.wav
#   → output/line_takes/line_s1.wav … line_s4.wav，以及串好的 output/line_takes/audition.wav
& $py audition.py output/line_takes --pick s3 -o output/line.wav
```

`audition.wav` 用嗶聲隔開各候選，印出來的表有每支「從第幾秒開始」、響度與高頻比；高頻比異常的標「可能有嘶聲」（門檻沿用 media-short-film 的挑音規則），但最後還是以耳朵為準。只想看送進模型的文字（不載模型）：`clone.py "句子" --dry-run`。

### 生成多聲音對話

```powershell
& $py dialogue.py
```

對話內容請直接編輯 `dialogue.py` 中的 `dialogue` 清單。每句會各自做後處理、調到同一個響度（−15 LUFS）再拼接，兩個人的聲音不會一大一小。

### 讀音替換表（念錯字時）

VoxCPM2 不支援注音標記，多音字、英文詞念錯只能改字。`clone.py`、`dialogue.py` 送進模型前會先套讀音替換表 `tts-lexicon.md`（Markdown 表格 `| 原文 | 改寫 | 說明 |`），並印出改寫後的文字；參考音的逐字稿不受影響。

| 原文 | 改寫 | 原因 |
|---|---|---|
| 奇數 | 基數 | 奇讀 jī |
| 十一少一 | 十一少了一 | 少讀 shǎo |
| GPT-4 | GPT four | 英文型號的數字用英文念 |

表放在本專案根目錄（`tts-lexicon.md`），單獨下載本 repo 就能用。同一份表在 [media-video-specs](https://github.com/changyiwu/media-video-specs)、[media-short-film](https://github.com/changyiwu/media-short-film)、[agent-speak](https://github.com/changyiwu/agent-speak) 也各有一份、內容相同。

找表的順序（近的覆蓋遠的，同一個原文以近的為準）：

1. 同層另外三份（四個 repo 放在同一個資料夾底下時，有就一起讀）
2. 本專案根目錄的 `tts-lexicon.md`（設了環境變數 `TTS_LEXICON` 就改讀它指的表）
3. 輸出資料夾往上每一層的 `tts-lexicon.md`

一份都找不到就印一行提示、照原文念。改寫盡量用同音字或補字；新增條目後重生那句實際聽過。試表不必載入模型：

```powershell
& $py tts_lexicon.py "奇數加奇數是偶數，十比十一少一。"
```

## 目錄結構

```
media-voice-clone/
├── record_ui.py              # 錄音 UI（唯一介面，僅用標準庫）
├── app.py                    # 舊版錄音 UI（gradio，見下方注意事項）
├── webui_record.py           # 舊版網頁錄音（gradio）
├── clone.py                  # Agent 工具：用聲音生成語音
├── dialogue.py               # Agent 工具：多聲音對話
├── tts_lexicon.py            # 讀音替換表載入（clone.py／dialogue.py 共用）
├── tts-lexicon.md            # 讀音替換表（四個 TTS repo 各一份、內容相同）
├── voice_prep.py             # 生成前後處理：繁轉簡、去靜音、濾波、−15 LUFS
├── audition.py               # 候選串成試聽檔、挑一支（搭配 clone.py --takes）
├── record.py                 # 命令列錄音（備案）
├── start.bat                 # 雙擊啟動錄音 UI
├── install.bat               # 雙擊安裝
├── install.ps1               # 自動偵測 GPU + 安裝依賴
├── AGENTS.md                 # Agent 使用指南
├── texts/sample_text.txt     # 錄音時朗讀的文字
├── voices/                   # 已錄製的聲音（本地，不進版控）
└── output/                   # 生成的語音
```

## GPU 支援對照

| GPU | 模式 | PyTorch | 效能 |
|-----|------|---------|------|
| NVIDIA RTX 5060 Ti | cuda | cu128 wheel | **RTF ~1.2（實測）** |
| NVIDIA (CUDA 12+) | cuda | cu128 wheel | RTF ~0.3（RTX 4090，未實測） |
| 無獨顯 | cpu | cpu wheel | RTF ~8.0（未實測） |
| Windows on ARM＋NVIDIA（RTX Spark） | cuda | 原生 ARM64＋NVIDIA 預覽版 cu134 wheel（Python 3.13） | **RTF ~1.7（實測）** |
| Windows on ARM（無 NVIDIA，或上面失敗時的退路） | cpu | x64 Python 模擬＋cpu wheel | 極慢：RTF 約 42（5.9 秒語音要 4 分鐘，實測），只適合短句 |

> 本專案只支援 NVIDIA CUDA 與 CPU。Intel Arc（XPU）支援已於 2026-08-08 移除。
>
> Windows on ARM＋NVIDIA GPU：`install.ps1` 建原生 ARM64 環境，裝 NVIDIA 的 Windows on ARM 預覽版 PyTorch
> （[NVIDIA/pytorch-windows-ci](https://github.com/NVIDIA/pytorch-windows-ci/releases)，固定在 `requirements-win-arm64.txt` 的那一版），
> 只裝生成用得到的套件。失敗會自動退回 x64 CPU 環境（Prism 模擬，用不到 GPU），原本的環境會先保留、失敗時還原。

> RTF = 生成 N 秒語音所需的時間倍率，越低越快。

RTX 5060 Ti 的實測細節（16GB，`inference_timesteps=10`、`cfg_value=2.0`）：

| 階段 | 數字 |
|------|------|
| 模型載入 | 約 10 秒（每次執行 `clone.py` 都要重載一次） |
| 每個行程的第一句 | RTF ~1.9 |
| 之後每句 | RTF ~1.2 |
| 尖峰 VRAM | 6.46 GB |

> 電腦上第一次跑 CUDA 時會額外建立 kernel 快取，那一次可能到 RTF ~3.5，屬正常現象，之後不會再發生。

## Smart App Control 注意事項

若你的 Windows 開啟了 **Smart App Control**（Windows 11 預設可能為開啟），Code Integrity 會封鎖 PyPI wheel 中未簽章的原生程式庫，症狀是：

```
ImportError: DLL load failed while importing ...: 應用程式控制原則已封鎖此檔案。
```

受影響的是 **pandas / pyarrow / datasets / gradio**，也就是舊版的 `app.py` 與 `webui_record.py` 會無法啟動。torch、voxcpm、soundfile、sounddevice 等核心套件不受影響。

`record_ui.py` 就是為此而寫：只用 Python 標準庫的 `http.server`，不經過 pandas/pyarrow，因此在 Smart App Control 開啟的機器上照常運作。**預設的 `start.bat` 已指向它，一般情況下你不需要做任何事。**

> 注意：在 Defender 新增排除項目**無法**解決此問題——攔截來自核心模式的 Code Integrity，而非防毒引擎，且 Smart App Control 不提供白名單機制。關閉 Smart App Control 是可行但**不可逆**的（除非重灌 Windows 才能再次啟用），不建議只為了錄音介面而關閉。

## 授權

- VoxCPM2 模型與程式碼：[Apache-2.0](https://github.com/OpenBMB/VoxCPM/blob/main/LICENSE)（可商用）
- 本專案腳本：[MIT](LICENSE)

### 衍生來源

本專案衍生自 [mathruffian-dot/voxcpm2-voice-cloner](https://github.com/mathruffian-dot/voxcpm2-voice-cloner)。該 repo 未附 LICENSE 檔（GitHub 判定為「無授權」），僅在 README 中聲明「本專案腳本：MIT」。本專案的 [LICENSE](LICENSE) 由本 repo 作者針對本 repo 內容做出 MIT 授權；若要再散布衍生自上游的部分，建議另向上游作者確認。詳見 [NOTICE](NOTICE)。
