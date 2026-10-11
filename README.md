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
2. 建立 Python 3.12 虛擬環境 `.venv`
3. 偵測加速裝置（CUDA／MPS／CPU）
4. 安裝對應版本的 PyTorch + voxcpm

### 2. 錄音

Windows 雙擊 `start.bat`；macOS 跑 `./.venv/bin/python record_ui.py --open`。
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

> **兩個平台的直譯器路徑不同**，以下範例寫的是 Windows 形式：
> - Windows：`.\.venv\Scripts\python.exe`
> - macOS：`./.venv/bin/python`
>
> 安裝腳本跑完會把這台該用的那一個印在「下一步」，照著複製即可。

### 錄製參考音

**方式 A：網頁介面**

```powershell
.\.venv\Scripts\python.exe record_ui.py --open
```

瀏覽器開啟後，有錄音按鈕、逐字稿顯示、試聽，滿意再存檔。

**方式 B：命令列**

```powershell
.\.venv\Scripts\python.exe record.py --voice 我的聲音
```

螢幕會顯示一段文字，對著麥克風自然地朗讀，念完按 Enter 停止。

### 生成克隆語音

```powershell
.\.venv\Scripts\python.exe clone.py "你好，這是我的克隆聲音。" --voice 我的聲音
```

或從文字檔生成：

```powershell
.\.venv\Scripts\python.exe clone.py --file my_script.txt
```

輸出檔案預設在 `output/cloned_voice.wav`。

### 生成多聲音對話

```powershell
.\.venv\Scripts\python.exe dialogue.py
```

對話內容請直接編輯 `dialogue.py` 中的 `dialogue` 清單。

## 目錄結構

```
voxcpm2-voice-cloner/
├── record_ui.py              # 錄音 UI（唯一介面，僅用標準庫）
├── app.py                    # 舊版錄音 UI（gradio，見下方注意事項）
├── webui_record.py           # 舊版網頁錄音（gradio）
├── clone.py                  # Agent 工具：用聲音生成語音
├── dialogue.py               # Agent 工具：多聲音對話
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
| Windows on ARM（RTX Spark、Snapdragon） | cpu | x64 Python 模擬＋cpu wheel | 比 x64 CPU 更慢（未實測） |

> 本專案只支援 NVIDIA CUDA 與 CPU。Intel Arc（XPU）支援已於 2026-08-08 移除。
>
> Windows on ARM 即使有 NVIDIA GPU 也只能跑 CPU：`torchaudio`、`pyarrow`、`tiktoken` 等相依沒有原生 ARM64 版，
> `install.ps1` 會自動改建 x64 環境靠 Prism 模擬執行，而模擬中的程式用不到 GPU。

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
