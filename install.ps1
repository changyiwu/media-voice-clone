# install.ps1 - VoxCPM2 Voice Cloner 自動安裝腳本
# 自動偵測加速裝置，安裝對應的 PyTorch + voxcpm
#
# 用法：Windows  pwsh -File install.ps1（或雙擊 install.bat）
#       macOS    pwsh -File install.ps1
#
# 環境建在本機、不建在專案裡：專案在雲端硬碟同步資料夾，而 venv 會寫死 base Python 的絕對路徑，
# 換電腦（或 uv 搬了 Python 的位置）就壞。
#   ~/.local/share/media-voice-clone/.venv（兩平台相同，Windows 的 ~ 是使用者資料夾）
#
# 裝置偵測邏輯：
#   NVIDIA (CUDA)      → pip install torch --index-url .../cu128
#   Apple Silicon      → pip install torch（PyPI 預設 wheel 已含 MPS，不可加 --index-url）
#   無獨顯 (CPU)       → pip install torch --index-url .../cpu
#   Windows on ARM     → 有 NVIDIA GPU（RTX Spark）：先建原生 ARM64＋CUDA（Python 3.13，NVIDIA 的預覽版
#     ＋NVIDIA GPU       PyTorch，套件見 requirements-win-arm64.txt）。失敗就退回 x64 CPU，原本的環境會先改名
#                        保留、失敗時還原。模擬中的 x64 載不到 ARM64 的 NVIDIA 驅動，所以 GPU 只有原生這條路
#   Windows on ARM     → 先試原生 ARM64 CPU，不行就改建 x64（Prism 模擬）。2026-10-28 實測原生 CPU 仍不行：
#     沒有 NVIDIA GPU    torchaudio 官方沒有 win_arm64 wheel

param(
    # 測試退回路徑用：指向一份會失敗的需求檔（例如 sha256 故意寫錯），確認原本的環境會被還原
    [string]$ArmRequirements = (Join-Path $PSScriptRoot 'requirements-win-arm64.txt')
)

$ErrorActionPreference = 'Stop'

if ($null -eq $IsWindows) {
    throw '需要 PowerShell 7（pwsh）；5.1 沒有 $IsWindows，平台判斷會靜默走錯分支'
}

# 兩平台同一個位置。Windows 不用 %LOCALAPPDATA%：從 Claude 桌面版（MSIX 封裝）裡跑時，寫進
# %LOCALAPPDATA% 的新資料夾會被虛擬化到 Packages/Claude_*/LocalCache，檔案總管和其他 Agent 都看不到
# （2026-10-28 LT-YI-RS 實測）；~/.local 不會。macOS 不用 ~/Library/Application Support：路徑有空白
$venvDir = Join-Path $HOME '.local' 'share' 'media-voice-clone' '.venv'
$venvPython = if ($IsWindows) { Join-Path $venvDir 'Scripts' 'python.exe' } else { Join-Path $venvDir 'bin' 'python' }
# 下面的 uv venv 都加 --python-preference only-managed（不用 --managed-python：設了 UV_PYTHON_PREFERENCE 會衝突）：venv 寫死 base Python 的路徑，用使用者自己裝的 Python 的話，
# 哪天移除或升級它環境就壞（2026-10-28 LT-YI-RS 的舊 .venv 就是這樣壞的）

# 只 import 不夠：Smart App Control 會擋延遲載入的原生模組，所以真的跑一下 torch、scipy、librosa
$smokeTest = @'
import torch, torchaudio, soundfile, sounddevice, resampy, opencc, librosa, numpy
from scipy.signal import butter, sosfiltfilt
from scipy.ndimage import minimum_filter1d
from voxcpm import VoxCPM
x = numpy.random.rand(16000).astype('float32')
sosfiltfilt(butter(4, 70, 'highpass', fs=16000, output='sos'), x)
minimum_filter1d(x, 5)
librosa.resample(x, orig_sr=16000, target_sr=48000)
(torch.from_numpy(x) * 2).sum().item()
'@
# GPU 環境另外要求 CUDA 真的能算：預覽版 torch 載得起來不代表驅動對得上
$gpuSmokeTest = $smokeTest + @'

assert torch.cuda.is_available(), 'torch.cuda.is_available() is False'
a = torch.randn(256, 256, device='cuda', dtype=torch.bfloat16)
(a @ a).float().sum().item()
'@

Write-Host ''
Write-Host '============================================' -ForegroundColor Cyan
Write-Host '  VoxCPM2 Voice Cloner - Auto Installer' -ForegroundColor Cyan
Write-Host '============================================' -ForegroundColor Cyan
Write-Host ''
Write-Host "環境位置：$venvDir"
Write-Host ''

# --- Step 1: 檢查 uv ---
Write-Host '[1/4] 檢查 uv 套件管理器...' -ForegroundColor Yellow
$uv = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uv) {
    Write-Host '  uv 未安裝，正在安裝...' -ForegroundColor Yellow
    pip install -U uv
} else {
    Write-Host "  uv 已安裝: $($uv.Source)" -ForegroundColor Green
}

# 看作業系統的架構，不看 pwsh 自己的：x64 版 pwsh 在 ARM 上模擬執行時，ProcessArchitecture 會回報 X64
$isWindowsArm = $IsWindows -and
    [Runtime.InteropServices.RuntimeInformation]::OSArchitecture -eq [Runtime.InteropServices.Architecture]::Arm64

# --- Step 2: 偵測加速裝置 ---
Write-Host '[2/4] 偵測加速裝置...' -ForegroundColor Yellow
$gpuType = 'cpu'
$gpuName = ''
$armGpu = $false

if ($IsWindows) {
    # Get-CimInstance Win32_VideoController 是 WMI，只有 Windows 有
    $videoControllers = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue
    foreach ($vc in $videoControllers) {
        if ($vc.Name -match 'NVIDIA') {
            $gpuType = 'cuda'
            $gpuName = $vc.Name
            break
        }
    }
    if ($isWindowsArm -and $gpuType -eq 'cuda') {
        # 先當 CPU 處理（下面的 x64 退路用 CPU 版 PyTorch），GPU 環境建成功才改回 cuda
        Write-Host "  偵測到 $gpuName（Windows on ARM）。" -ForegroundColor Green
        Write-Host '  → 先建原生 ARM64＋CUDA 環境（NVIDIA 預覽版 PyTorch），不行再退回 x64 CPU。' -ForegroundColor Green
        $armGpu = $true
        $gpuType = 'cpu'
    }
} elseif ($IsMacOS) {
    # MPS 只有 Apple Silicon 有；Intel Mac 沒有，維持 cpu
    if ((uname -m) -eq 'arm64') {
        $gpuType = 'mps'
        $gpuName = 'Apple Silicon (Metal)'
    }
}

switch ($gpuType) {
    'cuda' {
        Write-Host "  偵測到 NVIDIA GPU: $gpuName" -ForegroundColor Green
        Write-Host '  → 安裝 CUDA 版 PyTorch' -ForegroundColor Green
        $torchIndex = 'https://download.pytorch.org/whl/cu128'
    }
    'mps' {
        Write-Host "  偵測到 $gpuName" -ForegroundColor Green
        Write-Host '  → 安裝預設版 PyTorch（macOS wheel 內含 MPS 支援）' -ForegroundColor Green
        $torchIndex = $null
    }
    default {
        if (-not $gpuName) { Write-Host '  未偵測到加速裝置，使用 CPU 模式。' -ForegroundColor Yellow }
        if (-not $armGpu) { Write-Host '  → 安裝 CPU 版 PyTorch（推理會較慢）' -ForegroundColor Yellow }
        $torchIndex = 'https://download.pytorch.org/whl/cpu'
    }
}

function Get-PythonPlatform {
    # 'win-amd64'、'win-arm64'、'macosx-...'；跑不起來回 $null。
    # 不用 platform.machine()：Prism 模擬中的 x64 Python 會回報主機的 ARM64
    if (-not (Test-Path -LiteralPath $venvPython)) { return $null }
    try {
        $p = (& $venvPython -c 'import sysconfig; print(sysconfig.get_platform())' 2>$null | Out-String).Trim()
        if ($LASTEXITCODE -eq 0) { return $p }
    } catch {
        # Smart App Control 直接擋掉啟動器時 PowerShell 會丟例外，而不是回 exit code
    }
    return $null
}

function Remove-Venv {
    if (Test-Path -LiteralPath $venvDir) { [IO.Directory]::Delete($venvDir, $true) }
}

function Install-Packages {
    # 成功回 $null，失敗回原因。$Pins 是 sac-pins.txt 的路徑（只有 Smart App Control 擋下時才帶）
    param([string]$Pins)

    $pinArgs = if ($Pins) { @('--constraints', $Pins) } else { @() }
    Write-Host '  安裝 PyTorch...' -ForegroundColor Yellow
    if ($torchIndex) {
        uv pip install --python $venvPython torch torchaudio --index-url $torchIndex @pinArgs
    } else {
        # macOS 不指定 index：PyTorch 官方的 cpu/cu128 索引沒有 macOS wheel，指了反而裝不到含 MPS 的那份
        uv pip install --python $venvPython torch torchaudio @pinArgs
    }
    # 原生指令失敗不受 $ErrorActionPreference 管，不檢查的話會一路印「安裝完成」
    if ($LASTEXITCODE -ne 0) { return "PyTorch 安裝失敗（uv exit code: $LASTEXITCODE）" }

    # opencc-python-reimplemented：送進模型前繁轉簡（voice_prep.py；VoxCPM2 念簡體比較穩），純 Python
    Write-Host '  安裝 voxcpm + sounddevice + resampy + opencc...' -ForegroundColor Yellow
    uv pip install --python $venvPython voxcpm sounddevice resampy opencc-python-reimplemented @pinArgs
    if ($LASTEXITCODE -ne 0) { return "voxcpm 等套件安裝失敗（uv exit code: $LASTEXITCODE）" }
    return $null
}

function Test-Env {
    # 成功回 $null，失敗回錯誤訊息的最後一行
    param([string]$Code = $smokeTest)
    Write-Host '  驗證（實際跑 torch、scipy、librosa、voxcpm）...' -ForegroundColor Yellow
    $out = & $venvPython -W ignore -c $Code 2>&1 | Out-String
    if ($LASTEXITCODE -eq 0) { return $null }
    return @($out.Trim() -split "`r?`n")[-1]
}

function Install-Env {
    # 建立（若沒有）並裝好套件、跑過 $smokeTest。成功回 $null，失敗回原因，讓 ARM64 能退回 x64
    param([string]$PythonRequest)

    if (-not (Test-Path -LiteralPath $venvPython)) {
        uv venv --python-preference only-managed --python $PythonRequest $venvDir
        if ($LASTEXITCODE -ne 0) { return "建立 venv 失敗（uv exit code: $LASTEXITCODE）" }
    }
    if (-not (Get-PythonPlatform)) { return "$venvPython 跑不起來（可能被 Smart App Control 擋下）" }

    $failure = Install-Packages
    if ($failure) { return $failure }
    $err = Test-Env
    # WinError 4551／「應用程式控制原則已封鎖此檔案」：Smart App Control 擋了太新、信譽不夠的 wheel。
    # 重裝也沒用，只能換成已知放行的版本
    if ($err -and $IsWindows -and $err -match '4551|應用程式控制原則|Application Control policy') {
        Write-Host "  被 Smart App Control 擋下：$err" -ForegroundColor Yellow
        Write-Host '  改用 sac-pins.txt 的版本重裝...' -ForegroundColor Yellow
        $failure = Install-Packages -Pins (Join-Path $PSScriptRoot 'sac-pins.txt')
        if ($failure) { return $failure }
        $err = Test-Env
    }
    if ($err) { return "驗證失敗：$err" }
    return $null
}

function Install-ArmGpuEnv {
    # Windows on ARM＋NVIDIA：原生 ARM64、Python 3.13（NVIDIA 只出 cp313），套件見 requirements-win-arm64.txt。
    # 成功回 $null，失敗回原因
    if (-not (Test-Path -LiteralPath $venvPython)) {
        uv venv --python-preference only-managed --python 'cpython-3.13-windows-aarch64-none' $venvDir
        if ($LASTEXITCODE -ne 0) { return "建立 venv 失敗（uv exit code: $LASTEXITCODE）" }
    }
    if ((Get-PythonPlatform) -ne 'win-arm64') { return "$venvPython 不是可執行的原生 ARM64 環境" }

    # torch wheel 約 1.8 GB，預設 30 秒逾時不夠
    $env:UV_HTTP_TIMEOUT = '600'
    # 空殼要最先移除：librosa 相依 scikit-learn，下面就會裝進來，兩者共用 sklearn/ 資料夾，之後才移除空殼會刪掉真的 __init__.py
    uv pip uninstall --python $venvPython sklearn-unavailable 2>$null | Out-Null
    Write-Host '  安裝 NVIDIA 預覽版 PyTorch 與生成用套件（第一次要下載約 1.9 GB）...' -ForegroundColor Yellow
    uv pip install --python $venvPython --requirements $ArmRequirements
    if ($LASTEXITCODE -ne 0) { return "套件安裝失敗（uv exit code: $LASTEXITCODE）" }
    # voxcpm 宣告的相依大多只有訓練／ASR 才用、又沒有 win_arm64 wheel，所以不裝它的相依
    uv pip install --python $venvPython voxcpm --no-deps
    if ($LASTEXITCODE -ne 0) { return "voxcpm 安裝失敗（uv exit code: $LASTEXITCODE）" }

    # librosa 0.9.2 一 import 就要 sklearn。先裝真的；被 Smart App Control 擋（2026-10-28 cp313 每一版都被擋）
    # 才換成 shims/sklearn-unavailable：import 得到、真的用到才報清楚的錯
    $shim = Join-Path $PSScriptRoot 'shims' 'sklearn-unavailable'
    uv pip install --python $venvPython scikit-learn
    $sk = if ($LASTEXITCODE -eq 0) {
        & $venvPython -c 'import sklearn.decomposition, sklearn.cluster, sklearn.feature_extraction, sklearn.neighbors' 2>&1 | Out-String
    } else { 'install failed' }
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  scikit-learn 不能用（$(@($sk.Trim() -split "`r?`n")[-1])），改裝 sklearn-unavailable 空殼。" -ForegroundColor Yellow
        uv pip uninstall --python $venvPython scikit-learn 2>$null | Out-Null
        uv pip install --python $venvPython $shim
        if ($LASTEXITCODE -ne 0) { return "sklearn-unavailable 安裝失敗（uv exit code: $LASTEXITCODE）" }
    }

    $err = Test-Env -Code $gpuSmokeTest
    if ($err) { return "驗證失敗：$err" }
    return $null
}

function Restore-Backup {
    # GPU 環境失敗時把改名保留的原環境放回去
    param([string]$Backup)
    Remove-Venv
    if ($Backup -and (Test-Path -LiteralPath $Backup)) {
        Rename-Item -LiteralPath $Backup -NewName (Split-Path -Leaf $venvDir)
        Write-Host '  已還原原本的環境。' -ForegroundColor Green
    }
}

# --- Step 3: 建立 Python 環境並安裝 ---
Write-Host '[3/4] 建立 Python 環境並安裝套件...' -ForegroundColor Yellow
New-Item -ItemType Directory -Force (Split-Path $venvDir) | Out-Null

$backup = "$venvDir-backup"
# 上次中途被中斷、留下改名保留的環境：現在的能用就丟掉備份，不能用就把備份放回去
if (Test-Path -LiteralPath $backup) {
    if (Get-PythonPlatform) { [IO.Directory]::Delete($backup, $true) } else { Restore-Backup -Backup $backup }
}

if ($armGpu) {
    $current = Get-PythonPlatform
    if ($current -ne 'win-arm64') {
        # 改名不是複製：venv 裡的啟動器寫死了路徑，改回原名就完全還原
        if ($current) { Rename-Item -LiteralPath $venvDir -NewName (Split-Path -Leaf $backup) } else { Remove-Venv }
    }
    $failure = Install-ArmGpuEnv
    if ($failure -and $current -eq 'win-arm64' -and -not (Test-Env -Code $gpuSmokeTest)) {
        # 重跑時更新失敗（例如沒網路），但原本的 GPU 環境還能用：留著，不要因此退回 x64
        Write-Host "  更新失敗（$failure），沿用原本的 GPU 環境。" -ForegroundColor Yellow
        $failure = $null
    }
    if ($failure) {
        Write-Host "  原生 ARM64＋CUDA 不可用：$failure" -ForegroundColor Yellow
        Restore-Backup -Backup $backup
        Write-Host '  改用 x64 CPU 環境（Prism 模擬，會明顯較慢）...' -ForegroundColor Yellow
        if ((Get-PythonPlatform) -ne 'win-amd64') { Remove-Venv }
        $failure = Install-Env -PythonRequest 'cpython-3.12-windows-x86_64-none'
    } else {
        $gpuType = 'cuda'
        if (Test-Path -LiteralPath $backup) { [IO.Directory]::Delete($backup, $true) }
    }
} elseif ($isWindowsArm) {
    $current = Get-PythonPlatform
    if ($current -eq 'win-amd64') {
        # 已經是 x64 就沿用：它是上次原生失敗後退回來的，不必每次都重試原生
        Write-Host '  沿用既有的 x64 環境。' -ForegroundColor Green
        $failure = Install-Env -PythonRequest 'cpython-3.12-windows-x86_64-none'
    } else {
        if (-not $current) { Remove-Venv }   # 跑不起來的殘骸
        Write-Host '  Windows on ARM：先試原生 ARM64 環境...' -ForegroundColor Yellow
        $failure = Install-Env -PythonRequest 'cpython-3.12-windows-aarch64-none'
        if ($failure) {
            Write-Host "  原生 ARM64 不可用：$failure" -ForegroundColor Yellow
            Write-Host '  改建 x64 環境（Prism 模擬）...' -ForegroundColor Yellow
            Remove-Venv
            $failure = Install-Env -PythonRequest 'cpython-3.12-windows-x86_64-none'
        }
    }
} else {
    if ((Test-Path -LiteralPath $venvPython) -and -not (Get-PythonPlatform)) { Remove-Venv }
    $failure = Install-Env -PythonRequest '3.12'
}
if ($failure) { throw $failure }
Write-Host "  環境就緒（$(Get-PythonPlatform)）。" -ForegroundColor Green

# --- Step 4: 清掉舊版留在專案裡的 .venv ---
# 舊版把環境建在專案的 .venv；它在雲端硬碟同步範圍內，留著只會在別台電腦變成壞掉的殘骸
Write-Host '[4/4] 檢查專案內的舊環境...' -ForegroundColor Yellow
$oldVenv = Join-Path $PSScriptRoot '.venv'
if (Test-Path -LiteralPath $oldVenv) {
    try {
        # 雲端硬碟會把資料夾設成唯讀，Directory.Delete 遞迴刪除遇到唯讀資料夾會 Access denied，先清掉屬性
        Get-ChildItem -LiteralPath $oldVenv -Recurse -Force | ForEach-Object { $_.Attributes = $_.Attributes -band -bnot [IO.FileAttributes]::ReadOnly }
        (Get-Item -LiteralPath $oldVenv -Force).Attributes = 'Directory'
        [IO.Directory]::Delete($oldVenv, $true)
        Write-Host "  已刪除舊的 $oldVenv" -ForegroundColor Green
    } catch {
        # 雲端硬碟正在同步或鎖住檔案時刪不掉；新環境已經好了，不必因此中斷
        Write-Host "  刪不掉舊的 $oldVenv（$($_.Exception.InnerException.Message)），請稍後在檔案總管手動刪除。" -ForegroundColor Yellow
    }
} else {
    Write-Host '  沒有舊環境。' -ForegroundColor Green
}

# --- 完成 ---
Write-Host ''
Write-Host '============================================' -ForegroundColor Cyan
Write-Host '  安裝完成！' -ForegroundColor Cyan
Write-Host '============================================' -ForegroundColor Cyan
Write-Host ''
Write-Host '加速裝置: ' -NoNewline
switch ($gpuType) {
    'cuda'  { Write-Host 'NVIDIA CUDA' -ForegroundColor Green }
    'mps'   { Write-Host 'Apple Silicon MPS' -ForegroundColor Green }
    default { Write-Host 'CPU（較慢）' -ForegroundColor Yellow }
}
Write-Host ''
Write-Host '下一步：' -ForegroundColor Cyan
Write-Host "  1. 錄製參考音：& `"$venvPython`" record_ui.py --open"
Write-Host "  2. 生成語音：& `"$venvPython`" clone.py `"你想說的文字`""
Write-Host ''

# 舊版寫的 .gpu_type 不再使用：它跟著雲端硬碟同步，內容是最後一台跑安裝的電腦寫的。
# clone.py／dialogue.py 改成直接問 torch 這台有什麼
$gpuTypeFile = Join-Path $PSScriptRoot '.gpu_type'
if (Test-Path -LiteralPath $gpuTypeFile) { Remove-Item -LiteralPath $gpuTypeFile }
