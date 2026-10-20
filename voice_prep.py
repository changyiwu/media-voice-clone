#!/usr/bin/env python3
"""
voice_prep.py - VoxCPM2 生成前後的處理，clone.py、dialogue.py、audition.py 共用。

生成前（文字）：
  to_model_text(text, lexicon)  讀音替換表 → 繁轉簡。VoxCPM2 念簡體字比繁體穩（ai-video-workflow 實測），
                                沒裝 opencc 就跳過繁轉簡、印一行提示，不影響生成。字幕、逐字稿照原文。
生成後（聲音，numpy 陣列）：
  finish(wav, sr)  去頭尾靜音（保留開頭 80 ms、結尾 150 ms）→ highpass 70 Hz＋lowpass 11 kHz（濾低頻轟聲與嘶聲）
                   → 響度調到 TARGET_LUFS（−15，與 agent-video-specs、ai-video-workflow 的成片目標相同），
                   取樣峰值不超過 −1 dBFS（會超過就少加一點增益，不壓縮動態）
  lufs(wav, sr)    整合響度（ITU-R BS.1770，K 加權＋閘門；跟 ffmpeg ebur128 同一套量法）
  hf_ratio(wav, sr) 4 kHz 以上能量佔比：正常人聲 0.01–0.2，VoxCPM2 偶爾生出 0.5 以上的嘶聲版本

做法改編自 ai-video-workflow 的 voxcpm_batch.py（繁轉簡）、voice_pick.py（去靜音、高頻比）與
make_film.py cut（每句濾波、每句同響度再混）。需要：numpy、scipy、librosa（voxcpm 已帶）；
繁轉簡選用 opencc-python-reimplemented（install.ps1 會裝）。
"""
import numpy as np

TARGET_LUFS = -15.0
PEAK_MAX_DB = -1.0
_t2s = None


def t2s(text):
    """繁轉簡；沒有 opencc 就原樣回傳（只提示一次）"""
    global _t2s
    if _t2s is None:
        try:
            import opencc
            conv = None
            for cfg in ('t2s', 't2s.json'):          # 官方 opencc 與 opencc-python-reimplemented 的設定名寫法不同
                try:
                    conv = opencc.OpenCC(cfg).convert
                    break
                except Exception:
                    pass
            _t2s = conv or False
        except ImportError:
            _t2s = False
        if not _t2s:
            print('[prep] 沒有 opencc，跳過繁轉簡（pip install opencc-python-reimplemented 可裝）', flush=True)
    return _t2s(text) if _t2s else text


def to_model_text(text, lexicon):
    from tts_lexicon import speakable
    return t2s(speakable(text, lexicon))


def mono(x):
    x = np.asarray(x, dtype=np.float32)
    return x.mean(axis=1) if x.ndim > 1 else x


def trim(wav, sr, head=0.08, tail=0.15, top_db=40):
    import librosa
    _, (a, b) = librosa.effects.trim(wav, top_db=top_db)
    return wav[max(0, a - int(head * sr)):min(len(wav), b + int(tail * sr))]


def band(wav, sr, low=70.0, high=11000.0):
    from scipy.signal import butter, sosfiltfilt
    sos = butter(2, low, 'highpass', fs=sr, output='sos')
    if high < sr / 2:
        sos = np.vstack([sos, butter(4, high, 'lowpass', fs=sr, output='sos')])
    return sosfiltfilt(sos, wav).astype(np.float32)


def _biquad(b, a):
    a = np.asarray(a, dtype=np.float64)
    return np.asarray(b, dtype=np.float64) / a[0], a / a[0]


def k_weight(wav, sr):
    """BS.1770 的 K 加權：高架濾波（約 +4 dB @ 1.7 kHz）＋高通（38 Hz），係數依取樣率換算"""
    from scipy.signal import lfilter
    G, Q, fc = 3.999843853973347, 0.7071752369554196, 1681.974450955533
    A = 10 ** (G / 40); w0 = 2 * np.pi * fc / sr; al = np.sin(w0) / (2 * Q); c = np.cos(w0)
    b, a = _biquad([A * ((A + 1) + (A - 1) * c + 2 * np.sqrt(A) * al), -2 * A * ((A - 1) + (A + 1) * c),
                    A * ((A + 1) + (A - 1) * c - 2 * np.sqrt(A) * al)],
                   [(A + 1) - (A - 1) * c + 2 * np.sqrt(A) * al, 2 * ((A - 1) - (A + 1) * c),
                    (A + 1) - (A - 1) * c - 2 * np.sqrt(A) * al])
    y = lfilter(b, a, wav)
    Q, fc = 0.5003270373238773, 38.13547087602444
    w0 = 2 * np.pi * fc / sr; al = np.sin(w0) / (2 * Q); c = np.cos(w0)
    b, a = _biquad([(1 + c) / 2, -(1 + c), (1 + c) / 2], [1 + al, -2 * c, 1 - al])
    return lfilter(b, a, y)


def lufs(wav, sr):
    """整合響度（單聲道）；太短或太小聲量不到時回 None"""
    y = k_weight(np.asarray(wav, dtype=np.float64), sr)
    n, hop = int(0.4 * sr), int(0.1 * sr)                # 400 ms 區塊、75% 重疊
    if len(y) < n:
        return None
    z = np.array([np.mean(y[i:i + n] ** 2) for i in range(0, len(y) - n + 1, hop)])
    z = z[-0.691 + 10 * np.log10(z + 1e-12) > -70]       # 絕對閘門
    if not len(z):
        return None
    rel = -0.691 + 10 * np.log10(z.mean()) - 10          # 相對閘門
    z = z[-0.691 + 10 * np.log10(z + 1e-12) > rel]
    return float(-0.691 + 10 * np.log10(z.mean())) if len(z) else None


def limit(wav, sr, cap, look=0.01):
    """前瞻峰值限幅：只在超過 cap 的尖峰前後約 10 ms 把增益壓下來，其他地方不動"""
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    need = np.minimum(1.0, cap / np.maximum(np.abs(wav), 1e-9))
    w = max(1, int(look * sr))
    g = minimum_filter1d(need, size=2 * w + 1)           # 尖峰前後 w 個取樣都至少壓到所需
    g = uniform_filter1d(g, size=w + 1)                  # 平滑成漸進的增益變化；平均視窗在上一步範圍內，尖峰處不會壓不夠
    return (wav * g).astype(np.float32)


def normalize(wav, sr, target=TARGET_LUFS, peak_max_db=PEAK_MAX_DB):
    """調到目標響度；尖峰超過上限的部分用 limit() 壓，再量一次補回限幅掉的響度（最多 3 輪）"""
    l0 = lufs(wav, sr)
    if l0 is None:
        return wav, None
    cap = 10 ** (peak_max_db / 20)
    gain_db = target - l0
    for _ in range(3):
        out = limit(wav * 10 ** (gain_db / 20), sr, cap)
        l = lufs(out, sr)
        if l is None or abs(l - target) <= 0.2:
            break
        gain_db += target - l
    return out, gain_db


def finish(wav, sr, target=TARGET_LUFS):
    """去頭尾靜音 → 濾波 → 調響度；回傳 (處理後的聲音, 說明字串)"""
    wav = band(trim(mono(wav), sr), sr)
    wav, g = normalize(wav, sr, target)
    l = lufs(wav, sr)
    return wav, ('響度量不到，未調整' if g is None else f'{l:.1f} LUFS（增益 {g:+.1f} dB）')


def hf_ratio(wav, sr, cut=4000):
    x = mono(wav).astype(np.float64)
    S = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / sr)
    return float(S[f >= cut].sum() / (S.sum() + 1e-12))
