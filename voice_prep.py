#!/usr/bin/env python3
"""
voice_prep.py - VoxCPM2 生成前後的處理，clone.py、dialogue.py、audition.py 共用。

生成前（文字）：
  to_model_text(text, lexicon)  讀音替換表 → 數字符號轉中文 → 繁轉簡 → fix_zhe。
                                數字符號含 〇→零、整數／小數／百分比／分數／負號，保留常見標點停頓與英文詞。
                                VoxCPM2 念簡體字比繁體穩（media-short-film 實測），沒裝 opencc 就跳過繁轉簡。
                                繁轉簡後把念 zhe／zháo／zhuó 的「著」改簡體「着」，保留「著作／顯著」等念 zhù 的詞。
                                字幕、逐字稿照原文。
  poly_warnings(text)           輕量多音／兩岸異讀風險提醒（不阻斷執行；有裝 pypinyin 列出推測讀音，沒有則用內建清單）。
生成後（聲音，numpy 陣列）：
  finish(wav, sr)               去頭尾靜音（保留開頭 80 ms、結尾 150 ms）→ highpass 70 Hz＋lowpass 11 kHz（濾低頻轟聲與嘶聲）
                                → 響度調到 TARGET_LUFS（−15，與 media-video-specs、media-short-film 的成片目標相同），
                                取樣峰值不超過 −1 dBFS（會超過就少加一點增益，不壓縮動態）
  lufs(wav, sr)                 整合響度（ITU-R BS.1770，K 加權＋閘門；跟 ffmpeg ebur128 同一套量法）
  hf_ratio(wav, sr)             4 kHz 以上能量佔比：正常人聲 0.01–0.2，VoxCPM2 偶爾生出 0.5 以上的嘶聲版本
  buzz_db(wav, sr)              2.5–9 kHz 窄頻嗡嗡聲檢測（與鄰近 320 Hz 頻段中位數比較；> BUZZ_DB 10.0 可能有持續嗡嗡聲）
  verify_pronunciation(...)     選用（僅 --verify 呼叫）：本機 faster-whisper（無 initial_prompt）＋ pypinyin 讀音比對
模型位置：
  model_source()                VoxCPM2 權重的位置，給 VoxCPM.from_pretrained 用（見函式說明）
"""
import argparse
import difflib
import os
import re
import sys

import numpy as np

HF_MODEL_ID = 'openbmb/VoxCPM2'
TARGET_LUFS = -15.0
PEAK_MAX_DB = -1.0
BUZZ_DB = 10.0
DIG = '零一二三四五六七八九'

# OpenCC t2s 會把「活著、看著、睡著」的「著」原樣留下，簡體的「著」只念 zhù（著作）；
# 簡體裡 zhe／zháo／zhuó 都寫「着」，只有下面這些念 zhù 的詞保留「著」（移植自 media-short-film/tools/voxcpm_batch.py）。
ZHU_WORDS = (
    '著作', '著名', '著称', '著述', '著录', '著者', '著书', '著有',
    '名著', '原著', '巨著', '专著', '论著', '编著', '合著', '译著',
    '遗著', '新著', '拙著', '大著', '土著', '显著', '卓著', '昭著',
    '彰著', '撰著',
)
# 沒裝 opencc 時也保護繁體詞形，避免誤把「顯著」改成「顯着」
_ZHU_TRAD = ('著稱', '著錄', '著書', '專著', '論著', '編著', '譯著', '遺著', '顯著')
_ZHU_ALL = ZHU_WORDS + _ZHU_TRAD

# 常見高風險多音／異讀字（源自 media-video-specs/tools/polyphone_check.py WATCH 與教學旁白常見字，排除已處理的「著」與變調字「一不」）
POLY_WATCH = set(
    '奇少數還差重長調行便降種相應將當曾解樂好量率角處朝薄省卷背傳參給只正興覺似塞假藏'
    '都為曲空繫彈'
)
SANDHI = set('一不')

# 台灣與大陸常見異讀詞提醒（不含「著」，亦涵蓋單讀音但兩岸不同調的常用教學詞）
TW_POLY_HINTS = {
    '括號': '台灣「括」讀 guā（大陸常念 kuò，可於 tts-lexicon.md 改寫）',
    '係數': '台灣「係」讀 xì（部分模型偶念 jì）',
    '研究': '台灣「究」讀 jiù（大陸常念 jiū）',
    '質數': '台灣「質」讀 zhí（大陸常念 zhì）',
    '垃圾': '台灣讀 lè sè（大陸常念 lā jī）',
    '暴露': '台灣讀 pù lù（大陸常念 bào lù）',
    '企業': '台灣「企」讀 qì（大陸常念 qǐ）',
    '星期': '台灣「期」讀 qí（大陸常念 qī）',
    '微笑': '台灣「微」讀 wéi（大陸常念 wēi）',
    '危險': '台灣「危」讀 wéi（大陸常念 wēi）',
    '頭髮': '台灣「髮」讀 fǎ（大陸常念 fà）',
    '液體': '台灣「液」讀 yì（大陸常念 yè）',
    '大叔': '台灣「叔」讀 shú（大陸常念 shū）',
    '拚': '繁體「拚」(pīn) 轉簡體不會換成「拼」，建議於 tts-lexicon.md 改寫',
}

# 台灣讀音覆寫（源自 media-short-film/tools/pron_check.py，同時涵蓋繁簡寫法供 ASR 比對）
TW_OVERRIDE = {
    '垃圾': ['le4', 'se4'],
    '企': ['qi4'],
    '期': ['qi2'],
    '危': ['wei2'],
    '研': ['yan2'],
    '究': ['jiu4'],
    '髮': ['fa4'],
    '頭髮': ['tou2', 'fa4'],
    '头发': ['tou2', 'fa4'],
    '微': ['wei2'],
    '質': ['zhi2'],
    '质': ['zhi2'],
    '人質': ['ren2', 'zhi4'],
    '人质': ['ren2', 'zhi4'],
    '暴露': ['pu4', 'lu4'],
    '液': ['yi4'],
    '頭套': ['tou2', 'tao4'],
    '头套': ['tou2', 'tao4'],
    '什麼': ['shen2', 'me5'],
    '什么': ['shen2', 'me5'],
    '東西': ['dong1', 'xi5'],
    '东西': ['dong1', 'xi5'],
    '時候': ['shi2', 'hou4'],
    '时候': ['shi2', 'hou4'],
    '為什麼': ['wei4', 'shen2', 'me5'],
    '为什么': ['wei4', 'shen2', 'me5'],
    '這裡': ['zhe4', 'li3'],
    '这里': ['zhe4', 'li3'],
    '那裡': ['na4', 'li3'],
    '那里': ['na4', 'li3'],
    '哪裡': ['na3', 'li3'],
    '哪里': ['na3', 'li3'],
    '老師': ['lao3', 'shi1'],
    '老师': ['lao3', 'shi1'],
    '大叔': ['da4', 'shu2'],
    '叔': ['shu2'],
}

_t2s = None
_s2t = None
_asr_cache = {}


def model_source():
    """VoxCPM2 權重的位置：本機已經有一份就直接用，不再讓 HuggingFace 快取另存 4.7 GB。

    依序找：VOXCPM_MODEL 環境變數指的資料夾 → %AI_ROOT%\\Voice\\models\\VoxCPM2
    （media-short-film 裝好的語音模型，AI_ROOT 預設 C:\\AI）→ 都沒有就回傳 HuggingFace 名稱、照舊下載到快取。
    資料夾要有 config.json 跟 model.safetensors 才算數，缺檔的半成品不用。
    """
    cands = []
    if os.environ.get('VOXCPM_MODEL'):
        cands.append(os.environ['VOXCPM_MODEL'])
    ai_root = os.environ.get('AI_ROOT') or (r'C:\AI' if os.name == 'nt' else None)
    if ai_root:
        cands.append(os.path.join(ai_root, 'Voice', 'models', 'VoxCPM2'))
    for d in cands:
        if all(os.path.isfile(os.path.join(d, f)) for f in ('config.json', 'model.safetensors')):
            return d
    return HF_MODEL_ID


def _load_opencc(configs):
    try:
        import opencc
    except ImportError:
        return False
    for cfg in configs:
        try:
            return opencc.OpenCC(cfg).convert
        except Exception:
            pass
    return False


def t2s(text):
    """繁轉簡；沒有 opencc 就原樣回傳（只提示一次）"""
    global _t2s
    if _t2s is None:
        _t2s = _load_opencc(('t2s', 't2s.json'))
        if not _t2s:
            print('[prep] 沒有 opencc，跳過繁轉簡（pip install opencc-python-reimplemented 可裝）', flush=True)
    return _t2s(text) if _t2s else text


def _t2s_quiet(text):
    """靜默繁轉簡（供內部查字典／比對用，沒有 opencc 就原樣回傳）"""
    global _t2s
    if _t2s is None:
        _t2s = _load_opencc(('t2s', 't2s.json'))
    return _t2s(text) if _t2s else text


def _s2t_quiet(text):
    global _s2t
    if _s2t is None:
        _s2t = _load_opencc(('s2twp', 's2twp.json', 's2t', 's2t.json'))
    return _s2t(text) if _s2t else text


def fix_zhe(s):
    """在 t2s 繁轉簡後，將念 zhe/zháo/zhuó 的「著」改為簡體「着」，保留 ZHU_WORDS（zhù）。"""
    keep = {i for w in _ZHU_ALL for m in re.finditer(w, s) for i in range(m.start(), m.end())}
    return ''.join('着' if c == '著' and i not in keep else c for i, c in enumerate(s))


def int_zh(n):
    """非負整數轉中文讀法（參照 voxcpm_batch.py：0–999 照位值念，1000 以上逐字念以免誤讀年份）。"""
    n = int(n)
    if n < 0:
        return '負' + int_zh(-n)
    if n < 10:
        return DIG[n]
    if n < 100:
        t, o = divmod(n, 10)
        return ('' if t == 1 else DIG[t]) + '十' + (DIG[o] if o else '')
    if n < 1000:
        h, r = divmod(n, 100)
        return DIG[h] + '百' + ('' if r == 0 else ('零' + DIG[r] if r < 10 else ('一' + int_zh(r) if r < 20 else int_zh(r))))
    return ''.join(DIG[int(c)] for c in str(n))


def int_zh_full(n):
    """照位值念整數，四位一組，補上跨組的零。"""
    n = int(n)
    if n < 0:
        return '負' + int_zh_full(-n)
    if n < 1000 or n >= 10 ** 20:
        return int_zh(n)
    groups = []
    while n:
        n, group = divmod(n, 10000)
        groups.append(group)
    units = ('', '萬', '億', '兆', '京')
    parts = []
    need_zero = False
    for i in range(len(groups) - 1, -1, -1):
        group = groups[i]
        if not group:
            need_zero = True
            continue
        if parts and (need_zero or group < 1000):
            parts.append('零')
        thousands, rest = divmod(group, 1000)
        if thousands:
            parts.append(DIG[thousands] + '千')
            if 0 < rest < 100:
                parts.append('零')
        if rest:
            if 10 <= rest < 20 and parts:
                parts.append('一')
            parts.append(int_zh(rest))
        parts.append(units[i])
        need_zero = False
    return ''.join(parts)


def num_zh(tok, *, full=False):
    """數字字串（可含負號與小數點）轉中文讀法。"""
    s = str(tok).strip().replace(',', '')
    prefix = ''
    if s.startswith(('-', '−')):
        prefix = '負'
        s = s[1:]
    def read_integer(digits):
        if (len(digits) > 1 and int(digits[0]) == 0) or len(digits) > 20:
            return ''.join(DIG[int(c)] for c in digits)
        return (int_zh_full if full else int_zh)(int(digits))

    if '.' in s:
        a, b = s.split('.', 1)
        a_zh = read_integer(a) if a else '零'
        b_zh = ''.join(DIG[int(c)] for c in b)
        return prefix + a_zh + '點' + b_zh
    return prefix + read_integer(s)


_INTEGER = r'(?:\d{1,3}(?:,\d{3})+|\d+)'
_DECIMAL = rf'(?:{_INTEGER}(?:\.\d+)?|\.\d+)'
_NUM_START = r'(?<![A-Za-z\d.])(?<![A-Za-z]-)'
_NUM_END = r'(?![A-Za-z\d]|\.\d|,\d{3}(?!\d)|-[A-Za-z])'
_PCT_RE = re.compile(rf'{_NUM_START}([-−]?{_DECIMAL})\s*[%％]')
_FRAC_RE = re.compile(rf'{_NUM_START}(?<!/)([-−]?{_INTEGER})\s*/\s*({_INTEGER}){_NUM_END}(?!/)')
_NUM_RE = re.compile(rf'{_NUM_START}([-−]?{_DECIMAL}){_NUM_END}')
_YEAR_SUFFIX = re.compile(r'(?:\s*年(?!(?:級|期))|([/-])\d{1,2}\1\d{1,2}(?!\d))')


def normalize_numbers(text, *, full=False):
    """數量照位值、四位年份逐字念。full=True 也將年份照位值轉換，供 ASR 比對。"""
    s = text.replace('〇', '零').replace('……', '，').replace('——', '，')

    def _pct(m):
        raw = m.group(1)
        sign = '負' if raw.startswith(('-', '−')) else ''
        body = raw[1:] if sign else raw
        return sign + '百分之' + num_zh(body, full=True)

    def _frac(m):
        raw_num, den = m.group(1), m.group(2)
        sign = '負' if raw_num.startswith(('-', '−')) else ''
        num = raw_num[1:] if sign else raw_num
        return sign + num_zh(den, full=True) + '分之' + num_zh(num, full=True)

    def _number(m):
        raw = m.group(1)
        year = len(raw) == 4 and raw.isdecimal() and _YEAR_SUFFIX.match(m.string, m.end())
        return num_zh(raw, full=full or not year)

    s = _PCT_RE.sub(_pct, s)
    s = _FRAC_RE.sub(_frac, s)
    s = _NUM_RE.sub(_number, s)
    return s


def to_model_text(text, lexicon):
    """送進 VoxCPM2 前的文字正規化：speakable() → 數字符號轉中文 → t2s() → fix_zhe()。"""
    return prepare_text(text, lexicon)[1]


def prepare_text(text, lexicon):
    """回傳 (供多音提示用的文字, 送進模型的文字)。"""
    from tts_lexicon import speakable
    spoken = normalize_numbers(speakable(text, lexicon))
    return spoken, fix_zhe(t2s(spoken))


def generate_speech(model, text, reference, prompt_text, *, cfg_value, prompt_caches=None):
    """多句或多候選共用參考音編碼。舊版 VoxCPM 可沿用一般生成入口。"""
    tts = model.tts_model
    build_cache = getattr(tts, 'build_prompt_cache', None)
    generate_cached = getattr(tts, 'generate_with_prompt_cache', None)
    if prompt_caches is None or not callable(build_cache) or not callable(generate_cached):
        return model.generate(
            text=text, prompt_wav_path=reference, prompt_text=prompt_text,
            reference_wav_path=reference, cfg_value=cfg_value, inference_timesteps=10,
        )

    key = (os.path.normcase(os.path.realpath(reference)), prompt_text)
    if key not in prompt_caches:
        prompt_caches[key] = build_cache(
            prompt_wav_path=reference, prompt_text=prompt_text, reference_wav_path=reference,
        )
    wav, _, _ = generate_cached(
        target_text=re.sub(r'\s+', ' ', text),
        prompt_cache=prompt_caches[key],
        cfg_value=cfg_value,
        inference_timesteps=10,
        max_len=4096,
        retry_badcase=True,
    )
    return wav.squeeze(0).cpu().numpy()


def _is_cjk(ch):
    return '\u4e00' <= ch <= '\u9fff'


def poly_warnings(text):
    """檢查高風險多音／兩岸異讀風險，回傳提示字串清單（不阻斷執行）。
    有安裝 pypinyin 時補充推測讀音，沒有則僅靠內建高風險字詞清單零依賴運作。
    """
    if not text:
        return []
    warnings = []
    seen = set()
    covered = set()

    for word, hint in TW_POLY_HINTS.items():
        if word in text and ('hint', word) not in seen:
            seen.add(('hint', word))
            warnings.append(f'「{word}」：{hint}')
            for m in re.finditer(re.escape(word), text):
                covered.update(range(m.start(), m.end()))

    py_mod = None
    try:
        from pypinyin import Style, pinyin
        from pypinyin.constants import PHRASES_DICT
        py_mod = (Style, pinyin, PHRASES_DICT)
    except ImportError:
        pass

    if py_mod is None:
        for i, ch in enumerate(text):
            if i in covered or not _is_cjk(ch) or ch in SANDHI or ch in ('著', '着'):
                continue
            if ch not in POLY_WATCH:
                continue
            prev = text[i - 1] if i > 0 and _is_cjk(text[i - 1]) else ''
            nxt = text[i + 1] if i + 1 < len(text) and _is_cjk(text[i + 1]) else ''
            word = prev + ch if prev else ch + nxt
            key = (word, ch)
            if key not in seen:
                seen.add(key)
                warnings.append(f'「{word}」的「{ch}」：高風險多音字（未安裝 pypinyin，無法推估讀音）')
        return warnings

    Style, pinyin, PHRASES_DICT = py_mod
    simp = fix_zhe(_t2s_quiet(text))
    use_simp = len(simp) == len(text)
    base_str = simp if use_simp else text
    guess = [x[0] for x in pinyin(base_str, style=Style.TONE3, errors=lambda s: ['-'] * len(s))]

    for i, ch in enumerate(text):
        if i in covered or not _is_cjk(ch) or ch in SANDHI or ch in ('著', '着'):
            continue
        readings = list(dict.fromkeys(pinyin(base_str[i], style=Style.TONE3, heteronym=True)[0]))
        if len(readings) < 2:
            continue
        span = None
        if use_simp:
            for k in (4, 3, 2):
                span = next(
                    ((a, a + k) for a in range(i - k + 1, i + 1)
                     if a >= 0 and a + k <= len(base_str) and base_str[a:a + k] in PHRASES_DICT),
                    None,
                )
                if span:
                    break
        why = None
        if span:
            word = text[span[0]:span[1]]
            if guess[i] != readings[0] and guess[i][-1:].isdigit():
                why = f'推測讀 {guess[i]}（非最常見的 {readings[0]}）'
        else:
            prev = text[i - 1] if i > 0 and _is_cjk(text[i - 1]) else ''
            nxt = text[i + 1] if i + 1 < len(text) and _is_cjk(text[i + 1]) else ''
            word = prev + ch if prev else ch + nxt
            if ch in POLY_WATCH:
                why = f'詞典未收，可能念 {"／".join(readings[:4])}（猜 {guess[i]}）'
        if why:
            key = (word, ch)
            if key not in seen:
                seen.add(key)
                warnings.append(f'「{word}」的「{ch}」：{why}')
    return warnings


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


def _median_filter_np(x, k):
    pad = k // 2
    xp = np.pad(x, pad, mode='edge')
    return np.median(np.lib.stride_tricks.sliding_window_view(xp, k), axis=1)


def buzz_db(wav, sr, win_s=1.5):
    """2.5–9 kHz 滑動視窗窄頻峰檢測（純 numpy，移植自 media-short-film/tools/tone_check.py）。
    回傳最嚴重那一窗高出鄰近約 320 Hz 頻段中位數的 dB 數；> BUZZ_DB (10.0) 可能有持續嗡嗡聲。
    """
    x = mono(wav).astype(np.float64)
    n = 4096
    if len(x) < n or sr <= 0:
        return 0.0
    fr = np.fft.rfftfreq(n, 1 / sr)
    band_mask = (fr >= 2500) & (fr <= 9000)
    if not band_mask.any():
        return 0.0
    k = max(5, int(320 / (sr / n)) | 1)
    hop = max(n, int(win_s * sr))
    han = np.hanning(n)
    level = np.sqrt((x ** 2).mean()) + 1e-9
    peaks = []
    for a in range(0, max(1, len(x) - n), hop):
        seg = x[a:a + hop]
        if len(seg) < n or np.sqrt((seg ** 2).mean()) < level * 0.2:
            continue
        frames = seg[:len(seg) // n * n].reshape(-1, n) * han
        db = 20 * np.log10(np.abs(np.fft.rfft(frames, axis=1)).mean(axis=0) + 1e-9)
        pk = (db - _median_filter_np(db, k))[band_mask]
        peaks.append(float(pk.max()))
    return round(float(max(peaks)), 1) if peaks else 0.0


def _pron_tokens(text):
    """把文字轉成 [(顯示字, 比較鍵)]：漢字為 (拼音無聲調, 表面聲調)，英數字逐字大寫。
    套用台灣常用詞覆寫＋三三連讀＋「一／不」變調；沒有 pypinyin 時丟 ImportError。
    """
    from pypinyin import Style, pinyin

    s = re.sub(r'\[:[^\]]*\]', '', text)
    s = re.sub(r'兩(?=[千百萬億])', '二', s)
    s = fix_zhe(_t2s_quiet(s))

    # 先找出所有漢字的序列與斷點（標點／英數間不跨界變調）
    cjk_chars, brk = [], set()
    seq = []  # ('cjk', idx_in_cjk) or ('ascii', ch.upper())
    for ch in s:
        if _is_cjk(ch):
            cjk_chars.append(ch)
            seq.append(('cjk', len(cjk_chars) - 1))
        elif ch.isascii() and ch.isalnum():
            if cjk_chars:
                brk.add(len(cjk_chars) - 1)
            seq.append(('ascii', ch.upper()))
        else:
            if cjk_chars:
                brk.add(len(cjk_chars) - 1)

    if not cjk_chars:
        return [(v, v) for _, v in seq]

    cjk_str = ''.join(cjk_chars)
    py = [p[0] for p in pinyin(cjk_str, style=Style.TONE3, neutral_tone_with_five=True, heteronym=False)]
    # 短詞先套、長詞後套，確保「人質」覆蓋「質」
    for k, v in sorted(TW_OVERRIDE.items(), key=lambda kv: len(kv[0])):
        if not v:
            continue
        idx = cjk_str.find(k)
        while idx >= 0:
            py[idx:idx + len(k)] = v
            idx = cjk_str.find(k, idx + 1)

    tones = [int(p[-1]) if p[-1].isdigit() else 5 for p in py]
    for i, ch in enumerate(cjk_chars):
        if ch == '不':
            tones[i] = 4
        elif ch == '一':
            tones[i] = 1
    base = [re.sub(r'\d', '', p).replace('ü', 'v') for p in py]
    sur = tones[:]
    for i in range(len(sur) - 2, -1, -1):
        if tones[i] == 3 and sur[i + 1] == 3 and i not in brk:
            sur[i] = 2
    for i, ch in enumerate(cjk_chars):
        nxt = tones[i + 1] if i + 1 < len(tones) and i not in brk else None
        if ch == '一' and nxt is not None and not (i > 0 and cjk_chars[i - 1] in '第'):
            if cjk_chars[i + 1:i + 2] != ['次'] or nxt == 4:
                sur[i] = 2 if nxt == 4 else 4
        if ch == '不' and nxt == 4:
            sur[i] = 2

    out = []
    for kind, val in seq:
        if kind == 'cjk':
            out.append((cjk_chars[val], f'{base[val]}{sur[val]}'))
        else:
            out.append((val, val))
    return out


def _plain_chars(text):
    s = re.sub(r'\[:[^\]]*\]', '', text)
    s = re.sub(r'兩(?=[千百萬億])', '二', s)
    s = fix_zhe(_t2s_quiet(s))
    return [c.upper() for c in s if _is_cjk(c) or (c.isascii() and c.isalnum())]


def _diff_summary(want_items, heard_items, limit_n=5):
    w_keys = [k for _, k in want_items]
    h_keys = [k for _, k in heard_items]
    notes = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, w_keys, h_keys, autojunk=False).get_opcodes():
        if op == 'equal':
            continue
        w_txt = ''.join(f'{c}({k})' if c != k else c for c, k in want_items[i1:i2])
        h_txt = ''.join(f'{c}({k})' if c != k else c for c, k in heard_items[j1:j2])
        if op == 'replace':
            notes.append(f'預期「{w_txt}」≠ 聽寫「{h_txt}」')
        elif op == 'delete':
            notes.append(f'漏念「{w_txt}」')
        elif op == 'insert':
            notes.append(f'多念「{h_txt}」')
        if len(notes) >= limit_n:
            break
    return notes


def compare_pronunciation(heard_text, expected_text, orig_text=None):
    """比對聽寫結果與預期讀音；缺 pypinyin 時自動降級為純字面比對。"""
    want_raws = [expected_text] + ([orig_text] if orig_text and orig_text != expected_text else [])
    want_forms = []
    for w in want_raws:
        want_forms.append(normalize_numbers(w, full=False))
        w_full = normalize_numbers(w, full=True)
        if w_full not in want_forms:
            want_forms.append(w_full)

    heard_forms = [normalize_numbers(heard_text, full=False)]
    h_full = normalize_numbers(heard_text, full=True)
    if h_full not in heard_forms:
        heard_forms.append(h_full)

    try:
        want_toks = [_pron_tokens(w) for w in want_forms]
        heard_toks = [_pron_tokens(h) for h in heard_forms]
        partial = False
        reason = ''
    except ImportError:
        want_toks = [[(c, c) for c in _plain_chars(w)] for w in want_forms]
        heard_toks = [[(c, c) for c in _plain_chars(h)] for h in heard_forms]
        partial = True
        reason = '未安裝 pypinyin，降級為字面比對（未比對同音字與聲調）'

    best_sim = -1.0
    best_pair = (want_toks[0], heard_toks[0])
    matched = False
    for wt in want_toks:
        wk = [k for _, k in wt]
        for ht in heard_toks:
            hk = [k for _, k in ht]
            if wk == hk:
                return {
                    'ok': True,
                    'partial': partial,
                    'sim': 1.0,
                    'mismatch': [],
                    'reason': reason,
                }
            sim = difflib.SequenceMatcher(None, wk, hk, autojunk=False).ratio()
            if sim > best_sim:
                best_sim = sim
                best_pair = (wt, ht)

    return {
        'ok': matched,
        'partial': partial,
        'sim': round(max(0.0, best_sim), 3),
        'mismatch': _diff_summary(best_pair[0], best_pair[1]),
        'reason': reason,
    }


def _add_cuda_dll_dirs():
    """讓 Windows / Linux 上的 ctranslate2 找到 pip 安裝的 CUDA runtime DLL。"""
    import site
    roots = [os.path.join(sp, 'nvidia') for sp in site.getsitepackages() + [site.getusersitepackages()]]
    if os.name != 'nt':
        return
    for root in roots:
        if not os.path.isdir(root):
            continue
        for sub in os.listdir(root):
            bindir = os.path.join(root, sub, 'bin')
            if os.path.isdir(bindir):
                os.environ['PATH'] = bindir + os.pathsep + os.environ.get('PATH', '')
                try:
                    os.add_dll_directory(bindir)
                except Exception:
                    pass


def _load_whisper():
    """僅於 --verify 呼叫：自動尋找本機 faster-whisper；找不到時回傳 (None, 原因)。"""
    if 'model' in _asr_cache:
        return _asr_cache['model'], ''
    if 'error' in _asr_cache:
        return None, _asr_cache['error']

    _add_cuda_dll_dirs()
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        msg = '未安裝 faster-whisper，無法執行 ASR 讀音驗證（可選功能，不影響生成）'
        _asr_cache['error'] = msg
        return None, msg

    dev, ct = 'cpu', 'int8'
    try:
        import torch
        if torch.cuda.is_available():
            dev, ct = 'cuda', 'float16'
    except Exception:
        pass

    model_ref = os.environ.get('WHISPER_MODEL') or 'large-v3'
    ai_root = os.environ.get('AI_ROOT') or (r'C:\AI' if os.name == 'nt' else None)
    cache_dir = os.path.join(ai_root, 'Voice', 'models', 'whisper') if ai_root else None
    kw = {'device': dev, 'compute_type': ct}
    if cache_dir and os.path.isdir(cache_dir):
        kw['download_root'] = cache_dir

    try:
        try:
            m = WhisperModel(model_ref, local_files_only=True, **kw)
        except Exception:
            m = WhisperModel(model_ref, **kw)
        _asr_cache['model'] = m
        return m, ''
    except Exception as e:
        msg = f'faster-whisper 載入失敗：{e}'
        _asr_cache['error'] = msg
        return None, msg


def verify_pronunciation(wav_path, expected_text, orig_text=None):
    """僅供 --verify 呼叫：以無 initial_prompt 的 faster-whisper 聽寫並與 pypinyin 讀音比對。
    缺套件或模型不可用時清楚回傳不可用原因，不拋出例外中斷正常流程。
    """
    try:
        model, err = _load_whisper()
        if model is None:
            err = err or 'ASR 模型不可用'
        else:
            segs, _ = model.transcribe(
                str(wav_path),
                language='zh',
                beam_size=5,
                initial_prompt=None,
                condition_on_previous_text=False,
            )
            heard = _s2t_quiet(''.join(s.text for s in segs).strip())
            cmp_res = compare_pronunciation(heard, expected_text, orig_text=orig_text)
    except Exception as e:
        err = f'ASR 驗證失敗：{e}'
    if err:
        return {
            'available': False,
            'ok': None,
            'partial': False,
            'heard': '',
            'sim': None,
            'mismatch': [],
            'reason': err,
        }

    return {
        'available': True,
        'ok': cmp_res['ok'],
        'partial': cmp_res['partial'],
        'heard': heard,
        'sim': cmp_res['sim'],
        'mismatch': cmp_res['mismatch'],
        'reason': cmp_res['reason'],
    }


def selftest():
    """無模型單元驗證：fix_zhe、ZHU_WORDS、數字轉中文、to_model_text、poly_warnings、buzz_db（及可選 pypinyin 比對）。"""
    # 1. fix_zhe & ZHU_WORDS
    assert '著作' in ZHU_WORDS and '显著' in ZHU_WORDS
    assert fix_zhe('活著、看著、睡著了、著作、显著、活著的著作很顯著') == '活着、看着、睡着了、著作、显著、活着的著作很顯著'
    if _t2s_quiet('顯') == '显':
        assert fix_zhe(_t2s_quiet('活著的著作很顯著')) == '活着的著作很显著'

    # 2. int_zh / num_zh / normalize_numbers
    assert int_zh(0) == '零'
    assert int_zh(12) == '十二'
    assert int_zh(105) == '一百零五'
    assert int_zh(112) == '一百一十二'
    assert int_zh(2026) == '二零二六'
    assert num_zh('3.14') == '三點一四'
    assert num_zh('-0.5') == '負零點五'
    assert normalize_numbers('二〇二六年第 12 題，-3.5% 與 3/4，VoxCPM2, UTF-8, v2.0.1！') == (
        '二零二六年第 十二 題，負百分之三點五 與 四分之三，VoxCPM2, UTF-8, v2.0.1！'
    )
    assert normalize_numbers('2026年：1,234 個、.5%、-.75、1/1000。') == (
        '二零二六年：一千二百三十四 個、百分之零點五、負零點七五、一千分之一。'
    )
    assert normalize_numbers('收入 1200 元，100,010,001 人。') == (
        '收入 一千二百 元，一億零一萬零一 人。'
    )

    # 3. to_model_text
    lex = {'括號': '瓜號'}
    got = to_model_text('活著寫著作，第 12 題的括號佔 50%，用 VoxCPM2。', lex)
    if _t2s_quiet('著作') == '著作' and _t2s_quiet('題') == '题':
        assert got == '活着写著作，第 十二 题的瓜号占 百分之五十，用 VoxCPM2。', f'got={got}'
    else:
        assert '活着' in got and '著作' in got and '十二' in got and '瓜' in got and 'VoxCPM2' in got

    # 4. poly_warnings（含「著」以外的常用歧義）
    pw = poly_warnings('研究奇數、括號與重來，活著寫著作')
    assert pw and any('括號' in x for x in pw) and any('重' in x for x in pw), f'pw={pw}'
    assert not any('著' in x for x in pw), f'不應對已處理的「著」重複警示：{pw}'

    # 5. buzz_db（合成窄頻正弦峰值 > 10.0 vs 乾淨噪聲 <= 10.0）
    sr = 44100
    t = np.arange(int(2.0 * sr), dtype=np.float32) / sr
    rng = np.random.default_rng(42)
    clean = (0.05 * rng.standard_normal(len(t))).astype(np.float32)
    b_clean = buzz_db(clean, sr)
    assert b_clean <= BUZZ_DB, f'乾淨噪聲 buzz_db={b_clean} > {BUZZ_DB}'

    buzzy = (clean + 0.12 * np.sin(2 * np.pi * 4000.0 * t)).astype(np.float32)
    b_buzz = buzz_db(buzzy, sr)
    assert b_buzz > BUZZ_DB, f'合成窄頻峰值 buzz_db={b_buzz} <= {BUZZ_DB}'

    # 6. 若環境有 pypinyin，一併驗證讀音／變調／台灣覆寫邏輯
    try:
        import pypinyin  # noqa: F401
        r1 = compare_pronunciation('移路', '一路')
        assert r1['ok'] is True, f'一路變調比對失敗：{r1}'
        r2 = compare_pronunciation('直數', '質數')
        assert r2['ok'] is True, f'台灣質數覆寫比對失敗：{r2}'
        r3 = compare_pronunciation('什麼時候活着寫著作', '什么时候活着写著作')
        assert r3['ok'] is True, f'繁簡與 fix_zhe 比對失敗：{r3}'
        r4 = compare_pronunciation('完全不同', '應該吻合')
        assert r4['ok'] is False and r4['mismatch'], f'錯字比對失敗：{r4}'
    except ImportError:
        pass

    print(f'selftest OK（buzz_db 乾淨={b_clean} dB，窄頻={b_buzz} dB）')


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass
    ap = argparse.ArgumentParser(description='VoxCPM2 語音前後處理與聲學／文字檢核工具')
    ap.add_argument('--selftest', action='store_true', help='執行無模型單元自我測試')
    args = ap.parse_args()
    if args.selftest:
        selftest()
    else:
        ap.print_help()


if __name__ == '__main__':
    main()
