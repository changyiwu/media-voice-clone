#!/usr/bin/env python3
"""
audition.py - 多個候選（clone.py --takes N 的產出）串成一個試聽檔、用耳朵挑一支。

VoxCPM2 每次生成都有隨機性，同一句話換個 seed，聲調、停頓、有沒有嘶聲都會不同。
clone.py --takes 4 會在 <輸出檔名>_takes/ 生出 s1…s4，並自動建好試聽檔；本工具也可以單獨跑。

用法：
  python audition.py output/cloned_voice_takes                         → audition.wav＋每支的起始秒數、響度、高頻比
  python audition.py output/cloned_voice_takes --pick s3 -o output/cloned_voice.wav   → 選定的那支複製成正式檔

候選之間用嗶聲隔開（照順序 s1、s2…）；看印出來的「從第幾秒開始」對照。
高頻比（4 kHz 以上能量佔比）超過 max(0.30, 中位數 × 3) 標「可能有嘶聲」，響度跟中位數差 6 dB 以上標「音量離群」
——門檻沿用 media-short-film 的 voice_pick.py。標出來的通常不要選，但最後還是以耳朵為準。
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import soundfile as sf

import voice_prep as vp

for s in (sys.stdout, sys.stderr):
    try:
        s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


def takes(folder):
    fs = [f for f in os.listdir(folder) if f.endswith('.wav') and '_s' in f and f != 'audition.wav']
    seed = lambda f: int(f.rsplit('_s', 1)[1].split('.')[0]) if f.rsplit('_s', 1)[1].split('.')[0].isdigit() else 0
    return sorted(fs, key=seed)


def build(folder):
    files = takes(folder)
    if not files:
        sys.exit(f'{folder} 裡沒有候選（檔名要是 <名稱>_s<數字>.wav）')
    rows, parts, sr0, t = [], [], None, 0.0
    for f in files:
        x, sr = sf.read(os.path.join(folder, f), dtype='float32')
        x = vp.mono(x)
        if sr0 and sr != sr0:
            sys.exit(f'{f} 的取樣率 {sr} 和其他候選 {sr0} 不同')
        sr0 = sr
        beep = (0.25 * np.sin(2 * np.pi * 880 * np.arange(int(0.15 * sr)) / sr)).astype(np.float32)
        gap = np.zeros(int(0.35 * sr), np.float32)
        t += (len(beep) + len(gap)) / sr
        meta = os.path.join(folder, os.path.splitext(f)[0] + '.json')
        hf = None
        if os.path.isfile(meta):                         # clone.py 在濾波前量的高頻比（濾波後嘶聲會被低估）
            with open(meta, encoding='utf-8') as m:
                hf = json.load(m).get('hf_raw')
        rows.append((f, t, len(x) / sr, vp.lufs(x, sr), vp.hf_ratio(x, sr) if hf is None else hf))
        parts += [beep, gap, x, np.zeros(int(0.8 * sr), np.float32)]
        t += len(x) / sr + 0.8
    ls = [r[3] for r in rows if r[3] is not None]
    med_l = float(np.median(ls)) if ls else None
    hf_lim = max(0.30, 3 * float(np.median([r[4] for r in rows])))
    out = os.path.join(folder, 'audition.wav')
    sf.write(out, np.concatenate(parts), sr0, subtype='PCM_16')
    print(f'{"候選":<8}{"從第幾秒":>8}{"長度":>7}{"響度":>8}{"高頻比":>8}')
    for f, start, dur, l, hf in rows:
        tag = f.rsplit('_', 1)[1].replace('.wav', '')
        warn = []
        if hf > hf_lim:
            warn.append('可能有嘶聲')
        if l is not None and med_l is not None and abs(l - med_l) > 6:
            warn.append('音量離群')
        lt = '—' if l is None else f'{l:.1f}'
        print(f'{tag:<8}{start:>7.1f}s{dur:>6.1f}s{lt:>8}{hf:>8.2f}  {"、".join(warn)}')
    print(f'→ {out}')
    print(f'選好後：python audition.py "{folder}" --pick <候選> -o <正式檔名>.wav')


def pick(folder, tag, output):
    files = takes(folder)
    hit = [f for f in files if f.rsplit('_', 1)[1].replace('.wav', '') == tag or f == tag]
    if not hit:
        sys.exit(f'沒有候選 {tag}；有：{"、".join(files)}')
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    shutil.copyfile(os.path.join(folder, hit[0]), output)
    print(f'已用 {hit[0]} → {output}')


def main():
    ap = argparse.ArgumentParser(description='候選試聽與挑選')
    ap.add_argument('folder', help='候選資料夾（clone.py --takes 產生的 <名稱>_takes/）')
    ap.add_argument('--pick', help='選定的候選（s3 或完整檔名）')
    ap.add_argument('--output', '-o', help='選定後複製成這個檔（和 --pick 一起用）')
    a = ap.parse_args()
    if a.pick:
        if not a.output:
            sys.exit('--pick 要搭配 -o <正式檔名>.wav')
        pick(a.folder, a.pick, a.output)
    else:
        build(a.folder)


if __name__ == '__main__':
    main()
