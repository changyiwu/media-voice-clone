#!/usr/bin/env python3
"""
clone.py - 用 VoxCPM2 Ultimate Cloning 生成你的克隆語音。

預設使用「小吳」聲音。也可以用 --voice 指定其他聲音，或用 --reference/--text-file 自訂。

用法：
  python clone.py "你好，這是我的克隆聲音。"
  python clone.py "文字" --voice 小吳
  python clone.py --file my_script.txt
  python clone.py "文字" --output output/my_voice.wav
  python clone.py "文字" --reference my_ref.wav --text-file my_ref_text.txt
  python clone.py "文字" --takes 4                 生 4 個候選（seed 1–4）＋試聽檔，用 audition.py 挑一支
  python clone.py "文字" --dry-run                 只看送進模型的文字（讀音替換表＋繁轉簡），不載模型

預設會做後處理（voice_prep.py）：去頭尾靜音、濾低頻轟聲與嘶聲、響度調到 −15 LUFS；--raw 跳過。
"""

import os
import sys
import time
import argparse

REPO_DIR = os.path.dirname(os.path.abspath(__file__))

def detect_device():
    """問這台的 torch 有什麼可用：cuda → mps → cpu。

    不讀舊版 install.ps1 寫的 .gpu_type：它跟著雲端硬碟同步，是最後一台跑安裝的電腦寫的，
    兩個方向都會錯（別台寫 cpu 會讓有 GPU 的這台跑 CPU）。
    """
    import torch
    if torch.cuda.is_available():
        return 'cuda'
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return 'mps'          # Apple Silicon
    return 'cpu'


def resolve_voice_files(voice, reference_override, text_override):
    """根據 voice 名稱解析參考音和逐字稿路徑。"""
    voice_dir = os.path.join(REPO_DIR, 'voices', voice)
    reference = reference_override or os.path.join(voice_dir, 'ref_voice.wav')
    text_file = text_override or os.path.join(voice_dir, 'prompt.txt')
    return reference, text_file


def main():
    parser = argparse.ArgumentParser(description='VoxCPM2 Ultimate Cloning 語音生成')
    parser.add_argument('text', nargs='?', help='要生成的文字')
    parser.add_argument('--file', '-f', help='從文字檔讀取要生成的內容')
    parser.add_argument('--output', '-o', default='output/cloned_voice.wav',
                        help='輸出檔案路徑（預設: output/cloned_voice.wav）')
    parser.add_argument('--voice', '-v', default='小吳',
                        help='聲音名稱，對應 voices/ 目錄（預設: 小吳）')
    parser.add_argument('--reference', '-r',
                        help='覆蓋參考音檔路徑（預設由 --voice 決定）')
    parser.add_argument('--text-file', '-t',
                        help='覆蓋逐字稿檔案路徑（預設由 --voice 決定）')
    parser.add_argument('--device', '-d', help='強制指定裝置 (cuda/mps/cpu)')
    parser.add_argument('--takes', type=int, default=1,
                        help='生成幾個候選（不同 seed）給你用耳朵挑；>1 時存到 <輸出檔名>_takes/ 並建試聽檔（見 audition.py）')
    parser.add_argument('--raw', action='store_true',
                        help='不做後處理（預設會去頭尾靜音、濾低頻轟聲與嘶聲、響度調到 −15 LUFS）')
    parser.add_argument('--dry-run', action='store_true', help='只印出送進模型的文字（套讀音替換表＋繁轉簡），不載入模型')
    args = parser.parse_args()

    # 取得要生成的文字
    if args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            gen_text = f.read().strip()
    elif args.text:
        gen_text = args.text
    else:
        print('錯誤：請提供要生成的文字，或用 --file 指定文字檔。')
        print('範例: python clone.py "你好，這是我的克隆聲音。"')
        sys.exit(1)

    # 套讀音替換表（多音字、英文詞改字）再繁轉簡（VoxCPM2 念簡體比較穩）；只改送進模型的文字，逐字稿不動
    from tts_lexicon import load_lexicon
    import voice_prep as vp
    say_text = vp.to_model_text(gen_text, load_lexicon(os.path.dirname(os.path.abspath(args.output))))
    if say_text != gen_text:
        print(f'改寫後: {say_text}')
    if args.dry_run:
        return

    # 根據 voice 名稱解析參考音和逐字稿
    reference, text_file = resolve_voice_files(args.voice, args.reference, args.text_file)

    # 檢查參考音檔
    if not os.path.exists(reference):
        print(f'錯誤：找不到參考音檔 {reference}')
        if args.voice:
            print(f'請先錄製「{args.voice}」聲音: python record.py --voice {args.voice}')
        sys.exit(1)

    # 讀取逐字稿
    if not os.path.exists(text_file):
        print(f'錯誤：找不到逐字稿檔案 {text_file}')
        sys.exit(1)
    with open(text_file, 'r', encoding='utf-8') as f:
        prompt_text = f.read().strip()

    # 偵測裝置
    device = args.device or detect_device()

    # 載入模型
    from voxcpm import VoxCPM
    import soundfile as sf

    print(f'聲音: {args.voice}')
    print(f'裝置: {device}')
    print('載入 VoxCPM2 模型（首次會下載權重，約 4.7GB）...')
    t0 = time.time()
    model = VoxCPM.from_pretrained(
        'openbmb/VoxCPM2',
        load_denoiser=False,
        device=device,
        optimize=False,
    )
    print(f'模型載入完成，耗時 {time.time()-t0:.1f}s')

    sr = model.tts_model.sample_rate
    if args.takes > 1:
        stem, ext = os.path.splitext(args.output)
        folder = f'{stem}_takes'
        os.makedirs(folder, exist_ok=True)
        jobs = [(seed, os.path.join(folder, f'{os.path.basename(stem)}_s{seed}{ext or ".wav"}'))
                for seed in range(1, args.takes + 1)]
    else:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        jobs = [(None, args.output)]

    for seed, out_path in jobs:
        if seed is not None:                             # 固定 seed：同一句同一個 seed 每次結果相同，挑中的可以重現
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        label = f'候選 s{seed}' if seed is not None else '生成'
        print(f'{label}中（{len(say_text)} 字）...')
        t1 = time.time()
        wav = model.generate(
            text=say_text,
            prompt_wav_path=reference,
            prompt_text=prompt_text,
            reference_wav_path=reference,
            cfg_value=2.0,
            inference_timesteps=10,
        )
        elapsed = time.time() - t1
        duration = len(wav) / sr
        print(f'{label}完成，耗時 {elapsed:.1f}s（語音長度 {duration:.1f}s，RTF={elapsed/duration:.1f}）')
        if seed is not None:                             # 嘶聲要在濾波前量（濾掉 11 kHz 以上後就看不太出來），記給 audition.py
            import json
            with open(os.path.splitext(out_path)[0] + '.json', 'w', encoding='utf-8') as f:
                json.dump({'seed': seed, 'hf_raw': round(vp.hf_ratio(wav, sr), 3), 'text': say_text}, f, ensure_ascii=False)
        if not args.raw:
            wav, note = vp.finish(wav, sr)
            print(f'後處理：去頭尾靜音、濾波，{note}')
        sf.write(out_path, wav, sr)
        print(f'已存檔: {out_path}')

    if args.takes > 1:
        import audition
        audition.build(folder)


if __name__ == '__main__':
    main()
