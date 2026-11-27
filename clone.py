#!/usr/bin/env python3
"""
clone.py - 用 VoxCPM2 Ultimate Cloning 生成你的克隆語音。

預設使用「小吳」聲音。也可以用 --voice 指定其他聲音，或用 --reference/--text-file 自訂。

用法：
  python clone.py "你好，這是我的克隆聲音。"
  python clone.py "文字" --voice 小吳
  python clone.py --file my_script.txt
  python clone.py --batch jobs.json
  python clone.py "文字" --output output/my_voice.wav
  python clone.py "文字" --reference my_ref.wav --text-file my_ref_text.txt
  python clone.py "文字" --takes 4                 生 4 個候選（seed 1–4）＋試聽檔，用 audition.py 挑一支
  python clone.py "文字" --verify                  生成後用本機 faster-whisper 執行 ASR 聽寫與讀音比對
  python clone.py "文字" --dry-run                 只看送進模型的文字（讀音替換表＋數字符號轉中文＋繁轉簡＋fix_zhe）與多音提醒，不載模型

預設會做後處理（voice_prep.py）：去頭尾靜音、濾低頻轟聲與嘶聲、響度調到 −15 LUFS；--raw 跳過。
"""

import argparse
from itertools import count
import json
import os
from pathlib import Path
import sys
import tempfile
import time

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


def load_jobs(args):
    if not args.batch:
        text = Path(args.file).read_text(encoding='utf-8-sig').strip() if args.file else args.text
        if not text or not text.strip():
            raise ValueError('請提供要生成的文字，或用 --file 指定文字檔。')
        return [(text, Path(args.output or 'output/cloned_voice.wav'), args.voice)]

    entries = json.loads(Path(args.batch).read_text(encoding='utf-8-sig'))
    if not isinstance(entries, list) or not entries:
        raise ValueError('--batch 需要一個非空的 JSON 陣列。')
    jobs, outputs = [], set()
    for i, entry in enumerate(entries, 1):
        if not isinstance(entry, dict) or entry.keys() - {'text', 'output', 'voice'}:
            raise ValueError(f'第 {i} 筆需為包含 text、output 與選填 voice 的物件。')
        text, output = entry.get('text'), entry.get('output')
        voice = entry.get('voice', args.voice)
        if not all(isinstance(v, str) and v.strip() for v in (text, output, voice)):
            raise ValueError(f'第 {i} 筆的 text、output 與 voice 需為非空字串。')
        if '\0' in output or '\0' in voice:
            raise ValueError(f'第 {i} 筆的 output 或 voice 包含無效字元。')
        path = Path(output)
        key = os.path.normcase(os.path.realpath(path))
        if not path.name or path.is_dir():
            raise ValueError(f'第 {i} 筆的輸出必須是檔案路徑：{path}')
        if key in outputs:
            raise ValueError(f'第 {i} 筆的輸出路徑重複：{path}')
        outputs.add(key)
        jobs.append((text, path, voice))
    return jobs


def load_model(device):
    from voxcpm import VoxCPM
    import voice_prep as vp

    print(f'裝置: {device}')
    src = vp.model_source()
    if src == vp.HF_MODEL_ID:
        print('載入 VoxCPM2 模型（首次會下載權重，約 4.7GB）...')
    else:
        print(f'載入 VoxCPM2 模型（本機：{src}）...')
    t0 = time.time()
    model = VoxCPM.from_pretrained(
        src,
        load_denoiser=False,
        device=device,
        optimize=False,
    )
    print(f'模型載入完成，耗時 {time.time()-t0:.1f}s')
    return model


def render_job(model, args, text, say_text, output, reference, prompt_text, prompt_caches):
    import soundfile as sf
    import voice_prep as vp

    sr = model.tts_model.sample_rate

    def render(out_path, seed=None):
        if seed is not None:                             # 固定 seed：同一句同一個 seed 每次結果相同，挑中的可以重現
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        label = f'候選 s{seed}' if seed is not None else '生成'
        print(f'{label}中（{len(say_text)} 字）...')
        t1 = time.time()
        wav = vp.generate_speech(
            model, say_text, reference, prompt_text, cfg_value=2.0, prompt_caches=prompt_caches,
        )
        elapsed = time.time() - t1
        duration = len(wav) / sr
        if not duration:
            raise ValueError(f'{output}：模型傳回空音訊。')
        print(f'{label}完成，耗時 {elapsed:.1f}s（語音長度 {duration:.1f}s，RTF={elapsed/duration:.1f}）')

        # 嘶聲與嗡嗡聲要在濾波前量（濾掉 11 kHz 以上後就看不太出來）
        hf_raw = round(vp.hf_ratio(wav, sr), 3)
        buzz_raw = vp.buzz_db(wav, sr)
        if seed is not None:
            with out_path.with_suffix('.json').open('w', encoding='utf-8') as f:
                json.dump(
                    {
                        'seed': seed,
                        'hf_raw': hf_raw,
                        'buzz_db': buzz_raw,
                        'text': say_text,
                        'orig_text': text,
                    },
                    f,
                    ensure_ascii=False,
                )
        else:
            issues = []
            if hf_raw > 0.30:
                issues.append(f'高頻比 hf_raw={hf_raw:.3f} > 0.30（可能有嘶聲）')
            if buzz_raw > vp.BUZZ_DB:
                issues.append(f'窄頻峰值 buzz_db={buzz_raw:.1f} dB > {vp.BUZZ_DB:.1f}（可能有嗡嗡聲）')
            if issues:
                print(f'⚠️ 聲學品質警示：{"；".join(issues)}，建議改跑 --takes 4 挑版！')
            else:
                print(f'聲學檢核（濾波前）：hf_raw={hf_raw:.3f}、buzz_db={buzz_raw:.1f} dB（正常）')

        if not args.raw:
            wav, note = vp.finish(wav, sr)
            print(f'後處理：去頭尾靜音、濾波，{note}')
        sf.write(out_path, wav, sr)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.clone-', dir=output.parent) as tmp:
        staging = Path(tmp)
        if args.takes == 1:
            staged_file = staging / output.name
            render(staged_file)
            staged_file.replace(output)
        else:
            import audition
            for seed in range(1, args.takes + 1):
                render(staging / f'{output.stem}_s{seed}.wav', seed)
            audition.build(staging, verify=args.verify, show_path=False)
            folder = publish_takes(staging, output)
            audition.print_audition_path(folder)
            return

    print(f'已存檔: {output}')
    if args.verify:
        vr = vp.verify_pronunciation(output, say_text, orig_text=text)
        if not vr['available']:
            print(f"[verify] {vr['reason']}")
        else:
            extra = f"（{vr['reason']}）" if vr['reason'] else ''
            if vr['ok']:
                print(f"[verify] ✅ 讀音吻合（相似度 {vr['sim']:.2f}）：{vr['heard']}{extra}")
            else:
                diff = f"；差異：{'、'.join(vr['mismatch'])}" if vr['mismatch'] else ''
                print(f"[verify] ⚠️ 聽寫／讀音不符（相似度 {vr['sim']:.2f}）：聽寫={vr['heard']}{diff}{extra}，建議改跑 --takes 4 挑版！")


def publish_takes(staging, output):
    """只在整批完成後發布，保留已存在的候選批次。"""
    for i in count(1):
        suffix = '' if i == 1 else f'_{i}'
        folder = output.with_name(f'{output.stem}_takes{suffix}')
        if os.path.lexists(folder):
            continue
        try:
            return staging.rename(folder)
        except OSError:
            if not os.path.lexists(folder):
                raise


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

    parser = argparse.ArgumentParser(description='VoxCPM2 Ultimate Cloning 語音生成')
    parser.add_argument('text', nargs='?', help='要生成的文字')
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument('--file', '-f', help='從文字檔讀取要生成的內容')
    inputs.add_argument('--batch', help='批次 JSON 檔，每筆含 text、output 與選填的 voice')
    parser.add_argument('--output', '-o', help='單筆輸出路徑（預設: output/cloned_voice.wav）')
    parser.add_argument('--voice', '-v', default='小吳',
                        help='聲音名稱，對應 voices/ 目錄（預設: 小吳）')
    parser.add_argument('--reference', '-r', help='覆蓋參考音檔路徑（預設由 --voice 決定）')
    parser.add_argument('--text-file', '-t', help='覆蓋逐字稿檔案路徑（預設由 --voice 決定）')
    parser.add_argument('--device', '-d', help='強制指定裝置 (cuda/mps/cpu)')
    parser.add_argument('--takes', type=int, default=1,
                        help='每筆生成幾個候選；多候選時另外存一個候選資料夾與試聽檔')
    parser.add_argument('--raw', action='store_true', help='跳過去靜音、濾波與 −15 LUFS 後處理')
    parser.add_argument('--verify', action='store_true', help='生成後用本機 faster-whisper 執行聽寫檢核')
    parser.add_argument('--dry-run', action='store_true', help='顯示送入模型的文字與多音提示')
    args = parser.parse_args()
    if args.batch and (args.text is not None or args.output is not None):
        parser.error('--batch 已在清單裡指定文字與輸出，請移除單筆的文字或 --output。')
    if args.takes < 1:
        parser.error('--takes 至少為 1。')
    try:
        jobs = load_jobs(args)
    except (OSError, ValueError) as e:
        parser.error(str(e))

    from tts_lexicon import load_lexicon
    import voice_prep as vp

    prepared, lexicons, transcripts = [], {}, {}
    for i, (text, output, voice) in enumerate(jobs, 1):
        if args.batch:
            print(f'[{i}/{len(jobs)}] {voice} → {output}')
        parent = output.parent.resolve()
        if parent not in lexicons:
            lexicons[parent] = load_lexicon(parent)
        spoken_zh, say_text = vp.prepare_text(text, lexicons[parent])
        if args.dry_run:
            print(f'送入模型：{say_text}')
        elif say_text != text:
            print(f'改寫後: {say_text}')
        warnings = vp.poly_warnings(spoken_zh)
        if warnings:
            print('⚠️ 多音／異讀風險提醒（僅供試聽留意，不阻斷生成）：')
            for warning in warnings:
                print(f'  - {warning}')
        if not args.dry_run:
            reference, text_file = resolve_voice_files(voice, args.reference, args.text_file)
            if not os.path.isfile(reference):
                parser.error(f'找不到「{voice}」的參考音檔：{reference}')
            if text_file not in transcripts:
                try:
                    transcripts[text_file] = Path(text_file).read_text(encoding='utf-8-sig').strip()
                except (OSError, ValueError) as e:
                    parser.error(f'無法讀取「{voice}」的逐字稿：{e}')
            prepared.append((text, say_text, output, voice, reference, transcripts[text_file]))
    if args.dry_run:
        return

    model = load_model(args.device or detect_device())
    prompt_caches = {} if args.takes > 1 or len(prepared) > 1 else None
    for i, (text, say_text, output, voice, reference, prompt_text) in enumerate(prepared, 1):
        print(f'[{i}/{len(prepared)}] 生成 {voice} → {output}' if args.batch else f'聲音: {voice}')
        render_job(model, args, text, say_text, output, reference, prompt_text, prompt_caches)


if __name__ == '__main__':
    main()
