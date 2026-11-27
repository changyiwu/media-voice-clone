#!/usr/bin/env python3
"""
對話生成 — 兩個聲音的閒聊
模型只載入一次，逐句切換聲音生成，最後拼接成完整對話。

要換聲音就改 SPEAKER_A / SPEAKER_B（對應 voices/ 目錄名），要換內容就改 dialogue 清單。

每句送進模型前套讀音替換表＋數字符號轉中文＋繁轉簡＋fix_zhe（voice_prep.py）；開始前先集中掃描多音／異讀風險提醒；
生成後於濾波前檢查嘶聲與嗡嗡聲，每句各自去頭尾靜音、濾波、調到同一個響度（−15 LUFS）再拼接。
兩個人的聲音原本一大一小、句首句尾的空白長短不一，這樣處理後停頓只剩固定的 0.4 秒。
"""
import argparse
import os
import sys
import time

import numpy as np

from tts_lexicon import load_lexicon
import voice_prep as vp

REPO_DIR = os.path.dirname(os.path.abspath(__file__))

# 說話者：對應 voices/<名稱>/ 目錄
SPEAKER_A = "小吳"
SPEAKER_B = "老柯"

# 對話腳本：(說話者, 文字)
dialogue = [
    (SPEAKER_B, "嘿，你今天怎麼這麼晚才回來？"),
    (SPEAKER_A, "喔，剛剛在弄一個 AI 語音的東西，搞到忘記時間了。"),
    (SPEAKER_B, "AI 語音？就是那個可以模仿別人聲音的嗎？"),
    (SPEAKER_A, "對啊，叫 VoxCPM2，開源的，還可以商用。"),
    (SPEAKER_B, "聽起來很厲害耶，那它有辦法模仿我的聲音嗎？"),
    (SPEAKER_A, "已經模仿了啊，你現在聽到的就是你的聲音。"),
    (SPEAKER_B, "哇，真的假的！那我以後是不是不用自己錄音了？"),
    (SPEAKER_A, "哈，理論上是啦，但還是真人講的比較有感情啦。"),
]


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


def load_voice(voice_name):
    """讀取指定聲音的參考音與逐字稿。"""
    voice_dir = os.path.join(REPO_DIR, "voices", voice_name)
    ref_wav = os.path.join(voice_dir, "ref_voice.wav")
    prompt_file = os.path.join(voice_dir, "prompt.txt")
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt_text = f.read().strip()
    return ref_wav, prompt_text


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="多聲音對話語音生成")
    ap.add_argument("--dry-run", action="store_true",
                    help="只顯示文字正規化結果與多音／異讀風險提醒，不載入模型")
    args = ap.parse_args()

    # 讀音替換表（多音字、英文詞改字）；只改送進模型的文字，逐字稿不動
    output_path = os.path.join(REPO_DIR, "output", f"dialogue_{SPEAKER_A}_{SPEAKER_B}.wav")
    lexicon = load_lexicon(os.path.dirname(output_path))

    # 開始生成前先掃描全腳本並集中列出多音／異讀風險提醒（不阻斷執行）
    prepared = []
    script_warnings = []
    for i, (speaker, text) in enumerate(dialogue, 1):
        spoken_zh, say_text = vp.prepare_text(text, lexicon)
        prepared.append((speaker, text, say_text))
        for w in vp.poly_warnings(spoken_zh):
            script_warnings.append(f"  第 {i} 句（{speaker}）：{w}")
    if script_warnings:
        print("⚠️ 對話腳本多音／異讀風險提醒（僅供試聽留意，不阻斷生成）：")
        for line in script_warnings:
            print(line)
        print()

    if args.dry_run:
        for i, (speaker, text, say_text) in enumerate(prepared, 1):
            print(f"[{i}/{len(dialogue)}] {speaker}: {text}")
            if say_text != text:
                print(f"        改寫後: {say_text}")
        return

    from voxcpm import VoxCPM
    import soundfile as sf

    device = detect_device()
    print(f"裝置: {device}")
    src = vp.model_source()
    print(f"載入 VoxCPM2 模型（{src}）...")
    t0 = time.time()
    model = VoxCPM.from_pretrained(src, load_denoiser=False, device=device, optimize=False)
    print(f"模型載入完成，耗時 {time.time()-t0:.1f}s\n")

    # 預載兩個聲音的參考資料
    voices = {}
    for name in [SPEAKER_A, SPEAKER_B]:
        voices[name] = load_voice(name)
        print(f"  已載入聲音: {name}")

    print(f"\n開始生成對話（{len(dialogue)} 句）...\n")

    clips = []
    prompt_caches = {}
    sr = model.tts_model.sample_rate
    for i, (speaker, text, say_text) in enumerate(prepared, 1):
        ref_wav, prompt_text = voices[speaker]
        print(f"[{i}/{len(dialogue)}] {speaker}: {text}")
        if say_text != text:
            print(f"        改寫後: {say_text}")

        t1 = time.time()
        wav = vp.generate_speech(
            model, say_text, ref_wav, prompt_text, cfg_value=1.5, prompt_caches=prompt_caches,
        )
        elapsed = time.time() - t1
        duration = len(wav) / sr

        # 濾波前檢核高嘶聲／嗡嗡聲
        hf_raw = round(vp.hf_ratio(wav, sr), 3)
        buzz_raw = vp.buzz_db(wav, sr)
        issues = []
        if hf_raw > 0.30:
            issues.append(f"hf_raw={hf_raw:.3f} > 0.30（可能有嘶聲）")
        if buzz_raw > vp.BUZZ_DB:
            issues.append(f"buzz_db={buzz_raw:.1f} dB > {vp.BUZZ_DB:.1f}（可能有嗡嗡聲）")

        wav, note = vp.finish(wav, sr)
        print(f"        -> {duration:.1f}s ({elapsed:.1f}s)；後處理：{note}")
        if issues:
            print(f"        ⚠️ 聲學警示：{'；'.join(issues)}")

        # 句子之間加 0.4 秒停頓
        pause = np.zeros(int(0.4 * sr), dtype=wav.dtype)
        clips.append(wav)
        clips.append(pause)

    # 拼接所有片段
    full_audio = np.concatenate(clips)
    total_duration = len(full_audio) / sr
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, full_audio, sr)

    print("\n對話生成完成！")
    print(f"  總長度: {total_duration:.1f}s")
    print(f"  檔案: {output_path}")


if __name__ == "__main__":
    main()
