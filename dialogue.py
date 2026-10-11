#!/usr/bin/env python3
"""
對話生成 — 兩個聲音的閒聊
模型只載入一次，逐句切換聲音生成，最後拼接成完整對話。

要換聲音就改 SPEAKER_A / SPEAKER_B（對應 voices/ 目錄名），要換內容就改 dialogue 清單。
"""
from voxcpm import VoxCPM
import soundfile as sf
import numpy as np
import time
import os
import sys

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
    """從 install.ps1 產生的 .gpu_type 讀取，或自動偵測。"""
    gpu_type_file = os.path.join(REPO_DIR, '.gpu_type')
    if os.path.exists(gpu_type_file):
        with open(gpu_type_file, 'r', encoding='utf-8') as f:
            gpu_type = f.read().strip()
    else:
        import torch
        if torch.cuda.is_available():
            gpu_type = 'cuda'
        elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
            gpu_type = 'mps'          # Apple Silicon
        else:
            gpu_type = 'cpu'

    if gpu_type not in ('cuda', 'mps', 'cpu'):
        return 'cpu'
    # .gpu_type 跟著雲端硬碟同步，可能是別台寫的（例如桌機寫 cuda、Windows on ARM 筆電用不到），
    # 所以實際可不可用還要再確認一次
    if gpu_type != 'cpu':
        import torch
        ok = torch.cuda.is_available() if gpu_type == 'cuda' else (
            getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available())
        if not ok:
            print(f'⚠️ .gpu_type 是 {gpu_type}，但這台用不到，改用 CPU（會較慢）。', file=sys.stderr)
            return 'cpu'
    return gpu_type

def load_voice(voice_name):
    """讀取指定聲音的參考音與逐字稿。"""
    voice_dir = os.path.join(REPO_DIR, "voices", voice_name)
    ref_wav = os.path.join(voice_dir, "ref_voice.wav")
    prompt_file = os.path.join(voice_dir, "prompt.txt")
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt_text = f.read().strip()
    return ref_wav, prompt_text

def main():
    device = detect_device()
    print(f"裝置: {device}")
    print("載入 VoxCPM2 模型...")
    t0 = time.time()
    model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False, device=device, optimize=False)
    print(f"模型載入完成，耗時 {time.time()-t0:.1f}s\n")

    # 預載兩個聲音的參考資料
    voices = {}
    for name in [SPEAKER_A, SPEAKER_B]:
        voices[name] = load_voice(name)
        print(f"  已載入聲音: {name}")

    print(f"\n開始生成對話（{len(dialogue)} 句）...\n")

    clips = []
    for i, (speaker, text) in enumerate(dialogue, 1):
        ref_wav, prompt_text = voices[speaker]
        print(f"[{i}/{len(dialogue)}] {speaker}: {text}")

        t1 = time.time()
        wav = model.generate(
            text=text,
            prompt_wav_path=ref_wav,
            prompt_text=prompt_text,
            reference_wav_path=ref_wav,
            cfg_value=1.5,
            inference_timesteps=10,
        )
        elapsed = time.time() - t1
        duration = len(wav) / model.tts_model.sample_rate
        print(f"        -> {duration:.1f}s ({elapsed:.1f}s)")

        # 句子之間加 0.4 秒停頓
        pause = np.zeros(int(0.4 * model.tts_model.sample_rate), dtype=wav.dtype)
        clips.append(wav)
        clips.append(pause)

    # 拼接所有片段
    full_audio = np.concatenate(clips)
    total_duration = len(full_audio) / model.tts_model.sample_rate
    output_path = os.path.join(REPO_DIR, "output", f"dialogue_{SPEAKER_A}_{SPEAKER_B}.wav")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, full_audio, model.tts_model.sample_rate)

    print(f"\n對話生成完成！")
    print(f"  總長度: {total_duration:.1f}s")
    print(f"  檔案: {output_path}")

if __name__ == "__main__":
    main()
