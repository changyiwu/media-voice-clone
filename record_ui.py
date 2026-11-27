#!/usr/bin/env python3
"""
record_ui.py - VoxCPM2 錄音介面（不依賴 gradio）

與 app.py 功能相同，但只用 Python 標準庫的 http.server，避開 gradio →
pandas/pyarrow 的原生 DLL。在 Smart App Control 強制執行的機器上，
gradio 會被 Code Integrity 封鎖而無法啟動，此版本不受影響。

音訊處理分工：
  瀏覽器端  MediaRecorder 收音 → Web Audio 解碼 → 重採樣 16kHz 單聲道 → 編碼 WAV
  伺服器端  soundfile 讀取 → 峰值正規化 → 存成 voices/<名稱>/ref_voice.wav

用法：
  python record_ui.py              # http://127.0.0.1:7860
  python record_ui.py --port 7870
  python record_ui.py --open       # 啟動後自動開瀏覽器
"""

import argparse
import hashlib
import io
import json
import os
import socket
import webbrowser
from http.client import HTTPConnection, HTTPException
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
RECORDER_ID = hashlib.sha256(os.fsencode(os.path.normcase(os.path.realpath(REPO_DIR)))).hexdigest()
VOICES_DIR = os.path.join(REPO_DIR, "voices")
SAMPLE_RATE = 16000
MAX_UPLOAD = 200 * 1024 * 1024      # 200MB，約 100 分鐘的 16kHz 單聲道
MIN_DURATION = 2.0                  # 參考音太短會讓克隆品質崩掉
MIN_PEAK = 0.01                     # 低於此視為沒錄到聲音

# 檔名不能出現的字元（Windows 保留字元 + 路徑分隔）
INVALID_CHARS = set('<>:"/\\|?*')


def list_voices():
    """列出已錄製完成的聲音（需同時有參考音與逐字稿）。"""
    if not os.path.isdir(VOICES_DIR):
        return []
    out = []
    for d in sorted(os.listdir(VOICES_DIR)):
        vdir = os.path.join(VOICES_DIR, d)
        if all(os.path.isfile(os.path.join(vdir, f)) for f in ("ref_voice.wav", "prompt.txt")):
            out.append(d)
    return out


def load_sample_text():
    path = os.path.join(REPO_DIR, "texts", "sample_text.txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""


def safe_voice_name(name):
    """把使用者輸入的聲音名稱轉成安全的目錄名，不合法則回傳 None。"""
    name = (name or "").strip().rstrip(". ")
    if not name or len(name) > 64:
        return None
    if name in (".", ".."):
        return None
    if any(c in INVALID_CHARS for c in name):
        return None
    if any(ord(c) < 32 for c in name):
        return None
    return name


def voice_dir_for(name):
    """解析聲音目錄，並確認它確實落在 voices/ 底下（防路徑穿越）。"""
    vdir = os.path.abspath(os.path.join(VOICES_DIR, name))
    root = os.path.abspath(VOICES_DIR)
    if vdir != root and not vdir.startswith(root + os.sep):
        return None
    return vdir


def save_recording(name, wav_bytes):
    """把上傳的 WAV 存成參考音。回傳 (ok, message)。"""
    import soundfile as sf

    try:
        audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
    except Exception as e:
        return False, "無法讀取錄音資料：%s" % e

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if audio.size == 0:
        return False, "錄音是空的，請重新錄一次。"

    if sr != SAMPLE_RATE:
        import resampy
        audio = resampy.resample(audio, sr, SAMPLE_RATE)

    duration = len(audio) / SAMPLE_RATE
    if duration < MIN_DURATION:
        return False, "錄音只有 %.1f 秒，太短了。請完整念完整段文字（約 20 秒）。" % duration

    peak = float(np.abs(audio).max())
    if peak < MIN_PEAK:
        return False, "幾乎沒有錄到聲音（峰值 %.4f）。請確認麥克風有被選到，並靠近一點再錄一次。" % peak
    audio = audio / peak * 0.95

    vdir = voice_dir_for(name)
    if vdir is None:
        return False, "聲音名稱不合法。"

    overwrote = os.path.exists(os.path.join(vdir, "ref_voice.wav"))
    os.makedirs(vdir, exist_ok=True)
    sf.write(os.path.join(vdir, "ref_voice.wav"), audio, SAMPLE_RATE, subtype="PCM_16")
    with open(os.path.join(vdir, "prompt.txt"), "w", encoding="utf-8") as f:
        f.write(load_sample_text())

    lines = []
    if overwrote:
        lines.append("已覆蓋原本的「%s」。" % name)
    lines.append("聲音「%s」錄製成功！（%.0f 秒）" % (name, duration))
    lines.append("")
    lines.append("目前已錄製的聲音：%s" % "、".join(list_voices()))
    lines.append("")
    lines.append("接下來，你可以對 AI 說：")
    lines.append("「用 %s 的聲音說一段話」" % name)
    return True, "\n".join(lines)


PAGE = """<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VoxCPM2 語音錄製</title>
<style>
  * { box-sizing: border-box; }
  body {
    font-family: 'Segoe UI', 'Microsoft JhengHei', sans-serif;
    margin: 0; padding: 32px 16px; background: #fafafa; color: #222;
  }
  .wrap { max-width: 720px; margin: 0 auto; }
  header { text-align: center; margin-bottom: 24px; }
  header h1 { font-size: 2em; margin: 0 0 4px; }
  header p { font-size: 1.05em; color: #666; margin: 0; }
  .step-box {
    background: #fff; border: 1px solid #e0e0e0; border-radius: 12px;
    padding: 20px; margin: 12px 0;
  }
  .step-box h2 { font-size: 1.15em; margin: 0 0 12px; }
  input[type=text] {
    width: 100%; padding: 10px 12px; font-size: 1em; border-radius: 8px;
    border: 1px solid #ccc; font-family: inherit;
  }
  .hint { font-size: .85em; color: #888; margin-top: 8px; }
  .script {
    background: #fff3cd; padding: 16px; border-radius: 8px;
    font-size: 1.1em; line-height: 2; white-space: pre-wrap;
  }
  .rec-row { display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }
  .btn {
    padding: 12px 20px; font-size: 1em; font-family: inherit; cursor: pointer;
    border-radius: 8px; border: 1px solid #ccc; background: #fff;
  }
  .btn:hover:not(:disabled) { background: #f0f0f0; }
  .btn:disabled { opacity: .45; cursor: not-allowed; }
  .btn.recording { background: #d93025; border-color: #d93025; color: #fff; }
  .btn.primary {
    background: #2d6cdf; border-color: #2d6cdf; color: #fff;
    width: 100%; margin-top: 16px; font-size: 1.05em;
  }
  .btn.primary:hover:not(:disabled) { background: #2559b8; }
  .timer { font-variant-numeric: tabular-nums; font-size: 1.3em; color: #555; }
  audio { width: 100%; margin-top: 16px; }
  .msg {
    white-space: pre-wrap; word-break: break-word; padding: 16px;
    border-radius: 8px; font-family: inherit; font-size: .95em; line-height: 1.7;
    margin: 12px 0 0;
  }
  .msg.ok { background: #e6f4ea; border: 1px solid #b7dfc4; color: #1d6b3d; }
  .msg.err { background: #fce8e6; border: 1px solid #f3b8b2; color: #a5261c; }
  footer { text-align: center; margin-top: 24px; color: #999; font-size: .85em; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>&#127897;&#65039; VoxCPM2 語音錄製</h1>
    <p>錄下你的聲音，後續由 AI 幫你生成任何語音。</p>
  </header>

  <section class="step-box">
    <h2>&#9999;&#65039; 為聲音取名字</h2>
    <input type="text" id="name" placeholder="例如：王老師、林主任..." autocomplete="off">
    <div class="hint" id="existing"></div>
  </section>

  <section class="step-box">
    <h2>&#128214; 請念這段文字</h2>
    <div class="script" id="script">載入中...</div>
  </section>

  <section class="step-box">
    <h2>&#127908; 錄音並儲存</h2>
    <div class="rec-row">
      <button class="btn" id="recBtn">&#9679; 開始錄音</button>
      <span class="timer" id="timer">00:00</span>
      <span class="hint" id="recHint">念完後按「停止錄音」</span>
    </div>
    <audio id="preview" controls hidden></audio>
    <button class="btn primary" id="saveBtn" disabled>&#11015;&#65039; 儲存聲音</button>
  </section>

  <div class="msg" id="msg" hidden></div>

  <footer>VoxCPM2（Apache-2.0 可商用）</footer>
</div>

<script>
const $ = (id) => document.getElementById(id);
let recorder = null, chunks = [], stream = null;
let wavBlob = null, timerId = null, startedAt = 0;

function showMsg(text, ok) {
  const m = $('msg');
  m.textContent = text;
  m.className = 'msg ' + (ok ? 'ok' : 'err');
  m.hidden = false;
}

function fmtTime(sec) {
  const m = String(Math.floor(sec / 60)).padStart(2, '0');
  const s = String(Math.floor(sec % 60)).padStart(2, '0');
  return m + ':' + s;
}

async function loadInfo() {
  try {
    const r = await fetch('/api/info');
    const info = await r.json();
    $('script').textContent = info.sample_text || '(找不到 texts/sample_text.txt)';
    $('existing').textContent = info.voices.length
      ? '已錄製：' + info.voices.join('、')
      : '目前還沒有任何錄好的聲音。';
  } catch (e) {
    $('script').textContent = '無法載入稿件：' + e;
  }
}

/* MediaRecorder 產生的是 webm/opus，伺服器端讀不了。
   在瀏覽器解碼後重採樣成 16kHz 單聲道，再編成標準 WAV 送出。 */
async function toWav16k(blob) {
  const buf = await blob.arrayBuffer();
  const ctx = new AudioContext();
  let decoded;
  try {
    decoded = await ctx.decodeAudioData(buf);
  } finally {
    ctx.close();
  }
  const frames = Math.max(1, Math.ceil(decoded.duration * 16000));
  const off = new OfflineAudioContext(1, frames, 16000);
  const src = off.createBufferSource();
  src.buffer = decoded;          // 多聲道會由 OfflineAudioContext 自動降混成單聲道
  src.connect(off.destination);
  src.start();
  const rendered = await off.startRendering();
  return encodeWav(rendered.getChannelData(0), 16000);
}

function encodeWav(samples, sampleRate) {
  const bytes = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(bytes);
  const writeStr = (off, str) => {
    for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i));
  };
  writeStr(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);              // PCM
  view.setUint16(22, 1, true);              // 單聲道
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate
  view.setUint16(32, 2, true);              // block align
  view.setUint16(34, 16, true);             // 16-bit
  writeStr(36, 'data');
  view.setUint32(40, samples.length * 2, true);
  let off = 44;
  for (let i = 0; i < samples.length; i++, off += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([bytes], { type: 'audio/wav' });
}

async function startRec() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: false,
               noiseSuppression: false, autoGainControl: false }
    });
  } catch (e) {
    showMsg('無法使用麥克風：' + e.message +
            '\\n\\n請在瀏覽器網址列左側允許麥克風權限，然後重新整理頁面。', false);
    return;
  }
  chunks = [];
  wavBlob = null;
  $('preview').hidden = true;
  $('saveBtn').disabled = true;
  $('msg').hidden = true;

  recorder = new MediaRecorder(stream);
  recorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
  recorder.onstop = async () => {
    stream.getTracks().forEach((t) => t.stop());
    $('recBtn').disabled = true;
    $('recHint').textContent = '處理中...';
    try {
      wavBlob = await toWav16k(new Blob(chunks, { type: recorder.mimeType }));
      const url = URL.createObjectURL(wavBlob);
      $('preview').src = url;
      $('preview').hidden = false;
      $('saveBtn').disabled = false;
      $('recHint').textContent = '可以試聽，滿意就按儲存；不滿意可以重錄。';
    } catch (e) {
      showMsg('錄音轉檔失敗：' + e, false);
      $('recHint').textContent = '請重新錄一次。';
    }
    $('recBtn').disabled = false;
  };
  recorder.start();

  startedAt = Date.now();
  $('timer').textContent = '00:00';
  timerId = setInterval(() => {
    $('timer').textContent = fmtTime((Date.now() - startedAt) / 1000);
  }, 200);

  $('recBtn').textContent = '■ 停止錄音';
  $('recBtn').classList.add('recording');
  $('recHint').textContent = '錄音中... 念完後按「停止錄音」';
}

function stopRec() {
  if (recorder && recorder.state !== 'inactive') recorder.stop();
  clearInterval(timerId);
  $('recBtn').textContent = '● 重新錄音';
  $('recBtn').classList.remove('recording');
}

$('recBtn').addEventListener('click', () => {
  if (recorder && recorder.state === 'recording') stopRec();
  else startRec();
});

$('saveBtn').addEventListener('click', async () => {
  const name = $('name').value.trim();
  if (!name) { showMsg('請先為你的聲音取一個名字。', false); $('name').focus(); return; }
  if (!wavBlob) { showMsg('請先錄音。', false); return; }

  $('saveBtn').disabled = true;
  $('saveBtn').textContent = '儲存中...';
  try {
    const r = await fetch('/api/save?name=' + encodeURIComponent(name), {
      method: 'POST',
      headers: { 'Content-Type': 'audio/wav' },
      body: wavBlob
    });
    const res = await r.json();
    showMsg(res.msg, res.ok);
    if (res.ok) loadInfo();
  } catch (e) {
    showMsg('儲存失敗：' + e, false);
  }
  $('saveBtn').disabled = false;
  $('saveBtn').textContent = '⬇️ 儲存聲音';
});

loadInfo();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    server_version = "VoxCPM2Recorder/1.0"

    def log_message(self, fmt, *args):
        pass  # 不要把每個請求都印到終端機

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self._send(code, body, "application/json; charset=utf-8")

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/info":
            self._json(200, {"sample_text": load_sample_text(), "voices": list_voices()})
        elif path == "/api/status":
            self._send(200, RECORDER_ID.encode("ascii"), "text/plain; charset=ascii")
        else:
            self._send(404, b"not found", "text/plain; charset=utf-8")

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/save":
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return

        raw_name = (parse_qs(parsed.query).get("name") or [""])[0]
        name = safe_voice_name(raw_name)
        if name is None:
            self._json(400, {"ok": False,
                             "msg": '聲音名稱不合法。不能空白、不能超過 64 字，'
                                    '也不能包含 < > : " / \\ | ? * 這些字元。'})
            return

        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0:
            self._json(400, {"ok": False, "msg": "沒有收到錄音資料。"})
            return
        if length > MAX_UPLOAD:
            self._json(413, {"ok": False, "msg": "錄音檔太大（超過 200MB）。"})
            return

        data = self.rfile.read(length)
        if len(data) != length:
            self._json(400, {"ok": False, "msg": "錄音資料傳輸不完整，請再試一次。"})
            return

        try:
            ok, msg = save_recording(name, data)
        except Exception as e:
            ok, msg = False, "儲存失敗：%s" % e
        self._json(200 if ok else 400, {"ok": ok, "msg": msg})


def is_own_recorder(port):
    """確認連接埠上的錄音服務來自同一個專案目錄。"""
    connection = HTTPConnection("127.0.0.1", port, timeout=1)
    try:
        connection.request("GET", "/api/status")
        response = connection.getresponse()
        return response.status == 200 and response.read(len(RECORDER_ID) + 1) == RECORDER_ID.encode("ascii")
    except (OSError, HTTPException):
        return False
    finally:
        connection.close()


class RecorderServer(ThreadingHTTPServer):
    # Windows 的 SO_REUSEADDR 允許綁到其他行程已在監聽的位址。
    allow_reuse_address = os.name != 'nt'
    allow_reuse_port = False

    def server_bind(self):
        if os.name == 'nt':
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def main():
    p = argparse.ArgumentParser(description="VoxCPM2 錄音介面（不依賴 gradio）")
    p.add_argument("--port", "-p", type=int, default=7860)
    p.add_argument("--open", action="store_true", help="啟動後自動開啟瀏覽器")
    args = p.parse_args()
    if not 0 <= args.port <= 65535:
        p.error("--port 必須介於 0 和 65535")

    os.makedirs(VOICES_DIR, exist_ok=True)
    try:
        httpd = RecorderServer(("127.0.0.1", args.port), Handler)
    except OSError as e:
        if args.open and is_own_recorder(args.port):
            print("錄音介面已啟動，正在開啟瀏覽器。")
            webbrowser.open("http://127.0.0.1:%d" % args.port)
            return
        print("無法在 port %d 啟動：%s" % (args.port, e))
        print("若連接埠已被占用，可改用其他 port：")
        print("  python record_ui.py --open --port 7870")
        raise SystemExit(1)
    url = "http://127.0.0.1:%d" % httpd.server_port

    voices = list_voices()
    print("=" * 56)
    print("  VoxCPM2 錄音介面")
    print("=" * 56)
    print("  網址: %s" % url)
    print("  聲音存放: %s" % VOICES_DIR)
    print("  已錄製: %s" % ("、".join(voices) if voices else "(還沒有)"))
    print("")
    print("  按 Ctrl+C 停止伺服器。")
    print("=" * 56)

    if args.open:
        webbrowser.open(url)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
