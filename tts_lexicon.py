#!/usr/bin/env python3
"""
tts_lexicon.py - 讀音替換表：送進 VoxCPM2 前把念錯的詞換成念得對的寫法（多音字、英文詞）。

VoxCPM2 不支援注音標記，多音字、英文詞念錯只能改字。表的格式是 Markdown 表格
「| 原文 | 改寫 | 說明 |」，格式規則寫在主表開頭。只改送進模型的文字，
參考音的逐字稿（prompt.txt）不受影響。

找表的順序（近的覆蓋遠的，同一個原文以近的為準）：
  1. 主表：環境變數 TTS_LEXICON；沒設就從本專案資料夾往上找 ai-video-workflow/tts-lexicon.md
     （以本檔所在位置為起點，所以從任何工作目錄呼叫 clone.py 都找得到）
  2. 本專案根目錄的 tts-lexicon.md
  3. 輸出資料夾往上每一層的 tts-lexicon.md
找不到任何表就印一行提示、不做替換。

試表：python tts_lexicon.py "要念的句子"
"""
import os
import re
import sys
from pathlib import Path

NAME = 'tts-lexicon.md'
REPO = Path(__file__).resolve().parent
HEAD = ('原文', '改寫')


def master():
    p = os.environ.get('TTS_LEXICON')
    if p:
        return Path(p).expanduser()
    for d in (REPO, *REPO.parents):
        f = d / 'ai-video-workflow' / NAME
        if f.is_file():
            return f
    return None


def lexicon_files(start=None):
    """回傳要讀的表，遠的在前、近的在後"""
    files = [master(), REPO / NAME]
    if start:
        s = Path(start).resolve()
        files += reversed([d / NAME for d in (s, *s.parents)])
    out = []
    for f in files:
        if f and f.is_file():
            f = f.resolve()
            if f in out:            # 同一份表出現兩次，留近的那個位置
                out.remove(f)
            out.append(f)
    return out


def cells(line):
    s = line.strip()[1:]                                 # 去掉開頭的 |
    if s.endswith('|') and not s.endswith('\\|'):
        s = s[:-1]
    out = []
    for c in re.split(r'(?<!\\)\|', s):
        c = c.strip().replace('\\|', '|')
        if len(c) >= 2 and c[0] == c[-1] == '`':         # `GPT four` 這種用反引號標出空白的寫法
            c = c[1:-1]
        out.append(c)
    return out


def parse(path):
    """只讀表頭是「原文｜改寫」的 Markdown 表格；其他段落、表格都當說明"""
    table, inside = {}, False
    for n, line in enumerate(Path(path).read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.lstrip().startswith('|'):
            inside = False
            continue
        c = cells(line)
        if tuple(c[:2]) == HEAD:
            inside = True
            continue
        if not inside or all(re.fullmatch(r':?-+:?', x) for x in c if x):
            continue
        if len(c) < 2 or not c[0] or not c[1]:
            print(f'[lexicon] ⚠️ {path}:{n} 格式不對，略過（| 原文 | 改寫 | 說明 |，原文、改寫都不能空）', flush=True)
            continue
        table[c[0]] = c[1]
    return table


def load_lexicon(start=None):
    table = {}
    files = lexicon_files(start)
    for f in files:
        table.update(parse(f))
    if files:
        print(f'[lexicon] 讀音替換表 {len(table)} 條：' + '、'.join(str(f) for f in files), flush=True)
    else:
        print(f'[lexicon] 找不到 {NAME}（可設環境變數 TTS_LEXICON），不做替換', flush=True)
    return table


def speakable(text, table):
    """長的原文先比對，一次掃完：換過的結果不會再被其他條目換"""
    if not table:
        return text
    pat = re.compile('|'.join(re.escape(k) for k in sorted(table, key=len, reverse=True)))
    return pat.sub(lambda m: table[m.group(0)], text)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass
    print(speakable(sys.argv[1], load_lexicon()))
