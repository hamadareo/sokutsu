#!/usr/bin/env python3
"""配る Sokutsu.app を作るときの書き換えと最後の確かめ。

  sanitize.py rewrite <dir>...   写し(ビルドの作業場所)の中のテキストを書き換える。リポジトリは触らない。
  sanitize.py check <path>...    .app / DMG の中身を全部(バイナリも)探し、1つでもあれば 1 で終わる。

書き換え(PHRASES): 開発用の画面の文言(自分のリモートサーバーへの SSH 転送の案内)を、配る版の言い方
(config.json の server_url)へ。配る版ではリモートへの道(stt_url・とんねる)は通らない(config.apply_dist)。

確かめ(check): 次のどれかが .app の中に1つでもあれば失敗する。
  - 私的なアドレス(10.x / 172.16-31.x / 192.168.x / CGNAT 帯の 100.64-127.x / *.ts.net)
  - 環境変数 SOKUTSU_FORBIDDEN_WORDS(コンマ区切り・大文字小文字を問わない)に入れた語
    (ビルドする人が、自分の名前・所属・ホスト名などを入れておく。リポジトリには個人の語を置かない)
Python 3.9 以上(ビルドする Mac の python3)で動く。
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

TEXT_EXT = {".py", ".html", ".js", ".css", ".md", ".txt", ".sh", ".json", ".svg", ".plist", ".c"}

# 画面の訳の鍵と訳(1行)。Python の側(2行に割った文字列)と同じ文になるように(ずれると訳されない)
PHRASES = [
    ("(リモートサーバーなら scripts/remote_tunnel.sh、このMacなら LM Studio のサーバー起動)。\"",
     "(自分のサーバーなら config.json の server_url、このMacなら初回の準備が済んでいるか)。\""),
    ("(for the remote server: scripts/remote_tunnel.sh; for this Mac: start the LM Studio server).",
     "(for your server: server_url in config.json; for this Mac: make sure the first-time setup finished)."),
    ("(远程服务器请运行 scripts/remote_tunnel.sh,这台 Mac 请启动 LM Studio 服务器)。",
     "(自己的服务器请检查 config.json 的 server_url,这台 Mac 请确认首次准备已完成)。"),
    ("(원격 서버라면 scripts/remote_tunnel.sh, 이 Mac이라면 LM Studio 서버 시작).",
     "(내 서버라면 config.json의 server_url, 이 Mac이라면 처음 준비가 끝났는지 확인)."),
    ("(リモートサーバーなら \"\n", "(自分のサーバーなら config.json の server_url、\"\n"),
    ("scripts/remote_tunnel.sh、このMacなら LM Studio のサーバー起動)。", "このMacなら初回の準備が済んでいるか)。"),
    ("(リモートサーバー/LM Studio)", "(自分のサーバー/この Mac の翻訳エンジン)"),
    ("(the remote server / LM Studio)", "(your server / the engine on this Mac)"),
    ("(远程服务器/LM Studio)", "(自己的服务器/本机的翻译引擎)"),
    ("(원격 서버/LM Studio)", "(내 서버/이 Mac의 번역 엔진)"),
]

# 1つでも出たらビルド失敗(私的なアドレス・環境変数で指定した語)
_PRIVATE = [
    rb"(?<![0-9.])10\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}(?![0-9])",
    rb"(?<![0-9.])192\.168\.[0-9]{1,3}\.[0-9]{1,3}(?![0-9])",
    rb"(?<![0-9.])172\.(?:1[6-9]|2[0-9]|3[01])\.[0-9]{1,3}\.[0-9]{1,3}(?![0-9])",
    rb"(?<![0-9.])100\.(?:6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\.[0-9]{1,3}\.[0-9]{1,3}(?![0-9])",
    rb"\.ts\.net",
]


def extra_words() -> list:
    return [w.strip() for w in os.environ.get("SOKUTSU_FORBIDDEN_WORDS", "").split(",") if w.strip()]


def forbidden_rx() -> list:
    rx = [re.compile(p) for p in _PRIVATE]
    words = sorted({w.lower() for w in extra_words()}, key=len, reverse=True)
    if words:
        rx.append(re.compile(b"|".join(re.escape(w.encode()) for w in words), re.I))
    return rx


def _files(roots):
    for r in roots:
        r = Path(r)
        for p in ([r] if r.is_file() else sorted(r.rglob("*"))):
            if p.is_file() and not p.is_symlink() and p.suffix in TEXT_EXT:
                yield p


def rewrite(roots, phrases=True) -> int:
    changed = 0
    files = list(_files(roots))
    for p in files:
        t = p.read_text(encoding="utf-8")
        new = t
        for a, b in (PHRASES if phrases else ()):
            new = new.replace(a, b)
        if new != t:
            p.write_text(new, encoding="utf-8")
            changed += 1
    print(f"sanitize: {changed}/{len(files)} ファイルを書き換えました")
    return 0


def check(roots) -> int:
    bad = []
    FORBIDDEN_RX = forbidden_rx()
    for r in roots:
        r = Path(r)
        for p in ([r] if r.is_file() else sorted(r.rglob("*"))):
            if not p.is_file() or p.is_symlink():
                continue
            low = (p.relative_to(r).as_posix() if p != r else p.name).encode()
            for rx in FORBIDDEN_RX:
                if rx.search(low):
                    bad.append(f"{p} (名前に {rx.pattern!r})")
            try:
                data = p.read_bytes()
            except OSError:
                continue
            for rx in FORBIDDEN_RX:
                m = rx.search(data)
                if m:
                    bad.append(f"{p}: {m.group().decode()}  …{data[max(0, m.start() - 30): m.end() + 30]!r}…")
    if bad:
        print("check: 配ってはいけない文字が見つかりました:", file=sys.stderr)
        for b in bad[:60]:
            print("  " + b, file=sys.stderr)
        return 1
    print("check: 私的なアドレス・指定した語は見つかりませんでした")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("rewrite", "check"):
        print(__doc__)
        sys.exit(2)
    if sys.argv[1] == "check":
        sys.exit(check(sys.argv[2:]))
    sys.exit(rewrite(sys.argv[2:], phrases=True))
