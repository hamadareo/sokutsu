"""配る版(他人の Mac 向けの Sokutsu.app)かどうかと、そのデータの場所。

配る版のビルド(scripts/build_dist.sh)だけが tsuyaku/_dist.json を書く。リポジトリには無いので、
ユーザー本人の版(run.sh の道)では DIST=False のまま何も変わらない。
配る版では設定・記録・用語・モデルを ~/Library/Application Support/Sokutsu/ に置く
(.app の中に書くと署名が壊れて次から開けなくなる)。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

MARKER = Path(__file__).resolve().with_name("_dist.json")


def load_marker(path: Path = MARKER) -> "dict | None":
    """_dist.json の中身。無ければ None(=ユーザー本人の版)。壊れていても配る版とみなす。"""
    try:
        if not path.is_file():
            return None
    except OSError:
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


INFO = load_marker()
DIST = INFO is not None
VERSION = (INFO or {}).get("version", "dev")


def data_dir(home: "str | None" = None) -> Path:
    """配る版のデータの場所。SOKUTSU_DATA_DIR で上書きできる(試験用)。"""
    env = os.environ.get("SOKUTSU_DATA_DIR")
    if env:
        return Path(env)
    return Path(home or Path.home()) / "Library" / "Application Support" / "Sokutsu"


def app_logs_dir(home: "str | None" = None) -> Path:
    """起動役と llama-server の記録の場所(Console.app からも見える)。"""
    return Path(home or Path.home()) / "Library" / "Logs" / "Sokutsu"


DATA = data_dir() if DIST else None
