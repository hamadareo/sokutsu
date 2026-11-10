"""配る版の初回のモデル取得(11/07)。

大きいモデルは DMG に入れず、初回起動の画面(webui/setup.html)で公開元から取る:
  - 取得元は公式だけ(Hugging Face の ggml-org / ggerganov、GitHub の k2-fsa/sherpa-onnx リリース)。
  - <名前>.part に書き、HTTP Range で途中から続ける(窓を閉じても次の起動で続き)。
  - 全部受けたら SHA256 を確かめてから名前を変える。合わなければ .part を消して「壊れていました」。
  - tar.bz2 は一時フォルダに展開してから名前を変える(外へ書く・リンクの入った tar は拒む)。
  - 始める前に空き容量を確かめる。
推論はすべてこの Mac の中(クラウド API は使わない)。取るのはモデルのファイルだけ。
"""
from __future__ import annotations

import hashlib
import shutil
import tarfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

GB = 1024 ** 3


@dataclass(frozen=True)
class Item:
    key: str
    label: str            # 画面の名前(日本語。英語は labelEn)
    url: str
    sha256: str
    size: int
    dest: str             # models/ の下の最終の名前(ファイル、または tar の中の一番上のフォルダ)
    archive: bool = False
    label_en: str = ""


_GH = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
# Hugging Face はコミットで固定する(resolve/main のままだと公開元が差し替えた時に、全員が全部受けてから
# 「壊れていました」を繰り返す)。11/07 に各コミットの x-linked-etag・x-linked-size が下の SHA256・大きさと
# 一致することを確かめた。
_HF = "https://huggingface.co"
MANIFEST: "tuple[Item, ...]" = (
    Item("llm", "翻訳モデル(Gemma 3 4B)",
         f"{_HF}/ggml-org/gemma-3-4b-it-qat-GGUF/resolve/bbcac0d065076c47042838c0675c602411b0dd4c/"
         "gemma-3-4b-it-qat-Q4_0.gguf",
         "ee91c3e7a4ab95d8c95672f9fcb58bf236b257e9f217966bcf53a5a6df4ab49a", 2526080992,
         "gemma-3-4b-it-qat-Q4_0.gguf", label_en="Translation model (Gemma 3 4B)"),
    Item("whisper", "音声認識(Whisper large-v3-turbo)",
         f"{_HF}/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/"
         "ggml-large-v3-turbo-q5_0.bin",
         "394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2", 574041195,
         "ggml-large-v3-turbo-q5_0.bin", label_en="Speech recognition (Whisper large-v3-turbo)"),
    Item("sensevoice", "速い音声認識(SenseVoice)",
         f"{_GH}/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2",
         "7d1efa2138a65b0b488df37f8b89e3d91a60676e416f515b952358d83dfd347e", 163002883,
         "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17", archive=True,
         label_en="Fast speech recognition (SenseVoice)"),
    Item("kokoro", "読み上げの声(Kokoro)",
         f"{_GH}/tts-models/kokoro-int8-multi-lang-v1_0.tar.bz2",
         "72c347230107ef8a31098ec4d468a605244beef5f1ed00db97efa13518800653", 132353410,
         "kokoro-int8-multi-lang-v1_0", archive=True, label_en="Voices (Kokoro)"),
    Item("speaker", "話者の見分け(CAM++)",
         f"{_GH}/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
         "aa3cfc16963a10586a9393f5035d6d6b57e98d358b347f80c2a30bf4f00ceba2", 28281164,
         "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
         label_en="Speaker recognition (CAM++)"),
    Item("vad", "声の区切り(Silero VAD)",
         f"{_GH}/asr-models/silero_vad.onnx",
         "9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6", 643854,
         "silero_vad.onnx", label_en="Voice activity (Silero VAD)"),
)


def is_done(item: Item, models: Path) -> bool:
    return (Path(models) / item.dest).exists()


def missing(models: Path, items=MANIFEST) -> list:
    return [i for i in items if not is_done(i, models)]


def part_path(item: Item, models: Path) -> Path:
    name = f"{item.key}.tar.bz2.part" if item.archive else f"{item.dest}.part"
    return Path(models) / name


class FetchError(Exception):
    def __init__(self, kind: str, msg: str, msg_en: str):
        super().__init__(msg)
        self.kind, self.msg, self.msg_en = kind, msg, msg_en


class Cancelled(Exception):
    pass


def http_open(url: str, offset: int):
    """(chunks, Range が効いたか, close)。certifi の証明書で HTTPS(同梱の Python 向け)。"""
    import requests
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    r = requests.get(url, headers=headers, stream=True, timeout=(15, 60), allow_redirects=True)
    if r.status_code >= 400:
        r.close()
        raise ConnectionError(f"HTTP {r.status_code}")
    if "text/html" in (r.headers.get("Content-Type") or "").lower():
        # ホテル・駅の Wi-Fi のログイン画面など。.part に書くと後で続きを取っても頭が HTML のまま
        # になり、全部受けてから「壊れていました」になる(11/07 監査)
        r.close()
        raise FetchError("network", "モデルの代わりに Web ページが返りました — Wi-Fi のログインが要るか確かめてください",
                         "Got a web page instead of the model — the Wi-Fi may need you to log in")
    honored = offset == 0 or r.status_code == 206
    return r.iter_content(chunk_size=1 << 20), honored, r.close


def _safe_extract(tar_path: Path, dst: Path) -> None:
    with tarfile.open(tar_path, "r:bz2") as tf:
        root = dst.resolve()
        for m in tf.getmembers():
            target = (dst / m.name).resolve()
            if not (m.isfile() or m.isdir()) or m.name.startswith("/") or \
                    (target != root and root not in target.parents):
                raise FetchError("corrupt", f"展開できない中身がありました({m.name[:40]})",
                                 f"Unsafe archive entry ({m.name[:40]})")
        try:
            tf.extractall(dst, filter="data")
        except TypeError:          # Python 3.10.11 以前(ユーザー本人の venv)。上で確かめ済み
            tf.extractall(dst)


def _take_lock(models: Path):
    """同じ models/ を取る Sokutsu を1つにする(DMG の上と /Applications の2つを開くと、同じ .part
    に2つが書き足して壊していた)。取れなければ FetchError("busy")。プロセスが落ちれば OS が外す。"""
    import fcntl
    f = open(Path(models) / ".fetch.lock", "a+b")   # noqa: SIM115 - run の終わりまで持つ
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        raise FetchError("busy", "別の Sokutsu の窓がモデルを取得しています — そちらを閉じてからもう一度",
                         "Another Sokutsu window is downloading the models — close it and try again")
    return f


class Fetcher:
    """足りないモデルを順に取る。state() を画面が 0.3 秒ごとに読む。"""

    def __init__(self, models_dir: Path, items=MANIFEST, opener: "Callable | None" = None,
                 retries: int = 3, sleep=time.sleep,
                 free_bytes: "Callable[[Path], int] | None" = None, clock=time.monotonic):
        self.dir = Path(models_dir)
        self.items = list(items)
        self.opener = opener or http_open
        self.retries = retries
        self.sleep = sleep
        self.free_bytes = free_bytes or (lambda p: shutil.disk_usage(p).free)
        self.clock = clock
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._thread: "threading.Thread | None" = None
        self._phase = "idle"
        self._err: "FetchError | None" = None
        self._status = {i.key: "wait" for i in self.items}
        self._got = {i.key: 0 for i in self.items}
        self._marks: list = []          # (時刻, 合計バイト) 速さの計算用
        for i in self.items:
            if is_done(i, self.dir):
                self._status[i.key], self._got[i.key] = "done", i.size
            else:
                p = part_path(i, self.dir)
                if p.exists():
                    self._got[i.key] = min(p.stat().st_size, i.size)

    # ---------- 画面向け ----------

    def state(self) -> dict:
        with self._lock:
            got = sum(self._got.values())
            total = sum(i.size for i in self.items)
            speed = 0.0
            if len(self._marks) >= 2:
                (t0, b0), (t1, b1) = self._marks[0], self._marks[-1]
                if t1 > t0:
                    speed = (b1 - b0) / (t1 - t0)
            eta = (total - got) / speed if speed > 0 else None
            return {
                "phase": self._phase, "got": got, "total": total,
                "speed": speed, "eta": eta,
                "error": self._err.msg if self._err else "",
                "errorEn": self._err.msg_en if self._err else "",
                "errorKind": self._err.kind if self._err else "",
                "items": [{"key": i.key, "label": i.label, "labelEn": i.label_en or i.label,
                           "size": i.size, "got": self._got[i.key],
                           "status": self._status[i.key]} for i in self.items],
            }

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._cancel.clear()
            self._thread = threading.Thread(target=self.run, daemon=True, name="modelfetch")
            self._thread.start()

    def cancel(self) -> None:
        self._cancel.set()

    # ---------- 本体 ----------

    def _set(self, key: str, status: str) -> None:
        with self._lock:
            self._status[key] = status

    def _add(self, key: str, n: int) -> None:
        with self._lock:
            self._got[key] += n
            now = self.clock()
            self._marks.append((now, sum(self._got.values())))
            while len(self._marks) > 2 and now - self._marks[0][0] > 5.0:
                self._marks.pop(0)

    def run(self) -> None:
        with self._lock:
            self._phase, self._err = "running", None
            self._marks = []
        lock = None
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            lock = _take_lock(self.dir)
            todo = [i for i in self.items if not is_done(i, self.dir)]
            need = sum(i.size - min(self._got[i.key], i.size) for i in todo)
            need += sum(i.size * 2 for i in todo if i.archive) + GB // 2   # 展開の分と余裕
            free = self.free_bytes(self.dir)
            if todo and free < need:
                raise FetchError("disk", f"空きが足りません(あと {(need - free) / GB:.1f} GB 必要)",
                                 f"Not enough disk space ({(need - free) / GB:.1f} GB more needed)")
            for it in todo:
                self._one(it)
            with self._lock:
                self._phase = "done"
        except Cancelled:
            with self._lock:
                self._phase = "cancelled"
        except FetchError as e:
            with self._lock:
                self._phase, self._err = "error", e
        except Exception as e:  # noqa: BLE001
            with self._lock:
                self._phase = "error"
                self._err = FetchError("other", f"取得できませんでした({type(e).__name__}: {e})",
                                       f"Download failed ({type(e).__name__}: {e})")
        finally:
            if lock is not None:
                lock.close()           # flock も外れる
        for i in self.items:
            if self._status[i.key] in ("down", "verify", "extract"):
                self._set(i.key, "wait")

    def _one(self, it: Item) -> None:
        part = part_path(it, self.dir)
        tries = 0
        while True:
            try:
                self._download(it, part)
                break
            except (Cancelled, FetchError):
                raise
            except OSError as e:
                import errno
                if getattr(e, "errno", None) == errno.ENOSPC:
                    raise FetchError("disk", "空きが足りなくなりました", "Ran out of disk space")
                tries += 1
                if tries > self.retries:
                    raise FetchError("network", f"回線が切れました — {it.label}({e})",
                                     f"Connection lost — {it.label_en or it.label} ({e})")
                self.sleep(min(10.0, 2.0 * tries))
            except Exception as e:  # noqa: BLE001 - requests の例外など
                tries += 1
                if tries > self.retries:
                    raise FetchError("network", f"取得できませんでした — {it.label}({type(e).__name__})",
                                     f"Could not download {it.label_en or it.label} ({type(e).__name__})")
                self.sleep(min(10.0, 2.0 * tries))
        self._verify(it, part)
        self._finish(it, part)
        self._set(it.key, "done")

    def _download(self, it: Item, part: Path) -> None:
        self._set(it.key, "down")
        have = part.stat().st_size if part.exists() else 0
        if have > it.size:
            part.unlink()
            have = 0
        with self._lock:
            self._got[it.key] = have
        if have == it.size:
            return
        chunks, honored, close = self.opener(it.url, have)
        try:
            mode = "ab"
            if have and not honored:          # Range が効かなかった: 最初から
                mode = "wb"
                with self._lock:
                    self._got[it.key] = 0
            with open(part, mode) as f:
                for piece in chunks:
                    if self._cancel.is_set():
                        raise Cancelled()
                    if not piece:
                        continue
                    f.write(piece)
                    self._add(it.key, len(piece))
        finally:
            try:
                close()
            except Exception:  # noqa: BLE001
                pass
        size = part.stat().st_size
        if size < it.size:
            raise ConnectionError(f"途中で終わりました({size}/{it.size})")

    def _verify(self, it: Item, part: Path) -> None:
        self._set(it.key, "verify")
        h = hashlib.sha256()
        with open(part, "rb") as f:
            while True:
                if self._cancel.is_set():
                    raise Cancelled()
                b = f.read(4 << 20)
                if not b:
                    break
                h.update(b)
        if part.stat().st_size != it.size or h.hexdigest() != it.sha256:
            part.unlink(missing_ok=True)
            with self._lock:
                self._got[it.key] = 0
            raise FetchError("corrupt", f"{it.label} が壊れていました — もう一度取ります",
                             f"{it.label_en or it.label} was corrupted — please retry")

    def _finish(self, it: Item, part: Path) -> None:
        final = self.dir / it.dest
        if not it.archive:
            part.replace(final)
            return
        self._set(it.key, "extract")
        tmp = self.dir / f".x-{it.key}"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        try:
            _safe_extract(part, tmp)
            src = tmp / it.dest
            if not src.is_dir():
                raise FetchError("corrupt", f"{it.label} の中身が想定と違いました",
                                 f"Unexpected contents in {it.label_en or it.label}")
            src.replace(final)
        except FetchError:
            part.unlink(missing_ok=True)
            with self._lock:
                self._got[it.key] = 0
            raise
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        part.unlink(missing_ok=True)
