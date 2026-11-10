"""声を覚える(W4 11/07): 画面で名前を付けた話者の声を覚え、次の会議から同じ声に自動で名前を付ける。

しくみ:
  - 会議の後の付け直し(speakers.refine)のたびに、話者ごとの声の重心(CAM++ の特徴を区間の長さで
    重み付けした平均、正規化)を `session-….voices.npz` に残す(centroids)。
  - 利用者が話者に名前を付けると(Bridge.speaker_rename)、その話者の重心を声の帳面
    (logs/voices/book.json + book.npz、git に入れない)に「その名前の人の見本」として足す。
    名前を直せば見本はその名前へ移り、名前を消せば見本も消える。1人の見本は MAX_SAMPLES 個まで
    (マイク・部屋の違いに強くするため平均せずに見本のまま持ち、照合はいちばん近い見本で測る)。
    自動で付いた名前からは覚えない(間違いが自分を強めないように)。利用者が「合っている」と押した時だけ。
  - 次の会議では、話者の重心と帳面の全員の見本を比べ、次の条件を全部満たす時だけ名前を付ける
    (照合のしきい値は scripts/eval_voicebook.py の評価で決めた。数字はその docstring と
    eval/voicebook/ にある):
      * その話者が MIN_MATCH_S 秒・MIN_MATCH_N 区間以上話している(短い声の重心はぶれる)
      * いちばん近い人との類似が THRESHOLD 以上
      * 2番目に近い人より MARGIN 以上近い(似た声の2人のどちらか決められない時は付けない)
      * 1つの会議で同じ名前は1人だけ(いちばん近い話者に)
      * 同じ会議から覚えた見本は使わない
  - 付けた名前は「声から」の印つきで画面に出し、利用者は取り消せる(取り消した組は同じ会議で
    もう付けない。session-….voicematch.json に残す)。
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path

import numpy as np

# 照合の条件(scripts/eval_voicebook.py で決めた。変えるときは評価を回し直す。eval/voicebook/README.md)
# 11/07: 帳面にいない人(stranger)のいちばん近い別人との類似の最大は 合成会議(会場の雑音つき)0.639、
# AMI 0.422、声の束(静かな読み上げ・同じ言語と性別)0.742。同じ TTS エンジンの別の声は 0.70〜0.72。
# 0.60 だと合成会議の stranger の 6.8%(9/133)に違う名前が付いた。0.75 は見たどの別人よりも上:
# 間違った名前 0/133(合成)・0/12(AMI)、正しく付く割合 78%(合成)・60%(AMI、前半の口元マイクで覚えて
# 後半の卓上マイクで照合)。付かなければ名前を打つだけ(取り返せる)なので、間違いの少なさを取った。
THRESHOLD = 0.75
MARGIN = 0.05
MIN_MATCH_S = 20.0
MIN_MATCH_N = 6
MIN_ENROLL_S = 8.0
MAX_SAMPLES = 6
_lock = threading.RLock()


def centroids(embs: list, durs: list, labels: list, min_s: float = 0.6,
              max_weight_s: float = 6.0) -> dict:
    """区間ごとの特徴・長さ・話者 → {話者: (正規化した重心, 話した秒, 区間の数)}。

    重みは区間の長さ(上限 max_weight_s、speakers.refine_labels と同じ)。特徴の無い区間・
    min_s 未満の区間は使わない。"""
    acc: dict = {}
    for e, d, l in zip(embs, durs, labels):
        if l is None or e is None or d < min_s:
            continue
        w = min(float(d), max_weight_s)
        v = np.asarray(e, np.float32)
        s = acc.setdefault(l, [np.zeros_like(v), 0.0, 0])
        s[0] = s[0] + w * v
        s[1] += float(d)
        s[2] += 1
    out = {}
    for l, (v, secs, n) in acc.items():
        nv = float(np.linalg.norm(v))
        if nv > 0:
            out[l] = ((v / nv).astype(np.float32), round(secs, 2), n)
    return out


# ---------------- 会議ごとのファイル ----------------

def voices_file(jsonl: "str | Path") -> Path:
    return Path(f"{Path(jsonl).with_suffix('')}.voices.npz")


def matches_file(jsonl: "str | Path") -> Path:
    return Path(f"{Path(jsonl).with_suffix('')}.voicematch.json")


def save_meeting_voices(jsonl: "str | Path", cents: dict) -> None:
    if not cents:
        return
    ids = sorted(cents)
    f = voices_file(jsonl)
    tmp = f.with_name(f.name[:-4] + ".tmp.npz")
    np.savez(tmp, ids=np.array(ids), vecs=np.stack([np.asarray(cents[k][0], np.float32) for k in ids]),
             secs=np.array([float(cents[k][1]) for k in ids], np.float32),
             nseg=np.array([int(cents[k][2]) for k in ids], np.int32))
    tmp.replace(f)


def load_meeting_voices(jsonl: "str | Path") -> dict:
    f = voices_file(jsonl)
    if not f.exists():
        return {}
    try:
        with np.load(f) as z:
            return {str(k): (np.asarray(v, np.float32), float(s), int(n))
                    for k, v, s, n in zip(z["ids"], z["vecs"], z["secs"], z["nseg"])}
    except Exception:  # noqa: BLE001  (壊れた・書きかけのファイル)
        return {}


def load_matches(jsonl: "str | Path") -> dict:
    f = matches_file(jsonl)
    try:
        d = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
        return d if isinstance(d, dict) else {}
    except (ValueError, OSError):
        return {}


def save_matches(jsonl: "str | Path", m: dict) -> None:
    f = matches_file(jsonl)
    if not m:
        if f.exists():
            f.unlink()
        return
    f.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------- 声の帳面 ----------------

class VoiceBook:
    """覚えた人と声の見本。logs/voices/ に置く(git に入れない)。スレッド安全。"""

    def __init__(self, root: "str | Path"):
        self.root = Path(root)
        self._persons: list = []
        self._vecs: dict = {}
        self._noticed = False     # 初めて覚えた時の知らせ(この Mac だけに保存・いつでも消せる)を見せたか
        self._mtime = None
        self._load()

    # ---------- 読み書き ----------

    def _paths(self):
        return self.root / "book.json", self.root / "book.npz"

    def _sig(self):
        try:
            return self._paths()[0].stat().st_mtime_ns
        except OSError:
            return None

    def _load(self) -> None:
        jf, nf = self._paths()
        persons, vecs, noticed = [], {}, False
        try:
            if jf.exists():
                d = json.loads(jf.read_text(encoding="utf-8"))
                persons, noticed = d.get("persons", []), bool(d.get("noticed"))
            if nf.exists():
                with np.load(nf) as z:
                    vecs = {k: np.asarray(z[k], np.float32) for k in z.files}
        except Exception:  # noqa: BLE001
            persons, vecs = [], {}
        for p in persons:
            p["samples"] = [s for s in p.get("samples", []) if s.get("k") in vecs]
        self._persons = [p for p in persons if p["samples"]]
        self._vecs = vecs
        self._noticed = noticed
        self._mtime = self._sig()

    def _fresh(self) -> None:
        if self._sig() != self._mtime:      # 別の画面・別の Bridge が書いた
            self._load()

    def _save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        jf, nf = self._paths()
        keep = {s["k"] for p in self._persons for s in p["samples"]}
        tmpn = nf.with_name("book.tmp.npz")
        np.savez(tmpn, **{k: v for k, v in self._vecs.items() if k in keep})
        tmpn.replace(nf)
        tmpj = jf.with_name("book.tmp.json")
        tmpj.write_text(json.dumps({"version": 1, "noticed": self._noticed, "persons": self._persons},
                                   ensure_ascii=False, indent=1),
                        encoding="utf-8")
        tmpj.replace(jf)
        self._vecs = {k: v for k, v in self._vecs.items() if k in keep}
        self._mtime = self._sig()

    # ---------- 覚える・忘れる ----------

    def _drop_src(self, src: str) -> None:
        for p in self._persons:
            p["samples"] = [s for s in p["samples"] if s["src"] != src]
        self._persons = [p for p in self._persons if p["samples"]]

    def enroll(self, name: str, vec, secs: float, nseg: int, src: str) -> "dict | None":
        """name の人の見本として声を足す(src = "会議id:話者"。同じ src の古い見本は置き換える)。
        短すぎる声は覚えない(None)。"""
        name = (name or "").strip()
        if not name or secs < MIN_ENROLL_S or vec is None:
            return None
        v = np.asarray(vec, np.float32)
        v = v / (float(np.linalg.norm(v)) + 1e-9)
        with _lock:
            self._fresh()
            self._drop_src(src)
            p = next((x for x in self._persons if x["name"].casefold() == name.casefold()), None)
            if p is None:
                p = {"id": uuid.uuid4().hex[:12], "name": name, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "samples": []}
                self._persons.append(p)
            k = uuid.uuid4().hex
            self._vecs[k] = v
            p["samples"].append({"k": k, "src": src, "secs": round(float(secs), 1), "n": int(nseg),
                                 "added": time.strftime("%Y-%m-%d %H:%M:%S")})
            if len(p["samples"]) > MAX_SAMPLES:   # 短い見本から捨てる(最新の1つは残す)
                last = p["samples"][-1]
                rest = sorted(p["samples"][:-1], key=lambda s: -s["secs"])[:MAX_SAMPLES - 1]
                p["samples"] = rest + [last]
            self._save()
            return self._public(p)

    def unenroll(self, src: str) -> None:
        with _lock:
            self._fresh()
            self._drop_src(src)
            self._save()

    def forget(self, pid: str) -> bool:
        with _lock:
            self._fresh()
            n = len(self._persons)
            self._persons = [p for p in self._persons if p["id"] != pid]
            self._save()
            return len(self._persons) < n

    def forget_all(self) -> int:
        """覚えた声を全部消す(帳面のファイルごと)。消した人数を返す。知らせも初めからに戻す。"""
        with _lock:
            self._fresh()
            n = len(self._persons)
            self._persons, self._vecs, self._noticed = [], {}, False
            for f in list(self.root.glob("book*")) if self.root.exists() else []:
                f.unlink()
            self._mtime = self._sig()
            return n

    def noticed(self) -> bool:
        with _lock:
            self._fresh()
            return self._noticed

    def set_noticed(self) -> None:
        with _lock:
            self._fresh()
            self._noticed = True
            self._save()

    def rename(self, pid: str, name: str) -> bool:
        name = (name or "").strip()
        with _lock:
            self._fresh()
            for p in self._persons:
                if p["id"] == pid and name:
                    p["name"] = name
                    self._save()
                    return True
        return False

    def _public(self, p: dict) -> dict:
        return {"id": p["id"], "name": p["name"], "created": p.get("created", ""),
                "samples": len(p["samples"]), "secs": round(sum(s["secs"] for s in p["samples"]), 1),
                "meetings": sorted({s["src"].split(":")[0] for s in p["samples"]})}

    def persons(self) -> list:
        with _lock:
            self._fresh()
            return sorted((self._public(p) for p in self._persons), key=lambda x: x["name"].casefold())

    # ---------- 照合 ----------

    def scores(self, vec, exclude_mid: str = "") -> list:
        """声 → [(類似, 人)](近い順)。人ごとにいちばん近い見本で測る。"""
        v = np.asarray(vec, np.float32)
        v = v / (float(np.linalg.norm(v)) + 1e-9)
        out = []
        for p in self._persons:
            sims = [float(self._vecs[s["k"]] @ v) for s in p["samples"]
                    if s["k"] in self._vecs and s["src"].split(":")[0] != exclude_mid]
            if sims:
                out.append((max(sims), p))
        return sorted(out, key=lambda x: -x[0])

    def match(self, clusters: dict, exclude_mid: str = "", skip: "dict | None" = None) -> dict:
        """会議の話者の重心 {話者: (重心, 秒, 区間数)} → {話者: {name, person, sim, margin}}。

        skip: {話者: [付けない人の id]}(利用者が取り消した組)。"""
        skip = skip or {}
        with _lock:
            self._fresh()
            cand = []
            for spk, (vec, secs, n) in clusters.items():
                if secs < MIN_MATCH_S or n < MIN_MATCH_N or vec is None:
                    continue
                sc = [(s, p) for s, p in self.scores(vec, exclude_mid) if p["id"] not in skip.get(spk, [])]
                if not sc or sc[0][0] < THRESHOLD:
                    continue
                second = sc[1][0] if len(sc) > 1 else -1.0
                if sc[0][0] - second < MARGIN:
                    continue
                cand.append((sc[0][0], spk, sc[0][1], sc[0][0] - second))
        out: dict = {}
        used: set = set()
        for sim, spk, p, mg in sorted(cand, key=lambda x: -x[0]):
            if p["id"] in used or spk in out:
                continue
            used.add(p["id"])
            out[spk] = {"name": p["name"], "person": p["id"], "sim": round(sim, 3),
                        "margin": round(min(mg, 1.0), 3)}
        return out
