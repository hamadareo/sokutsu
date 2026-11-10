"""話者の見分け(会議メモ 11/06): 1区間の声 → 話者1/話者2/…(このMacの CPU で動く)。

方式: 声の特徴(speaker embedding, sherpa-onnx の SpeakerEmbeddingExtractor)を区間ごとに
取り、会話の中でオンラインにまとめ分ける(OnlineClusters)。
  - 今までの話者の重心(特徴の平均)とのコサイン類似が threshold 以上 → いちばん近い話者
    (重心を更新。長い区間ほど重く、1区間の重みには上限 max_weight_s を置く)。
  - どの話者とも似ていない区間は、new_min_s 秒以上あるときだけ新しい話者の候補にし、
    候補に confirm 区間(候補どうしも threshold 以上で似ているもの)が集まったら話者にする。
    短い区間や雑音まじりの1区間から話者を作ると「幽霊の話者」が増えるため(候補のうちは None)。
  - min_s 秒未満の区間は判断しない(None)。似ていない短い区間も None。
  - 話者が max_speakers 人に達したら、似ていなくても(new_min_s 秒以上なら)一番近い話者に付ける。
モデル・しきい値などの既定値は scripts/eval_speakers.py(話者の分かるコーパス JVNV/LibriSpeech/
AISHELL-1/Zeroth-Korean から作った合成会議 56本、静か/会場、2〜4人、日英中韓)で決めた(11/06):
  モデル 3D-Speaker CAM++(zh+en, common advanced, 28MB): 声の照合の EER 全体 1.3%
  (ja 7.5 / en 1.9 / zh 0.1 / ko 0.3%)。ERes2Net-base は同程度で2倍遅く、TitaNet-small は
  まとめ分けで1〜2pt 下、VoxCeleb の CAM++ 2種は日中韓英の混ざる声でほぼ当てずっぽう(EER 40%台)。
  threshold 0.3 / update_th 0.5 / merge_th 0.7 / new_min_s 1.2 / min_s 0.6:
  合成会議の区間ごとの正解率 静か2人 95.4% / 静か3〜4人 90.0% / 会場2人 90.8% / 会場3〜4人 80.1%。
  本物の4人の英語会議 AMI(ES2004a/TS3003a/IS1009a/EN2002a、手作業の書き起こしが正解):
  口元マイクを混ぜた音 91.5% / 卓上マイク1本(このアプリの使い方に近い) 87.7%。
  0.35 は合成の静か3〜4人で +2pt だが、AMI 卓上 −1.8pt・会場 2人 −1.9pt なので 0.3 にした。
  弱いのは同じ言語の同性が3〜4人(JVNV ja4)と、卓上マイクで声の小さい人(AMI ES2004a 75%)。
  速さ: 特徴1回 平均27ms、10秒で約80ms(M3, 2スレッド)。

アプリへの組み込みは __main__.py 側(通訳の道の外のスレッドで呼ぶ)。モデルが無ければ
find_model が None を返し、アプリは話者なしで動く。
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

# 使うモデル(上から順に探す)。scripts/get_models.sh で models/ に置く。
MODEL_NAMES = (
    "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
)
SR = 16000


def find_model(models_dir) -> str | None:
    """models_dir(と models_dir/speaker)から話者モデルを探す。無ければ None。"""
    d = Path(models_dir)
    for sub in (d, d / "speaker"):
        for name in MODEL_NAMES:
            p = sub / name
            if p.is_file():
                return str(p)
    return None


def _resample(x: np.ndarray, sr_in: int, sr_out: int = SR) -> np.ndarray:
    if sr_in == sr_out or len(x) == 0:
        return x
    n_out = int(round(len(x) * sr_out / sr_in))
    return np.interp(np.arange(n_out) / sr_out, np.arange(len(x)) / sr_in, x).astype(np.float32)


def refine_labels(embs: list, durs: list, labels: list, min_frac: float = 0.05, iters: int = 5,
                  min_s: float = 0.6, max_weight_s: float = 6.0) -> list:
    """会議の後の付け直し: 話した時間が全体の min_frac 以上の話者だけを残し、各区間を一番近い
    話者の重心へ付け直して重心を作り直す(k-means を iters 回)。オンラインでは決められなかった
    区間(None)にも、特徴があれば話者を付ける。embs[i] は正規化した特徴か None。

    11/06 A/B(AMI 4会議×2マイク + 合成会議56本): 区間の正解率の平均 89.4 → 92.1%
    (AMI 卓上マイク 87.7 → 93.8、会場2人 90.8 → 94.2、静か2人 96.5 → 97.3)。
    小さい話者を近い話者へ合わせるだけ(区間は付け直さない)だと 90.3% 止まり。
    """
    tot: dict = {}
    for l, d in zip(labels, durs):
        if l is not None:
            tot[l] = tot.get(l, 0.0) + d
    if not tot:
        return list(labels)
    total = sum(tot.values())
    major = [l for l, v in tot.items() if v >= min_frac * total] or [max(tot, key=tot.get)]
    E = [e if e is not None and d >= min_s else None for e, d in zip(embs, durs)]

    def centroid(k, labs):
        v = None
        for e, d, l in zip(E, durs, labs):
            if l == k and e is not None:
                v = min(d, max_weight_s) * e if v is None else v + min(d, max_weight_s) * e
        return None if v is None else v / (np.linalg.norm(v) + 1e-9)

    cents = {k: c for k in major if (c := centroid(k, labels)) is not None}
    if not cents:
        return list(labels)
    out = list(labels)
    for _ in range(max(1, iters)):
        ks = list(cents)
        C = np.stack([cents[k] for k in ks])
        out = [ks[int(np.argmax(C @ e))] if e is not None else (l if l in cents else None)
               for e, l in zip(E, labels)]
        for k in ks:
            c = centroid(k, out)
            if c is not None:
                cents[k] = c
    return out


class OnlineClusters:
    """正規化した声の特徴と区間の長さ → 話者 id(オンラインのまとめ分け)。スレッド安全ではない。

    update_th: 重心を更新するのは類似がこれ以上のときだけ(既定 = threshold)。付けるのは
      threshold 以上、重心を育てるのはもっと確かなときだけ、にすると雑音まじりの区間で
      重心が濁らない。
    merge_th: 2人の重心がこれ以上似てきたら同じ人とみなして合わせる(後から作った方を
      先の方へ。aliases に {"S3": "S1"} のように残る。以後の assign は合わせた先の id を返す)。
      None = 合わせない。
    absorb_frac / absorb_after: 作ってから absorb_after 区間たっても、話した量(重み)が全体の
      absorb_frac 未満の話者は、一番近い話者へ合わせる(aliases に残る)。None = しない。
    prov_n / prov_s: 新しい話者は、区間が prov_n 個かつ prov_s 秒たまるまで「仮」。仮のあいだは
      一番近い確かな話者の id を返す(仮の id は表に出ない)。0 = しない。
    """

    def __init__(self, threshold: float = 0.5, max_speakers: int = 8, min_s: float = 0.6,
                 new_min_s: float = 1.2, max_weight_s: float = 6.0, confirm: int = 1,
                 max_candidates: int = 4, update_th: float | None = None,
                 merge_th: float | None = None, absorb_frac: float | None = None,
                 absorb_after: int = 30, prov_n: int = 0, prov_s: float = 0.0,
                 absorb_sim: float | None = None):
        self.threshold = threshold
        self.max_speakers = max_speakers
        self.min_s = min_s
        self.new_min_s = new_min_s
        self.max_weight_s = max_weight_s
        self.confirm = max(1, confirm)
        self.max_candidates = max_candidates
        self.update_th = update_th
        self.merge_th = merge_th
        self.absorb_frac = absorb_frac
        self.absorb_after = absorb_after
        self.absorb_sim = absorb_sim       # 合わせ先との類似がこれ未満なら合わせない(本当に別の人)
        self.prov_n = prov_n
        self.prov_s = prov_s
        self._sums: list[np.ndarray] = []    # 話者ごとの「重み × 正規化した特徴」の和
        self._ids: list[str] = []            # その話者の id(合わせても番号は使い回さない)
        self._w: list[float] = []            # 話者ごとの付けた重み(秒)の和
        self._nseg: list[int] = []           # 話者ごとの付けた区間の数
        self._born: list[int] = []           # 作ったときの assign の通し番号
        self._prov: list[bool] = []          # 仮の話者か
        self._n = 0
        self._next = 1
        self._cands: list[list] = []         # 話者の候補 [和, 区間の数](confirm > 1 のとき)
        self.aliases: dict[str, str] = {}    # 合わせた話者 → 合わせ先

    def __len__(self) -> int:
        return len(self._sums)

    def reset(self) -> None:
        for lst in (self._sums, self._ids, self._w, self._nseg, self._born, self._prov,
                    self._cands):
            lst.clear()
        self.aliases.clear()
        self._n = 0
        self._next = 1

    def sims(self, e: np.ndarray) -> dict:
        """状態を変えずに、確かな話者ごとの類似(コサイン){id: 類似}。"""
        if e is None or not self._sums:
            return {}
        e = np.asarray(e, dtype=np.float32)
        e = e / (float(np.linalg.norm(e)) + 1e-9)
        return {self._ids[i]: float((sm / (np.linalg.norm(sm) + 1e-9)) @ e)
                for i, sm in enumerate(self._sums) if not (self._prov and self._prov[i])}

    def nearest(self, e: np.ndarray) -> "tuple[str, float] | None":
        """状態を変えずに、一番近い確かな話者の id と類似(コサイン)。話者がいなければ None。"""
        sm = self.sims(e)
        if not sm:
            return None
        best = max(sm, key=sm.get)
        return self.canonical(best), sm[best]

    def canonical(self, spk: str | None) -> str | None:
        """過去に返した id → 今の id(合わせた話者をたどる)。"""
        seen = set()
        while spk in self.aliases and spk not in seen:
            seen.add(spk)
            spk = self.aliases[spk]
        return spk

    def _new(self, s: np.ndarray, w: float = 0.0) -> str:
        sid = f"S{self._next}"
        self._next += 1
        self._sums.append(s)
        self._ids.append(sid)
        self._w.append(w)
        self._nseg.append(1)
        self._born.append(self._n)
        self._prov.append(self.prov_n > 0 and any(not p for p in self._prov))
        return self._shown(len(self._sums) - 1)

    def _drop(self, j: int, into: int) -> None:
        """話者 j を話者 into へ合わせる(aliases に残す)。"""
        self._sums[into] = self._sums[into] + self._sums[j]
        self._w[into] += self._w[j]
        self._nseg[into] += self._nseg[j]
        self._prov[into] = self._prov[into] and self._prov[j]
        if not self._prov[j]:
            self.aliases[self._ids[j]] = self._ids[into]
        for lst in (self._sums, self._ids, self._w, self._nseg, self._born, self._prov):
            del lst[j]

    def _shown(self, i: int) -> str:
        """表に出す id: 仮の話者なら一番近い確かな話者の id。"""
        if not self._prov[i]:
            return self._ids[i]
        firm = [k for k in range(len(self._sums)) if not self._prov[k]]
        if not firm:
            return self._ids[i]
        c = self._sums[i] / (np.linalg.norm(self._sums[i]) + 1e-9)
        best = max(firm, key=lambda k: float(c @ (self._sums[k] / (np.linalg.norm(self._sums[k]) + 1e-9))))
        return self._ids[best]

    def _upkeep(self) -> None:
        # 仮の話者が育ったら確かな話者に
        for i in range(len(self._sums)):
            if self._prov[i] and self._nseg[i] >= self.prov_n and self._w[i] >= self.prov_s:
                self._prov[i] = False
        # 育たない小さい話者は近い話者へ合わせる
        if self.absorb_frac is None or len(self._sums) < 2:
            return
        tot = sum(self._w)
        for j in sorted(range(len(self._sums)), key=lambda k: self._w[k]):
            if j >= len(self._sums) or len(self._sums) < 2:
                break
            if self._n - self._born[j] < self.absorb_after or self._w[j] >= self.absorb_frac * tot:
                continue
            c = self._sums[j] / (np.linalg.norm(self._sums[j]) + 1e-9)
            others = [k for k in range(len(self._sums)) if k != j]
            sim = {k: float(c @ (self._sums[k] / (np.linalg.norm(self._sums[k]) + 1e-9))) for k in others}
            into = max(others, key=sim.get)
            if self.absorb_sim is not None and sim[into] < self.absorb_sim:
                continue    # 誰とも似ていない小さい話者 = 発言の少ない別の人。合わせない
            self._drop(j, into)
            return          # 1回に1人(番号がずれるため)

    def _merge(self) -> None:
        if self.merge_th is None or len(self._sums) < 2:
            return
        while len(self._sums) >= 2:
            c = np.stack([s / (np.linalg.norm(s) + 1e-9) for s in self._sums])
            m = c @ c.T
            np.fill_diagonal(m, -1.0)
            i, j = np.unravel_index(int(np.argmax(m)), m.shape)
            if m[i, j] < self.merge_th:
                return
            i, j = min(i, j), max(i, j)
            self._drop(j, i)

    def assign(self, e: np.ndarray, dur: float) -> str | None:
        if e is None or dur < self.min_s:
            return None
        self._n += 1
        out = self._assign(e, dur)
        if out is not None and (self.absorb_frac is not None or self.prov_n > 0):
            self._upkeep()
            out = self.canonical(out)
        return out

    def _assign(self, e: np.ndarray, dur: float) -> str | None:
        e = np.asarray(e, dtype=np.float32)
        e = e / (float(np.linalg.norm(e)) + 1e-9)
        w = min(dur, self.max_weight_s)
        if self._sums:
            cents = np.stack([s / (np.linalg.norm(s) + 1e-9) for s in self._sums])
            sims = cents @ e
            best = int(np.argmax(sims))
            full = len(self._sums) >= self.max_speakers
            if sims[best] >= self.threshold or (full and dur >= self.new_min_s):
                shown = self._shown(best)
                self._nseg[best] += 1
                self._w[best] += w       # 話した量は付けた区間すべてで数える(重心の更新とは別)
                if self.update_th is None or sims[best] >= self.update_th:
                    self._sums[best] += w * e
                    self._merge()
                return self.canonical(shown)
            if full:
                return None
        if dur < self.new_min_s:
            return None          # 短い区間からは新しい話者を作らない
        if self.confirm > 1:
            # 候補: 似た区間が confirm 個そろうまでは話者にしない(None)
            bi, best_s = -1, -1.0
            for i, c in enumerate(self._cands):
                sim = float(np.dot(c[0] / (np.linalg.norm(c[0]) + 1e-9), e))
                if sim > best_s:
                    bi, best_s = i, sim
            if bi >= 0 and best_s >= self.threshold:
                best_c = self._cands[bi]
                best_c[0] = best_c[0] + w * e
                best_c[1] += 1
                if best_c[1] >= self.confirm:
                    del self._cands[bi]
                    return self._new(best_c[0], float(np.linalg.norm(best_c[0])))
                return None
            self._cands.append([w * e, 1])
            if len(self._cands) > self.max_candidates:
                self._cands.pop(0)           # 古い候補(続かなかった雑音など)は忘れる
            return None
        return self._new(w * e, w)


class SpeakerTracker:
    """1区間の声 → 話者 id。assign はスレッド安全。

    merge_th で2人を合わせたとき、以後の assign は合わせ先の id を返す。過去に返した id を
    直すには canonical(spk) か aliases({"S3": "S1"})を使う(会議メモ・記録の表示用)。

    話者が増えすぎる対策(11/06、AMI 4会議×2マイク + 合成会議56本で A/B):
    作ってから30区間たっても話した量が全体の3%未満の話者は、類似0.2以上の一番近い話者へ合わせる
    (absorb_frac 0.03 / absorb_after 30 / absorb_sim 0.2。aliases に残るので過去の行も直せる)。
    AMI で余計なまとまり(全部で50個中21個が5%未満)を調べると、多くは相づち・重なり・短い区間
    (作った区間の中央 1.5秒、本物の話者は 2.1秒)から生まれた「同じ人の2つ目」だった。
    人数の差(5%未満も含む全部)の平均 1.02 → 0.56、人数が当たった会話 29/64 → 36/64、
    区間の正解率(合わせた後)89.4 → 90.0%(どの組も下がらない)、会話中の表示 89.1 → 89.1%。
    15区間で合わせると人数は 0.42 まで減るが、遠い声を合わせて作り直しを繰り返し、会話中の
    表示が AMI で 91.5 → 70.6% に落ちた(類似の下限と30区間の待ちはその対策)。
    不採用: 仮の話者(育つまで近い話者の id を出す)は正解率 −6pt、merge_th 0.5 は −4.5pt、
    new_min_s 2.0 は人数 0.47 だが正解率 −0.4pt(合成の静か2人 −3.8pt)。
    """

    def __init__(self, model_path: str, threshold: float = 0.3, max_speakers: int = 8,
                 min_s: float = 0.6, new_min_s: float = 1.2, max_weight_s: float = 6.0,
                 confirm: int = 1, update_th: float | None = 0.5,
                 merge_th: float | None = 0.7, absorb_frac: float | None = 0.03,
                 absorb_after: int = 30, absorb_sim: float | None = 0.2,
                 max_embed_s: float = 10.0, num_threads: int = 2):
        import sherpa_onnx
        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(model_path), num_threads=num_threads, debug=False, provider="cpu")
        if not cfg.validate():
            raise ValueError(f"話者モデルを読めません: {model_path}")
        self._ext = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
        self._cl = OnlineClusters(threshold, max_speakers, min_s, new_min_s, max_weight_s, confirm,
                                  update_th=update_th, merge_th=merge_th,
                                  absorb_frac=absorb_frac, absorb_after=absorb_after,
                                  absorb_sim=absorb_sim)
        self._lock = threading.Lock()
        # 特徴を取る長さの上限(中央を使う)。CAM++ 2スレッドで 10秒 ≈ 80ms(M3)
        self.max_embed_s = max_embed_s
        self._hist: list = []    # assign ごとの [特徴 or None, 秒, 返した id](会議の後の付け直し用)

    @property
    def threshold(self) -> float:
        return self._cl.threshold

    @threshold.setter
    def threshold(self, v: float) -> None:
        self._cl.threshold = v

    @property
    def n_speakers(self) -> int:
        return len(self._cl)

    def assign(self, samples: np.ndarray, sr: int = SR) -> str | None:
        """1区間の声 → "S1"/"S2"/…(短すぎ・判断できなければ None)。スレッド安全。"""
        return self.assign_indexed(samples, sr)[1]

    def assign_indexed(self, samples: np.ndarray, sr: int = SR) -> tuple[int, str | None]:
        """assign と同じ。(この会話で何番目の assign か, id) を返す。番号は refine() の並びと同じ。"""
        x = np.asarray(samples)
        if x.dtype == np.int16:
            x = x.astype(np.float32) / 32768.0
        x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
        dur = len(x) / float(sr)
        e = None
        if dur >= self._cl.min_s:
            if sr != SR:
                x = _resample(x, sr, SR)
            with self._lock:
                e = self.embed(x)
        with self._lock:
            spk = self._cl.assign(e, dur) if e is not None else None
            self._hist.append([e, dur, spk])
            return len(self._hist) - 1, spk

    def guess(self, samples: np.ndarray, sr: int = SR, min_s: float = 0.25) -> "tuple[str, float] | None":
        """状態を変えない話者の見当: (一番近い話者の id, 類似)。assign と違い短い区間(0.6秒未満)も
        見るが、話者は作らず重心も動かさず記録にも残さない。特徴が取れない・話者がまだいない → None。

        使い道(11/07 擬似会議 v2): Endpointer の「話者が違えば繋がない」で、短い区間の話者が
        None のとき。判定は different_speaker() を使うのがよい(id が違うだけで別とみると、同じ人の
        短い区間の13%を別の人と言ってしまう)。"""
        x = np.asarray(samples)
        if x.dtype == np.int16:
            x = x.astype(np.float32) / 32768.0
        x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
        if len(x) / float(sr) < min_s:
            return None
        if sr != SR:
            x = _resample(x, sr, SR)
        with self._lock:
            e = self.embed(x)
            return self._cl.nearest(e) if e is not None else None

    def different_speaker(self, samples: np.ndarray, spk: str | None, sr: int = SR,
                          margin: float = 0.1, min_s: float = 0.25) -> "bool | None":
        """この区間は話者 spk とは別の人か(状態は変えない)。True = 別の人、False = 同じか分からない、
        None = 判断できない(特徴が取れない・spk が知らない話者)。

        規則: spk 以外のいちばん近い話者の方が、spk より margin 以上近いとき「別の人」。
        11/07 `scripts/eval_speakers.py guess`(AMI 4会議×2マイク + 擬似会議4本、1.2秒未満の区間が
        話者の付いた直前の区間から1.2秒以内に続いた 606組、正解は書き起こし):
          margin 0.1(既定): 別の人を別と言えた 71% / 同じ人を別と言ってしまった 6%
                            (AMI 口元 74/5、AMI 卓上 66/7、擬似会議 80/0)
          margin 0.0: 90% / 12%   margin 0.2: 48% / 5%
        合成会議56本を足した 1150組でも margin 0.1 で 78% / 8%。他の規則: id が違うだけ(= margin 0)
        90% / 13%、id が違い類似 0.3 以上 72% / 9%、spk との類似 0.2 未満 62% / 8%。"""
        x = np.asarray(samples)
        if x.dtype == np.int16:
            x = x.astype(np.float32) / 32768.0
        x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
        if not spk or len(x) / float(sr) < min_s:
            return None
        if sr != SR:
            x = _resample(x, sr, SR)
        with self._lock:
            e = self.embed(x)
            if e is None:
                return None
            sm = {self._cl.canonical(k): v for k, v in self._cl.sims(e).items()}
            me = sm.get(self._cl.canonical(spk))
        if me is None:
            return None
        others = [v for k, v in sm.items() if k != self._cl.canonical(spk)]
        return bool(others) and max(others) - me >= margin

    def refine(self, min_frac: float = 0.05, iters: int = 5) -> list:
        """会議の後(または会議中にときどき)の付け直し: 今までの assign の番号順に、付け直した id。
        話した時間が min_frac 未満の話者は消え、その区間は近い話者に付く。None だった区間にも付く。
        オンラインの状態(以後の assign)は変えない。refine_labels を参照。"""
        with self._lock:
            embs = [h[0] for h in self._hist]
            durs = [h[1] for h in self._hist]
            labs = [self._cl.canonical(h[2]) for h in self._hist]
            min_s, mw = self._cl.min_s, self._cl.max_weight_s
        return refine_labels(embs, durs, labs, min_frac=min_frac, iters=iters,
                             min_s=min_s, max_weight_s=mw)

    def centroids(self, labels: list | None = None) -> dict:
        """話者ごとの声の重心 {id: (正規化した重心, 話した秒, 区間数)}(声を覚える W4 11/07)。
        labels は refine() の結果(assign の番号順)。None なら今の id(合わせた先)で数える。"""
        from .voicebook import centroids as _centroids
        with self._lock:
            embs = [h[0] for h in self._hist]
            durs = [h[1] for h in self._hist]
            labs = list(labels) if labels is not None else [self._cl.canonical(h[2]) for h in self._hist]
            min_s, mw = self._cl.min_s, self._cl.max_weight_s
        labs = (labs + [None] * len(embs))[:len(embs)]
        return _centroids(embs, durs, labs, min_s=min_s, max_weight_s=mw)

    def reset(self) -> None:
        with self._lock:
            self._cl.reset()
            self._hist.clear()

    def canonical(self, spk: str | None) -> str | None:
        """過去に返した id → 今の id(後から同じ人と分かって合わせた話者を直す)。"""
        with self._lock:
            return self._cl.canonical(spk)

    @property
    def aliases(self) -> dict[str, str]:
        with self._lock:
            return dict(self._cl.aliases)

    def embed(self, x: np.ndarray) -> np.ndarray | None:
        """16kHz の声 → 正規化した特徴(評価からも使う)。長すぎる区間は中央 max_embed_s 秒だけ使う。"""
        n = int(self.max_embed_s * SR)
        if self.max_embed_s > 0 and len(x) > n:
            a = (len(x) - n) // 2
            x = x[a:a + n]
        st = self._ext.create_stream()
        st.accept_waveform(sample_rate=SR, waveform=np.ascontiguousarray(x, dtype=np.float32))
        st.input_finished()
        if not self._ext.is_ready(st):
            return None
        e = np.asarray(self._ext.compute(st), dtype=np.float32)
        n = float(np.linalg.norm(e))
        return e / n if n > 0 else None
