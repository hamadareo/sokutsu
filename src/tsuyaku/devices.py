"""入力・出力デバイスの一覧と、名前での選び方(11/08 hear)。

見つけた問題: マイクとスピーカーは macOS の「システムの既定」しか使えず、アプリの中で替えられなかった。
しかも PortAudio(sounddevice)は起動した時のデバイス一覧と既定を覚えたままなので、後から AirPods を
つないで既定が替わっても、会話は古い既定のマイクを開いていた。

やり方:
  一覧   CoreAudio に毎回直接聞く(ctypes。PortAudio の覚えた一覧ではなく「今」の一覧。数ミリ秒)。
  保存   名前で残す(番号は抜き差しで変わる)。"" = システムの設定に合わせる。
  開く   名前 → PortAudio の番号。PortAudio の一覧に無い(後からつないだ)ときは、PortAudio の
         ストリームが1本も開いていない時だけ PortAudio を初期化し直して探す(開いている最中に
         やり直すとそのストリームが壊れる)。見つからなければシステムの既定へ落とす。
  読み上げ CoreAudio の UID(AVAudioPlayer.currentDevice)と、say -a に渡す名前。
"""
from __future__ import annotations

import ctypes
import ctypes.util
import struct
import threading

INPUT, OUTPUT = "input", "output"


def _fourcc(s: str) -> int:
    return struct.unpack(">I", s.encode("ascii"))[0]


class _Addr(ctypes.Structure):
    _fields_ = [("mSelector", ctypes.c_uint32), ("mScope", ctypes.c_uint32),
                ("mElement", ctypes.c_uint32)]


_SYS = 1   # kAudioObjectSystemObject
_GLOB, _IN, _OUT = _fourcc("glob"), _fourcc("inpt"), _fourcc("outp")
_DEVS, _DEF_IN, _DEF_OUT = _fourcc("dev#"), _fourcc("dIn "), _fourcc("dOut")
_NAME, _UID, _CONF = _fourcc("lnam"), _fourcc("uid "), _fourcc("slay")
_UTF8 = 0x08000100

_ca = _cf = None
_ca_lock = threading.Lock()


def _libs():
    global _ca, _cf
    if _ca is None:
        ca = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/CoreAudio.framework/CoreAudio")
        cf = ctypes.cdll.LoadLibrary(
            "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
        ca.AudioObjectGetPropertyDataSize.argtypes = [
            ctypes.c_uint32, ctypes.POINTER(_Addr), ctypes.c_uint32, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32)]
        ca.AudioObjectGetPropertyData.argtypes = [
            ctypes.c_uint32, ctypes.POINTER(_Addr), ctypes.c_uint32, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p]
        cf.CFStringGetCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long,
                                          ctypes.c_uint32]
        cf.CFStringGetCString.restype = ctypes.c_bool
        cf.CFRelease.argtypes = [ctypes.c_void_p]
        _ca, _cf = ca, cf
    return _ca, _cf


def _get(obj: int, sel: int, scope: int = _GLOB) -> "bytes | None":
    ca, _cf2 = _libs()
    a = _Addr(sel, scope, 0)
    n = ctypes.c_uint32(0)
    if ca.AudioObjectGetPropertyDataSize(obj, ctypes.byref(a), 0, None, ctypes.byref(n)) != 0:
        return None
    buf = ctypes.create_string_buffer(max(n.value, 1))
    if ca.AudioObjectGetPropertyData(obj, ctypes.byref(a), 0, None, ctypes.byref(n), buf) != 0:
        return None
    return buf.raw[:n.value]


def _cfstr(obj: int, sel: int) -> str:
    ca, cf = _libs()
    a = _Addr(sel, _GLOB, 0)
    ref = ctypes.c_void_p()
    n = ctypes.c_uint32(ctypes.sizeof(ref))
    if ca.AudioObjectGetPropertyData(obj, ctypes.byref(a), 0, None, ctypes.byref(n),
                                     ctypes.byref(ref)) != 0 or not ref.value:
        return ""
    out = ctypes.create_string_buffer(1024)
    ok = cf.CFStringGetCString(ref, out, 1024, _UTF8)
    cf.CFRelease(ref)
    return out.value.decode("utf-8", "replace") if ok else ""


def _channels(dev: int, scope: int) -> int:
    raw = _get(dev, _CONF, scope)
    if not raw or len(raw) < 4:
        return 0
    nbuf = struct.unpack_from("I", raw, 0)[0]
    total = 0
    for i in range(nbuf):   # AudioBufferList: UInt32 数 + (8 に揃えて) AudioBuffer{UInt32 ch, UInt32 size, void*}
        off = 8 + 16 * i
        if off + 4 > len(raw):
            break
        total += struct.unpack_from("I", raw, off)[0]
    return total


def coreaudio_devices() -> list[dict]:
    """今つながっているデバイス [{id, name, uid, inputs, outputs}](CoreAudio に直接聞く)。"""
    with _ca_lock:
        raw = _get(_SYS, _DEVS) or b""
        ids = struct.unpack(f"{len(raw) // 4}I", raw[:len(raw) // 4 * 4])
        out = []
        for d in ids:
            name = _cfstr(d, _NAME)
            if not name:
                continue
            out.append({"id": d, "name": name, "uid": _cfstr(d, _UID),
                        "inputs": _channels(d, _IN), "outputs": _channels(d, _OUT)})
        return out


def _default_id(kind: str) -> int:
    with _ca_lock:
        raw = _get(_SYS, _DEF_IN if kind == INPUT else _DEF_OUT)
    return struct.unpack("I", raw[:4])[0] if raw and len(raw) >= 4 else 0


# ---------- 外から(試験はこの3つを差し替える) ----------

_cache: dict = {}


def list_devices(kind: str, max_age: float = 0.0) -> list[dict]:
    """kind の向きに使えるデバイス [{name, uid, isDefault}]。取れなければ PortAudio の一覧。
    max_age 秒以内に聞いた一覧があればそれを使う(読み上げの度に聞かない。画面は毎回 0 で聞き直す)。"""
    import time as _t
    hit = _cache.get(kind)
    if max_age > 0 and hit and _t.monotonic() - hit[0] < max_age:
        return [dict(d) for d in hit[1]]
    out = _list_devices(kind)
    _cache[kind] = (_t.monotonic(), out)
    return [dict(d) for d in out]


def _list_devices(kind: str) -> list[dict]:
    try:
        dflt = _default_id(kind)
        key = "inputs" if kind == INPUT else "outputs"
        return [{"name": d["name"], "uid": d["uid"], "isDefault": d["id"] == dflt}
                for d in coreaudio_devices() if d[key] > 0]
    except Exception:  # noqa: BLE001 - CoreAudio に聞けない(試験の環境など)
        return _pa_list(kind)


def _pa_list(kind: str) -> list[dict]:
    try:
        import sounddevice as sd
        key = "max_input_channels" if kind == INPUT else "max_output_channels"
        di = sd.default.device[0 if kind == INPUT else 1]
        return [{"name": d["name"], "uid": "", "isDefault": i == di}
                for i, d in enumerate(sd.query_devices()) if d.get(key, 0) > 0]
    except Exception:  # noqa: BLE001
        return []


def default_name(kind: str) -> str:
    for d in list_devices(kind):
        if d["isDefault"]:
            return d["name"]
    return ""


def find(kind: str, name: str, max_age: float = 0.0) -> "dict | None":
    """名前のデバイスが今つながっていれば {name, uid, isDefault}。"""
    if not name:
        return None
    for d in list_devices(kind, max_age):
        if d["name"] == name:
            return d
    return None


def resolve(kind: str, name: str) -> "tuple[str, bool]":
    """保存した名前 → 実際に使う名前と「保存したものが無かった」か。"" = システムの既定。
    返す名前は、システムの既定を使うときもその時点の既定の名前(分かれば)。"""
    if name and find(kind, name) is not None:
        return name, False
    return default_name(kind), bool(name)


# ---------- PortAudio(マイクのストリームを開く番号) ----------

_pa_lock = threading.Lock()
_pa_open = 0      # 今開いている PortAudio のストリーム数(初期化し直してよいか)


def stream_opened() -> None:
    global _pa_open
    with _pa_lock:
        _pa_open += 1


def stream_closed() -> None:
    global _pa_open
    with _pa_lock:
        _pa_open = max(0, _pa_open - 1)


def _pa_find(sd, name: str, kind: str) -> "int | None":
    key = "max_input_channels" if kind == INPUT else "max_output_channels"
    for i, d in enumerate(sd.query_devices()):
        if d["name"] == name and d.get(key, 0) > 0:
            return i
    return None


def pa_index(sd, kind: str, name: str) -> "int | None":
    """名前 → PortAudio のデバイス番号。PortAudio が覚えた一覧に無ければ(後からつないだ)、
    開いているストリームが無い時だけ初期化し直して探す。無ければ None(= PortAudio の既定)。"""
    if not name:
        return None
    try:
        i = _pa_find(sd, name, kind)
        if i is not None:
            return i
        with _pa_lock:
            if _pa_open == 0 and hasattr(sd, "_terminate") and hasattr(sd, "_initialize"):
                sd._terminate()
                sd._initialize()
        return _pa_find(sd, name, kind)
    except Exception:  # noqa: BLE001
        return None


# ---------- 出力が「聞き手だけに届く」か(11/08 pace との継ぎ目: ヘッドホンなら読み上げ中も聞き続ける) ----------

_TRAN, _SSRC = _fourcc("tran"), _fourcc("ssrc")


def _prop4(dev: int, sel: str, scope=None) -> str:
    """4文字の値の属性(transport type 'tran'・data source 'ssrc')。取れなければ ""。"""
    raw = None
    with _ca_lock:
        raw = _get(dev, _fourcc(sel), _OUT if scope is None and sel == "ssrc" else (scope or _GLOB))
    if not raw or len(raw) < 4:
        return ""
    v = struct.unpack("I", raw[:4])[0]
    try:
        return struct.pack(">I", v).decode("ascii")
    except Exception:  # noqa: BLE001
        return ""


def classify_output(transport: str, source: str = "") -> "bool | None":
    """True = 聞き手の耳だけ(ヘッドホン端子・Bluetooth)/ False = 部屋に鳴る(内蔵スピーカー・HDMI・
    DisplayPort・AirPlay)/ None = 分からない(USB・仮想・まとめた機器など)。
    Bluetooth はほぼイヤホン・ヘッドホンだが、Bluetooth スピーカーも True になる(見分けられない)。"""
    if transport == "bltn":
        return {"hdpn": True, "ispk": False}.get(source)
    if transport in ("blue", "blea"):
        return True
    if transport in ("hdmi", "dprt", "airp"):
        return False
    return None


def output_is_private(name: str = "") -> "bool | None":
    """読み上げを鳴らす出力(name。"" = 今のシステムの既定)が聞き手だけに届くか。classify_output を見よ。"""
    try:
        if name:
            d = next((x for x in coreaudio_devices() if x["name"] == name and x["outputs"] > 0), None)
            if d is None:
                return None
            dev = d["id"]
        else:
            dev = _default_id(OUTPUT)
            if not dev:
                return None
        return classify_output(_prop4(dev, "tran"), _prop4(dev, "ssrc", _OUT))
    except Exception:  # noqa: BLE001
        return None
