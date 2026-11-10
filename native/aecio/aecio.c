// sokutsu-aecio: 読み上げを鳴らしながらマイクを聞く小さな部品(エコー消し付き)。
//
// 見つけた問題: 読み上げ中はマイクを止めている(自分の声を拾って訳し直さないため)。そのため
// 読み上げ中に相手が話した言葉が落ちる。
//
// やり方: macOS の VoiceProcessingIO(AudioUnit 'vpio' = FaceTime などと同じエコー消し)を1つ開き、
//   ・出力(バス 0)… 標準入力で受けた読み上げの音を鳴らす
//   ・入力(バス 1)… マイクの音から「いまバス 0 で鳴らした音」の回り込みを引いたものを標準出力へ
// VPIO が引けるのは自分のバス 0 で鳴らした音だけ(afplay など別プロセスの音は引けない)なので、
// 鳴らすのもこの部品でやる。音の形式はどちらも 16 kHz・モノラル・float32(機器との変換は VPIO 任せ)。
//
// C で書いている理由: このMacのコマンドラインツールは Swift / Foundation のヘッダが壊れている。
//
// 標準入力(鳴らす音。1件ずつ):
//   4バイトの uint32 リトルエンディアン = サンプル数 n、続けて float32 16 kHz モノラルを n 個
//   n = 0 は「いま鳴っている分と待っている分を捨てる(止める)」
//   標準入力が閉じたら(親の Python が終わった)片付けて終わる
// 標準出力: エコーを引いたマイクの音(float32 16 kHz モノラル、生の PCM)
// 標準エラーへ1行ずつ:
//   READY rate=16000 in="…" out="…" in_hw=48000 out_hw=48000 ...  … 動き始めた
//   PLAY_START id=1 n=16000              … 1件目の最初のサンプルを鳴らし始めた
//   PLAY_DONE id=1 n=16000 cancelled=0   … 1件目を鳴らし終えた(cancelled=1 は止めた)
//   FLUSH pending=1                      … 止める命令を受けた(待っていた件数)
//   INFO ... / ERROR ...                 … 参考 / 終わる理由(終了コードとセット)
//   id は受けた順に 1, 2, 3 …(Python 側も同じ数え方をする)
// 終了コード(sysaudio と同じ並び):
//   0 正常(標準入力が閉じた / SIGTERM)  2 引数・約束の誤り  3 マイクの許可がない
//   4 この Mac では使えない(VPIO が無い・入出力の機器が無い)  5 その他の失敗(AudioUnit のエラー)
//   6 入力か出力の機器が変わった(作り直してほしい)  7 標準出力が閉じた
//
// 試験用:
//   --check     … マイクの許可の状態だけ表示(問い合わせのダイアログは出さない)
//   --selftest  … VPIO を作って形式を決め、初期化して読み戻すだけ(開始しない = 音を出さない・録らない)
//   --fake      … 音声機器を使わず、10 ms ごとの偽の時計で同じ道を動かす。マイクの音 =
//                 鳴らした音 × 0.5(エコーのつもり)。約束(標準入出力・状態の行)の試験用
// 設定:
//   --agc       … VPIO の自動音量(AGC)を使う(既定は切る。アプリ側に autogain があるため)
//   --bypass    … エコー消しを素通しにする(比べる試験用)
//   --tail-ms N … 最後のサンプルを渡してから PLAY_DONE までの待ち(機器の中の遅れ分。既定 60)
#include <AudioToolbox/AudioToolbox.h>
#include <CoreAudio/CoreAudio.h>
#include <CoreFoundation/CoreFoundation.h>
#include <dispatch/dispatch.h>
#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define SR 16000.0
#define MAX_CLIP (16000u * 600u)  // 1件は10分まで(それより長いのは約束の誤りとみなす)

static void logline(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void logline(const char *fmt, ...) {
  char buf[1024];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof buf, fmt, ap);
  va_end(ap);
  fprintf(stderr, "%s\n", buf);
  fflush(stderr);
}

static double now_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// ---------- 一方通行のリングバッファ(書く側1本・読む側1本。リアルタイムの側でロックしない) ----------
typedef struct {
  float *buf;
  uint64_t mask;            // 容量 - 1(容量は2の累乗)
  _Atomic uint64_t w, r;    // これまでに書いた/読んだ総数
} ring_t;

static int ring_init(ring_t *q, uint64_t cap_pow2) {
  q->buf = calloc(cap_pow2, sizeof(float));
  q->mask = cap_pow2 - 1;
  atomic_init(&q->w, 0);
  atomic_init(&q->r, 0);
  return q->buf != NULL;
}

// 書けた数を返す(空きが足りなければある分だけ)
static uint32_t ring_push(ring_t *q, const float *x, uint32_t n) {
  uint64_t w = atomic_load_explicit(&q->w, memory_order_relaxed);
  uint64_t r = atomic_load_explicit(&q->r, memory_order_acquire);
  uint64_t room = (q->mask + 1) - (w - r);
  if (n > room) n = (uint32_t)room;
  for (uint32_t i = 0; i < n; i++) q->buf[(w + i) & q->mask] = x[i];
  atomic_store_explicit(&q->w, w + n, memory_order_release);
  return n;
}

// 読めた数を返す
static uint32_t ring_pop(ring_t *q, float *x, uint32_t n) {
  uint64_t r = atomic_load_explicit(&q->r, memory_order_relaxed);
  uint64_t w = atomic_load_explicit(&q->w, memory_order_acquire);
  if (n > w - r) n = (uint32_t)(w - r);
  for (uint32_t i = 0; i < n; i++) x[i] = q->buf[(r + i) & q->mask];
  atomic_store_explicit(&q->r, r + n, memory_order_release);
  return n;
}

static ring_t g_mic;    // マイク(入力コールバック → 標準出力へ書くスレッド)
static ring_t g_play;   // 鳴らす音(標準入力を読むスレッド → 出力コールバック)
static _Atomic uint64_t g_flush_to = 0;   // 0 以外: 出力コールバックがここまで読み飛ばす(止める命令)
static _Atomic uint64_t g_mic_dropped = 0;
static double g_tail_s = 0.06;

// ---------- 鳴らす件の表(リアルタイムの側は触らない。ロックしてよい) ----------
#define MAX_CLIPS 256
typedef struct {
  uint32_t id, n;
  uint64_t start, end;   // g_play の位置で [start, end)
  int started, cancelled;
  double done_at;        // end まで渡し終えた時刻(0 = まだ)
} clip_t;
static clip_t g_clips[MAX_CLIPS];
static int g_nclips = 0;
static uint32_t g_next_id = 1;
static pthread_mutex_t g_clip_lock = PTHREAD_MUTEX_INITIALIZER;

// ---------- 後片付け ----------
static AudioUnit g_unit = NULL;
static int g_started = 0;
static pthread_mutex_t g_fin = PTHREAD_MUTEX_INITIALIZER;

static void finish(int code, const char *msg) __attribute__((noreturn));
static void finish(int code, const char *msg) {
  pthread_mutex_lock(&g_fin);  // 2本目以降はここで止まる(exit まで待つ)
  if (g_unit) {
    if (g_started) AudioOutputUnitStop(g_unit);
    AudioUnitUninitialize(g_unit);
    AudioComponentInstanceDispose(g_unit);
  }
  if (msg) logline("%s", msg);
  uint64_t d = atomic_load(&g_mic_dropped);
  if (d) logline("INFO マイクの音を %llu サンプル捨てました(標準出力が遅れた)", (unsigned long long)d);
  _exit(code);
}

// ---------- 標準出力 ----------
static void write_all(const void *p, size_t len) {
  const char *c = p;
  while (len > 0) {
    ssize_t w = write(1, c, len);
    if (w < 0) {
      if (errno == EINTR) continue;
      finish(7, "ERROR 標準出力が閉じました");
    }
    c += w;
    len -= (size_t)w;
  }
}

// ---------- マイクの許可(TCC。非公開の関数を dlopen で使う。無ければ OS の判断に任せる) ----------
enum { PERM_GRANTED, PERM_DENIED, PERM_UNKNOWN, PERM_UNAVAILABLE };
static const char *PERM_NAMES[] = {"granted", "denied", "unknown", "unavailable"};
static void *g_tcc;

static void *tcc(const char *name) {
  if (!g_tcc) g_tcc = dlopen("/System/Library/PrivateFrameworks/TCC.framework/Versions/A/TCC", RTLD_NOW);
  return g_tcc ? dlsym(g_tcc, name) : NULL;
}

static int preflight(void) {
  typedef long (*fn_t)(CFStringRef, CFDictionaryRef);
  fn_t f = (fn_t)tcc("TCCAccessPreflight");
  if (!f) return PERM_UNAVAILABLE;
  long r = f(CFSTR("kTCCServiceMicrophone"), NULL);
  return r == 0 ? PERM_GRANTED : r == 1 ? PERM_DENIED : PERM_UNKNOWN;
}

// まだ決まっていなければ OS の許可ダイアログを出して答えを待つ。-1 = 関数が無い/時間切れ
static int request_permission(double timeout_s) {
  typedef void (*fn_t)(CFStringRef, CFDictionaryRef, void (^)(Boolean));
  fn_t f = (fn_t)tcc("TCCAccessRequest");
  if (!f) return -1;
  dispatch_semaphore_t sem = dispatch_semaphore_create(0);
  __block int ok = 0;
  f(CFSTR("kTCCServiceMicrophone"), NULL, ^(Boolean granted) {
    ok = granted ? 1 : 0;
    dispatch_semaphore_signal(sem);
  });
  if (dispatch_semaphore_wait(sem, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(timeout_s * 1e9))))
    return -1;
  return ok;
}

// ---------- Core Audio の小道具 ----------
static AudioObjectID default_device(int input) {
  AudioObjectPropertyAddress a = {
      input ? kAudioHardwarePropertyDefaultInputDevice : kAudioHardwarePropertyDefaultOutputDevice,
      kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain};
  AudioObjectID dev = kAudioObjectUnknown;
  UInt32 size = sizeof dev;
  AudioObjectGetPropertyData(kAudioObjectSystemObject, &a, 0, NULL, &size, &dev);
  return dev;
}

static void device_name(AudioObjectID dev, char *out, size_t cap) {
  snprintf(out, cap, "?");
  AudioObjectPropertyAddress a = {kAudioObjectPropertyName, kAudioObjectPropertyScopeGlobal,
                                  kAudioObjectPropertyElementMain};
  CFStringRef s = NULL;
  UInt32 size = sizeof s;
  if (AudioObjectGetPropertyData(dev, &a, 0, NULL, &size, &s) == noErr && s) {
    CFStringGetCString(s, out, (CFIndex)cap, kCFStringEncodingUTF8);
    CFRelease(s);
  }
  for (char *c = out; *c; c++)
    if (*c == '"') *c = '\'';
}

static double device_rate(AudioObjectID dev) {
  AudioObjectPropertyAddress a = {kAudioDevicePropertyNominalSampleRate,
                                  kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain};
  Float64 r = 0;
  UInt32 size = sizeof r;
  AudioObjectGetPropertyData(dev, &a, 0, NULL, &size, &r);
  return r;
}

static AudioStreamBasicDescription client_format(void) {
  AudioStreamBasicDescription f;
  memset(&f, 0, sizeof f);
  f.mSampleRate = SR;
  f.mFormatID = kAudioFormatLinearPCM;
  f.mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked | kAudioFormatFlagIsNonInterleaved;
  f.mBytesPerPacket = 4;
  f.mFramesPerPacket = 1;
  f.mBytesPerFrame = 4;
  f.mChannelsPerFrame = 1;
  f.mBitsPerChannel = 32;
  return f;
}

// ---------- 鳴らす側(出力コールバック。リアルタイムなのでロック・malloc・I/O をしない) ----------
static void render_play(float *out, uint32_t frames) {
  uint64_t to = atomic_exchange_explicit(&g_flush_to, 0, memory_order_acq_rel);
  if (to) {  // 止める命令: そこまで書かれた分を読み飛ばす
    uint64_t r = atomic_load_explicit(&g_play.r, memory_order_relaxed);
    if (to > r) atomic_store_explicit(&g_play.r, to, memory_order_release);
  }
  uint32_t got = ring_pop(&g_play, out, frames);
  if (got < frames) memset(out + got, 0, (frames - got) * sizeof(float));  // 無ければ無音
}

static OSStatus on_render(void *ref, AudioUnitRenderActionFlags *flags, const AudioTimeStamp *ts,
                          UInt32 bus, UInt32 frames, AudioBufferList *io) {
  (void)ref; (void)ts; (void)bus;
  for (UInt32 b = 0; b < io->mNumberBuffers; b++) {
    float *out = io->mBuffers[b].mData;
    uint32_t n = io->mBuffers[b].mDataByteSize / sizeof(float);
    if (n > frames) n = frames;
    if (b == 0) {
      render_play(out, n);
    } else {
      memcpy(out, io->mBuffers[0].mData, n * sizeof(float));
    }
  }
  (void)flags;  // 無音でも OutputIsSilence は立てない(VPIO の参照はいつも渡す)
  return noErr;
}

// ---------- 聞く側(入力コールバック) ----------
#define MAX_FRAMES 16384
static float g_inbuf[MAX_FRAMES];

static void push_mic(const float *x, uint32_t n) {
  uint32_t put = ring_push(&g_mic, x, n);
  if (put < n) atomic_fetch_add_explicit(&g_mic_dropped, n - put, memory_order_relaxed);
}

static OSStatus on_input(void *ref, AudioUnitRenderActionFlags *flags, const AudioTimeStamp *ts,
                         UInt32 bus, UInt32 frames, AudioBufferList *unused) {
  (void)ref; (void)unused;
  if (frames > MAX_FRAMES) frames = MAX_FRAMES;
  AudioBufferList abl;
  abl.mNumberBuffers = 1;
  abl.mBuffers[0].mNumberChannels = 1;
  abl.mBuffers[0].mDataByteSize = frames * sizeof(float);
  abl.mBuffers[0].mData = g_inbuf;
  OSStatus st = AudioUnitRender(g_unit, flags, ts, bus, frames, &abl);
  if (st != noErr) return st;
  push_mic(g_inbuf, frames);
  return noErr;
}

// ---------- 標準出力へ書く・件の始まり/終わりを知らせるスレッド ----------
static void check_clips(void) {
  uint64_t r = atomic_load_explicit(&g_play.r, memory_order_acquire);
  double t = now_s();
  pthread_mutex_lock(&g_clip_lock);
  int k = 0;
  for (int i = 0; i < g_nclips; i++) {
    clip_t *c = &g_clips[i];
    if (!c->started && !c->cancelled && r > c->start) {
      c->started = 1;
      logline("PLAY_START id=%u n=%u", c->id, c->n);
    }
    if (r >= c->end && c->done_at == 0) c->done_at = t;
    // 止めた件はすぐ、鳴らし終えた件は機器の中の遅れ(tail)を待ってから知らせる
    if (c->done_at != 0 && (c->cancelled || t - c->done_at >= g_tail_s)) {
      logline("PLAY_DONE id=%u n=%u cancelled=%d", c->id, c->n, c->cancelled);
      continue;
    }
    g_clips[k++] = *c;
  }
  g_nclips = k;
  pthread_mutex_unlock(&g_clip_lock);
}

static void *writer_thread(void *arg) {
  (void)arg;
  static float buf[4096];
  for (;;) {
    uint32_t n;
    int any = 0;
    while ((n = ring_pop(&g_mic, buf, 4096)) > 0) {
      write_all(buf, n * sizeof(float));
      any = 1;
    }
    check_clips();
    if (!any) usleep(5000);
  }
  return NULL;
}

// ---------- 標準入力を読むスレッド ----------
static int read_full(void *p, size_t len) {  // 1 = 読めた、0 = 閉じた
  char *c = p;
  while (len > 0) {
    ssize_t r = read(0, c, len);
    if (r == 0) return 0;
    if (r < 0) {
      if (errno == EINTR) continue;
      return 0;
    }
    c += r;
    len -= (size_t)r;
  }
  return 1;
}

static void do_flush(void) {
  pthread_mutex_lock(&g_clip_lock);
  int pending = 0;
  for (int i = 0; i < g_nclips; i++) {
    if (g_clips[i].done_at == 0) {  // まだ渡し終えていない件 = 止めた件(渡し終えた件は鳴った扱い)
      g_clips[i].cancelled = 1;
      g_clips[i].done_at = now_s();
      pending++;
    }
  }
  pthread_mutex_unlock(&g_clip_lock);
  // ここまで書いた分を出力コールバックが読み飛ばす
  atomic_store_explicit(&g_flush_to, atomic_load(&g_play.w), memory_order_release);
  logline("FLUSH pending=%d", pending);
}

static void *stdin_thread(void *arg) {
  (void)arg;
  static float buf[4096];
  for (;;) {
    unsigned char h[4];
    if (!read_full(h, 4)) finish(0, NULL);  // 親が閉じた
    uint32_t n = (uint32_t)h[0] | ((uint32_t)h[1] << 8) | ((uint32_t)h[2] << 16) | ((uint32_t)h[3] << 24);
    if (n == 0) {
      do_flush();
      continue;
    }
    if (n > MAX_CLIP) {
      char m[128];
      snprintf(m, sizeof m, "ERROR 約束の誤り: 1件 %u サンプルは長すぎます", n);
      finish(2, m);
    }
    // 件を表に入れてから音をリングへ(PLAY_START が書き終わりより先に来てもよいように)
    pthread_mutex_lock(&g_clip_lock);
    while (g_nclips >= MAX_CLIPS) {  // 待ちすぎ(普通は無い): 空くまで待つ
      pthread_mutex_unlock(&g_clip_lock);
      usleep(10000);
      pthread_mutex_lock(&g_clip_lock);
    }
    clip_t *c = &g_clips[g_nclips++];
    memset(c, 0, sizeof *c);
    c->id = g_next_id++;
    c->n = n;
    c->start = atomic_load(&g_play.w);
    c->end = c->start + n;
    pthread_mutex_unlock(&g_clip_lock);
    uint32_t left = n;
    while (left > 0) {
      uint32_t k = left < 4096 ? left : 4096;
      if (!read_full(buf, k * sizeof(float))) finish(0, NULL);
      uint32_t off = 0;
      while (off < k) {  // リングが一杯なら鳴って空くまで待つ(とても長い件だけ)
        off += ring_push(&g_play, buf + off, k - off);
        if (off < k) usleep(5000);
      }
      left -= k;
    }
  }
  return NULL;
}

// ---------- 偽の時計(--fake): 機器を使わずに同じ道を動かす ----------
static void *fake_clock(void *arg) {
  (void)arg;
  float out[160], mic[160];
  double t0 = now_s();
  uint64_t ticks = 0;
  for (;;) {
    render_play(out, 160);
    for (int i = 0; i < 160; i++) mic[i] = 0.5f * out[i];  // エコーのつもり
    push_mic(mic, 160);
    ticks++;
    double wait = t0 + ticks * 0.01 - now_s();
    if (wait > 0) usleep((useconds_t)(wait * 1e6));
  }
  return NULL;
}

// ---------- VPIO を作る ----------
static int g_agc = 0, g_bypass = 0;

#define CHECK(expr, what)                                              \
  do {                                                                 \
    OSStatus _st = (expr);                                             \
    if (_st != noErr) {                                                \
      logline("ERROR %s に失敗しました (OSStatus %d)", what, (int)_st); \
      return 5;                                                        \
    }                                                                  \
  } while (0)

static int make_unit(void) {
  if (default_device(1) == kAudioObjectUnknown || default_device(0) == kAudioObjectUnknown) {
    logline("ERROR 入力か出力の機器がありません");
    return 4;
  }
  AudioComponentDescription d = {kAudioUnitType_Output, kAudioUnitSubType_VoiceProcessingIO,
                                 kAudioUnitManufacturer_Apple, 0, 0};
  AudioComponent comp = AudioComponentFindNext(NULL, &d);
  if (!comp) {
    logline("ERROR この Mac には VoiceProcessingIO がありません");
    return 4;
  }
  CHECK(AudioComponentInstanceNew(comp, &g_unit), "VPIO の作成");
  UInt32 one = 1;
  CHECK(AudioUnitSetProperty(g_unit, kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Input, 1,
                             &one, sizeof one), "入力(バス 1)を有効にする");
  CHECK(AudioUnitSetProperty(g_unit, kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Output, 0,
                             &one, sizeof one), "出力(バス 0)を有効にする");
  AudioStreamBasicDescription f = client_format();
  // マイク側: バス 1 の出口(アプリが受け取る形)/ スピーカー側: バス 0 の入口(アプリが渡す形)
  CHECK(AudioUnitSetProperty(g_unit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Output, 1, &f,
                             sizeof f), "マイク側の形式 16 kHz を決める");
  CHECK(AudioUnitSetProperty(g_unit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input, 0, &f,
                             sizeof f), "スピーカー側の形式 16 kHz を決める");
  UInt32 agc = g_agc ? 1 : 0, bypass = g_bypass ? 1 : 0;
  // 失敗しても続ける(Global なので要素番号は効かないはずだが、資料によって 0 と 1 があるので両方)
  for (UInt32 el = 0; el < 2; el++) {
    AudioUnitSetProperty(g_unit, kAUVoiceIOProperty_VoiceProcessingEnableAGC, kAudioUnitScope_Global, el,
                         &agc, sizeof agc);
    AudioUnitSetProperty(g_unit, kAUVoiceIOProperty_BypassVoiceProcessing, kAudioUnitScope_Global, el,
                         &bypass, sizeof bypass);
  }
  // VPIO は既定で他のアプリの音(Zoom の相手の声など)を小さくする。いちばん弱くする(macOS 14+)
  if (__builtin_available(macOS 14.0, *)) {
    AUVoiceIOOtherAudioDuckingConfiguration duck = {false, kAUVoiceIOOtherAudioDuckingLevelMin};
    OSStatus st = AudioUnitSetProperty(g_unit, kAUVoiceIOProperty_OtherAudioDuckingConfiguration,
                                       kAudioUnitScope_Global, 0, &duck, sizeof duck);
    if (st != noErr) logline("INFO 他のアプリの音を小さくする設定を変えられません (OSStatus %d)", (int)st);
  } else {
    logline("INFO macOS 13 では他のアプリの音が小さくなります(VPIO の既定)");
  }
  AURenderCallbackStruct in = {on_input, NULL};
  CHECK(AudioUnitSetProperty(g_unit, kAudioOutputUnitProperty_SetInputCallback, kAudioUnitScope_Global,
                             0, &in, sizeof in), "入力コールバックの登録");
  AURenderCallbackStruct out = {on_render, NULL};
  CHECK(AudioUnitSetProperty(g_unit, kAudioUnitProperty_SetRenderCallback, kAudioUnitScope_Input, 0,
                             &out, sizeof out), "出力コールバックの登録");
  CHECK(AudioUnitInitialize(g_unit), "VPIO の初期化");
  return 0;
}

// 初期化の後の形式を読み戻して確かめる。0 = 16 kHz モノラル float32 のまま
static int check_formats(char *desc, size_t cap) {
  AudioStreamBasicDescription m, s, mh, sh;
  UInt32 z = sizeof m;
  if (AudioUnitGetProperty(g_unit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Output, 1, &m, &z))
    return -1;
  z = sizeof s;
  if (AudioUnitGetProperty(g_unit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input, 0, &s, &z))
    return -1;
  // 機器側(VPIO の中で使う形。参考)
  memset(&mh, 0, sizeof mh);
  memset(&sh, 0, sizeof sh);
  z = sizeof mh;
  AudioUnitGetProperty(g_unit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input, 1, &mh, &z);
  z = sizeof sh;
  AudioUnitGetProperty(g_unit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Output, 0, &sh, &z);
  snprintf(desc, cap, "mic=%.0fHz/%uch spk=%.0fHz/%uch vp_in=%.0fHz/%uch vp_out=%.0fHz/%uch",
           m.mSampleRate, (unsigned)m.mChannelsPerFrame, s.mSampleRate, (unsigned)s.mChannelsPerFrame,
           mh.mSampleRate, (unsigned)mh.mChannelsPerFrame, sh.mSampleRate, (unsigned)sh.mChannelsPerFrame);
  int ok = m.mSampleRate == SR && m.mChannelsPerFrame == 1 && (m.mFormatFlags & kAudioFormatFlagIsFloat) &&
           s.mSampleRate == SR && s.mChannelsPerFrame == 1 && (s.mFormatFlags & kAudioFormatFlagIsFloat);
  return ok ? 0 : 1;
}

static void device_info(char *out, size_t cap) {
  AudioObjectID in = default_device(1), outd = default_device(0);
  char ni[128], no[128];
  device_name(in, ni, sizeof ni);
  device_name(outd, no, sizeof no);
  snprintf(out, cap, "in=\"%s\" out=\"%s\" in_hw=%.0f out_hw=%.0f", ni, no, device_rate(in),
           device_rate(outd));
}

static int run_selftest(void) {
  char dev[512], fmt[256];
  device_info(dev, sizeof dev);
  logline("INFO 機器 %s 許可=%s", dev, PERM_NAMES[preflight()]);
  int rc = make_unit();
  if (rc) return rc;
  int f = check_formats(fmt, sizeof fmt);
  if (f < 0) finish(5, "ERROR 形式を読み戻せません");
  logline("SELFTEST %s %s", f == 0 ? "ok" : "format_mismatch", fmt);
  return f == 0 ? 0 : 5;
}

// 入力か出力の既定の機器が変わったら作り直してもらう(VPIO は開いたときの機器のまま)
static void watch_devices(void) {
  AudioObjectPropertySelector sels[] = {kAudioHardwarePropertyDefaultInputDevice,
                                        kAudioHardwarePropertyDefaultOutputDevice};
  for (int i = 0; i < 2; i++) {
    AudioObjectPropertyAddress a = {sels[i], kAudioObjectPropertyScopeGlobal,
                                    kAudioObjectPropertyElementMain};
    AudioObjectAddPropertyListenerBlock(kAudioObjectSystemObject, &a, dispatch_get_main_queue(),
                                        ^(UInt32 n, const AudioObjectPropertyAddress *addrs) {
                                          (void)n; (void)addrs;
                                          finish(6, "ERROR 入力か出力の機器が変わりました");
                                        });
  }
}

static int run_io(int fake) {
  if (!ring_init(&g_mic, 1u << 18) || !ring_init(&g_play, 1u << 21)) {  // 16 秒 / 131 秒
    logline("ERROR メモリが足りません");
    return 5;
  }
  char dev[512] = "in=fake out=fake", fmt[256] = "";
  if (!fake) {
    int perm = preflight();
    if (perm == PERM_DENIED) {
      logline("ERROR PERMISSION_DENIED マイクが許可されていません");
      return 3;
    }
    if (perm == PERM_UNKNOWN) {
      logline("INFO 許可を問い合わせます(macOS のダイアログ)");
      if (request_permission(600) == 0) {
        logline("ERROR PERMISSION_DENIED マイクが許可されませんでした");
        return 3;
      }
    }
    int rc = make_unit();
    if (rc) return rc;
    if (check_formats(fmt, sizeof fmt) != 0) {
      logline("ERROR 16 kHz モノラル float32 にできません (%s)", fmt);
      return 5;
    }
    device_info(dev, sizeof dev);
  }
  pthread_t tw, tr, tf;
  pthread_create(&tw, NULL, writer_thread, NULL);
  if (fake) {
    pthread_create(&tf, NULL, fake_clock, NULL);
  } else {
    CHECK(AudioOutputUnitStart(g_unit), "VPIO の開始");
    g_started = 1;
    watch_devices();
  }
  logline("READY rate=16000 %s agc=%d bypass=%d fake=%d %s", dev, g_agc, g_bypass, fake, fmt);
  pthread_create(&tr, NULL, stdin_thread, NULL);
  dispatch_main();
}

int main(int argc, char **argv) {
  signal(SIGPIPE, SIG_IGN);
  int check = 0, selftest = 0, fake = 0;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--check")) {
      check = 1;
    } else if (!strcmp(argv[i], "--selftest")) {
      selftest = 1;
    } else if (!strcmp(argv[i], "--fake")) {
      fake = 1;
    } else if (!strcmp(argv[i], "--agc")) {
      g_agc = 1;
    } else if (!strcmp(argv[i], "--bypass")) {
      g_bypass = 1;
    } else if (!strcmp(argv[i], "--tail-ms") && i + 1 < argc) {
      g_tail_s = atof(argv[++i]) / 1000.0;
    } else {
      logline("ERROR 知らない引数 %s", argv[i]);
      return 2;
    }
  }
  if (check) {
    printf("PERMISSION %s\n", PERM_NAMES[preflight()]);
    return 0;
  }
  if (selftest) finish(run_selftest(), NULL);

  int sigs[] = {SIGTERM, SIGINT, SIGHUP};
  for (int i = 0; i < 3; i++) {
    signal(sigs[i], SIG_IGN);
    dispatch_source_t s = dispatch_source_create(DISPATCH_SOURCE_TYPE_SIGNAL, (uintptr_t)sigs[i], 0,
                                                 dispatch_get_main_queue());
    dispatch_source_set_event_handler(s, ^{ finish(0, NULL); });
    dispatch_resume(s);
  }
  finish(run_io(fake), NULL);
}
