// sokutsu-sysaudio: この Mac で鳴っている音(Zoom・Teams・Meet の相手の声など)を聞いて、
// モノラル・float32 の生の PCM を標準出力へ流す小さな部品。16 kHz への変換は Python 側。
//
// 仕組み: Core Audio の「プロセスタップ」(macOS 14.2+)。全部のアプリの音をモノラルに
// まとめ、Sokutsu 自身(Python と、その子の afplay / say = 読み上げの声)だけは外す。
// 画面収録(ScreenCaptureKit)と違って映像は扱わず、要る許可は「システムオーディオ録音のみ」。
//
// C で書いている理由: このMacのコマンドラインツールは Swift / Foundation のヘッダが壊れていて
// (NSString.h が無い・SwiftBridging の二重定義)Swift も Objective-C も通らない。
// CATapDescription(Objective-C のクラス)は objc ランタイムを直接呼んで作る。
//
// 標準エラーへ1行ずつ:
//   READY rate=48000 excluded=2 ...   … 音が流れ始めた(rate = 標準出力の PCM の周波数)
//   INFO ...                           … 参考
//   ERROR ...                          … 終わる理由(終了コードとセット)
// 終了コード:
//   0 正常(標準入力が閉じた / SIGTERM)  2 引数の誤り  3 許可がない
//   4 この macOS では使えない  5 タップ・集約デバイスを作れない
//   6 出力先が変わった(作り直してほしい)  7 標準出力が閉じた
//
// 試験用(許可も音声機器も使わない):
//   --tone 秒   … 48 kHz の 440 Hz を同じ出力の道で出す(速く、時間を待たずに)
//   --check     … 許可の状態だけ表示(問い合わせのダイアログは出さない)
//   --list      … 音を出すプロセスの一覧と、外す対象か
#include <CoreAudio/CoreAudio.h>
#include <CoreFoundation/CoreFoundation.h>
#include <dispatch/dispatch.h>
#include <dlfcn.h>
#include <errno.h>
#include <math.h>
#include <objc/message.h>
#include <objc/runtime.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysctl.h>
#include <unistd.h>

// CATapDescription / AudioHardwareTapping.h は Objective-C 専用ヘッダなので自前で宣言する
extern OSStatus AudioHardwareCreateProcessTap(void *inDescription, AudioObjectID *outTapID)
    __attribute__((weak_import));
extern OSStatus AudioHardwareDestroyProcessTap(AudioObjectID inTapID) __attribute__((weak_import));
extern void *objc_autoreleasePoolPush(void);
extern void objc_autoreleasePoolPop(void *);
#define kTapPropertyDescription 'tdsc'
#define kTapPropertyFormat 'tfmt'

typedef void *id_t_;
#define MSG(ret, ...) ((ret(*)(__VA_ARGS__))objc_msgSend)

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

// ---------- 引数 ----------
#define MAX_ROOTS 16
static pid_t g_roots[MAX_ROOTS];
static int g_nroots = 0;

// ---------- 後片付け ----------
static AudioObjectID g_tap = kAudioObjectUnknown, g_agg = kAudioObjectUnknown;
static AudioDeviceIOProcID g_proc = NULL;
static pthread_mutex_t g_fin = PTHREAD_MUTEX_INITIALIZER;

static void finish(int code, const char *msg) __attribute__((noreturn));
static void finish(int code, const char *msg) {
  pthread_mutex_lock(&g_fin);  // 2本目以降はここで止まる(exit まで待つ)
  if (g_agg != kAudioObjectUnknown && g_proc) {
    AudioDeviceStop(g_agg, g_proc);
    AudioDeviceDestroyIOProcID(g_agg, g_proc);
  }
  if (g_agg != kAudioObjectUnknown) AudioHardwareDestroyAggregateDevice(g_agg);
  if (g_tap != kAudioObjectUnknown && AudioHardwareDestroyProcessTap)
    AudioHardwareDestroyProcessTap(g_tap);
  if (msg) logline("%s", msg);
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

// ---------- 許可(TCC。非公開の関数を dlopen で使う。無ければ OS の判断に任せる) ----------
enum { PERM_GRANTED, PERM_DENIED, PERM_UNKNOWN, PERM_UNAVAILABLE };
static void *g_tcc;
static const char *PERM_NAMES[] = {"granted", "denied", "unknown", "unavailable"};

static int preflight(void) {
  typedef long (*fn_t)(CFStringRef, CFDictionaryRef);
  if (!g_tcc) g_tcc = dlopen("/System/Library/PrivateFrameworks/TCC.framework/Versions/A/TCC", RTLD_NOW);
  fn_t f = g_tcc ? (fn_t)dlsym(g_tcc, "TCCAccessPreflight") : NULL;
  if (!f) return PERM_UNAVAILABLE;
  long r = f(CFSTR("kTCCServiceAudioCapture"), NULL);
  return r == 0 ? PERM_GRANTED : r == 1 ? PERM_DENIED : PERM_UNKNOWN;
}

// まだ決まっていなければ OS の許可ダイアログを出して答えを待つ。-1 = 関数が無い/時間切れ
static int request_permission(double timeout_s) {
  typedef void (*fn_t)(CFStringRef, CFDictionaryRef, void (^)(Boolean));
  if (!g_tcc) g_tcc = dlopen("/System/Library/PrivateFrameworks/TCC.framework/Versions/A/TCC", RTLD_NOW);
  fn_t f = g_tcc ? (fn_t)dlsym(g_tcc, "TCCAccessRequest") : NULL;
  if (!f) return -1;
  dispatch_semaphore_t sem = dispatch_semaphore_create(0);
  __block int ok = 0;
  f(CFSTR("kTCCServiceAudioCapture"), NULL, ^(Boolean granted) {
    ok = granted ? 1 : 0;
    dispatch_semaphore_signal(sem);
  });
  if (dispatch_semaphore_wait(sem, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(timeout_s * 1e9))))
    return -1;
  return ok;
}

// ---------- Core Audio の小道具 ----------
static AudioObjectPropertyAddress addr(AudioObjectPropertySelector sel) {
  AudioObjectPropertyAddress a = {sel, kAudioObjectPropertyScopeGlobal,
                                  kAudioObjectPropertyElementMain};
  return a;
}

static OSStatus get_prop(AudioObjectID obj, AudioObjectPropertySelector sel, void *out, UInt32 size) {
  AudioObjectPropertyAddress a = addr(sel);
  return AudioObjectGetPropertyData(obj, &a, 0, NULL, &size, out);
}

static int process_objects(AudioObjectID *ids, int cap) {
  AudioObjectPropertyAddress a = addr(kAudioHardwarePropertyProcessObjectList);
  UInt32 size = 0;
  if (AudioObjectGetPropertyDataSize(kAudioObjectSystemObject, &a, 0, NULL, &size) != noErr) return 0;
  if (size > (UInt32)(cap * sizeof(AudioObjectID))) size = (UInt32)(cap * sizeof(AudioObjectID));
  if (AudioObjectGetPropertyData(kAudioObjectSystemObject, &a, 0, NULL, &size, ids) != noErr) return 0;
  return (int)(size / sizeof(AudioObjectID));
}

static pid_t parent_pid(pid_t pid) {
  struct kinfo_proc info;
  size_t size = sizeof info;
  int mib[4] = {CTL_KERN, KERN_PROC, KERN_PROC_PID, pid};
  memset(&info, 0, sizeof info);
  if (sysctl(mib, 4, &info, &size, NULL, 0) != 0 || size == 0) return 0;
  return info.kp_eproc.e_ppid;
}

// pid が roots(Sokutsu の Python など)かその子孫か
static int ours(pid_t pid) {
  pid_t p = pid;
  for (int depth = 0; depth < 32; depth++) {
    for (int i = 0; i < g_nroots; i++)
      if (g_roots[i] == p) return 1;
    if (p <= 1) return 0;
    p = parent_pid(p);
  }
  return 0;
}

#define MAX_PROCS 512
static int our_objects(AudioObjectID *out, int cap) {
  AudioObjectID ids[MAX_PROCS];
  int n = process_objects(ids, MAX_PROCS), k = 0;
  for (int i = 0; i < n && k < cap; i++) {
    pid_t pid = 0;
    if (get_prop(ids[i], kAudioProcessPropertyPID, &pid, sizeof pid) == noErr && ours(pid))
      out[k++] = ids[i];
  }
  // 並びを揃える(変化の比較用)
  for (int i = 1; i < k; i++)
    for (int j = i; j > 0 && out[j - 1] > out[j]; j--) {
      AudioObjectID t = out[j];
      out[j] = out[j - 1];
      out[j - 1] = t;
    }
  return k;
}

static CFArrayRef number_array(const AudioObjectID *ids, int n) {
  CFMutableArrayRef arr = CFArrayCreateMutable(NULL, n, &kCFTypeArrayCallBacks);
  for (int i = 0; i < n; i++) {
    SInt32 v = (SInt32)ids[i];
    CFNumberRef num = CFNumberCreate(NULL, kCFNumberSInt32Type, &v);
    CFArrayAppendValue(arr, num);
    CFRelease(num);
  }
  return arr;
}

static void cfstr_to_c(CFStringRef s, char *buf, size_t len) {
  buf[0] = 0;
  if (s) CFStringGetCString(s, buf, (CFIndex)len, kCFStringEncodingUTF8);
}

// ---------- 試験用の音 ----------
static int run_tone(double sec) {
  const double sr = 48000.0;
  logline("READY rate=48000 excluded=0 tone=1");
  float block[480];
  long total = (long)(sec * sr), t = 0;
  while (t < total) {
    int n = (int)((total - t) < 480 ? (total - t) : 480);
    for (int i = 0; i < n; i++) block[i] = (float)(0.5 * sin(2.0 * M_PI * 440.0 * (double)(t + i) / sr));
    write_all(block, (size_t)n * sizeof(float));
    t += n;
  }
  return 0;
}

static int run_list(void) {
  AudioObjectID ids[MAX_PROCS];
  int n = process_objects(ids, MAX_PROCS);
  for (int i = 0; i < n; i++) {
    pid_t pid = -1;
    UInt32 out = 0;
    CFStringRef bid = NULL;
    char b[256];
    get_prop(ids[i], kAudioProcessPropertyPID, &pid, sizeof pid);
    get_prop(ids[i], kAudioProcessPropertyIsRunningOutput, &out, sizeof out);
    get_prop(ids[i], kAudioProcessPropertyBundleID, &bid, sizeof bid);
    cfstr_to_c(bid, b, sizeof b);
    if (bid) CFRelease(bid);
    printf("obj=%u pid=%d output=%u ours=%d %s\n", ids[i], pid, out, ours(pid), b);
  }
  return 0;
}

// ---------- 本番 ----------
static void *g_desc;                // CATapDescription*
static AudioObjectID g_excl[64];
static int g_nexcl = 0;
static float *g_mono;
static size_t g_mono_cap;
static int g_got_audio = 0;

static void refresh_exclusions(void) {
  AudioObjectID now[64];
  int n = our_objects(now, 64);
  if (n == g_nexcl && memcmp(now, g_excl, sizeof(AudioObjectID) * (size_t)n) == 0) return;
  memcpy(g_excl, now, sizeof(AudioObjectID) * (size_t)n);
  g_nexcl = n;
  void *pool = objc_autoreleasePoolPush();
  CFArrayRef arr = number_array(now, n);
  MSG(void, void *, SEL, CFArrayRef)(g_desc, sel_registerName("setProcesses:"), arr);
  CFRelease(arr);
  AudioObjectPropertyAddress a = addr(kTapPropertyDescription);
  void *ref = g_desc;
  OSStatus r = AudioObjectSetPropertyData(g_tap, &a, 0, NULL, sizeof ref, &ref);
  if (r != noErr) logline("INFO 外す相手を更新できません (OSStatus %d)", (int)r);
  else logline("INFO 外す相手 %d 個", n);
  objc_autoreleasePoolPop(pool);
}

static int run_capture(void) {
  if (!AudioHardwareCreateProcessTap) {
    logline("ERROR この macOS ではシステムの音を聞けません(14.2 以降が必要)");
    return 4;
  }
  Class cls = objc_getClass("CATapDescription");
  if (!cls) {
    logline("ERROR この macOS ではシステムの音を聞けません(CATapDescription が無い)");
    return 4;
  }
  int perm = preflight();
  if (perm == PERM_DENIED) {
    logline("ERROR PERMISSION_DENIED システムオーディオ録音が許可されていません");
    return 3;
  }
  if (perm == PERM_UNKNOWN) {
    logline("INFO 許可を問い合わせます(macOS のダイアログ)");
    if (request_permission(600) == 0) {
      logline("ERROR PERMISSION_DENIED システムオーディオ録音が許可されませんでした");
      return 3;
    }
  }

  void *pool = objc_autoreleasePoolPush();
  g_nexcl = our_objects(g_excl, 64);
  CFArrayRef arr = number_array(g_excl, g_nexcl);
  void *d = MSG(void *, Class, SEL)(cls, sel_registerName("alloc"));
  d = MSG(void *, void *, SEL, CFArrayRef)(d, sel_registerName("initMonoGlobalTapButExcludeProcesses:"), arr);
  CFRelease(arr);
  if (!d) finish(5, "ERROR タップの設定を作れません");
  g_desc = d;  // 終わるまで持つ(解放しない)
  MSG(void, void *, SEL, CFStringRef)(d, sel_registerName("setName:"), CFSTR("Sokutsu system audio"));
  MSG(void, void *, SEL, BOOL)(d, sel_registerName("setPrivate:"), YES);
  MSG(void, void *, SEL, long)(d, sel_registerName("setMuteBehavior:"), 0);  // 鳴らしたまま
  void *uuid = MSG(void *, void *, SEL)(d, sel_registerName("UUID"));
  CFStringRef tapUID = (CFStringRef)MSG(void *, void *, SEL)(uuid, sel_registerName("UUIDString"));
  CFRetain(tapUID);
  objc_autoreleasePoolPop(pool);

  OSStatus st = AudioHardwareCreateProcessTap(d, &g_tap);
  if (st != noErr) {
    char m[128];
    snprintf(m, sizeof m, "ERROR タップを作れません (OSStatus %d)", (int)st);
    finish(5, m);
  }

  AudioObjectID outDev = 0;
  CFStringRef outUID = NULL;
  if (get_prop(kAudioObjectSystemObject, kAudioHardwarePropertyDefaultOutputDevice, &outDev,
               sizeof outDev) != noErr || !outDev ||
      get_prop(outDev, kAudioDevicePropertyDeviceUID, &outUID, sizeof outUID) != noErr || !outUID)
    finish(5, "ERROR 出力デバイスが見つかりません");

  CFStringRef aggUID = CFSTR("app.sokutsu.dev.sysaudio");
  char aggb[96];
  snprintf(aggb, sizeof aggb, "app.sokutsu.dev.sysaudio.%d", (int)getpid());
  aggUID = CFStringCreateWithCString(NULL, aggb, kCFStringEncodingUTF8);

  CFMutableDictionaryRef sub = CFDictionaryCreateMutable(NULL, 0, &kCFTypeDictionaryKeyCallBacks,
                                                         &kCFTypeDictionaryValueCallBacks);
  CFDictionarySetValue(sub, CFSTR(kAudioSubDeviceUIDKey), outUID);
  CFArrayRef subs = CFArrayCreate(NULL, (const void **)&sub, 1, &kCFTypeArrayCallBacks);
  CFMutableDictionaryRef tap = CFDictionaryCreateMutable(NULL, 0, &kCFTypeDictionaryKeyCallBacks,
                                                         &kCFTypeDictionaryValueCallBacks);
  CFDictionarySetValue(tap, CFSTR(kAudioSubTapUIDKey), tapUID);
  CFDictionarySetValue(tap, CFSTR(kAudioSubTapDriftCompensationKey), kCFBooleanTrue);
  CFArrayRef taps = CFArrayCreate(NULL, (const void **)&tap, 1, &kCFTypeArrayCallBacks);
  CFMutableDictionaryRef agg = CFDictionaryCreateMutable(NULL, 0, &kCFTypeDictionaryKeyCallBacks,
                                                         &kCFTypeDictionaryValueCallBacks);
  CFDictionarySetValue(agg, CFSTR(kAudioAggregateDeviceNameKey), CFSTR("Sokutsu-SystemAudio"));
  CFDictionarySetValue(agg, CFSTR(kAudioAggregateDeviceUIDKey), aggUID);
  CFDictionarySetValue(agg, CFSTR(kAudioAggregateDeviceMainSubDeviceKey), outUID);
  CFDictionarySetValue(agg, CFSTR(kAudioAggregateDeviceIsPrivateKey), kCFBooleanTrue);
  CFDictionarySetValue(agg, CFSTR(kAudioAggregateDeviceIsStackedKey), kCFBooleanFalse);
  CFDictionarySetValue(agg, CFSTR(kAudioAggregateDeviceTapAutoStartKey), kCFBooleanTrue);
  CFDictionarySetValue(agg, CFSTR(kAudioAggregateDeviceSubDeviceListKey), subs);
  CFDictionarySetValue(agg, CFSTR(kAudioAggregateDeviceTapListKey), taps);
  st = AudioHardwareCreateAggregateDevice(agg, &g_agg);
  if (st != noErr) {
    g_agg = kAudioObjectUnknown;
    char m[128];
    snprintf(m, sizeof m, "ERROR 集約デバイスを作れません (OSStatus %d)", (int)st);
    finish(5, m);
  }

  AudioStreamBasicDescription tf;
  memset(&tf, 0, sizeof tf);
  get_prop(g_tap, kTapPropertyFormat, &tf, sizeof tf);
  Float64 aggRate = 0;
  get_prop(g_agg, kAudioDevicePropertyNominalSampleRate, &aggRate, sizeof aggRate);
  double rate = aggRate > 0 ? aggRate : (tf.mSampleRate > 0 ? tf.mSampleRate : 48000.0);

  dispatch_queue_t q = dispatch_queue_create("sokutsu.sysaudio.io", DISPATCH_QUEUE_SERIAL);
  g_mono_cap = 8192;
  g_mono = malloc(g_mono_cap * sizeof(float));
  st = AudioDeviceCreateIOProcIDWithBlock(&g_proc, g_agg, q,
      ^(const AudioTimeStamp *now, const AudioBufferList *in, const AudioTimeStamp *it,
        AudioBufferList *out, const AudioTimeStamp *ot) {
        (void)now; (void)it; (void)out; (void)ot;
        if (!in || in->mNumberBuffers == 0) return;
        // 出力デバイス側に入力(ヘッドセットのマイク等)があると前に並ぶ。タップはいちばん後ろ。
        const AudioBuffer *b = &in->mBuffers[in->mNumberBuffers - 1];
        if (!b->mData) return;
        UInt32 ch = b->mNumberChannels ? b->mNumberChannels : 1;
        size_t n = b->mDataByteSize / (sizeof(float) * ch);
        if (!n) return;
        if (n > g_mono_cap) {
          g_mono_cap = n;
          g_mono = realloc(g_mono, n * sizeof(float));
        }
        const float *src = b->mData;
        if (ch == 1) {
          memcpy(g_mono, src, n * sizeof(float));
        } else {
          for (size_t i = 0; i < n; i++) {
            float s = 0;
            for (UInt32 c = 0; c < ch; c++) s += src[i * ch + c];
            g_mono[i] = s / (float)ch;
          }
        }
        if (!g_got_audio) {
          g_got_audio = 1;
          logline("INFO 最初の音の塊 %zu フレーム %uch", n, ch);
        }
        write_all(g_mono, n * sizeof(float));
      });
  if (st != noErr || !g_proc) {
    g_proc = NULL;
    char m[128];
    snprintf(m, sizeof m, "ERROR 受け取りを作れません (OSStatus %d)", (int)st);
    finish(5, m);
  }
  st = AudioDeviceStart(g_agg, g_proc);
  if (st != noErr) {
    char m[128];
    snprintf(m, sizeof m, "ERROR 集約デバイスを開始できません (OSStatus %d)", (int)st);
    finish(5, m);
  }
  char ub[256];
  cfstr_to_c(outUID, ub, sizeof ub);
  logline("READY rate=%d tap_rate=%d tap_ch=%u excluded=%d out=%s", (int)rate, (int)tf.mSampleRate,
          tf.mChannelsPerFrame, g_nexcl, ub);

  // 出力先(ヘッドホンを挿した等)が変わったら作り直してもらう(集約デバイスの時計が古いまま)
  AudioObjectPropertyAddress da = addr(kAudioHardwarePropertyDefaultOutputDevice);
  AudioObjectAddPropertyListenerBlock(kAudioObjectSystemObject, &da, dispatch_get_main_queue(),
      ^(UInt32 n, const AudioObjectPropertyAddress *a) {
        (void)n; (void)a;
        finish(6, "ERROR OUTPUT_CHANGED 出力先が変わりました");
      });

  // 読み上げ(afplay / say)は1回ごとに新しいプロセス: 0.3 秒ごとに外す相手を更新する。
  // あわせて、途中で許可が取り消されたら知らせて終わる。
  dispatch_source_t timer = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0,
                                                   dispatch_get_main_queue());
  dispatch_source_set_timer(timer, dispatch_time(DISPATCH_TIME_NOW, 300 * NSEC_PER_MSEC),
                            300 * NSEC_PER_MSEC, 50 * NSEC_PER_MSEC);
  __block int ticks = 0;
  dispatch_source_set_event_handler(timer, ^{
    ticks++;
    refresh_exclusions();
    if (ticks % 10 == 0 && preflight() == PERM_DENIED)
      finish(3, "ERROR PERMISSION_DENIED システムオーディオ録音が取り消されました");
  });
  dispatch_resume(timer);
  dispatch_main();
}

static void *stdin_watch(void *arg) {
  (void)arg;
  char b[256];
  for (;;) {
    ssize_t r = read(0, b, sizeof b);
    if (r == 0 || (r < 0 && errno != EINTR)) finish(0, NULL);
  }
  return NULL;
}

int main(int argc, char **argv) {
  signal(SIGPIPE, SIG_IGN);
  g_roots[g_nroots++] = getpid();
  g_roots[g_nroots++] = getppid();
  double tone = -1;
  int check = 0, list = 0;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--exclude-tree") && i + 1 < argc) {
      if (g_nroots < MAX_ROOTS) g_roots[g_nroots++] = (pid_t)atoi(argv[++i]);
    } else if (!strcmp(argv[i], "--tone") && i + 1 < argc) {
      tone = atof(argv[++i]);
    } else if (!strcmp(argv[i], "--check")) {
      check = 1;
    } else if (!strcmp(argv[i], "--list")) {
      list = 1;
    } else {
      logline("ERROR 知らない引数 %s", argv[i]);
      return 2;
    }
  }
  if (check) {
    printf("PERMISSION %s\n", PERM_NAMES[preflight()]);
    return 0;
  }
  if (list) return run_list();
  if (tone >= 0) return run_tone(tone);

  // 標準入力が閉じたら(親の Python が終わった)片付けて終わる。SIGTERM なども同じ
  pthread_t th;
  pthread_create(&th, NULL, stdin_watch, NULL);
  int sigs[] = {SIGTERM, SIGINT, SIGHUP};
  for (int i = 0; i < 3; i++) {
    signal(sigs[i], SIG_IGN);
    dispatch_source_t s = dispatch_source_create(DISPATCH_SOURCE_TYPE_SIGNAL, (uintptr_t)sigs[i], 0,
                                                 dispatch_get_main_queue());
    dispatch_source_set_event_handler(s, ^{ finish(0, NULL); });
    dispatch_resume(s);
  }
  int rc = run_capture();
  finish(rc, NULL);
}
