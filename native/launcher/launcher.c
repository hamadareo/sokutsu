/* 配る Sokutsu.app の起動役(Contents/MacOS/Sokutsu)。
 *
 * 同梱の libpython(Contents/Resources/python/lib/libpython3.12.dylib)を読み込んで Py_BytesMain を呼ぶ。
 * Python の実行ファイルへ exec すると「Python」という別のアプリに見え、メニュー名・アイコン・
 * 許可の文言が Sokutsu にならないので、プロセスの実体は Sokutsu.app のままにする。
 *   引数なし(Finder・Dock から)     → -P -B -s -m tsuyaku.dist_main。記録は ~/Library/Logs/Sokutsu/run.log
 *   引数あり(sys.executable として)  → そのまま Python へ(例: -m tsuyaku.export pdf ...)
 * .app の中に .pyc を書くと署名が壊れるので PYTHONDONTWRITEBYTECODE=1(.pyc はビルドで作り済み)。
 */
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

extern int Py_BytesMain(int argc, char **argv);

static void mkdirs(const char *path) {
    char buf[PATH_MAX];
    strlcpy(buf, path, sizeof buf);
    for (char *p = buf + 1; *p; p++) {
        if (*p == '/') {
            *p = 0;
            mkdir(buf, 0755);
            *p = '/';
        }
    }
    mkdir(buf, 0755);
}

static void log_to_file(void) {
    const char *home = getenv("HOME");
    if (!home || !*home) return;
    char dir[PATH_MAX], log[PATH_MAX], prev[PATH_MAX];
    snprintf(dir, sizeof dir, "%s/Library/Logs/Sokutsu", home);
    mkdirs(dir);
    snprintf(log, sizeof log, "%s/run.log", dir);
    snprintf(prev, sizeof prev, "%s/run.prev.log", dir);
    rename(log, prev);
    int fd = open(log, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return;
    dup2(fd, 1);
    dup2(fd, 2);
    close(fd);
}

int main(int argc, char **argv) {
    char exe[PATH_MAX], real[PATH_MAX];
    uint32_t n = sizeof exe;
    if (_NSGetExecutablePath(exe, &n) != 0 || !realpath(exe, real)) {
        fprintf(stderr, "Sokutsu: cannot find itself\n");
        return 1;
    }
    /* real = …/Sokutsu.app/Contents/MacOS/Sokutsu → …/Sokutsu.app/Contents */
    char contents[PATH_MAX];
    strlcpy(contents, real, sizeof contents);
    for (int i = 0; i < 2; i++) {
        char *s = strrchr(contents, '/');
        if (!s) return 1;
        *s = 0;
    }
    char home[PATH_MAX];
    snprintf(home, sizeof home, "%s/Resources/python", contents);
    setenv("PYTHONHOME", home, 1);
    setenv("PYTHONDONTWRITEBYTECODE", "1", 1);
    setenv("PYTHONNOUSERSITE", "1", 1);
    setenv("PYTHONUTF8", "1", 1);
    unsetenv("PYTHONPATH");
    unsetenv("PYTHONSTARTUP");
    unsetenv("PYTHONEXECUTABLE");
    unsetenv("__PYVENV_LAUNCHER__");
    /* Finder から開くと PATH が最小。say・afplay・osascript は /usr/bin にある */
    const char *path = getenv("PATH");
    if (!path || !*path) setenv("PATH", "/usr/bin:/bin:/usr/sbin:/sbin", 1);

    int app = argc < 2 || strncmp(argv[1], "-psn_", 5) == 0;
    if (!app) return Py_BytesMain(argc, argv);
    log_to_file();
    char *args[] = {argv[0], "-P", "-B", "-s", "-m", "tsuyaku.dist_main", NULL};
    return Py_BytesMain(6, args);
}
