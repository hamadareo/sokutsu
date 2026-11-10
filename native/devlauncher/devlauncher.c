/* 手元用 /Applications/Sokutsu.app(scripts/install_app.sh)の起動役 Contents/MacOS/Sokutsu。
 *
 * 見つけた問題(11/08「音声が入っていない」): 起動役がシェルスクリプトで、最後に Python へ exec していた。
 * exec するとプロセスの実体は Python.app(org.python.python)になり、そこにはマイクを使う理由の文
 * (NSMicrophoneUsageDescription)が無いので、macOS は許可を聞く窓を出さずに無音を渡す
 * (システム設定のマイクの一覧にも Sokutsu が載らない)。
 * 直し方: この起動役は exec せずに run.sh を子として起こし、終わるまで待つ。子(Python)のマイクの
 * 許可は「起こした側」= Sokutsu.app の名前と理由の文で聞かれ、一覧にも Sokutsu として載る
 * (ターミナルから ./run.sh で動かすとターミナルの許可になるのと同じ仕組み)。
 * 記録: ~/Library/Logs/Sokutsu/run.log(前回の分は run.prev.log)。
 */
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <signal.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#ifndef RUN_SH
#error "RUN_SH (the absolute path of run.sh) must be given with -DRUN_SH=..."
#endif

extern char **environ;
static pid_t child = 0;

static void forward(int sig) {
    if (child > 0) kill(child, sig);
}

static void log_to_file(void) {
    const char *home = getenv("HOME");
    if (!home || !*home) return;
    char dir[PATH_MAX], log[PATH_MAX], prev[PATH_MAX];
    snprintf(dir, sizeof dir, "%s/Library/Logs", home);
    mkdir(dir, 0755);
    snprintf(dir, sizeof dir, "%s/Library/Logs/Sokutsu", home);
    mkdir(dir, 0755);
    snprintf(log, sizeof log, "%s/run.log", dir);
    snprintf(prev, sizeof prev, "%s/run.prev.log", dir);
    rename(log, prev);
    int fd = open(log, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return;
    dup2(fd, 1);
    dup2(fd, 2);
    close(fd);
}

int main(void) {
    const char *path = getenv("PATH");
    char newpath[8192];
    snprintf(newpath, sizeof newpath, "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:%s",
             path ? path : "");
    setenv("PATH", newpath, 1);
    log_to_file();

    /* arch -arm64: シェルの起動役だと Rosetta で立ち上がり arm64 の numpy が読めなかった(11/06) */
    char *argv[] = {"/usr/bin/arch", "-arm64", "/bin/bash", RUN_SH, NULL};
    int rc = posix_spawn(&child, argv[0], NULL, NULL, argv, environ);
    if (rc != 0) {
        fprintf(stderr, "Sokutsu: run.sh を起こせませんでした (%s)\n", strerror(rc));
        return 1;
    }
    signal(SIGTERM, forward);
    signal(SIGINT, forward);
    signal(SIGHUP, forward);
    int st = 0;
    while (waitpid(child, &st, 0) < 0) {
        if (errno != EINTR) return 1;   /* シグナルで起こされただけなら待ち直す */
    }
    return WIFEXITED(st) ? WEXITSTATUS(st) : 1;
}
