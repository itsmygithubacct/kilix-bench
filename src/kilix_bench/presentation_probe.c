/* Bounded Kitty graphics probe. Links kitty-framebuffer; does not copy it.
 * A fixture reply is not a terminal. A Kitty reply is not scanout.
 * GPL-3.0-or-later; see LICENSE. */
#define _POSIX_C_SOURCE 200809L
#include "kitty_framebuffer.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <pty.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>

/* Matches kitty-framebuffer's graphics query image id. */
static const char graphics_reply[] = "\033_Gi=31;OK\033\\\033[?62;4c";

static uint64_t now_ns(void)
{
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
        return 0;
    }
    return (uint64_t)now.tv_sec * UINT64_C(1000000000) + (uint64_t)now.tv_nsec;
}

static int shm_count(pid_t pid)
{
    char prefix[64];
    DIR *directory;
    struct dirent *entry;
    int count = 0;
    int length = snprintf(prefix, sizeof(prefix), "kilix-fb-%ld-", (long)pid);
    if (length < 0 || (size_t)length >= sizeof(prefix)) {
        return -1;
    }
    directory = opendir("/dev/shm");
    if (directory == NULL) {
        return -1;
    }
    while ((entry = readdir(directory)) != NULL) {
        if (strncmp(entry->d_name, prefix, (size_t)length) == 0) {
            count++;
        }
    }
    closedir(directory);
    return count;
}

static void options_for_probe(kittyfb_options *options)
{
    kittyfb_options_init(options);
    options->manage_alt_screen = false;
    options->hide_cursor = false;
    options->install_winch_handler = false;
    options->probe_graphics = true;
    options->probe_timeout_ms = 2000;
    options->transport = KITTYFB_TRANSPORT_SHM;
    options->shm_slots = 2;
    options->min_width = 64;
    options->min_height = 32;
    options->max_width = 320;
    options->max_height = 200;
}

struct reply_job {
    int master;
};

static void *reply_main(void *opaque)
{
    struct reply_job *job = opaque;
    char seen[512];
    size_t used = 0;
    struct pollfd descriptor = {job->master, POLLIN, 0};
    uint64_t deadline = now_ns() + UINT64_C(3000000000);
    while (now_ns() < deadline && used + 1 < sizeof(seen)) {
        int ready = poll(&descriptor, 1, 50);
        if (ready < 0 && errno == EINTR) {
            continue;
        }
        if (ready <= 0) {
            continue;
        }
        ssize_t count = read(job->master, seen + used, sizeof(seen) - 1 - used);
        if (count > 0) {
            used += (size_t)count;
            seen[used] = '\0';
            if (strstr(seen, "a=q") != NULL) {
                (void)write(job->master, graphics_reply, sizeof(graphics_reply) - 1);
                return NULL;
            }
        }
    }
    return NULL;
}

static int start_session(kittyfb_session *session, int input, int output)
{
    kittyfb_options options;
    options_for_probe(&options);
    kittyfb_session_init(session);
    return kittyfb_start(session, input, output, &options);
}

static void json_string(FILE *file, const char *text)
{
    fputc('"', file);
    for (const unsigned char *cursor = (const unsigned char *)text; *cursor; cursor++) {
        if (*cursor == '"' || *cursor == '\\') {
            fprintf(file, "\\%c", *cursor);
        } else if (*cursor >= 32 && *cursor < 127) {
            fputc(*cursor, file);
        }
    }
    fputc('"', file);
}

static void write_record(FILE *file, const char *peer, const char *query,
                         const char *shm, uint64_t session_ns,
                         uint64_t enqueue_ns, uint64_t written_ns,
                         uint64_t consumed_ns, const char *transport,
                         const char *reason)
{
    fprintf(file,
            "{\"schema\":1,\"peer\":");
    json_string(file, peer);
    fprintf(file, ",\"graphics_query\":");
    json_string(file, query);
    fprintf(file, ",\"shm_consumed\":");
    json_string(file, shm);
    fprintf(file, ",\"session_start_ns\":%" PRIu64, session_ns);
    fprintf(file, ",\"frame_enqueue_ns\":%" PRIu64, enqueue_ns);
    if (written_ns == 0) {
        fprintf(file, ",\"frame_written_ns\":null");
    } else {
        fprintf(file, ",\"frame_written_ns\":%" PRIu64, written_ns);
    }
    if (consumed_ns == 0) {
        fprintf(file, ",\"shm_consumed_ns\":null");
    } else {
        fprintf(file, ",\"shm_consumed_ns\":%" PRIu64, consumed_ns);
    }
    fprintf(file, ",\"transport\":");
    json_string(file, transport);
    fprintf(file,
            ",\"display_latency\":\"unavailable\""
            ",\"scanout\":\"unavailable\""
            ",\"gpu_latency\":\"unavailable\",\"reason\":");
    json_string(file, reason);
    fprintf(file, "}\n");
}

static const char *transport_name(kittyfb_transport transport)
{
    if (transport == KITTYFB_TRANSPORT_SHM) {
        return "shm";
    }
    if (transport == KITTYFB_TRANSPORT_INLINE) {
        return "inline";
    }
    return "unknown";
}

static void present_and_watch(kittyfb_session *session, const char *peer, FILE *out,
                              uint64_t session_ns)
{
    int width = kittyfb_width(session);
    int height = kittyfb_height(session);
    size_t pixels = (size_t)width * (size_t)height;
    uint8_t *frame = calloc(pixels, 4);
    kittyfb_stats before, after;
    uint64_t enqueue, written = 0, consumed = 0;
    const char *shm = "not-a-consumer";
    const char *transport = transport_name(kittyfb_active_transport(session));
    int saw_name = 0;
    if (frame == NULL || width <= 0 || height <= 0) {
        write_record(out, peer, "answered", "not-presented", session_ns, 0, 0, 0, transport,
                     "framebuffer size was unusable");
        free(frame);
        return;
    }
    for (size_t index = 0; index < pixels; index++) {
        frame[index * 4 + 3] = 255;
    }
    kittyfb_get_stats(session, &before);
    uint64_t started = now_ns();
    if (!kittyfb_present(session, frame, width, height)) {
        write_record(out, peer, "answered", "present-failed", session_ns, now_ns() - started,
                     0, 0, transport, "present returned failure");
        free(frame);
        return;
    }
    enqueue = now_ns() - started;
    if (strcmp(peer, "kilix-kitty") == 0 && strcmp(transport, "shm") != 0) {
        shm = "not-shared-memory";
    } else if (strcmp(peer, "kilix-kitty") == 0) {
        uint64_t deadline = now_ns() + UINT64_C(2000000000);
        int spins = 0;
        shm = "timeout";
        while (now_ns() < deadline) {
            kittyfb_get_stats(session, &after);
            if (written == 0 && after.frames_encoded > before.frames_encoded) {
                written = now_ns() - started;
            }
            int names = shm_count(getpid());
            if (names > 0) {
                saw_name = 1;
            }
            if (saw_name && names == 0 && written != 0) {
                consumed = now_ns() - started;
                shm = "observed";
                break;
            }
            /* The name can appear and disappear in one presenter turn.
             * Spin first so that turn is not missed, then back off. */
            if (spins < 200) {
                spins++;
            } else {
                struct timespec pause = {0, spins < 4000 ? 200000L : 1000000L};
                nanosleep(&pause, NULL);
                spins++;
            }
        }
    }
    write_record(out, peer, "answered", shm, session_ns, enqueue, written, consumed, transport,
                 strcmp(shm, "observed") == 0
                     ? "shared-memory name disappeared after the write"
                     : "consumer read was not observed");
    free(frame);
}

static int run_fixture(void)
{
    int master = -1, slave = -1;
    struct winsize size = {24, 80, 720, 432};
    kittyfb_session session;
    pthread_t thread;
    struct reply_job job;
    uint64_t started, elapsed;
    int result;
    if (openpty(&master, &slave, NULL, NULL, &size) != 0) {
        return 1;
    }
    job.master = master;
    if (pthread_create(&thread, NULL, reply_main, &job) != 0) {
        return 1;
    }
    started = now_ns();
    result = start_session(&session, slave, slave);
    elapsed = now_ns() - started;
    pthread_join(thread, NULL);
    if (result != 0) {
        write_record(stdout, "protocol-fixture", "error", "not-a-consumer",
                     elapsed, 0, 0, 0, "", "graphics query was not accepted");
        return 1;
    }
    write_record(stdout, "protocol-fixture", "answered", "not-a-consumer",
                 elapsed, 0, 0, 0, transport_name(kittyfb_active_transport(&session)),
                 "in-process reply; not a terminal acknowledgement");
    kittyfb_stop(&session);
    close(master);
    close(slave);
    return 0;
}

static int run_live(const char *path)
{
    kittyfb_session session;
    char temporary[1024];
    FILE *file;
    uint64_t started, elapsed;
    int result;
    int length = snprintf(temporary, sizeof(temporary), "%s.partial", path);
    if (length < 0 || (size_t)length >= sizeof(temporary)) {
        return 1;
    }
    started = now_ns();
    result = start_session(&session, STDIN_FILENO, STDOUT_FILENO);
    elapsed = now_ns() - started;
    file = fopen(temporary, "w");
    if (file == NULL) {
        return 1;
    }
    if (result != 0) {
        write_record(file, "kilix-kitty", "error", "not-presented", elapsed, 0, 0, 0,
                     "", "graphics query failed");
        fclose(file);
        rename(temporary, path);
        return 1;
    }
    /* session_start_ns is the whole start, including the query. */
    present_and_watch(&session, "kilix-kitty", file, elapsed);
    fclose(file);
    rename(temporary, path);
    kittyfb_stop(&session);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc == 2 && strcmp(argv[1], "fixture") == 0) {
        return run_fixture();
    }
    if (argc == 3 && strcmp(argv[1], "live") == 0) {
        return run_live(argv[2]);
    }
    fprintf(stderr, "usage: presentation-probe fixture | live RESULT\n");
    return 2;
}
