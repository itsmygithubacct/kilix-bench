/* Native loops for desktop-face experiments. One fresh process per job.
 * The calls are the ones a desktop face makes through Neuronte's public API:
 * neuronte_open_file and neuronte_set_binary when the face opens, then for
 * each character neuronte_text_width (regular and bold) and one
 * neuronte_draw_text at the face's em into a cleared ARGB canvas. Coverage
 * cases call the public uncached neuronte_coverage at the same em x 2em cell.
 * No window, terminal, FreeType or desktop code is involved.
 * GPL-3.0-or-later; see LICENSE. */
#define _POSIX_C_SOURCE 200809L
#include <dlfcn.h>
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

typedef struct neuronte neuronte;
_Static_assert(sizeof(float) == sizeof(uint32_t), "face outputs require 32-bit floats");

typedef struct {
    void *module;
    neuronte *net;
    int (*open)(const char *, neuronte **);
    void (*close)(neuronte *);
    int (*set_binary)(neuronte *, int);
    int (*coverage)(const neuronte *, uint32_t, int, int, int, float *);
    int (*text_width)(neuronte *, const char *, size_t, int, int);
    int (*draw_text)(neuronte *, uint32_t *, int, int, int, int, int, int, int, const char *, size_t, uint32_t);
    int (*route)(const neuronte *, uint32_t);
    int em, binary, cw, ch, canvas_w;
} Face;

static void fail(const char *message)
{
    fprintf(stderr, "face worker: %s\n", message);
    exit(1);
}

static uint64_t ns(clockid_t clock)
{
    struct timespec t;
    if (clock_gettime(clock, &t)) fail("clock unavailable");
    return (uint64_t)t.tv_sec * UINT64_C(1000000000) + (uint64_t)t.tv_nsec;
}

static int number(const char *text, int low, int high)
{
    char *end;
    errno = 0;
    long value = strtol(text, &end, 10);
    if (errno || !*text || *end || value < low || value > high)
        fail("argument outside its bounds");
    return (int)value;
}

static void *allocate(size_t count, size_t size)
{
    void *p = calloc(count, size);
    if (!p) fail("allocation failed");
    return p;
}

static void symbol(void *module, const char *name, void *target, size_t size)
{
    void *address = dlsym(module, name);
    if (!address || size != sizeof address) fail("Neuronte API unavailable");
    memcpy(target, &address, size);
}

/* UTF-8 of one scalar; the corpus holds valid scalars only. */
static size_t utf8(uint32_t s, char *out)
{
    if (s < 0x80) { out[0] = (char)s; return 1; }
    if (s < 0x800) { out[0] = (char)(0xC0 | s >> 6); out[1] = (char)(0x80 | (s & 63)); return 2; }
    if (s < 0x10000) {
        out[0] = (char)(0xE0 | s >> 12); out[1] = (char)(0x80 | (s >> 6 & 63));
        out[2] = (char)(0x80 | (s & 63)); return 3;
    }
    out[0] = (char)(0xF0 | s >> 18); out[1] = (char)(0x80 | (s >> 12 & 63));
    out[2] = (char)(0x80 | (s >> 6 & 63)); out[3] = (char)(0x80 | (s & 63)); return 4;
}

/* What the desktop does for one character: both advances, then the regular
 * letter drawn alone at the pen, x 0, y 0, white, into a cleared canvas
 * em + 2 pixels wide and 2 em high (the whole cell, so every pixel the
 * library can set is kept). Returns the regular advance. */
static int face_draw(Face *f, uint32_t scalar, uint32_t *canvas, int *bold_advance)
{
    char text[4];
    size_t n = utf8(scalar, text);
    int adv = f->text_width(f->net, text, n, 0, f->em);
    int badv = f->text_width(f->net, text, n, 1, f->em);
    if (adv < 0 || badv < 0) fail("text width failed");
    memset(canvas, 0, (size_t)f->canvas_w * f->ch * sizeof *canvas);
    if (f->draw_text(f->net, canvas, f->canvas_w, f->canvas_w, f->ch, 0, 0, f->em, 0, text, n, UINT32_MAX) != adv)
        fail("draw_text moved the pen by other than the regular advance");
    *bold_advance = badv;
    return adv;
}

int main(int argc, char **argv)
{
    /* library asset mode em binary calls corpus.txt output.bin */
    if (argc != 9) fail("usage: library asset mode em binary calls corpus.txt output.bin");
    Face f = {0};
    const char *mode = argv[3];
    f.em = number(argv[4], 1, 128);
    f.binary = number(argv[5], 0, 1);
    int calls = number(argv[6], 1, 1000000);
    f.cw = f.em; f.ch = 2 * f.em; f.canvas_w = f.em + 2;
    uint32_t scalars[4096];
    int count = 0, code;
    unsigned scalar;
    FILE *corpus = fopen(argv[7], "r");
    if (!corpus) fail("cannot read corpus");
    while ((code = fscanf(corpus, "%x", &scalar)) == 1) {
        if (count == 4096 || scalar > 0x10ffff || (scalar >= 0xd800 && scalar <= 0xdfff))
            fail("invalid corpus scalar or too many scalars");
        scalars[count++] = scalar;
    }
    if (ferror(corpus) || code != EOF || !count) fail("invalid or empty corpus");
    fclose(corpus);

    /* Opening: dlopen, the weights, the binary switch and the buffers. */
    uint64_t init_cpu_started = ns(CLOCK_PROCESS_CPUTIME_ID);
    uint64_t started = ns(CLOCK_MONOTONIC);
    f.module = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!f.module) fail("cannot load Neuronte library");
    symbol(f.module, "neuronte_open_file", &f.open, sizeof f.open);
    symbol(f.module, "neuronte_close", &f.close, sizeof f.close);
    symbol(f.module, "neuronte_set_binary", &f.set_binary, sizeof f.set_binary);
    symbol(f.module, "neuronte_coverage", &f.coverage, sizeof f.coverage);
    symbol(f.module, "neuronte_text_width", &f.text_width, sizeof f.text_width);
    symbol(f.module, "neuronte_draw_text", &f.draw_text, sizeof f.draw_text);
    symbol(f.module, "neuronte_route", &f.route, sizeof f.route);
    if (f.open(argv[2], &f.net)) fail("cannot open Neuronte face");
    if (f.set_binary(f.net, f.binary)) fail("cannot set binary drawing");
    size_t pixels = (size_t)f.cw * f.ch, canvas_pixels = (size_t)f.canvas_w * f.ch;
    float *coverage = allocate(pixels, sizeof *coverage);
    uint32_t *canvas = allocate(canvas_pixels, sizeof *canvas);
    uint64_t init_ns = ns(CLOCK_MONOTONIC) - started;
    uint64_t init_cpu_ns = ns(CLOCK_PROCESS_CPUTIME_ID) - init_cpu_started;

    if (!strcmp(mode, "open")) {
        printf("{\"initialize_ns\":%" PRIu64 ",\"initialize_cpu_ns\":%" PRIu64 "}\n", init_ns, init_cpu_ns);
        goto done;
    }
    if (!strcmp(mode, "routes")) {
        printf("{\"routes\":[");
        for (int i = 0; i < count; i++) {
            int index = f.route(f.net, scalars[i]);
            if (index < 0) fail("cannot route face scalar");
            printf("%s%d", i ? "," : "", index);
        }
        puts("]}");
        goto done;
    }
    int draw = !strcmp(mode, "draw-cold") || !strcmp(mode, "draw-warm");
    int validate = !strcmp(mode, "validate");
    int direct = !strcmp(mode, "coverage");
    if (!draw && !validate && !direct) fail("unknown face mode");
    if (calls % count) fail("face batches must cover complete corpus passes");
    if (validate || !strcmp(mode, "draw-cold")) calls = count;
    /* validate: every character's coverage, then every character drawn, each
     * from an empty draw cache in this process (the first draw of each). */
    size_t out_bytes = validate ? (size_t)count * (pixels * sizeof(float) + canvas_pixels * sizeof(uint32_t) + 8)
                                : direct ? pixels * sizeof(float) : canvas_pixels * sizeof(uint32_t);
    unsigned char *out = allocate(out_bytes, 1);
    int adv = 0, badv = 0, priming = 0;
    uint64_t first_ns = 0, prime_ns = 0;
    if (direct) {
        /* The first direct coverage is timed apart; it cannot fill the draw cache. */
        started = ns(CLOCK_MONOTONIC);
        if (f.coverage(f.net, scalars[0], f.cw, f.ch, 0, coverage)) fail("first coverage failed");
        first_ns = ns(CLOCK_MONOTONIC) - started;
    }
    if (!strcmp(mode, "draw-warm")) {
        started = ns(CLOCK_MONOTONIC);
        for (int i = 0; i < count; i++) face_draw(&f, scalars[i], canvas, &badv);
        prime_ns = ns(CLOCK_MONOTONIC) - started;
        priming = count;
    }
    if (validate) {
        unsigned char *p = out;
        for (int i = 0; i < count; i++) {
            if (f.coverage(f.net, scalars[i], f.cw, f.ch, 0, coverage)) fail("coverage failed");
            for (size_t k = 0; k < pixels; k++)
                if (!isfinite(coverage[k]) || coverage[k] < 0 || coverage[k] > 1) fail("invalid coverage");
            memcpy(p, coverage, pixels * sizeof(float));
            p += pixels * sizeof(float);
        }
        for (int i = 0; i < count; i++) {
            int32_t a[2];
            a[0] = face_draw(&f, scalars[i], canvas, &badv);
            a[1] = badv;
            memcpy(p, canvas, canvas_pixels * sizeof *canvas);
            p += canvas_pixels * sizeof *canvas;
            memcpy(p, a, 8);
            p += 8;
        }
        calls = count;
    } else {
        uint64_t cpu_started = ns(CLOCK_PROCESS_CPUTIME_ID);
        started = ns(CLOCK_MONOTONIC);
        for (int i = 0; i < calls; i++) {
            if (draw) adv = face_draw(&f, scalars[i % count], canvas, &badv);
            else if (f.coverage(f.net, scalars[i % count], f.cw, f.ch, 0, coverage)) fail("coverage failed");
        }
        uint64_t elapsed = ns(CLOCK_MONOTONIC) - started;
        uint64_t cpu_elapsed = ns(CLOCK_PROCESS_CPUTIME_ID) - cpu_started;
        (void)adv;
        if (direct) {
            for (size_t k = 0; k < pixels; k++)
                if (!isfinite(coverage[k]) || coverage[k] < 0 || coverage[k] > 1) fail("invalid coverage");
            memcpy(out, coverage, out_bytes);
        } else memcpy(out, canvas, out_bytes);
        FILE *file = fopen(argv[8], "wb");
        if (!file || fwrite(out, 1, out_bytes, file) != out_bytes || fclose(file)) fail("cannot write face pixels");
        struct rusage usage;
        if (getrusage(RUSAGE_SELF, &usage)) fail("cannot read process counters");
        printf("{\"elapsed_ns\":%" PRIu64 ",\"cpu_ns\":%" PRIu64
               ",\"initialize_ns\":%" PRIu64 ",\"initialize_cpu_ns\":%" PRIu64
               ",\"first_coverage_ns\":%" PRIu64 ",\"prime_ns\":%" PRIu64
               ",\"priming_calls\":%d,\"calls\":%d,\"coverage_calls\":%d,\"draw_calls\":%d"
               ",\"width_calls\":%d,\"output_bytes\":%zu"
               ",\"peak_rss_kib\":%ld,\"minor_faults\":%ld,\"major_faults\":%ld}\n",
               elapsed, cpu_elapsed, init_ns, init_cpu_ns, first_ns, prime_ns,
               priming, calls, direct ? calls : 0, draw ? calls : 0, draw ? 2 * calls : 0,
               out_bytes, usage.ru_maxrss, usage.ru_minflt, usage.ru_majflt);
        goto done_out;
    }
    {
        FILE *file = fopen(argv[8], "wb");
        if (!file || fwrite(out, 1, out_bytes, file) != out_bytes || fclose(file)) fail("cannot write face pixels");
        printf("{\"calls\":%d,\"coverage_calls\":%d,\"draw_calls\":%d,\"output_bytes\":%zu}\n",
               calls, count, count, out_bytes);
    }
done_out:
    free(out);
done:
    free(coverage);
    free(canvas);
    f.close(f.net);
    dlclose(f.module);
    return 0;
}
