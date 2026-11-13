/* Native loops for the font suite. Both engines return a cell-sized u8 mask
 * and use the same cache and compositor. No window or terminal is opened.
 * GPL-3.0-or-later; see LICENSE. */
#define _POSIX_C_SOURCE 200809L
#include <ft2build.h>
#include FT_FREETYPE_H
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
_Static_assert(sizeof(float) == sizeof(uint32_t), "expert outputs require 32-bit floats");
typedef struct {
    void *module;
    neuronte *net;
    neuronte *right;
    int (*open)(const char *, neuronte **);
    void (*close)(neuronte *);
    int (*coverage)(const neuronte *, uint32_t, int, int, int, float *);
    FT_Library ft;
    FT_Face face;
    int w, h, samples, baseline;
    FT_Int32 flags;
    float *coverage_buffer;
    float *left_buffer, *right_buffer;
    uint64_t raster_calls, clipped;
} Engine;

static void fail(const char *message)
{
    fprintf(stderr, "font worker: %s\n", message);
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

static void open_engine(Engine *e, const char *kind, const char *library,
                        const char *asset, int hinting)
{
    if (!strcmp(kind, "neuronte")) {
        e->module = dlopen(library, RTLD_NOW | RTLD_LOCAL);
        if (!e->module) fail("cannot load Neuronte library");
        symbol(e->module, "neuronte_open_file", &e->open, sizeof e->open);
        symbol(e->module, "neuronte_close", &e->close, sizeof e->close);
        symbol(e->module, "neuronte_coverage", &e->coverage, sizeof e->coverage);
        if (e->open(asset, &e->net)) fail("cannot open Neuronte weights");
        e->coverage_buffer = allocate((size_t)e->w * e->h, sizeof(float));
    } else if (!strcmp(kind, "freetype")) {
        FT_Size_RequestRec request = {FT_SIZE_REQUEST_TYPE_CELL,
                                      e->w * 64, e->h * 64, 0, 0};
        if (FT_Init_FreeType(&e->ft) || FT_New_Face(e->ft, asset, 0, &e->face))
            fail("cannot open FreeType face");
        if (!FT_IS_SCALABLE(e->face) || !FT_IS_FIXED_WIDTH(e->face))
            fail("terminal-cell comparison requires a scalable monospace font");
        if (FT_Select_Charmap(e->face, FT_ENCODING_UNICODE) ||
            FT_Request_Size(e->face, &request)) fail("cannot size Unicode face");
        e->baseline = (int)((e->face->size->metrics.ascender + 63) / 64);
        e->flags = FT_LOAD_RENDER | FT_LOAD_NO_BITMAP | FT_LOAD_TARGET_NORMAL;
        if (!hinting) e->flags |= FT_LOAD_NO_HINTING | FT_LOAD_NO_AUTOHINT;
    } else fail("unknown engine");
}

static void open_right(Engine *e, const char *asset)
{
    if (!e->net || e->w % 2) fail("split glyphs require Neuronte and an even width");
    if (e->open(asset, &e->right)) fail("cannot open right-column weights");
    size_t half = (size_t)(e->w / 2) * e->h;
    e->left_buffer = allocate(half, sizeof(float));
    e->right_buffer = allocate(half, sizeof(float));
    int (*set_samples)(neuronte *, int);
    symbol(e->module, "neuronte_set_samples", &set_samples, sizeof set_samples);
    int samples = e->samples ? e->samples : e->w < 24 ? 2 : 1;
    if (set_samples(e->net, samples) || set_samples(e->right, samples))
        fail("cannot set split glyph sampling");
}

static int expert_coverage(Engine *e, uint32_t scalar, float *output)
{
    if (!e->right)
        return e->coverage(e->net, scalar, e->w, e->h, e->samples, output);
    if (scalar < 0x2800 || scalar > 0x28ff) fail("split glyph corpus is not Braille");
    uint32_t bits = scalar - 0x2800;
    int half = e->w / 2, samples = e->samples ? e->samples : e->w < 24 ? 2 : 1;
    int rc = e->coverage(e->net, 0x2800 | (bits & 0x47), half, e->h, samples, e->left_buffer);
    if (rc) return rc;
    rc = e->coverage(e->right, 0x2800 | (bits & 0xb8), half, e->h, samples, e->right_buffer);
    if (rc) return rc;
    for (int y = 0; y < e->h; y++) {
        memcpy(output + (size_t)y * e->w, e->left_buffer + (size_t)y * half, (size_t)half * sizeof(float));
        memcpy(output + (size_t)y * e->w + half, e->right_buffer + (size_t)y * half, (size_t)half * sizeof(float));
    }
    return 0;
}

typedef int (*DrawScalar)(neuronte *, uint32_t *, int, int, int, int, int,
                          int, int, uint32_t, uint32_t);

static void expert_draw(Engine *e, DrawScalar draw, uint32_t scalar, uint32_t *canvas)
{
    memset(canvas, 0, (size_t)e->w * e->h * sizeof *canvas);
    if (!e->right) {
        if (draw(e->net, canvas, e->w, e->w, e->h, 0, 0, e->w, e->h, scalar, UINT32_MAX))
            fail("expert drawing failed");
        return;
    }
    if (scalar < 0x2800 || scalar > 0x28ff) fail("split glyph corpus is not Braille");
    uint32_t bits = scalar - 0x2800;
    int half = e->w / 2;
    if (draw(e->net, canvas, e->w, e->w, e->h, 0, 0, half, e->h,
             0x2800 | (bits & 0x47), UINT32_MAX) ||
        draw(e->right, canvas, e->w, e->w, e->h, half, 0, half, e->h,
             0x2800 | (bits & 0xb8), UINT32_MAX)) fail("split glyph drawing failed");
}

static int supported(Engine *e, uint32_t scalar)
{
    /* Neuronte support comes from a SHA256-bound corpus profile in the parent.
     * A successful route or fallback drawing is never evidence of support. */
    return !e->face || FT_Get_Char_Index(e->face, scalar) != 0;
}

static void raster(Engine *e, uint32_t scalar, unsigned char *out)
{
    size_t pixels = (size_t)e->w * e->h;
    e->raster_calls++;
    memset(out, 0, pixels);
    if (e->net) {
        if (e->coverage(e->net, scalar, e->w, e->h, e->samples,
                        e->coverage_buffer)) fail("Neuronte rasterization failed");
        for (size_t i = 0; i < pixels; i++) {
            float a = e->coverage_buffer[i];
            if (!isfinite(a)) fail("non-finite coverage");
            a = a < 0 ? 0 : a > 1 ? 1 : a;
            out[i] = (unsigned char)(a * 255 + .5f);
        }
        return;
    }
    FT_UInt glyph = FT_Get_Char_Index(e->face, scalar);
    if (!glyph || FT_Load_Glyph(e->face, glyph, e->flags))
        fail("missing glyph or FreeType rasterization failure");
    FT_GlyphSlot slot = e->face->glyph;
    FT_Bitmap *b = &slot->bitmap;
    if (!b->width || !b->rows) return; /* spaces */
    if (b->pixel_mode != FT_PIXEL_MODE_GRAY || b->num_grays != 256)
        fail("expected an 8-bit grayscale outline raster");
    for (unsigned y = 0; y < b->rows; y++) {
        const unsigned char *row = b->buffer + (ptrdiff_t)y * b->pitch;
        for (unsigned x = 0; x < b->width; x++) {
            int dx = slot->bitmap_left + (int)x;
            int dy = e->baseline - slot->bitmap_top + (int)y;
            if (dx < 0 || dy < 0 || dx >= e->w || dy >= e->h) {
                e->clipped += row[x] != 0;
            } else out[(size_t)dy * e->w + dx] = row[x];
        }
    }
}

typedef struct { uint32_t scalar; uint64_t used; unsigned char *mask; } Entry;
typedef struct {
    Entry *entries;
    unsigned char *storage;
    int capacity;
    uint64_t tick, hits, misses;
} Cache;

static const unsigned char *cached(Engine *e, Cache *c, uint32_t scalar)
{
    /* Identical four-way set-associative LRU policy for both engines. */
    unsigned set = (scalar * UINT32_C(2654435761)) % (unsigned)(c->capacity / 4);
    Entry *victim = c->entries + set * 4;
    for (int way = 0; way < 4; way++) {
        Entry *entry = c->entries + set * 4 + way;
        if (entry->used && entry->scalar == scalar) {
            entry->used = ++c->tick;
            c->hits++;
            return entry->mask;
        }
        if (entry->used < victim->used) victim = entry;
    }
    c->misses++;
    raster(e, scalar, victim->mask);
    victim->scalar = scalar;
    victim->used = ++c->tick;
    return victim->mask;
}

static void composite(unsigned char *frame, const unsigned char *mask,
                       int w, int h, int cols, int cell)
{
    int x0 = cell % cols * w, y0 = cell / cols * h;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            unsigned a = mask[(size_t)y * w + x];
            unsigned char value = (unsigned char)((233 * a + 17 * (255 - a) + 127) / 255);
            size_t at = ((size_t)(y0 + y) * cols * w + x0 + x) * 3;
            frame[at] = frame[at + 1] = frame[at + 2] = value;
        }
}

/* The expert suite uses the public float coverage and draw APIs directly.
 * Its draw cases deliberately bypass the comparison suite's u8 cache. */
static void neuronte_case(Engine *e, const char *mode, const uint32_t *scalars,
                         int count, int calls, const char *path,
                         uint64_t init_ns, uint64_t init_cpu_ns)
{
    if (!e->net) fail("expert cases require Neuronte");
    if (!strcmp(mode, "nrt-open")) {
        printf("{\"initialize_ns\":%" PRIu64 ",\"initialize_cpu_ns\":%" PRIu64 "}\n",
               init_ns, init_cpu_ns);
        return;
    }
    if (!strcmp(mode, "nrt-routes") || !strcmp(mode, "nrt-route-bench")) {
        if (e->right) fail("split glyphs execute both models without scalar routing");
        int (*route)(const neuronte *, uint32_t);
        symbol(e->module, "neuronte_route", &route, sizeof route);
        if (!strcmp(mode, "nrt-route-bench")) {
            volatile unsigned sink = 0;
            uint64_t cpu_started = ns(CLOCK_PROCESS_CPUTIME_ID);
            uint64_t started = ns(CLOCK_MONOTONIC);
            for (int i = 0; i < calls; i++) {
                int index = route(e->net, scalars[i % count]);
                if (index < 0) fail("cannot route expert scalar");
                sink += (unsigned)index;
            }
            uint64_t elapsed = ns(CLOCK_MONOTONIC) - started;
            uint64_t cpu_elapsed = ns(CLOCK_PROCESS_CPUTIME_ID) - cpu_started;
            printf("{\"elapsed_ns\":%" PRIu64 ",\"cpu_ns\":%" PRIu64
                   ",\"calls\":%d,\"route_checksum\":%u}\n", elapsed, cpu_elapsed, calls, sink);
            return;
        }
        printf("{\"routes\":[");
        for (int i = 0; i < count; i++) {
            int index = route(e->net, scalars[i]);
            if (index < 0) fail("cannot route expert scalar");
            printf("%s%d", i ? "," : "", index);
        }
        puts("]}");
        return;
    }
    int draw = !strcmp(mode, "nrt-draw-cold") || !strcmp(mode, "nrt-draw-warm");
    int validate = !strcmp(mode, "nrt-validate");
    if (!draw && !validate && strcmp(mode, "nrt-coverage")) fail("unknown expert mode");
    DrawScalar draw_scalar = NULL;
    if (draw) symbol(e->module, "neuronte_draw_scalar", &draw_scalar, sizeof draw_scalar);
    if (calls % count) fail("expert batches must cover complete corpus passes");
    size_t pixels = (size_t)e->w * e->h;
    size_t output_bytes = pixels * (validate ? (size_t)count : 1) * sizeof(float);
    float *output = allocate(output_bytes, 1);
    uint32_t *canvas = allocate(pixels, sizeof *canvas);
    uint64_t started = ns(CLOCK_MONOTONIC);
    if (expert_coverage(e, scalars[0], e->coverage_buffer))
        fail("first expert coverage failed");
    uint64_t first_ns = ns(CLOCK_MONOTONIC) - started;
    int prime = !strcmp(mode, "nrt-draw-warm") || !strcmp(mode, "nrt-coverage");
    started = ns(CLOCK_MONOTONIC);
    for (int i = 0; prime && i < count; i++) {
        if (draw) {
            expert_draw(e, draw_scalar, scalars[i], canvas);
        } else if (expert_coverage(e, scalars[i], e->coverage_buffer))
            fail("expert coverage priming failed");
    }
    uint64_t prime_ns = ns(CLOCK_MONOTONIC) - started;
    if (validate) calls = count;
    uint64_t cpu_started = ns(CLOCK_PROCESS_CPUTIME_ID);
    started = ns(CLOCK_MONOTONIC);
    for (int i = 0; i < calls; i++) {
        if (draw) {
            expert_draw(e, draw_scalar, scalars[i % count], canvas);
        } else if (expert_coverage(e, scalars[i % count], output + (validate ? (size_t)i * pixels : 0)))
            fail("expert coverage failed");
    }
    uint64_t elapsed = ns(CLOCK_MONOTONIC) - started;
    uint64_t cpu_elapsed = ns(CLOCK_PROCESS_CPUTIME_ID) - cpu_started;
    if (!draw)
        for (size_t i = 0; i < output_bytes / sizeof(float); i++)
            if (!isfinite(output[i]) || output[i] < 0 || output[i] > 1)
                fail("invalid expert coverage");
    FILE *file = fopen(path, "wb");
    if (!file || fwrite(draw ? (void *)canvas : (void *)output, 1, output_bytes, file) != output_bytes || fclose(file))
        fail("cannot write expert pixels");
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage)) fail("cannot read expert process counters");
    printf("{\"elapsed_ns\":%" PRIu64 ",\"cpu_ns\":%" PRIu64
           ",\"initialize_ns\":%" PRIu64 ",\"initialize_cpu_ns\":%" PRIu64
           ",\"first_coverage_ns\":%" PRIu64 ",\"prime_ns\":%" PRIu64
           ",\"priming_calls\":%d,\"calls\":%d,\"coverage_calls\":%d,\"draw_calls\":%d"
           ",\"cache_hits\":null,\"cache_misses\":null,\"output_bytes\":%zu"
           ",\"peak_rss_kib\":%ld,\"minor_faults\":%ld,\"major_faults\":%ld}\n",
           elapsed, cpu_elapsed, init_ns, init_cpu_ns, first_ns, prime_ns,
           prime ? count : 0, calls, draw ? 0 : calls * (e->right ? 2 : 1), draw ? calls * (e->right ? 2 : 1) : 0,
           output_bytes, usage.ru_maxrss, usage.ru_minflt, usage.ru_majflt);
    free(canvas); free(output);
}

int main(int argc, char **argv)
{
    /* engine library asset mode width height samples hinting capacity calls
     * columns rows corpus.txt output.bin */
    if (argc != 15 && argc != 16) fail("expected fourteen or fifteen internal arguments");
    Engine e = {0};
    e.w = number(argv[5], 1, 128); e.h = number(argv[6], 1, 256);
    e.samples = number(argv[7], 0, 8);
    int hinting = number(argv[8], 0, 1);
    int capacity = number(argv[9], 4, 4096);
    int calls = number(argv[10], 1, 1000000);
    int cols = number(argv[11], 1, 256), rows = number(argv[12], 1, 256);
    if (capacity % 4 || (size_t)cols * rows * e.w * e.h > 16777216)
        fail("cache or frame geometry outside its bounds");
    uint32_t scalars[4096];
    int count = 0, code;
    unsigned scalar;
    FILE *corpus = fopen(argv[13], "r");
    if (!corpus) fail("cannot read corpus");
    while ((code = fscanf(corpus, "%x", &scalar)) == 1) {
        if (count == 4096 || scalar > 0x10ffff || (scalar >= 0xd800 && scalar <= 0xdfff))
            fail("invalid corpus scalar or too many scalars");
        scalars[count++] = scalar;
    }
    if (ferror(corpus) || code != EOF || !count) fail("invalid or empty corpus");
    fclose(corpus);
    uint64_t init_cpu_started = ns(CLOCK_PROCESS_CPUTIME_ID);
    uint64_t started = ns(CLOCK_MONOTONIC);
    open_engine(&e, argv[1], argv[2], argv[3], hinting);
    if (argc == 16) {
        if (strncmp(argv[4], "nrt-", 4)) fail("split glyphs require an expert case");
        open_right(&e, argv[15]);
    }
    uint64_t init_ns = ns(CLOCK_MONOTONIC) - started;
    uint64_t init_cpu_ns = ns(CLOCK_PROCESS_CPUTIME_ID) - init_cpu_started;
    int major = 0, minor = 0, patch = 0;
    if (e.ft) FT_Library_Version(e.ft, &major, &minor, &patch);
    const char *mode = argv[4];
    if (!strncmp(mode, "nrt-", 4)) {
        neuronte_case(&e, mode, scalars, count, calls, argv[14], init_ns, init_cpu_ns);
        goto done;
    }
    if (!strcmp(mode, "probe")) {
        printf("{\"supported\":[");
        for (int i = 0; i < count; i++) printf("%s%s", i ? "," : "", supported(&e, scalars[i]) ? "true" : "false");
        printf("],\"freetype_version\":\"%d.%d.%d\"}\n", major, minor, patch);
        goto done;
    }
    for (int i = 0; i < count; i++) if (!supported(&e, scalars[i])) fail("unfiltered missing glyph");
    int draw = !strcmp(mode, "draw-warm") || !strcmp(mode, "draw-cold");
    int validate = !strcmp(mode, "validate");
    int load = !strcmp(mode, "load");
    if (!draw && !validate && !load && strcmp(mode, "raster")) fail("unknown mode");
    if (draw && calls % (cols * rows)) fail("draw calls must cover whole frames");
    size_t pixels = (size_t)e.w * e.h;
    unsigned char *mask = allocate(pixels, 1);
    size_t output_bytes = draw ? (size_t)cols * rows * pixels * 3 : validate ? (size_t)count * pixels : pixels;
    unsigned char *output = allocate(output_bytes, 1);
    Cache cache = {0};
    if (draw) {
        cache.capacity = capacity;
        cache.entries = allocate((size_t)capacity, sizeof(Entry));
        cache.storage = allocate((size_t)capacity, pixels);
        for (int i = 0; i < capacity; i++) cache.entries[i].mask = cache.storage + i * pixels;
    }
    started = ns(CLOCK_MONOTONIC);
    raster(&e, scalars[0], mask);
    uint64_t first_ns = ns(CLOCK_MONOTONIC) - started;
    if (!strcmp(mode, "draw-warm"))
        for (int i = 0; i < count; i++) (void)cached(&e, &cache, scalars[i]);
    if (!strcmp(mode, "raster"))
        for (int i = 0; i < count; i++) raster(&e, scalars[i], mask);
    cache.hits = cache.misses = e.raster_calls = e.clipped = 0;
    if (validate || load) calls = count;
    uint64_t cpu_started = ns(CLOCK_PROCESS_CPUTIME_ID);
    started = ns(CLOCK_MONOTONIC);
    for (int i = 0; i < calls; i++) {
        uint32_t current = scalars[i % count];
        if (draw) composite(output, cached(&e, &cache, current), e.w, e.h, cols, i % (cols * rows));
        else raster(&e, current, validate ? output + i * pixels : output);
    }
    uint64_t elapsed = ns(CLOCK_MONOTONIC) - started;
    uint64_t cpu_elapsed = ns(CLOCK_PROCESS_CPUTIME_ID) - cpu_started;
    volatile uint32_t sink = 0;
    started = ns(CLOCK_MONOTONIC);
    for (int i = 0; i < calls; i++) sink ^= scalars[i % count];
    uint64_t loop = ns(CLOCK_MONOTONIC) - started;
    (void)sink;
    FILE *file = fopen(argv[14], "wb");
    if (!file || fwrite(output, 1, output_bytes, file) != output_bytes || fclose(file))
        fail("cannot write result pixels");
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage)) fail("cannot read process counters");
    printf("{\"schema\":1,\"elapsed_ns\":%" PRIu64 ",\"cpu_ns\":%" PRIu64
           ",\"initialize_ns\":%" PRIu64 ",\"first_glyph_ns\":%" PRIu64
           ",\"loop_ns\":%" PRIu64 ",\"calls\":%d,\"raster_calls\":%" PRIu64
           ",\"cache_hits\":%" PRIu64 ",\"cache_misses\":%" PRIu64
           ",\"clipped_ink_samples\":%" PRIu64 ",\"output_bytes\":%zu"
           ",\"peak_rss_kib\":%ld,\"minor_faults\":%ld,\"major_faults\":%ld"
           ",\"freetype_version\":\"%d.%d.%d\",\"x_ppem\":%u,\"y_ppem\":%u,\"baseline\":%d}\n",
           elapsed, cpu_elapsed, init_ns, first_ns, loop, calls, e.raster_calls,
           cache.hits, cache.misses, e.clipped, output_bytes, usage.ru_maxrss,
           usage.ru_minflt, usage.ru_majflt, major, minor, patch,
           e.face ? e.face->size->metrics.x_ppem : 0,
           e.face ? e.face->size->metrics.y_ppem : 0, e.baseline);
    free(mask); free(output); free(cache.entries); free(cache.storage);
done:
    free(e.left_buffer); free(e.right_buffer);
    free(e.coverage_buffer);
    if (e.right) e.close(e.right);
    if (e.net) e.close(e.net);
    if (e.module) dlclose(e.module);
    if (e.face) FT_Done_Face(e.face);
    if (e.ft) FT_Done_FreeType(e.ft);
    return 0;
}
