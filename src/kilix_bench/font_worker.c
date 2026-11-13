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
#include <unistd.h>

typedef struct neuronte neuronte;
typedef struct {
    void *module;
    neuronte *net;
    int (*open)(const char *, neuronte **);
    void (*close)(neuronte *);
    int (*coverage)(const neuronte *, uint32_t, int, int, int, float *);
    FT_Library ft;
    FT_Face face;
    int w, h, samples, baseline, fixed_width;
    FT_Int32 flags;
    float *coverage_buffer;
    uint64_t raster_calls, clipped;
} Engine;

typedef struct {
    uint32_t ink, clipped;
    int placed_min_x, placed_min_y, placed_max_x, placed_max_y;
    int source_min_x, source_min_y, source_max_x, source_max_y;
    int edges, advance, bitmap_w, bitmap_h, bitmap_left, bitmap_top;
    int has_ink, has_source, outline;
} GlyphGeom;

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
                        const char *asset, int hinting, int require_fixed)
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
        if (!FT_IS_SCALABLE(e->face) || (require_fixed && !FT_IS_FIXED_WIDTH(e->face)))
            fail(require_fixed
                 ? "terminal-cell comparison requires a scalable monospace font"
                 : "cell-forced coverage requires a scalable font");
        e->fixed_width = FT_IS_FIXED_WIDTH(e->face) ? 1 : 0;
        if (FT_Select_Charmap(e->face, FT_ENCODING_UNICODE) ||
            FT_Request_Size(e->face, &request)) fail("cannot size Unicode face");
        e->baseline = (int)((e->face->size->metrics.ascender + 63) / 64);
        e->flags = FT_LOAD_RENDER | FT_LOAD_NO_BITMAP | FT_LOAD_TARGET_NORMAL;
        if (!hinting) e->flags |= FT_LOAD_NO_HINTING | FT_LOAD_NO_AUTOHINT;
    } else fail("unknown engine");
}

static int supported(Engine *e, uint32_t scalar)
{
    /* Neuronte support comes from a SHA256-bound corpus profile in the parent.
     * A successful route or fallback drawing is never evidence of support. */
    return !e->face || FT_Get_Char_Index(e->face, scalar) != 0;
}

static void note_sample(GlyphGeom *geom, int x, int y, int w, int h,
                        unsigned char value, int inside)
{
    if (!geom || !value) return;
    if (!geom->has_source) {
        geom->source_min_x = geom->source_max_x = x;
        geom->source_min_y = geom->source_max_y = y;
        geom->has_source = 1;
    } else {
        if (x < geom->source_min_x) geom->source_min_x = x;
        if (y < geom->source_min_y) geom->source_min_y = y;
        if (x > geom->source_max_x) geom->source_max_x = x;
        if (y > geom->source_max_y) geom->source_max_y = y;
    }
    if (!inside) {
        geom->clipped++;
        return;
    }
    geom->ink++;
    if (!geom->has_ink) {
        geom->placed_min_x = geom->placed_max_x = x;
        geom->placed_min_y = geom->placed_max_y = y;
        geom->has_ink = 1;
    } else {
        if (x < geom->placed_min_x) geom->placed_min_x = x;
        if (y < geom->placed_min_y) geom->placed_min_y = y;
        if (x > geom->placed_max_x) geom->placed_max_x = x;
        if (y > geom->placed_max_y) geom->placed_max_y = y;
    }
    if (x == 0) geom->edges |= 1;
    if (x == w - 1) geom->edges |= 2;
    if (y == 0) geom->edges |= 4;
    if (y == h - 1) geom->edges |= 8;
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
    if (!b->width || !b->rows) return; /* spaces and empty cmap glyphs */
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

/* Geometry belongs to validation. Keep its per-pixel branches and bounds
 * bookkeeping out of the timed raster and cache-miss paths. */
static void raster_geometry(Engine *e, uint32_t scalar, unsigned char *out, GlyphGeom *geom)
{
    raster(e, scalar, out);
    memset(geom, 0, sizeof *geom);
    if (e->net) {
        geom->advance = e->w;
        for (int y = 0; y < e->h; y++)
            for (int x = 0; x < e->w; x++)
                note_sample(geom, x, y, e->w, e->h, out[(size_t)y * e->w + x], 1);
        return;
    }
    FT_GlyphSlot slot = e->face->glyph;
    FT_Bitmap *b = &slot->bitmap;
    geom->outline = 1;
    geom->advance = (int)(slot->advance.x / 64);
    geom->bitmap_w = (int)b->width;
    geom->bitmap_h = (int)b->rows;
    geom->bitmap_left = slot->bitmap_left;
    geom->bitmap_top = slot->bitmap_top;
    if (!b->width || !b->rows) return;
    for (unsigned y = 0; y < b->rows; y++) {
        const unsigned char *row = b->buffer + (ptrdiff_t)y * b->pitch;
        for (unsigned x = 0; x < b->width; x++) {
            int dx = slot->bitmap_left + (int)x;
            int dy = e->baseline - slot->bitmap_top + (int)y;
            int inside = dx >= 0 && dy >= 0 && dx < e->w && dy < e->h;
            note_sample(geom, dx, dy, e->w, e->h, row[x], inside);
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

static void print_box(int present, int a, int b, int c, int d)
{
    if (!present) printf("null");
    else printf("[%d,%d,%d,%d]", a, b, c, d);
}

static void print_edges(int edges)
{
    printf("\"");
    if (edges & 1) printf("L");
    if (edges & 2) printf("R");
    if (edges & 4) printf("T");
    if (edges & 8) printf("B");
    printf("\"");
}

static void print_glyph(uint32_t scalar, const GlyphGeom *geom)
{
    printf("{\"scalar\":%" PRIu32 ",\"ink\":%" PRIu32 ",\"clipped\":%" PRIu32
           ",\"edges\":", scalar, geom->ink, geom->clipped);
    print_edges(geom->edges);
    printf(",\"placed\":");
    print_box(geom->has_ink, geom->placed_min_x, geom->placed_min_y,
              geom->placed_max_x, geom->placed_max_y);
    printf(",\"source\":");
    print_box(geom->has_source, geom->source_min_x, geom->source_min_y,
              geom->source_max_x, geom->source_max_y);
    printf(",\"advance\":%d,\"bitmap\":[%d,%d,%d,%d],\"outline\":%s}",
           geom->advance, geom->bitmap_w, geom->bitmap_h, geom->bitmap_left,
           geom->bitmap_top, geom->outline ? "true" : "false");
}

int main(int argc, char **argv)
{
    /* engine library asset mode width height samples hinting capacity calls
     * columns rows corpus.txt output.bin pitch */
    if (argc != 16) fail("expected 15 worker arguments");
    Engine e = {0};
    e.w = number(argv[5], 1, 128); e.h = number(argv[6], 1, 256);
    e.samples = number(argv[7], 0, 8);
    int hinting = number(argv[8], 0, 1);
    int capacity = number(argv[9], 4, 4096);
    int calls = number(argv[10], 1, 1000000);
    int cols = number(argv[11], 1, 256), rows = number(argv[12], 1, 256);
    int require_fixed = number(argv[15], 0, 1) == 0;
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
    uint64_t started = ns(CLOCK_MONOTONIC);
    open_engine(&e, argv[1], argv[2], argv[3], hinting, require_fixed);
    uint64_t init_ns = ns(CLOCK_MONOTONIC) - started;
    int major = 0, minor = 0, patch = 0;
    if (e.ft) FT_Library_Version(e.ft, &major, &minor, &patch);
    const char *mode = argv[4];
    if (!strcmp(mode, "probe")) {
        printf("{\"supported\":[");
        for (int i = 0; i < count; i++) printf("%s%s", i ? "," : "", supported(&e, scalars[i]) ? "true" : "false");
        printf("],\"freetype_version\":\"%d.%d.%d\",\"fixed_width\":%s}\n",
               major, minor, patch, e.fixed_width ? "true" : "false");
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
    GlyphGeom *glyphs = validate ? allocate((size_t)count, sizeof(GlyphGeom)) : NULL;
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
    struct rusage batch_before, batch_after;
    if (getrusage(RUSAGE_SELF, &batch_before)) fail("cannot read batch counters");
    uint64_t realtime_started = ns(CLOCK_REALTIME);
    uint64_t cpu_started = ns(CLOCK_PROCESS_CPUTIME_ID);
    uint64_t batch_started = ns(CLOCK_MONOTONIC);
    for (int i = 0; i < calls; i++) {
        uint32_t current = scalars[i % count];
        if (draw) composite(output, cached(&e, &cache, current), e.w, e.h, cols, i % (cols * rows));
        else if (validate) raster_geometry(&e, current, output + i * pixels, &glyphs[i]);
        else raster(&e, current, output);
    }
    uint64_t elapsed = ns(CLOCK_MONOTONIC) - batch_started;
    uint64_t cpu_elapsed = ns(CLOCK_PROCESS_CPUTIME_ID) - cpu_started;
    if (getrusage(RUSAGE_SELF, &batch_after)) fail("cannot read batch counters");
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
           ",\"pid\":%ld,\"batch_realtime_ns\":%" PRIu64 ",\"batch_monotonic_ns\":%" PRIu64
           ",\"batch_voluntary_switches\":%ld,\"batch_involuntary_switches\":%ld"
           ",\"batch_minor_faults\":%ld,\"batch_major_faults\":%ld"
           ",\"freetype_version\":\"%d.%d.%d\",\"x_ppem\":%u,\"y_ppem\":%u,\"baseline\":%d"
           ",\"fixed_width\":%s",
           elapsed, cpu_elapsed, init_ns, first_ns, loop, calls, e.raster_calls,
           cache.hits, cache.misses, e.clipped, output_bytes, usage.ru_maxrss,
           usage.ru_minflt, usage.ru_majflt, (long)getpid(), realtime_started, batch_started,
           batch_after.ru_nvcsw - batch_before.ru_nvcsw,
           batch_after.ru_nivcsw - batch_before.ru_nivcsw,
           batch_after.ru_minflt - batch_before.ru_minflt,
           batch_after.ru_majflt - batch_before.ru_majflt, major, minor, patch,
           e.face ? e.face->size->metrics.x_ppem : 0,
           e.face ? e.face->size->metrics.y_ppem : 0, e.baseline,
           e.fixed_width ? "true" : "false");
    if (glyphs) {
        printf(",\"glyphs\":[");
        for (int i = 0; i < count; i++) {
            if (i) printf(",");
            print_glyph(scalars[i], &glyphs[i]);
        }
        printf("]");
    }
    printf("}\n");
    free(mask); free(output); free(glyphs); free(cache.entries); free(cache.storage);
done:
    free(e.coverage_buffer);
    if (e.net) e.close(e.net);
    if (e.module) dlclose(e.module);
    if (e.face) FT_Done_Face(e.face);
    if (e.ft) FT_Done_FreeType(e.ft);
    return 0;
}
