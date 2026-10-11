# mav1 build. `make` builds the library, CLI and tests; `make asan` builds a
# CPU-only sanitizer variant of the CLI.
CC      ?= clang
CFLAGS  ?= -O3 -mcpu=apple-m1 -g
CFLAGS  += -Wall -Wextra -Wno-unused-parameter -Wno-unused-function -Isrc -std=gnu11
BUILD   := build

CORE := src/encoder.c src/obu.c src/tile_writer.c src/cdfs.c src/entenc.c src/bitwriter.c \
        src/fwd_matrices.c src/mvpred.c src/backend_cpu.c src/loopfilter_cpu.c src/lr_search.c
KERNELS := $(wildcard src/kernels/*.h src/kernels/*.metal) src/spec_tables.h src/spec_enums.h
HDRS := $(wildcard src/*.h) $(KERNELS)

CORE_OBJS := $(patsubst src/%.c,$(BUILD)/%.o,$(CORE))
METAL_OBJ := $(BUILD)/backend_metal.o

all: $(BUILD)/libmav1.a $(BUILD)/mav1enc $(BUILD)/test_entropy $(BUILD)/test_txfm $(BUILD)/test_neon

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/%.o: src/%.c $(HDRS) | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

# The Metal backend embeds the kernel sources (compiled at runtime).
$(BUILD)/kernels_src.h: $(KERNELS) tools/embed_kernels.py | $(BUILD)
	python3 tools/embed_kernels.py src $@

$(METAL_OBJ): src/backend_metal.m $(HDRS) $(BUILD)/kernels_src.h | $(BUILD)
	$(CC) $(CFLAGS) -fobjc-arc -I$(BUILD) -c $< -o $@

$(BUILD)/libmav1.a: $(CORE_OBJS) $(METAL_OBJ)
	rm -f $@ && ar rcs $@ $^

LIBS := -framework Metal -framework Foundation

$(BUILD)/mav1enc: tools/mav1enc.c $(BUILD)/libmav1.a
	$(CC) $(CFLAGS) $< $(BUILD)/libmav1.a $(LIBS) -o $@

$(BUILD)/test_entropy: tests/test_entropy.c src/entenc.c
	$(CC) $(CFLAGS) $^ -o $@

$(BUILD)/test_txfm: tests/test_txfm.c src/fwd_matrices.c $(HDRS)
	$(CC) $(CFLAGS) tests/test_txfm.c src/fwd_matrices.c -o $@

# NEON kernels vs the scalar reference implementations (aarch64 only).
$(BUILD)/test_neon: tests/test_neon.c src/fwd_matrices.c $(HDRS)
	$(CC) $(CFLAGS) tests/test_neon.c src/fwd_matrices.c -o $@

asan: $(BUILD)/mav1enc_asan
$(BUILD)/mav1enc_asan: tools/mav1enc.c $(CORE) src/backend_metal_stub.c $(HDRS) | $(BUILD)
	$(CC) -O1 -g -fsanitize=address,undefined -Wall -Isrc -std=gnu11 tools/mav1enc.c $(CORE) src/backend_metal_stub.c -o $@

# `make install PREFIX=/usr/local` stages mav1.h, libmav1.a and a pkg-config
# file for consumers such as the FFmpeg wrapper (docs/ffmpeg-*-libmav1.patch).
PREFIX  ?= /usr/local

install: $(BUILD)/libmav1.a
	mkdir -p $(DESTDIR)$(PREFIX)/include $(DESTDIR)$(PREFIX)/lib/pkgconfig
	cp src/mav1.h $(DESTDIR)$(PREFIX)/include/mav1.h
	cp $(BUILD)/libmav1.a $(DESTDIR)$(PREFIX)/lib/libmav1.a
	printf 'libdir=%s\nincludedir=%s\n\nName: libmav1\nDescription: AV1 encoder (Metal GPU backend, Apple Silicon)\nVersion: 0.1\nLibs: -L$${libdir} -lmav1 -framework Metal -framework Foundation\nCflags: -I$${includedir}\n' "$(PREFIX)/lib" "$(PREFIX)/include" > $(DESTDIR)$(PREFIX)/lib/pkgconfig/libmav1.pc

clean:
	rm -rf $(BUILD)

.PHONY: all asan install clean
