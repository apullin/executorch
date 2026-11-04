/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// The softmax override's exp (newlib_expf.h espressif_softmax_expf: newlib's
// own expf on the chip, its transcription on the host) against
// quantizer/mirrors.py expf_newlib's bits (expf_golden.h, from
// expf_golden.py). On the chip (the parity firmware, under QEMU and on the
// board) this is the one test of the transcription against the code it
// transcribes; on the host (--selftest) it holds the C transcription to the
// numpy one. Compiled -ffp-contract=off: the stream map below is the
// operations it spells.

#include "newlib_expf.h"

#include <cstdio>
#include <cstring>

#include "expf_golden.h"

namespace espressif {
namespace expf_check {

namespace {

float as_float(uint32_t u) {
  float f;
  memcpy(&f, &u, sizeof f);
  return f;
}

uint32_t as_bits(float f) {
  uint32_t u;
  memcpy(&u, &f, sizeof u);
  return u;
}

// Behind a volatile operand, so no case is folded at build time by the
// compiler's own exp.
float chip_expf(float x) {
  volatile float v = x;
  return espressif_softmax_expf(v);
}

uint32_t xorshift32(uint32_t& state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

// expf_golden.py stream_inputs: the same states, the same map.
float stream_input(uint32_t family, uint32_t state) {
  if (family == 0) {
    return static_cast<float>(state >> 8) * 0x1p-17f - 104.0f;
  }
  const uint32_t exponent = ((state >> 23) & 0xffu) % 0x86u;
  return as_float((state & 0x807fffffu) | (exponent << 23));
}

uint32_t fnv1a(uint32_t hash, uint32_t word) {
  for (int byte = 0; byte < 4; ++byte) {
    hash ^= (word >> (8 * byte)) & 0xffu;
    hash *= 16777619u;
  }
  return hash;
}

}  // namespace

uint32_t run_check() {
  uint32_t cases = 0;
  uint32_t mismatches = 0;
  for (const Case& c : kBoundary) {
    ++cases;
    const uint32_t got = as_bits(chip_expf(as_float(c.x)));
    if (got != c.y) {
      if (mismatches < 16) {
        std::printf("  MISMATCH expf: x %08lx got %08lx want %08lx\n",
                    (unsigned long)c.x, (unsigned long)got,
                    (unsigned long)c.y);
      }
      ++mismatches;
    }
  }
  uint32_t fnvs[sizeof(kStreams) / sizeof(kStreams[0])];
  for (size_t k = 0; k < sizeof(kStreams) / sizeof(kStreams[0]); ++k) {
    const Stream& stream = kStreams[k];
    uint32_t state = stream.seed;
    uint32_t hash = 2166136261u;
    for (uint32_t i = 0; i < stream.count; ++i) {
      const float x = stream_input(stream.family, xorshift32(state));
      hash = fnv1a(hash, as_bits(chip_expf(x)));
    }
    cases += stream.count;
    fnvs[k] = hash;
    if (hash != stream.fnv) {
      std::printf("  MISMATCH expf: %s stream (%lu values) fnv %08lx want %08lx\n",
                  stream.name, (unsigned long)stream.count,
                  (unsigned long)hash, (unsigned long)stream.fnv);
      ++mismatches;
    }
  }
  std::printf("expf check: %lu case(s), %lu mismatch(es) (expf vs "
              "mirrors.expf_newlib: %lu boundaries, %s fnv %08lx, %s fnv %08lx)\n",
              (unsigned long)cases, (unsigned long)mismatches,
              (unsigned long)(sizeof(kBoundary) / sizeof(kBoundary[0])),
              kStreams[0].name, (unsigned long)fnvs[0], kStreams[1].name,
              (unsigned long)fnvs[1]);
  return mismatches;
}

}  // namespace expf_check
}  // namespace espressif
