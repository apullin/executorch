/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// The weight tiers' kernel rates: how fast the backend's decode rows read
// their weights from flash, from PSRAM and from internal SRAM, on one core
// and on two. The backend copies weights from flash into PSRAM within a
// budget (its weight-tier options), and what a copy is worth is the
// difference between two of these rates. target/chips.py records them
// (MEASURED, keyed by board, tier, kernel, cores, caches and the PSRAM timing
// point); utils/weight_tiers.py turns them into time per inference.
//
// Hardware only, like the rest of bench_memory: emulated flash is RAM. Built
// only into ET_BENCH_MEM firmware (run_hardware.py's parity build, and a
// story build with sdkconfig.bench.defaults), so no shipped build carries it.
// At -O2 when the backend is (ESP32_BACKEND_O2), so these loops run at the
// backend's speed.
//
// The rows are the backend's at decode (one activation row), k = 256. The
// integer rows are in the story's staged layout -- groups of 128, the codes
// of every row and then their fp16 scales, 4-aligned, as a weight-tier copy
// holds them -- and sum (dot * sx) * scale per group in addmm_q_row_range's
// order:
//   i4    packed int4 rows through the fused kernel (dot_i4_groups_s3)
//         against the deinterleaved activations, less 8 * sum(x) per group
//   i8    int8 rows through dot_i8_s3, a group at a time: an int4 head that
//         FLAG_STAGE_I8 expands runs exactly this
// The float rows are the backend's own loop, esp32_float::matmul, on 2-byte
// weights [n, k] (no scales), on one core as the backend runs it:
//   f16   widened exactly, through a slow path for subnormals, infinities
//         and NaNs (the row reports how many of its values take it)
//   bf16  widened by a shift
// Every tier reads the same bytes -- the model's flash partition, and copies
// of it -- so the checksums must agree across tiers and cores; the f16 rows'
// slow-path share is the same in every tier, so their tiers compare like for
// like even where the share is not a real model's.
//
// One line per row, for run_hardware.py:
//   TIER_RATE tier=<flash|psram|sram> kernel=<i4|i8> cores=<1|2> k=256
//       rows=<n> bytes=<read per pass> us=<mean of 3> best_us=<best>
//       mb_s=<bytes/us> check=<checksum of the first rows of every tier>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>

#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
// After FreeRTOS: esp_partition.h first breaks specreg.h (main.cpp's note).
#include "esp_partition.h"
#include "soc/soc.h"
#include "soc/spi_mem_reg.h"

#include "esp32_float.h"
#include "esp32_kernels.h"

namespace {

constexpr int kK = 256;              // the story's commonest row length
constexpr int kGroup = 128;          // its scale group (group_weights)
constexpr int kGroups = kK / kGroup;
constexpr size_t kStreamBytes = 1u << 20;  // codes read per pass: 16x the cache
constexpr int kPasses = 3;
constexpr float kSx = 0.01f;         // an activation scale

// The loops timed, each with its row bytes and the cores the backend runs it
// on.
enum class Kernel { kI4, kI8, kF16, kBf16 };
constexpr Kernel kKernels[] = {Kernel::kI4, Kernel::kI8, Kernel::kF16,
                               Kernel::kBf16};

const char* kernel_name(Kernel kernel) {
  switch (kernel) {
    case Kernel::kI4:
      return "i4";
    case Kernel::kI8:
      return "i8";
    case Kernel::kF16:
      return "f16";
    default:
      return "bf16";
  }
}

int row_bytes_of(Kernel kernel) {
  return kernel == Kernel::kI4 ? kK / 2 : kernel == Kernel::kI8 ? kK : kK * 2;
}

bool has_scales(Kernel kernel) {
  return kernel == Kernel::kI4 || kernel == Kernel::kI8;
}

// One tier's rows: the codes of every row, then (integer rows) their fp16
// scales, 4-aligned.
struct Rows {
  const uint8_t* codes;
  const uint16_t* scales;
  int rows;
  int row_bytes;
};

inline float f16_to_f32(uint16_t h) {
  // esp32_backend.cpp's conversion, by hand for the same reason.
  const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  const uint32_t exp = (h >> 10) & 0x1Fu;
  const uint32_t mant = h & 0x3FFu;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      uint32_t m = mant;
      int e = -1;
      do {
        m <<= 1;
        ++e;
      } while ((m & 0x400u) == 0);
      bits = sign | ((uint32_t)(127 - 15 - e) << 23) | ((m & 0x3FFu) << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7F800000u | (mant << 13);
  } else {
    bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
  }
  const union {
    uint32_t bits;
    float value;
  } pun = {bits};
  return pun.value;
}

// The activation row, int8, and the fused kernel's form of it: per 32
// inputs the 16 even then the 16 odd, and 8 * sum(x) per group; and the float
// rows' activation, in [-1, 1).
alignas(16) int8_t g_x[kK];
alignas(16) int8_t g_xd[kK];
int32_t g_corr[kGroups];
float g_xf[kK];

void prepare_activation() {
  uint32_t s = 0x9E3779B9u;
  for (int i = 0; i < kK; ++i) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    g_x[i] = (int8_t)s;
    g_xf[i] = (float)(int32_t)(s >> 8) / 8388608.0f - 1.0f;
  }
  for (int c = 0; c < kK; c += 32) {
    for (int t = 0; t < 16; ++t) {
      g_xd[c + t] = g_x[c + 2 * t];
      g_xd[c + 16 + t] = g_x[c + 2 * t + 1];
    }
  }
  for (int g = 0; g < kGroups; ++g) {
    int32_t sum = 0;
    for (int t = g * kGroup; t < (g + 1) * kGroup; ++t) {
      sum += g_x[t];
    }
    g_corr[g] = 8 * sum;
  }
}

void rows_i4(const Rows& w, int n0, int n1, float* y) {
  int32_t sums[kGroups];
  for (int n = n0; n < n1; ++n) {
    dot_i4_groups_s3(w.codes + (size_t)n * w.row_bytes, g_xd, sums, kGroups,
                     kGroup / 2);
    float acc = 0.0f;
    for (int g = 0; g < kGroups; ++g) {
      acc += (float)(sums[g] - g_corr[g]) * kSx *
             f16_to_f32(w.scales[(size_t)n * kGroups + g]);
    }
    y[n] = acc;
  }
}

void rows_i8(const Rows& w, int n0, int n1, float* y) {
  for (int n = n0; n < n1; ++n) {
    const int8_t* row =
        reinterpret_cast<const int8_t*>(w.codes + (size_t)n * w.row_bytes);
    float acc = 0.0f;
    for (int g = 0; g < kGroups; ++g) {
      const int32_t dot = dot_i8_s3(g_x + g * kGroup, row + g * kGroup, kGroup);
      acc += (float)dot * kSx * f16_to_f32(w.scales[(size_t)n * kGroups + g]);
    }
    y[n] = acc;
  }
}

// Output columns [n0, n1) of the backend's float loop, from weight row n0.
void rows_float(esp32_float::Storage s, const Rows& w, int n0, int n1,
                float* y) {
  esp32_float::matmul(s, w.codes + (size_t)n0 * w.row_bytes, g_xf, nullptr,
                      y + n0, 1, kK, (uint32_t)(n1 - n0), nullptr);
}

struct Job {
  const Rows* w;
  Kernel kernel;
  int n0, n1;
  float* y;
};

void run_job(const Job& job) {
  switch (job.kernel) {
    case Kernel::kI4:
      rows_i4(*job.w, job.n0, job.n1, job.y);
      break;
    case Kernel::kI8:
      rows_i8(*job.w, job.n0, job.n1, job.y);
      break;
    case Kernel::kF16:
      rows_float(esp32_float::Storage::kF16, *job.w, job.n0, job.n1, job.y);
      break;
    case Kernel::kBf16:
      rows_float(esp32_float::Storage::kBf16, *job.w, job.n0, job.n1, job.y);
      break;
  }
}

// The second core's half, as the backend's split worker takes it: a task on
// core 1 at the caller's priority, woken per call; the caller spins on the
// join.
TaskHandle_t g_worker = nullptr;
Job g_worker_job;
volatile int g_worker_done = 0;

void worker_main(void*) {
  for (;;) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    run_job(g_worker_job);
    __sync_synchronize();
    g_worker_done = 1;
  }
}

int64_t time_rows(const Job& job, int cores) {
  if (cores == 1 || g_worker == nullptr) {
    const int64_t t0 = esp_timer_get_time();
    run_job(job);
    return esp_timer_get_time() - t0;
  }
  Job mine = job;
  mine.n1 = job.n0 + (job.n1 - job.n0) / 2;
  g_worker_job = job;
  g_worker_job.n0 = mine.n1;
  g_worker_done = 0;
  __sync_synchronize();
  const int64_t t0 = esp_timer_get_time();
  xTaskNotifyGive(g_worker);
  run_job(mine);
  while (!g_worker_done) {
  }
  return esp_timer_get_time() - t0;
}

// The layout in `base` for n rows of a kernel: codes, then any scales
// 4-aligned.
Rows layout(const uint8_t* base, int n, Kernel kernel) {
  const size_t codes = (size_t)n * row_bytes_of(kernel);
  return Rows{base,
              has_scales(kernel) ? reinterpret_cast<const uint16_t*>(
                                       base + ((codes + 3) & ~(size_t)3))
                                 : nullptr,
              n, row_bytes_of(kernel)};
}

size_t scale_bytes_of(int n, Kernel kernel) {
  return has_scales(kernel) ? (size_t)n * kGroups * sizeof(uint16_t) : 0;
}

size_t layout_bytes(int n, Kernel kernel) {
  return (((size_t)n * row_bytes_of(kernel) + 3) & ~(size_t)3) +
         scale_bytes_of(n, kernel);
}

void bench_rows(const char* tier, Kernel kernel, const Rows& w,
                int check_rows, float* y) {
  const char* name = kernel_name(kernel);
  const size_t bytes = (size_t)w.rows * w.row_bytes + scale_bytes_of(w.rows, kernel);
  // The backend splits integer rows across the cores (min_latency) and runs
  // the float loop on one.
  const int max_cores = has_scales(kernel) ? 2 : 1;
  for (int cores = 1; cores <= max_cores; ++cores) {
    if (cores == 2 && g_worker == nullptr) {
      printf("  TIER_RATE tier=%s kernel=%s cores=2 skipped: no worker task\n",
             tier, name);
      continue;
    }
    const Job job{&w, kernel, 0, w.rows, y};
    time_rows(job, cores);  // the code, the TLB and the cache's state warm
    int64_t total = 0, best = INT64_MAX;
    for (int pass = 0; pass < kPasses; ++pass) {
      const int64_t us = time_rows(job, cores);
      total += us;
      best = us < best ? us : best;
    }
    double check = 0;
    for (int i = 0; i < check_rows; ++i) {
      check += (double)y[i] * (double)(i + 1);
    }
    const double mean = (double)total / kPasses;
    printf("  TIER_RATE tier=%s kernel=%s cores=%d k=%d rows=%d bytes=%u "
           "us=%.0f best_us=%lld mb_s=%.2f check=%.6e\n",
           tier, name, cores, kK, w.rows, (unsigned)bytes, mean,
           (long long)best, mean > 0 ? (double)bytes / mean : 0.0, check);
  }
}

}  // namespace

// The model's weights partition: the story's (0x44), else the parity
// firmware's int4 decode weights (0x46), else its largest. Null without one.
extern "C" const esp_partition_t* bench_model_partition(void) {
  static const uint8_t kSubtypes[] = {0x44, 0x46, 0x42, 0x41, 0x40};
  for (const uint8_t subtype : kSubtypes) {
    const esp_partition_t* part = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)subtype, nullptr);
    if (part != nullptr) {
      return part;
    }
  }
  return nullptr;
}

extern "C" void bench_weight_tiers(void) {
  printf("\n=== weight tiers: kernel rates (hardware only) ===\n");
#if !defined(ET_STORY)
  // The story prints this at boot; any other bench build prints it here, so
  // every rate is filed with the PSRAM timing point it ran at.
  printf("[mspi] psram din_mode 0x%08x din_num 0x%08x timing_cali 0x%08x "
         "core_clk 0x%08x sram_clk 0x%08x\n",
         (unsigned)REG_READ(SPI_MEM_SPI_SMEM_DIN_MODE_REG(0)),
         (unsigned)REG_READ(SPI_MEM_SPI_SMEM_DIN_NUM_REG(0)),
         (unsigned)REG_READ(SPI_MEM_SPI_SMEM_TIMING_CALI_REG(0)),
         (unsigned)REG_READ(SPI_MEM_CORE_CLK_SEL_REG(0)),
         (unsigned)REG_READ(SPI_MEM_SRAM_CLK_REG(0)));
#endif
  const esp_partition_t* part = bench_model_partition();
  // kStreamBytes of codes a pass, whatever the kernel: the int4 rows' scales
  // make theirs the widest span read.
  const int rows4 = (int)(kStreamBytes / row_bytes_of(Kernel::kI4));
  const size_t span = layout_bytes(rows4, Kernel::kI4);
  if (part == nullptr || part->size < span) {
    printf("  no weights partition of %u B to read\n", (unsigned)span);
    return;
  }
  const void* mapped = nullptr;
  esp_partition_mmap_handle_t handle = 0;
  if (esp_partition_mmap(part, 0, span, ESP_PARTITION_MMAP_DATA, &mapped,
                         &handle) != ESP_OK) {
    printf("  mmap of partition 0x%02x failed\n", (unsigned)part->subtype);
    return;
  }
  const uint8_t* flash = static_cast<const uint8_t*>(mapped);
  printf("  rows read from partition 0x%02x at 0x%x (%u B) and copies of it\n",
         (unsigned)part->subtype, (unsigned)part->address,
         (unsigned)part->size);
  prepare_activation();
  if (xTaskCreatePinnedToCore(worker_main, "tier_bench", 4096, nullptr,
                              uxTaskPriorityGet(nullptr), &g_worker,
                              1) != pdPASS) {
    g_worker = nullptr;
  }

  // The most rows any kernel runs: the int4 rows'.
  float* y = static_cast<float*>(heap_caps_malloc(
      sizeof(float) * (size_t)rows4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
  uint8_t* psram = static_cast<uint8_t*>(heap_caps_aligned_alloc(
      16, span, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  // Internal SRAM: as much as fits, halving from 128 KB. Its rows are the
  // first ones, so every tier's checksum covers them.
  uint8_t* sram = nullptr;
  size_t sram_bytes = 0;
  for (size_t want = 128u * 1024u; want >= 16u * 1024u && sram == nullptr;
       want /= 2) {
    sram = static_cast<uint8_t*>(heap_caps_aligned_alloc(
        16, want, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    sram_bytes = sram != nullptr ? want : 0;
  }
  if (y == nullptr) {
    printf("  no internal SRAM for the outputs\n");
  } else {
    for (const Kernel kernel : kKernels) {
      const char* name = kernel_name(kernel);
      const int row_bytes = row_bytes_of(kernel);
      const int rows = (int)(kStreamBytes / row_bytes);
      int sram_rows = 0;
      while (layout_bytes(sram_rows + 1, kernel) <= sram_bytes &&
             sram_rows < rows) {
        ++sram_rows;
      }
      const int check_rows = sram_rows > 0 ? sram_rows : rows;
      if (kernel == Kernel::kF16) {
        const uint16_t* half = reinterpret_cast<const uint16_t*>(flash);
        size_t rare = 0;
        const size_t values = (size_t)rows * kK;
        for (size_t t = 0; t < values; ++t) {
          const uint32_t em = half[t] & 0x7FFFu;
          rare += em < 0x0400u || em >= 0x7C00u;
        }
        printf("  f16 rows: %.1f%% of their values are subnormal, infinite "
               "or NaN (the widening's slow path)\n",
               100.0 * (double)rare / (double)values);
      }
      // Flash: the codes, then (integer rows) the bytes after them as the
      // scales, both read in place through the mapping, as the weights file
      // is.
      const Rows on_flash = layout(flash, rows, kernel);
      bench_rows("flash", kernel, on_flash, check_rows, y);
      if (psram != nullptr) {
        std::memcpy(psram, flash, layout_bytes(rows, kernel));
        bench_rows("psram", kernel, layout(psram, rows, kernel), check_rows,
                   y);
      } else {
        printf("  TIER_RATE tier=psram kernel=%s skipped: no PSRAM\n", name);
      }
      if (sram_rows > 0) {
        // The same first rows and their scales, in the same layout.
        const Rows head = layout(sram, sram_rows, kernel);
        std::memcpy(sram, flash, (size_t)sram_rows * row_bytes);
        if (head.scales != nullptr) {
          std::memcpy(const_cast<uint16_t*>(head.scales), on_flash.scales,
                      scale_bytes_of(sram_rows, kernel));
        }
        bench_rows("sram", kernel, head, check_rows, y);
      } else {
        printf("  TIER_RATE tier=sram kernel=%s skipped: no internal block\n",
               name);
      }
    }
  }
  if (g_worker != nullptr) {
    vTaskDelete(g_worker);
    g_worker = nullptr;
  }
  heap_caps_free(sram);
  heap_caps_free(psram);
  heap_caps_free(y);
  esp_partition_munmap(handle);
}
