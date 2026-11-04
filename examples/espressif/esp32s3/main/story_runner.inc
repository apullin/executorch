/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Story mode: the big Glimmer story model served from the console.
//
// Included by main.cpp under STORY_INPUTS, so it shares that file's
// allocators, loaders and types. The program is a two-method prefill/decode
// pair (shared mutable KV in mem-id-2, weights in one PTD) whose head is the
// full vocabulary; the next token is sampled.
//
// Startup runs the exporter's golden: the same prompt, greedy, with per-step
// logit checksums from the host modules; tokens and checksums are compared
// and reported as STORY_GOLDEN lines, with a digest of every step's logits
// (story_digest.h) and the cold start's times (STORY_COLD). Then, after
// "=== DONE ===", the console loop serves "R <prompt with \n escapes>",
// "T <temperature>", "K <top-k>", "S <seed>", "N <max new tokens>", "Q".
#include <new>
#include <string_view>

#include "bpe.h"
#include "driver/gpio.h"
#include "esp32_weight_tiers.h"
#include "esp_attr.h"
#include "esp_random.h"
#include "nvs.h"
#include "nvs_flash.h"
#include "story_digest.h"

namespace {

constexpr size_t kStoryMaxPrompt = 200;
constexpr size_t kStoryMaxNew = 200;
constexpr size_t kStoryMaxLine = 1024;
constexpr size_t kStoryMaxText = 1400;
constexpr int kStoryTopKMax = 64;
// The story's text buffers and drafts (26 KB) are EXT_RAM_BSS_ATTR: PSRAM on
// the board (sdkconfig.story.defaults), touched a few times per token at
// most. The internal SRAM they took belongs to the KV cache and the display.

// ---- Where the story's memory goes -----------------------------------------
// Internal SRAM: about 250 KB is free at boot with the display on ([mem]
// boot). The KV cache takes it first (mem-id-2, 147 KB, read every token);
// then, within a floor, decode's own activations (33.8 KB) and the small
// constants outside the delegate (StoryHotConstants, 45.6 KB), in that order,
// each as far as the floor allows: on the board the activations saved 0.048
// ms a token per KB and the constants 0.027 (board A/Bs). The floor is
// free internal SRAM kept after load for what is made later: the split
// worker's stack at the first decode init, NVS writes, the display's
// refreshes (a short internal heap can break the boot handshake). A
// placement that would cross it goes to PSRAM, or stays on flash. Free
// internal SRAM is printed after load and after the first story.
//
// PSRAM: the larger method's activations (prefill's mem-id-1, 890 KB), which
// decode shares unless it has its own; the method arena; the temp tier; then
// the weight copies (weight_tier_budget).
//
// These settings, the weight tier's and the temp tier's with them, are data,
// not code: kStoryPlacement is read through a volatile, so two builds that
// differ only in them run the same code at the same addresses, and a board A/B
// of one of them sees only where the bytes are. Code layout alone moves decode
// by up to 0.6% through the 32 KB instruction cache, more than the 0.5% an A/B
// is held to.
#ifndef ET_STORY_INTERNAL_FLOOR
#define ET_STORY_INTERNAL_FLOOR (40 * 1024)
#endif
// What a load still takes from internal SRAM after the placements: the split
// worker's 4 KB stack and its task, made at the first decode init.
#ifndef ET_STORY_LOAD_RESERVE
#define ET_STORY_LOAD_RESERVE (6 * 1024)
#endif
#ifndef ET_STORY_DECODE_ACT_INTERNAL
#define ET_STORY_DECODE_ACT_INTERNAL 1
#endif
#ifndef ET_STORY_HOT_CONSTANTS
#define ET_STORY_HOT_CONSTANTS 1
#endif
#ifndef ET_STORY_ARENA_BYTES
#define ET_STORY_ARENA_BYTES (128 * 1024)
#endif
#ifndef ET_STORY_TEMP_BYTES
#define ET_STORY_TEMP_BYTES (2 * 1024 * 1024)
#endif
#ifndef ET_STORY_STAGE_BYTES
#define ET_STORY_STAGE_BYTES 0
#endif
// Ranked weight placement by default is this line: run_story_hardware.py
// reads its --tier-rank default from here. On: at the 6 MiB budget it cost
// 0.03% of a decode step on the board, and at 4.5 MiB it took decode
// from 110.10 to 98.85 ms per token.
#ifndef ET_STORY_TIER_RANK
#define ET_STORY_TIER_RANK 1
#endif
// Under an auto weight budget (ET_STORY_STAGE_BYTES -1), the PSRAM the copies
// leave free for what is allocated after load.
#ifndef ET_STORY_TIER_RESERVE
#define ET_STORY_TIER_RESERVE 0
#endif

struct StoryPlacement {
  uint32_t floor;               // internal SRAM left free after load
  uint32_t load_reserve;        // what a load takes from it after the placements
  uint32_t decode_act_internal; // 1: decode's activations in a buffer of their own, internal first
  uint32_t hot_constants;       // 1: small constants outside the delegate from internal SRAM
  uint32_t hot_constant_max;    // the largest of those; cos and sin (16 KB, a row a token) stay on flash
  uint32_t arena_bytes;         // the method arena, in PSRAM
  uint32_t temp_bytes;          // the temp allocator's PSRAM tier (0: none)
  int32_t stage_bytes;          // weight_tier_budget: PSRAM for weight copies, -1 auto
  uint32_t tier_reserve;        // weight_tier_reserve, under auto
  uint32_t tier_rank;           // weight_tier_rank
};
static const volatile StoryPlacement kStoryPlacement = {
    ET_STORY_INTERNAL_FLOOR, ET_STORY_LOAD_RESERVE, ET_STORY_DECODE_ACT_INTERNAL,
    ET_STORY_HOT_CONSTANTS,  4096,                  ET_STORY_ARENA_BYTES,
    ET_STORY_TEMP_BYTES,     ET_STORY_STAGE_BYTES,  ET_STORY_TIER_RESERVE,
    ET_STORY_TIER_RANK};

// A heap block's header and alignment, beyond the bytes asked for.
constexpr size_t kStoryHeapOverhead = 32;

static size_t story_internal_free(void) { return heap_caps_get_free_size(MALLOC_CAP_INTERNAL); }
static size_t story_internal_largest(void) {
  return heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
}

// Whether `bytes` of internal SRAM can go now and still leave the floor once
// the load has made its own allocations.
static bool story_internal_room(size_t bytes) {
  return story_internal_largest() >= bytes + kStoryHeapOverhead &&
         story_internal_free() >= bytes + kStoryHeapOverhead + kStoryPlacement.load_reserve +
                                      kStoryPlacement.floor;
}

static void story_report_internal(const char* when) {
  const size_t free_now = story_internal_free();
  printf("  [mem] %s: internal free %u B, largest block %u B, lowest since boot %u B; floor %u B %s\n",
         when, (unsigned)free_now, (unsigned)story_internal_largest(),
         (unsigned)heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL),
         (unsigned)kStoryPlacement.floor, free_now >= kStoryPlacement.floor ? "held" : "BROKEN");
}

enum StoryTier { kInternalFirst, kInternalOnly, kPsramFirst };

// A 16-aligned buffer. kInternalFirst: internal SRAM within the floor, else
// PSRAM, else internal SRAM whatever the floor. kInternalOnly: internal SRAM
// within the floor, or nothing. kPsramFirst: PSRAM, else internal SRAM. Says
// where it went and what internal SRAM is left.
static uint8_t* story_alloc(const char* what, size_t bytes, StoryTier order) {
  const size_t allocation = (bytes + 15u) & ~15u;
  constexpr uint32_t kInternal = MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT;
  uint8_t* p = nullptr;
  const char* tier = "internal";
  if (order != kPsramFirst && story_internal_room(allocation)) {
    p = static_cast<uint8_t*>(heap_caps_aligned_alloc(16, allocation, kInternal));
  }
  if (p == nullptr && order != kInternalOnly) {
    p = static_cast<uint8_t*>(heap_caps_aligned_alloc(16, allocation, MALLOC_CAP_SPIRAM));
    tier = "PSRAM";
    if (p == nullptr) {
      p = static_cast<uint8_t*>(heap_caps_aligned_alloc(16, allocation, kInternal));
      tier = "internal";
    }
  }
  if (p == nullptr) {
    if (order != kInternalOnly) printf("  %s allocation FAILED (%u B)\n", what, (unsigned)bytes);
    return nullptr;
  }
  printf("  %s: %u B (%s); internal free %u B, largest block %u B\n", what, (unsigned)bytes, tier,
         (unsigned)story_internal_free(), (unsigned)story_internal_largest());
  return p;
}

// Small constants outside the delegate, served from internal SRAM: decode's
// RMSNorm weights (42 of 1 KB) and the caches' position ranges its attention
// masks compare with (2,560 B). From the weights file they are read through
// the flash cache, and the 5 MB weight stream evicts them every token, about
// 1.1-1.6 ms per token by estimate. A wrapper of the weights file:
// a constant of at most hot_constant_max bytes that the delegate does not own
// (its keys are esp32_*) is copied into internal SRAM at its first request,
// while the floor allows, and every later request (each load) gets the copy.
// The bytes are the same, so the numbers are; the delegate's own requests,
// and everything else, pass through untouched.
class StoryHotConstants final : public NamedDataMap {
 public:
  explicit StoryHotConstants(const NamedDataMap* inner) : inner_(inner) {}
  // Once no method holds a copy.
  ~StoryHotConstants() override {
    while (copies_ != nullptr) {
      Copy* next = copies_->next;
      heap_caps_free(copies_);
      copies_ = next;
    }
  }
  StoryHotConstants(const StoryHotConstants&) = delete;
  StoryHotConstants& operator=(const StoryHotConstants&) = delete;

  executorch::runtime::Result<const executorch::runtime::TensorLayout> get_tensor_layout(
      std::string_view key) const override {
    return inner_->get_tensor_layout(key);
  }
  executorch::runtime::Result<FreeableBuffer> get_data(std::string_view key) const override {
    if (!kStoryPlacement.hot_constants ||
        (key.size() >= 6 && memcmp(key.data(), "esp32_", 6) == 0)) {
      return inner_->get_data(key);
    }
    const uint64_t hash = fnv1a(key);
    for (const Copy* copy = copies_; copy != nullptr; copy = copy->next) {
      if (copy->hash == hash) return FreeableBuffer(copy + 1, copy->size, nullptr);
    }
    executorch::runtime::Result<FreeableBuffer> source = inner_->get_data(key);
    if (!source.ok() || source->size() > kStoryPlacement.hot_constant_max) return source;
    const size_t size = source->size();
    void* raw = nullptr;
    if (story_internal_room(sizeof(Copy) + size)) {
      raw = heap_caps_aligned_alloc(16, sizeof(Copy) + size, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    }
    if (raw == nullptr) {
      ++left_;
      left_bytes_ += size;
      return source;
    }
    Copy* copy = new (raw) Copy{hash, copies_, (uint32_t)size};
    memcpy(copy + 1, source->data(), size);
    copies_ = copy;
    ++copied_;
    copied_bytes_ += size;
    return FreeableBuffer(copy + 1, size, nullptr);
  }
  executorch::runtime::Error load_data_into(std::string_view key, void* buffer,
                                            size_t size) const override {
    return inner_->load_data_into(key, buffer, size);
  }
  executorch::runtime::Result<uint32_t> get_num_keys() const override {
    return inner_->get_num_keys();
  }
  executorch::runtime::Result<const char*> get_key(uint32_t index) const override {
    return inner_->get_key(index);
  }

  void report() const {
    printf("  [mem] hot constants: %u in internal SRAM (%u B), %u left on flash by the floor (%u B)%s\n",
           (unsigned)copied_, (unsigned)copied_bytes_, (unsigned)left_, (unsigned)left_bytes_,
           kStoryPlacement.hot_constants ? "" : "; off");
  }

 private:
  // A copy's header (16 B), followed by its bytes, 16-aligned as the block
  // is. A key is known by its 64-bit FNV-1a hash: a collision among the
  // weights file's few hundred keys is a 1e-15 chance.
  struct alignas(16) Copy {
    uint64_t hash;
    Copy* next;
    uint32_t size;
  };

  static uint64_t fnv1a(std::string_view key) {
    uint64_t hash = 0xcbf29ce484222325ull;
    for (const char c : key) hash = (hash ^ (uint8_t)c) * 0x100000001b3ull;
    return hash;
  }

  const NamedDataMap* inner_;
  mutable Copy* copies_ = nullptr;
  mutable uint32_t copied_ = 0, left_ = 0;
  mutable size_t copied_bytes_ = 0, left_bytes_ = 0;
};

struct StoryModel {
  const Program* program;
  Span<uint8_t>* prefill_spans;
  Span<uint8_t>* decode_spans;
  size_t n_planned;
  const NamedDataMap* named_data;
  const StoryHotConstants* hot;
  uint8_t* arena;
  size_t arena_size;
  uint8_t* temp;
  size_t temp_size;
};

// The weight tiers in one line, while a method holds its weights: the budget,
// what is copied and what stayed on flash, and the ranked placements.
static void story_report_tiers(void) {
  Esp32WeightTierStats stats;
  esp32_backend_weight_tier_stats(&stats);
  uint64_t staged = 0, on_flash = 0;
  for (int c = 0; c < ESP32_TIER_CLASSES; ++c) {
    staged += stats.staged[c];
    on_flash += stats.on_flash[c];
  }
  // Byte counts fit 32 bits here: the chip has 8 MB of PSRAM.
  printf("  [tier] budget %s%u B%s, rank %s: %u of %u B wanted in PSRAM (%u B left on flash), "
         "resident %u B; copies %u, reuses %u, evictions %u, placements %u (%u by an execute), "
         "waiting %u\n",
         stats.auto_budget ? "auto keeping " : "",
         (unsigned)(stats.auto_budget ? stats.reserve : stats.budget), stats.auto_budget ? " free" : "",
         stats.rank ? "on" : "off", (unsigned)staged, (unsigned)stats.wanted, (unsigned)on_flash,
         (unsigned)stats.resident, (unsigned)stats.copies, (unsigned)stats.hits,
         (unsigned)stats.evictions, (unsigned)stats.placements, (unsigned)stats.late_placements,
         (unsigned)stats.waiting);
}

#ifdef ESP32_OP_TIMING
// Portable-op attribution for decode (instrumented build only): every
// instruction's time by its index in the decode plan, reported after the
// golden check. Names come from the .pte on the host (story_op_profile.py).
static TimingEventTracer g_story_op_tracer;
#endif

struct StoryStats {
  int prefill_blocks = 0;
  int64_t prefill_us = 0;
  int decode_steps = 0;
  int64_t decode_us = 0;
  // Outside the step clocks: each method's load_method, and the ranked
  // placement after it (story_place_weights).
  int64_t prefill_load_us = 0;
  int64_t prefill_place_us = 0;
  int64_t decode_load_us = 0;
  int64_t decode_place_us = 0;
  // Golden mode: FNV-1a over every step's logits, in order (story_digest.h).
  uint64_t digest = STORY_DIGEST_BASIS;
  const char* stopped = "max";
  const char* error = nullptr;
};

static uint32_t g_story_rng = 0x9e3779b9u;
static uint32_t story_rng(void) {
  uint32_t x = g_story_rng;
  x ^= x << 13; x ^= x >> 17; x ^= x << 5;
  g_story_rng = x;
  return x;
}

// A seed's bits spread over the whole state (splitmix32's finaliser): boot
// seeds differ little from one power-on to the next, and xorshift's first
// outputs from close seeds stay close (successive boots would open alike).
static void story_seed(uint32_t seed) {
  seed += 0x9e3779b9u;
  seed = (seed ^ (seed >> 16)) * 0x85ebca6bu;
  seed = (seed ^ (seed >> 13)) * 0xc2b2ae35u;
  seed ^= seed >> 16;
  g_story_rng = seed | 1u;
  for (int i = 0; i < 8; ++i) story_rng();
}

// A uniform pick from n, from the generator's high bits (its weakest are the
// low ones that `% n` would use).
static uint32_t story_below(uint32_t n) {
  return (uint32_t)(((uint64_t)story_rng() * n) >> 32);
}

// One method step on a two-input method: returns the logits (STORY_VOCAB
// floats) or nullptr with *error set.
static const float* story_step(executorch::runtime::Method& method,
                               const char* name, const int64_t* tokens,
                               size_t count, int64_t position,
                               int64_t* elapsed_us, const char** error) {
  EValue inputs[2];
  if (method.get_inputs(inputs, 2) != Error::Ok) { *error = "get_inputs"; return nullptr; }
  Tensor token_tensor = inputs[0].toTensor();
  Tensor position_tensor = inputs[1].toTensor();
  if (token_tensor.numel() != (int64_t)count || position_tensor.numel() != 1) {
    *error = "input geometry"; return nullptr;
  }
  memcpy(token_tensor.mutable_data_ptr<int64_t>(), tokens, count * sizeof(int64_t));
  position_tensor.mutable_data_ptr<int64_t>()[0] = position;
  const int64_t begin = esp_timer_get_time();
  const Error status = method.execute();
  *elapsed_us += esp_timer_get_time() - begin;
  if (status != Error::Ok) {
    printf("  %s execute FAILED at pos %lld: %d\n", name, (long long)position, (int)status);
    *error = "execute"; return nullptr;
  }
  EValue outputs[1];
  if (method.get_outputs(outputs, 1) != Error::Ok) { *error = "get_outputs"; return nullptr; }
  Tensor logits = outputs[0].toTensor();
  if (logits.numel() != STORY_VOCAB) { *error = "output width"; return nullptr; }
  return logits.const_data_ptr<float>();
}

static double story_checksum(const float* values) {
  double sum = 0.0;
  for (int lane = 0; lane < STORY_VOCAB; ++lane) sum += (double)values[lane] * (double)(lane + 1);
  return sum;
}

// Temperature + top-k sampling; temperature <= 0 is the argmax.
static int story_sample(const float* logits, float temperature, int top_k) {
  int best_index[kStoryTopKMax];
  float best_value[kStoryTopKMax];
  if (top_k < 1) top_k = 1;
  if (top_k > kStoryTopKMax) top_k = kStoryTopKMax;
  if (temperature <= 0.0f) top_k = 1;
  int count = 0;
  for (int i = 0; i < STORY_VOCAB; ++i) {
    const float v = logits[i];
    if (count == top_k && v <= best_value[count - 1]) continue;
    int at = count < top_k ? count : top_k - 1;
    while (at > 0 && best_value[at - 1] < v) {
      best_value[at] = best_value[at - 1]; best_index[at] = best_index[at - 1]; --at;
    }
    best_value[at] = v; best_index[at] = i;
    if (count < top_k) ++count;
  }
  if (top_k == 1) return best_index[0];
  float probs[kStoryTopKMax];
  float total = 0.0f;
  for (int k = 0; k < count; ++k) {
    probs[k] = expf((best_value[k] - best_value[0]) / temperature);
    total += probs[k];
  }
  const float u = (float)(story_rng() >> 8) / 16777216.0f * total;
  float acc = 0.0f;
  for (int k = 0; k < count; ++k) {
    acc += probs[k];
    if (u < acc) return best_index[k];
  }
  return best_index[count - 1];
}

static bool story_is_stop(char c) { return c == '.' || c == '!' || c == '?'; }

// Where a sentence ends, read byte by byte because a byte-level token can
// split a UTF-8 character. Straight quotes toggle; curly quotes open (“) and
// close (”). A sentence end counts only outside every quote, or as a closing
// quote right after . ! or ? -- otherwise a card could end inside a
// character's speech: said, “I can make one for you.
struct StoryEnding {
  size_t straight = 0;
  int curly = 0;
  char last = 0, prev = 0;
  unsigned char utf8[4] = {0, 0, 0, 0};
  int utf8_have = 0, utf8_need = 0;
  void add(const char* piece) {
    for (const unsigned char* p = reinterpret_cast<const unsigned char*>(piece); *p; ++p) {
      const unsigned char b = *p;
      if (b < 0x80) {
        utf8_need = 0;
        if (b == '"') ++straight;
        if (b != ' ' && b != '\n') { prev = last; last = (char)b; }
      } else if (b >= 0xC0) {
        utf8[0] = b;
        utf8_have = 1;
        utf8_need = b >= 0xF0 ? 4 : b >= 0xE0 ? 3 : 2;
      } else if (utf8_need > 0 && utf8_have < utf8_need) {
        utf8[utf8_have++] = b;
        if (utf8_have == utf8_need) {
          const bool curly = utf8_need == 3 && utf8[0] == 0xE2 && utf8[1] == 0x80;
          const bool opens = curly && utf8[2] == 0x9C;
          const bool closes = curly && utf8[2] == 0x9D;
          if (opens) ++this->curly;
          if (closes && this->curly > 0) --this->curly;
          prev = last;
          last = (opens || closes) ? '"' : 'x';
          utf8_need = 0;
        }
      }
    }
  }
  bool clean() const {
    return straight % 2 == 0 && curly == 0 &&
           (story_is_stop(last) || (last == '"' && story_is_stop(prev)));
  }
};

// Called with the tokens generated so far every `every` tokens: the reader
// shows the story being written.
struct StoryProgress {
  void (*fn)(const int64_t* tokens, size_t count, void* ctx);
  void* ctx;
  size_t every;
};

// The backend's ranked weight placement (weight_tier_rank, set in run_story):
// a load leaves its weights for this call, which copies the ones worth most
// into PSRAM first. Made after each load and before its first execute, so no
// copy lands inside a timed step; with ranking off nothing waits for it and
// it does nothing.
static void story_place_weights(void) {
  executorch::runtime::BackendOptions<1> apply;
  apply.set_option("weight_tier_apply", true);
  (void)executorch::runtime::set_option("Esp32Backend", apply.view());
}

// Prefill whole chunks, decode the rest of the prompt, then generate. Golden
// mode (temperature 0) also records per-step checksums when arrays are given,
// and then folds each step's raw logits into stats->digest, every step in the
// order it ran, after the step's clock has stopped.
//
// Whole sentences: a story runs to the model's own end (most do, at a median
// of about 180 tokens), and one cut by the token guard or the context instead
// ends at its last clean sentence end (a . ! or ? outside quotes, or a
// closing quote right after one), never mid-sentence. whole_sentences false
// keeps the exact token count the golden needs.
//
// forced (the golden): after the prompt the device is fed these tokens, not
// its own choices, while out[] still records what it would have chosen. A
// near-tie the two sides break differently then costs one choice, instead of
// a different continuation from there on in which no later checksum compares.
static int story_generate(const StoryModel& model, const int64_t* prompt, size_t n,
                          size_t max_new, bool whole_sentences, float temperature, int top_k,
                          int64_t* out, StoryStats* stats,
                          double* prefill_sums, double* decode_sums, size_t max_sums,
                          const int64_t* forced = nullptr,
                          const StoryProgress* progress = nullptr) {
  const size_t n_full = ((n - 1) / STORY_PREFILL_CHUNK) * STORY_PREFILL_CHUNK;
  size_t generated = 0;
  if (n_full > 0) {
    TrackingAllocator method_allocator(model.arena_size, model.arena);
    TieredTempAllocator temp_allocator(sizeof(temp_arena), temp_arena, model.temp_size, model.temp);
    HierarchicalAllocator planned_allocator({model.prefill_spans, model.n_planned});
    MemoryManager memory_manager(&method_allocator, &planned_allocator, &temp_allocator);
    const int64_t load_begin = esp_timer_get_time();
    auto method = model.program->load_method("prefill", &memory_manager, nullptr, model.named_data);
    stats->prefill_load_us += esp_timer_get_time() - load_begin;
    if (!method.ok()) { stats->error = "load prefill"; return 0; }
    const int64_t place_begin = esp_timer_get_time();
    story_place_weights();
    stats->prefill_place_us += esp_timer_get_time() - place_begin;
    for (size_t start = 0; start < n_full; start += STORY_PREFILL_CHUNK) {
      const float* logits = story_step(*method, "prefill", prompt + start, STORY_PREFILL_CHUNK,
                                       (int64_t)start, &stats->prefill_us, &stats->error);
      if (logits == nullptr) return 0;
      if (prefill_sums && (size_t)stats->prefill_blocks < max_sums) {
        prefill_sums[stats->prefill_blocks] = story_checksum(logits);
      }
      if (prefill_sums) {
        stats->digest = story_digest_bytes(stats->digest, logits, STORY_VOCAB * sizeof(float));
      }
      ++stats->prefill_blocks;
      vTaskDelay(1);
    }
    // The arena is sized for both methods, and the board's prompts are too
    // short for prefill (it needs 33 tokens): a long --request prints this.
    static bool prefill_reported = false;
    if (!prefill_reported) {
      printf("  method arena: prefill uses %u of %u B\n", (unsigned)method_allocator.peak(),
             (unsigned)model.arena_size);
      printf("  temp: prefill peaks at %u B (%u B internal + %u B PSRAM)\n",
             (unsigned)temp_allocator.peak(), (unsigned)sizeof(temp_arena), (unsigned)model.temp_size);
      prefill_reported = true;
    }
  }
  memset(model.arena, 0, model.arena_size);
  TrackingAllocator method_allocator(model.arena_size, model.arena);
  TieredTempAllocator temp_allocator(sizeof(temp_arena), temp_arena, model.temp_size, model.temp);
  HierarchicalAllocator planned_allocator({model.decode_spans, model.n_planned});
  MemoryManager memory_manager(&method_allocator, &planned_allocator, &temp_allocator);
  const int64_t load_begin = esp_timer_get_time();
#ifdef ESP32_OP_TIMING
  auto method = model.program->load_method("decode", &memory_manager, &g_story_op_tracer, model.named_data);
#else
  auto method = model.program->load_method("decode", &memory_manager, nullptr, model.named_data);
#endif
  stats->decode_load_us += esp_timer_get_time() - load_begin;
  if (!method.ok()) { stats->error = "load decode"; return 0; }
  const int64_t place_begin = esp_timer_get_time();
  story_place_weights();
  stats->decode_place_us += esp_timer_get_time() - place_begin;
  static bool decode_reported = false;
  if (!decode_reported) {
    // The first decode load: everything it allocates, the split worker
    // included, is made, and its weights are placed.
    printf("  method arena: decode uses %u of %u B\n", (unsigned)method_allocator.peak(),
           (unsigned)model.arena_size);
    if (model.hot != nullptr) model.hot->report();
    story_report_internal("after load");
    story_report_tiers();
  }
  int64_t token = 0;
  size_t last_clean = 0;
  StoryEnding ending;
  for (size_t pos = n_full; ; ++pos) {
    if (pos >= (size_t)STORY_CONTEXT - 1) { stats->stopped = "context"; break; }
    const int64_t current = pos < n ? prompt[pos] : (forced != nullptr ? forced[pos - n] : token);
    const float* logits = story_step(*method, "decode", &current, 1, (int64_t)pos,
                                     &stats->decode_us, &stats->error);
    if (logits == nullptr) return (int)generated;
    if (decode_sums && (size_t)stats->decode_steps < max_sums) {
      decode_sums[stats->decode_steps] = story_checksum(logits);
    }
    if (decode_sums) {
      stats->digest = story_digest_bytes(stats->digest, logits, STORY_VOCAB * sizeof(float));
    }
    ++stats->decode_steps;
    vTaskDelay(1);
    if (pos + 1 < n) continue;               // still feeding the prompt
    token = story_sample(logits, temperature, top_k);
    if (token == STORY_EOT && forced == nullptr) { stats->stopped = "eot"; break; }
    out[generated++] = token;
    if (whole_sentences && token >= 0 && token < STORY_VOCAB) {
      ending.add(kStoryVocabText[token]);
      if (ending.clean()) last_clean = generated;
    }
    if (progress != nullptr && progress->fn != nullptr && progress->every > 0 &&
        generated % progress->every == 0) {
      progress->fn(out, generated, progress->ctx);
    }
    if (generated >= max_new) { stats->stopped = "max"; break; }
  }
  if (!decode_reported) {
    printf("  temp: decode peaks at %u B (%u B internal + %u B PSRAM)\n",
           (unsigned)temp_allocator.peak(), (unsigned)sizeof(temp_arena), (unsigned)model.temp_size);
    decode_reported = true;
  }
  if (whole_sentences && last_clean > 0 &&
      (strcmp(stats->stopped, "max") == 0 || strcmp(stats->stopped, "context") == 0)) {
    generated = last_clean;
    stats->stopped = "trimmed";
  }
  return (int)generated;
}

static void story_detokenize(const int64_t* tokens, size_t count, char* text, size_t cap) {
  size_t used = 0;
  text[0] = 0;
  for (size_t i = 0; i < count; ++i) {
    const int64_t id = tokens[i];
    if (id < 0 || id >= STORY_VOCAB) continue;
    const char* piece = kStoryVocabText[id];
    const size_t n = strlen(piece);
    if (used + n + 1 > cap) break;
    memcpy(text + used, piece, n);
    used += n;
    text[used] = 0;
  }
}

static void story_print_escaped(const char* text) {
  for (const unsigned char* p = (const unsigned char*)text; *p; ++p) {
    if (*p == '\n') printf("\\n");
    else if (*p == '\\') printf("\\\\");
    else if (*p == '"') printf("\\\"");
    else if (*p < 32 || *p >= 127) printf("\\x%02x", *p);
    else putchar(*p);
  }
}

// "\n" and "\\" escapes in a request line become the real characters.
static void story_unescape(char* text) {
  char* w = text;
  for (const char* r = text; *r; ++r) {
    if (*r == '\\' && r[1] == 'n') { *w++ = '\n'; ++r; }
    else if (*r == '\\' && r[1] == '\\') { *w++ = '\\'; ++r; }
    else *w++ = *r;
  }
  *w = 0;
}

// What the card shows: an opening line the model continued is part of the
// story (without it the card starts mid-scene: " She was so proud..."); an
// instruction prompt ("...Story:\n") is not.
// The card's fonts (LVGL's built-in Montserrat) are ASCII only: a curly quote
// draws as an empty box ("□Let□s keep playing,□"). Curly quotes, apostrophes,
// dashes and the ellipsis are 99.6% of the non-ASCII characters in the
// training stories; they map to ASCII, a few accented letters to their base
// letter, anything else is dropped. In place; the text only shrinks.
static void story_ascii(char* text) {
  unsigned char* w = reinterpret_cast<unsigned char*>(text);
  for (const unsigned char* r = w; *r;) {
    if (*r < 0x80) { *w++ = *r++; continue; }
    const int len = *r >= 0xF0 ? 4 : *r >= 0xE0 ? 3 : *r >= 0xC0 ? 2 : 1;
    uint32_t cp = len == 1 ? *r : (*r & (0x7F >> len));
    int i = 1;
    for (; i < len && (r[i] & 0xC0) == 0x80; ++i) cp = (cp << 6) | (r[i] & 0x3F);
    r += i;
    const char* map = "";
    switch (cp) {
      case 0x201C: case 0x201D: case 0x00AB: case 0x00BB: map = "\""; break;
      case 0x2018: case 0x2019: map = "'"; break;
      case 0x2013: case 0x2014: map = "-"; break;
      case 0x2026: map = "..."; break;
      case 0x00A0: map = " "; break;
      case 0x00E0: case 0x00E1: case 0x00E2: case 0x00E4: map = "a"; break;
      case 0x00E8: case 0x00E9: case 0x00EA: case 0x00EB: map = "e"; break;
      case 0x00EC: case 0x00ED: case 0x00EE: case 0x00EF: map = "i"; break;
      case 0x00F2: case 0x00F3: case 0x00F4: case 0x00F6: map = "o"; break;
      case 0x00F9: case 0x00FA: case 0x00FB: case 0x00FC: map = "u"; break;
      case 0x00F1: map = "n"; break;
      case 0x00E7: map = "c"; break;
      default: break;
    }
    // "..." is the only mapping longer than one byte, and it replaces a
    // three-byte character, so writing never overtakes reading.
    for (; *map; ++map) *w++ = static_cast<unsigned char>(*map);
  }
  *w = 0;
}

static void story_compose(const char* prompt_text, const char* generated,
                          char* card, size_t cap) {
  size_t used = 0;
  card[0] = 0;
  if (strstr(prompt_text, "Story:\n") == nullptr) {
    used = strlen(prompt_text);
    if (used > cap - 1) used = cap - 1;
    memcpy(card, prompt_text, used);
  } else {
    while (*generated == ' ' || *generated == '\n') ++generated;
  }
  size_t n = strlen(generated);
  if (used + n > cap - 1) n = cap - 1 - used;
  memcpy(card + used, generated, n);
  card[used + n] = 0;
  story_ascii(card);
}

// Token guard: bounds a story's time and the 256-token context. Most
// stories end on their own well before it.
constexpr size_t kStoryMaxNewTokens = 230;
constexpr int kStoryMaxPages = 6;
constexpr const char* kStoryBrand = "nanoGlimmer";

// The premise of a new story: a random opening line the base model carries
// on from, and a title made from its parts ("Zoe and the Magic Stone": the
// model never writes one, so no training change). Names match the character
// (a girl's name for "a little girl": a mismatch makes the model mix she and
// he); animals come as "a little fox named Mia who found ...", because "Mia
// the fox found ..." reads as two characters.
static const char* const kStoryGirls[] = {
    "Lily", "Mia", "Lucy", "Anna", "Zoe", "Emma", "Rosie", "Sara",
    "Ella", "Amy", "Kate", "Nora", "Ruby", "Grace", "Molly", "Daisy"};
static const char* const kStoryBoys[] = {
    "Tom", "Ben", "Sam", "Max", "Tim", "Jack", "Leo", "Finn",
    "Noah", "Owen", "Jake", "Luke", "Eli", "Hugo", "Ollie", "Joe"};
static const char* const kStoryAnimals[] = {
    "dog", "cat", "bird", "bear", "mouse", "frog", "fox", "bunny",
    "duck", "owl", "puppy", "kitten", "turtle", "squirrel", "lion", "pig"};
static const char* const kStoryObjects[] = {
    "a red ball", "a striped scarf", "a shiny key", "a blue kite", "a toy boat",
    "a big box", "a yellow hat", "a magic stone", "an old map", "a paper crown",
    "a silver bell", "a tiny drum", "a golden feather", "a green umbrella",
    "a wooden flute", "a little seed", "a lost glove", "a bright lantern",
    "a funny sock", "a pair of red boots"};
static const char* const kStoryObjectTitles[] = {
    "the Red Ball", "the Striped Scarf", "the Shiny Key", "the Blue Kite", "the Toy Boat",
    "the Big Box", "the Yellow Hat", "the Magic Stone", "the Old Map", "the Paper Crown",
    "the Silver Bell", "the Tiny Drum", "the Golden Feather", "the Green Umbrella",
    "the Wooden Flute", "the Little Seed", "the Lost Glove", "the Bright Lantern",
    "the Funny Sock", "the Red Boots"};
static_assert(sizeof(kStoryObjects) == sizeof(kStoryObjectTitles), "one title per object");

template <size_t N>
static const char* story_pick(const char* const (&list)[N]) {
  return list[story_below(N)];
}

struct StoryPremise {
  char text[160];
  char title[64];
};

static void story_premise(StoryPremise* premise) {
  const uint32_t kind = story_below(10);
  const size_t object = story_below(sizeof(kStoryObjects) / sizeof(kStoryObjects[0]));
  const char* name;
  if (kind < 4) {
    name = story_pick(kStoryGirls);
    snprintf(premise->text, sizeof(premise->text),
             "Once upon a time, there was a little girl named %s who had %s.", name, kStoryObjects[object]);
  } else if (kind < 8) {
    name = story_pick(kStoryBoys);
    snprintf(premise->text, sizeof(premise->text),
             "Once upon a time, there was a little boy named %s who had %s.", name, kStoryObjects[object]);
  } else {
    name = story_below(2) ? story_pick(kStoryGirls) : story_pick(kStoryBoys);
    snprintf(premise->text, sizeof(premise->text),
             "Once upon a time, there was a little %s named %s who found %s.",
             story_pick(kStoryAnimals), name, kStoryObjects[object]);
  }
  snprintf(premise->title, sizeof(premise->title), "%s and %s", name, kStoryObjectTitles[object]);
}

// A finished story: what the reader sees and how it was written. Kept in NVS
// between stories, so the next story is ready even after a reboot.
constexpr uint32_t kStoryDraftMagic = 0x53545931;  // "STY1"
struct StoryDraft {
  uint32_t magic;
  char title[64];
  char text[kStoryMaxText];
  char stopped[16];
  uint16_t prompt_tokens;
  uint16_t generated;
  uint16_t decode_steps;
  uint32_t decode_ms;
};

static void story_draft_save(const StoryDraft& draft) {
  nvs_handle_t handle;
  esp_err_t error = nvs_open("story", NVS_READWRITE, &handle);
  if (error == ESP_OK) {
    error = nvs_set_blob(handle, "next", &draft, sizeof(draft));
    if (error == ESP_OK) error = nvs_commit(handle);
    nvs_close(handle);
  }
  printf("STORY_SAVED title=\"%s\" result=%s error=%d\n", draft.title,
         error == ESP_OK ? "OK" : "FAIL", (int)error);
}

static bool story_draft_take(StoryDraft* draft) {
  nvs_handle_t handle;
  if (nvs_open("story", NVS_READWRITE, &handle) != ESP_OK) return false;
  size_t size = sizeof(*draft);
  const bool ok = nvs_get_blob(handle, "next", draft, &size) == ESP_OK &&
                  size == sizeof(*draft) && draft->magic == kStoryDraftMagic;
  // A story is shown once: the next boot or key gets a new one.
  nvs_erase_key(handle, "next");
  nvs_commit(handle);
  nvs_close(handle);
  return ok;
}

// The story on the panel: its title, text, and where each page starts.
struct StoryBook {
  char title[64];
  char text[kStoryMaxText];
  int starts[kStoryMaxPages];
  int pages;
  int page;
};
EXT_RAM_BSS_ATTR static StoryBook g_book = {};

static void story_page_text(const StoryBook& book, int page, char* out, size_t cap) {
  const int len = (int)strlen(book.text);
  const int from = book.starts[page];
  int to = page + 1 < book.pages ? book.starts[page + 1] : len;
  while (to > from && (book.text[to - 1] == ' ' || book.text[to - 1] == '\n')) --to;
  size_t n = (size_t)(to - from);
  if (n > cap - 1) n = cap - 1;
  memcpy(out, book.text + from, n);
  out[n] = 0;
}

// Lay a text out in pages: balanced, measured with the display's own fonts.
static void story_book_set(StoryBook* book, const char* title, const char* text) {
  snprintf(book->title, sizeof(book->title), "%s", title);
  snprintf(book->text, sizeof(book->text), "%s", text);
#ifdef ET_PAPERCOLOR_DISPLAY
  book->pages = et_papercolor::paginateStory(book->text, book->starts, kStoryMaxPages);
#else
  book->starts[0] = 0;
  book->pages = 1;
#endif
  book->page = 0;
}

// The current page to the panel: title, page number (or `corner`), text.
// fast: the interrupted waveform (0.57 s) for page turns; a new story's first
// page goes out with the complete one (about 15 s, clean). wait: block until
// the panel shows it.
static void story_book_show(const StoryBook& book, const char* corner, bool fast, bool wait,
                            const char* title = nullptr) {
#ifdef ET_PAPERCOLOR_DISPLAY
  EXT_RAM_BSS_ATTR static char page[kStoryMaxText];
  char label[16];
  story_page_text(book, book.page, page, sizeof(page));
  if (corner != nullptr) snprintf(label, sizeof(label), "%s", corner);
  else if (book.pages > 1) snprintf(label, sizeof(label), "%d/%d", book.page + 1, book.pages);
  else label[0] = 0;
  et_papercolor::postStoryPage(title != nullptr ? title : book.title, label, page, fast);
  if (wait) {
    const bool shown = et_papercolor::flush(60'000);
    printf("PAPERCOLOR_STATUS page=%d/%d final=%s\n", book.page + 1, book.pages, shown ? "YES" : "NO");
  }
#else
  (void)book; (void)corner; (void)fast; (void)wait; (void)title;
#endif
}

struct StorySettings {
  float temperature = 0.8f;
  int top_k = 40;
  size_t max_new = kStoryMaxNewTokens;
};

// Write a whole story from an opening line into `draft`. Nothing reaches the
// panel while it is written: stories appear finished. `hook` runs every token
// (the reader's keys stay live during a background write).
static bool story_write(const StoryModel& model, const char* prompt_text, const char* title,
                        const StorySettings& settings, StoryDraft* draft, const StoryProgress* hook) {
  EXT_RAM_BSS_ATTR static uint16_t ids[kStoryMaxPrompt];
  EXT_RAM_BSS_ATTR static int64_t prompt[kStoryMaxPrompt];
  EXT_RAM_BSS_ATTR static int64_t out[kStoryMaxNew + 64];
  EXT_RAM_BSS_ATTR static char text[kStoryMaxText];
  const size_t n = bpe_encode(prompt_text, strlen(prompt_text), ids, kStoryMaxPrompt);
  if (n == 0 || n > kStoryMaxPrompt || n + 8 >= (size_t)STORY_CONTEXT) {
    printf("STORY_RESULT error=\"prompt length\" prompt_tokens=%u\n", (unsigned)n);
    return false;
  }
  for (size_t i = 0; i < n; ++i) prompt[i] = ids[i];
  StoryStats stats;
  const size_t room = (size_t)STORY_CONTEXT - 1 - n;
  const size_t max_new = std::min(std::min(settings.max_new, room), kStoryMaxNew + 63);
  const int generated = story_generate(model, prompt, n, max_new, true, settings.temperature,
                                       settings.top_k, out, &stats, nullptr, nullptr, 0,
                                       nullptr, hook);
  if (stats.error != nullptr) {
    printf("STORY_RESULT error=\"%s\" prompt_tokens=%u\n", stats.error, (unsigned)n);
    return false;
  }
  story_detokenize(out, (size_t)generated, text, sizeof(text));
  memset(draft, 0, sizeof(*draft));
  draft->magic = kStoryDraftMagic;
  snprintf(draft->title, sizeof(draft->title), "%s", title);
  story_compose(prompt_text, text, draft->text, sizeof(draft->text));
  snprintf(draft->stopped, sizeof(draft->stopped), "%s", stats.stopped);
  draft->prompt_tokens = (uint16_t)n;
  draft->generated = (uint16_t)generated;
  draft->decode_steps = (uint16_t)stats.decode_steps;
  draft->decode_ms = (uint32_t)(stats.decode_us / 1000);
  printf("STORY_WRITTEN title=\"%s\" generated=%d stopped=%s decode_ms=%lu\n",
         draft->title, generated, stats.stopped, (unsigned long)draft->decode_ms);
  return true;
}

// Put a finished story on the panel, first page, complete refresh.
static void story_show(const StoryDraft& draft) {
  story_book_set(&g_book, draft.title, draft.text);
  printf("STORY_SHOWN title=\"%s\" pages=%d\n", g_book.title, g_book.pages);
  printf("STORY_TEXT \"");
  story_print_escaped(g_book.text);
  printf("\"\n");
  printf("STORY_RESULT prompt_tokens=%u generated=%u stopped=%s pages=%d prefill_blocks=0 prefill_us=0 "
         "decode_steps=%u decode_us=%llu\n",
         (unsigned)draft.prompt_tokens, (unsigned)draft.generated, draft.stopped, g_book.pages,
         (unsigned)draft.decode_steps, (unsigned long long)draft.decode_ms * 1000ULL);
  story_book_show(g_book, nullptr, false, true);
}

// Whether page turns and the Writing page use the fast refresh. Off: on this
// colour panel the cut-short waveform leaves the page half black at 340 ms and
// mostly black at 2000 ms, so they use the complete refresh (~15 s, always
// clean). Console `X <ms>` turns fast on for testing,
// `X 0` off again.
static bool g_turn_fast = false;

static void story_turn(int delta, bool wait) {
  const int page = g_book.page + delta;
  if (g_book.pages > 1 && page >= 0 && page < g_book.pages) {
    g_book.page = page;
    story_book_show(g_book, nullptr, g_turn_fast, wait);
  }
  printf("STORY_PAGE %d/%d\n", g_book.page + 1, g_book.pages);
}

// The PaperColor's three user keys (M5Stack pin map): button A on GPIO10,
// B on GPIO9, C on GPIO1. On the side, A is the top key and B the middle one
// (the lowest side key is power/reset, not ours), so the side pair turns
// pages: A back, B next; C shows a new story. The idle
// level is read at start-up (no key held), so a press is any change from it,
// however the board wires the keys. A small task polls them every 10 ms and
// queues presses: between tokens (about 250 ms) a quick press would be missed.
static const gpio_num_t kStoryKeys[3] = {GPIO_NUM_10, GPIO_NUM_9, GPIO_NUM_1};
static const char kStoryKeyNames[3] = {'A', 'B', 'C'};
static int g_key_idle[3];
static int g_key_state[3];  // debounced: 1 while held
static int g_key_count[3];  // consecutive polls that disagree with the state
static QueueHandle_t g_key_queue = nullptr;

static char story_keys_poll(void) {
  char pressed = 0;
  for (int i = 0; i < 3; ++i) {
    const int held = gpio_get_level(kStoryKeys[i]) != g_key_idle[i];
    if (held == g_key_state[i]) { g_key_count[i] = 0; continue; }
    if (++g_key_count[i] < 3) continue;
    g_key_state[i] = held;
    g_key_count[i] = 0;
    if (held && pressed == 0) pressed = kStoryKeyNames[i];
  }
  return pressed;
}

static void story_key_task(void*) {
  for (;;) {
    const char key = story_keys_poll();
    if (key != 0) xQueueSend(g_key_queue, &key, 0);
    vTaskDelay(pdMS_TO_TICKS(10));
  }
}

static void story_keys_start(void) {
  for (int i = 0; i < 3; ++i) {
    gpio_config_t config = {};
    config.pin_bit_mask = 1ULL << kStoryKeys[i];
    config.mode = GPIO_MODE_INPUT;
    config.pull_up_en = GPIO_PULLUP_ENABLE;
    gpio_config(&config);
  }
  vTaskDelay(pdMS_TO_TICKS(5));
  for (int i = 0; i < 3; ++i) {
    g_key_idle[i] = gpio_get_level(kStoryKeys[i]);
    g_key_state[i] = 0;
    g_key_count[i] = 0;
  }
  printf("STORY_KEYS idle A(GPIO10)=%d B(GPIO9)=%d C(GPIO1)=%d\n",
         g_key_idle[0], g_key_idle[1], g_key_idle[2]);
  g_key_queue = xQueueCreate(8, sizeof(char));
  xTaskCreatePinnedToCore(story_key_task, "story_keys", 2048, nullptr, 5, nullptr, 0);
}

static char story_key_take(void) {
  char key = 0;
  if (g_key_queue != nullptr && xQueueReceive(g_key_queue, &key, 0) == pdTRUE) {
    printf("STORY_KEY %c\n", key);
    return key;
  }
  return 0;
}

// A fixed two-page document for testing page turns without the model: the
// console's 'L' puts it on the panel (the reader's keys then turn its pages).
static const char kStoryNavSample[] =
    "This is the page turning test. It is not a story the model wrote: it is "
    "fixed text, so it always fills two pages and always looks the same.\n"
    "Press the lower side key to go to the next page. The screen flashes for "
    "about fifteen seconds, then the number in the top right corner should "
    "say 2/2 and the text should change.\n"
    "Then press the upper side key to come back. The corner should say 1/2 "
    "again, and this page should look the same as it did before.\n"
    "After each turn the letters should be black and the paper white. A grey, "
    "dark or half black page means something is wrong.\n"
    "If you are reading this on page 2/2, the next page key works. This page "
    "is the second half of the test, and it has a few more lines than it needs "
    "so that it is not almost empty.\n"
    "Now press the upper side key to go back to page one. Then press the key on "
    "the top edge to get a new story from the model.\n"
    "The end of the page turning test.";

// The next story, written ahead in the background, and whether the reader has
// asked for it while it was still being written.
EXT_RAM_BSS_ATTR static StoryDraft g_next = {};
static bool g_story_shown_at_boot = false;
static bool g_next_ready = false;
static bool g_want_new = false;

// The panel while a story the reader asked for is still being written: one
// fast frame, a plain "Writing" page, so the press visibly did something;
// the story replaces it with a complete refresh when it is done.
static void story_show_writing(void) {
#ifdef ET_PAPERCOLOR_DISPLAY
  et_papercolor::postStoryPage(kStoryBrand, "", "Writing a new story...", g_turn_fast);
#endif
  printf("STORY_WRITING\n");
}

// Keys while a story is written in the background: pages still turn, and a
// new-story press puts up the Writing page and is remembered until the story
// is done.
static void story_key_hook(const int64_t*, size_t, void*) {
  const char key = story_key_take();
  if (key == 'A') {
    story_turn(-1, false);
  } else if (key == 'B') {
    story_turn(+1, false);
  } else if (key == 'C' && !g_want_new) {
    g_want_new = true;
    story_show_writing();
  }
}

// Every story fills two pages, so the side keys always have a page to turn
// to, and its second page is at least a third full (pages fill like a book's,
// so a story just over one page would otherwise leave a few lines alone on
// page two). One that falls short is put aside and another one written (in
// the background, where nobody waits for it); one that runs past two pages
// loses whole paragraphs, then sentences, from its end.
constexpr int kStoryTargetPages = 2;
constexpr int kStoryMinLastFill = 33;  // percent of a page
constexpr int kStoryAttempts = 1;  // rewrites off

static int story_pages_of(const char* text, int* last_fill = nullptr) {
#ifdef ET_PAPERCOLOR_DISPLAY
  static int starts[kStoryMaxPages];
  const int pages = et_papercolor::paginateStory(text, starts, kStoryMaxPages);
  if (last_fill != nullptr) *last_fill = et_papercolor::storyLastFill(text, starts, pages);
  return pages;
#else
  (void)text;
  if (last_fill != nullptr) *last_fill = 100;
  return kStoryTargetPages;
#endif
}

static void story_trim_to_pages(char* text) {
  while (story_pages_of(text) > kStoryTargetPages) {
    const size_t len = strlen(text);
    size_t cut = 0;
    for (size_t i = len; i-- > 1;) {           // the last paragraph break
      if (text[i] == '\n') { cut = i; break; }
    }
    if (cut == 0) {                             // one paragraph: its last sentence
      for (size_t i = len - 1; i-- > 1;) {
        if ((text[i] == '.' || text[i] == '!' || text[i] == '?') &&
            (text[i + 1] == ' ' || text[i + 1] == '"')) {
          cut = i + (text[i + 1] == '"' ? 2 : 1);
          break;
        }
      }
    }
    if (cut == 0) return;
    text[cut] = 0;
    while (cut > 0 && (text[cut - 1] == ' ' || text[cut - 1] == '\n')) text[--cut] = 0;
  }
}

static bool story_write_paged(const StoryModel& model, const StorySettings& settings,
                              StoryDraft* draft, const StoryProgress* hook, int attempts) {
  for (int attempt = 1; attempt <= attempts; ++attempt) {
    StoryPremise premise;
    story_premise(&premise);
    printf("STORY_PREMISE \"%s\"\n", premise.text);
    if (!story_write(model, premise.text, premise.title, settings, draft, hook)) return false;
    int last_fill = 0;
    const int pages = story_pages_of(draft->text, &last_fill);
    if (pages > kStoryTargetPages) {
      story_trim_to_pages(draft->text);
      printf("STORY_TRIMMED title=\"%s\" from=%d to=%d\n", draft->title, pages, story_pages_of(draft->text));
      return true;
    }
    // Someone is looking at the Writing page: show this one, whatever its length.
    const bool fits = pages == kStoryTargetPages && last_fill >= kStoryMinLastFill;
    if (fits || attempt == attempts || g_want_new) return true;
    printf("STORY_REJECTED title=\"%s\" pages=%d last_fill=%d%%\n", draft->title, pages, last_fill);
  }
  return true;
}

// Write the next story now, keys live; show it at once if it was asked for.
static void story_write_next(const StoryModel& model, const StorySettings& settings) {
  const StoryProgress hook = {story_key_hook, nullptr, 1};
  g_next_ready = story_write_paged(model, settings, &g_next, &hook, kStoryAttempts);
  if (!g_next_ready) return;
  story_draft_save(g_next);
  if (g_want_new) {
    g_want_new = false;
    g_next_ready = false;
    story_draft_take(&g_next);  // consumed: drop the saved copy
    story_show(g_next);
  }
}

// The reader: keys and console lines, until 'Q'. Whenever no story is waiting
// it writes the next one (keys stay live); otherwise it waits for input.
//   G              show the next story (key C)
//   R <text>       write and show a story from this opening line (\n escapes)
//   D / U          next / previous page (keys B / A: the side pair)
//   F <px>         story face 14, 16, 18 or 20 px; the story is laid out again
//   T / K / S / N  temperature, top-k, seed, token guard
static void story_serve(const StoryModel& model) {
  EXT_RAM_BSS_ATTR static char line[kStoryMaxLine];
  size_t line_len = 0;
  StorySettings settings;
  printf("=== REQUEST READY ===\n");
  fflush(stdout);
  for (;;) {
    if (!g_next_ready) {
      story_write_next(model, settings);
      continue;
    }
    char command = 0;
    const int byte = getchar();
    if (byte == EOF) {
      clearerr(stdin);
      const char key = story_key_take();
      if (key == 0) {
        vTaskDelay(pdMS_TO_TICKS(10));
        continue;
      }
      command = key == 'A' ? 'U' : key == 'B' ? 'D' : 'G';
      line[0] = 0;
    } else {
      if (byte == '\r') continue;
      if (byte != '\n') {
        if (line_len + 1 < sizeof(line)) line[line_len++] = (char)byte;
        continue;
      }
      line[line_len] = 0;
      line_len = 0;
      if (line[0] == 0) {
        // A blank line asks "ready?" (the host's first nudge after connecting).
        printf("=== REQUEST READY ===\n");
        fflush(stdout);
        continue;
      }
      command = line[0];
    }
    char* arg = (line[0] != 0 && line[1] == ' ') ? line + 2 : line + strlen(line);
    switch (command) {
      case 'Q':
        return;
      case 'G':
        g_next_ready = false;
        story_draft_take(&g_next);  // consumed: drop the saved copy
        story_show(g_next);
        break;
      case 'R': {
        EXT_RAM_BSS_ATTR static StoryDraft custom;
        story_unescape(arg);
        if (story_write(model, arg, kStoryBrand, settings, &custom, nullptr)) story_show(custom);
        break;
      }
      case 'D':
        story_turn(+1, true);
        break;
      case 'U':
        story_turn(-1, true);
        break;
#ifdef ET_PAPERCOLOR_DISPLAY
      case 'F': {
        et_papercolor::setStoryFontPx(atoi(arg));
        static char title[64];
        snprintf(title, sizeof(title), "%s", g_book.title);
        story_book_set(&g_book, title, g_book.text);
        printf("STORY_SET font_px=%d pages=%d\n", et_papercolor::storyFontPx(), g_book.pages);
        story_book_show(g_book, nullptr, false, true);
        break;
      }
      case 'X': {
        // Refresh test: page turns go fast with this cut-off (0: complete,
        // the default), and this page is redrawn that way, the cut-off named
        // in its title.
        const int ms = atoi(arg);
        g_turn_fast = ms > 0;
        if (g_turn_fast) et_papercolor::setFastCutoffMs((uint16_t)ms);
        static char title[64];
        if (g_turn_fast) snprintf(title, sizeof(title), "Fast refresh %u ms", (unsigned)et_papercolor::fastCutoffMs());
        else snprintf(title, sizeof(title), "%s", g_book.title);
        story_book_show(g_book, nullptr, g_turn_fast, true, title);
        break;
      }
      case 'P':
        story_book_show(g_book, nullptr, false, true);  // a clean, complete redraw
        break;
      case 'L':
        story_book_set(&g_book, "Page turning test", kStoryNavSample);
        printf("STORY_SHOWN title=\"%s\" pages=%d\n", g_book.title, g_book.pages);
        story_book_show(g_book, nullptr, false, true);
        break;
#endif
      case 'T':
        settings.temperature = strtof(arg, nullptr);
        printf("STORY_SET temperature=%.3f\n", (double)settings.temperature);
        break;
      case 'K':
        settings.top_k = atoi(arg);
        printf("STORY_SET top_k=%d\n", settings.top_k);
        break;
      case 'S':
        story_seed((uint32_t)strtoul(arg, nullptr, 10));
        printf("STORY_SET seed=%lu\n", (unsigned long)g_story_rng);
        break;
      case 'N':
        settings.max_new = std::min((size_t)atoi(arg), kStoryMaxNewTokens);
        printf("STORY_SET max_new=%u\n", (unsigned)settings.max_new);
        break;
      default:
        printf("STORY_RESULT error=\"bad line\"\n");
        break;
    }
    printf("=== REQUEST DONE ===\n=== REQUEST READY ===\n");
    fflush(stdout);
  }
}

static void run_story(void) {
#ifdef ET_PAPERCOLOR_DISPLAY
  // Every story card is read by a person: the complete waveform from the
  // first frame, never the interrupted one that goes half black.
  et_papercolor::setFastRefresh(false);
#endif
  // A different run of premises every boot (a fixed seed would repeat the same
  // stories, in the same order, after each power-on).
  story_seed(esp_random() ^ (uint32_t)esp_timer_get_time());
  story_keys_start();
  // A story written before the last power-off goes on the panel at once,
  // while the model loads and checks itself.
  nvs_flash_init();
  {
    EXT_RAM_BSS_ATTR static StoryDraft saved;
    if (story_draft_take(&saved)) {
      story_book_set(&g_book, saved.title, saved.text);
      printf("STORY_SHOWN title=\"%s\" pages=%d saved=YES\n", g_book.title, g_book.pages);
      story_book_show(g_book, nullptr, false, false);
      g_story_shown_at_boot = true;
    }
  }
  printf("--- ple.pte  [story Glimmer V%d D%d L%d, context %d, prefill %d] ---\n",
         STORY_VOCAB, STORY_D_MODEL, STORY_LAYERS, STORY_CONTEXT, STORY_PREFILL_CHUNK);
#if defined(ET_STORY_DUAL_CORE) && ET_STORY_DUAL_CORE
  {
    // The spare core belongs to inference here: every projection's column
    // loop is split across both LX7 cores (the backend's min-latency posture,
    // a runtime option so no re-export is needed).
    executorch::runtime::BackendOptions<1> on;
    on.set_option("min_latency", true);
    const Error e = executorch::runtime::set_option("Esp32Backend", on.view());
    printf("  dual-core min_latency: %s\n", e == Error::Ok ? "ON" : "REFUSED");
  }
#endif
  printf("  checkpoint sha256=%s\n", kStoryCheckpointSha256);
  auto prog_loader = esp32s3::EspPartitionDataLoader::from_subtype(0x43);
  auto wts_loader = esp32s3::EspPartitionDataLoader::from_subtype(0x44);
  if (!prog_loader.ok() || !wts_loader.ok()) { printf("  partitions 0x43/0x44 absent\n"); return; }
  const int64_t program_begin = esp_timer_get_time();
  auto program = Program::load(&prog_loader.get());
  if (!program.ok()) { printf("  Program::load FAILED: %d\n", (int)program.error()); return; }
  if (program->num_methods() != 2) { printf("  expected decode + prefill, found %u\n", (unsigned)program->num_methods()); return; }
  auto weights = executorch::extension::FlatTensorDataMap::load(&wts_loader.get());
  if (!weights.ok()) { printf("  FlatTensorDataMap::load FAILED: %d\n", (int)weights.error()); return; }
  const int64_t program_load_us = esp_timer_get_time() - program_begin;
  auto decode_meta = program->method_meta("decode");
  auto prefill_meta = program->method_meta("prefill");
  if (!decode_meta.ok() || !prefill_meta.ok()) { printf("  method metadata FAILED\n"); return; }
  const size_t n_planned = decode_meta->num_memory_planned_buffers();
  if (n_planned != 2 || prefill_meta->num_memory_planned_buffers() != n_planned) {
    printf("  expected shared mem-id-1/2 plans in both methods\n"); return;
  }
  auto decode_act = decode_meta->memory_planned_buffer_size(0);
  auto prefill_act = prefill_meta->memory_planned_buffer_size(0);
  auto decode_kv = decode_meta->memory_planned_buffer_size(1);
  auto prefill_kv = prefill_meta->memory_planned_buffer_size(1);
  if (!decode_act.ok() || !prefill_act.ok() || !decode_kv.ok() || !prefill_kv.ok() ||
      *decode_kv != *prefill_kv) {
    printf("  planned-buffer ABI mismatch\n"); return;
  }
  // The KV cache first, while internal SRAM is whole. Then the activations:
  // one buffer for both methods, in PSRAM, sized to the larger (prefill's);
  // or, with decode_act_internal, decode's own in internal SRAM when it fits
  // within the floor (prefill's stays in PSRAM, where it was), before the
  // loads copy any small constant. PSRAM gets the same allocations in the
  // same order either way.
  uint8_t* kv = story_alloc("shared mem-id-2", (size_t)*decode_kv, kInternalFirst);
  if (kv == nullptr) return;
  memset(kv, 0, (size_t)*decode_kv);
  const size_t act_size = std::max((size_t)*decode_act, (size_t)*prefill_act);
  const bool own_decode = kStoryPlacement.decode_act_internal != 0;
  uint8_t* act = story_alloc(own_decode ? "prefill mem-id-1" : "shared mem-id-1", act_size, kPsramFirst);
  if (act == nullptr) return;
  memset(act, 0, act_size);
  uint8_t* decode_own = own_decode
      ? story_alloc("decode mem-id-1", (size_t)*decode_act, kInternalOnly) : nullptr;
  if (decode_own != nullptr) {
    memset(decode_own, 0, (size_t)*decode_act);
  } else if (own_decode) {
    printf("  decode mem-id-1: %u B shares prefill's (PSRAM): no internal room within the floor\n",
           (unsigned)*decode_act);
  }
  Span<uint8_t> prefill_spans[2] = {Span<uint8_t>(act, act_size), Span<uint8_t>(kv, (size_t)*decode_kv)};
  Span<uint8_t> decode_spans[2] = {
      decode_own != nullptr ? Span<uint8_t>(decode_own, (size_t)*decode_act) : prefill_spans[0],
      prefill_spans[1]};
  uint8_t* planned[3] = {kv, act, decode_own};
  const size_t arena_size = kStoryPlacement.arena_bytes;
  uint8_t* arena = story_alloc("method arena", arena_size, kPsramFirst);
  const size_t temp_bytes = kStoryPlacement.temp_bytes;
  ScopedCapsBuffer large_temp(temp_bytes, MALLOC_CAP_SPIRAM);
  const int stage_bytes = kStoryPlacement.stage_bytes;
  const bool rank = kStoryPlacement.tier_rank != 0;
  char stage_budget[48];
  if (stage_bytes < 0) {
    snprintf(stage_budget, sizeof(stage_budget), "auto keeping %u B free", (unsigned)kStoryPlacement.tier_reserve);
  } else {
    snprintf(stage_budget, sizeof(stage_budget), "%u B", (unsigned)stage_bytes);
  }
  {
    // Weight tier and cache are deployment choices, made here through the
    // backend's option interface (the same door as min_latency). Ranked, the
    // copies are made by story_place_weights after each load instead of
    // inside it, the weights worth most first.
    executorch::runtime::BackendOptions<4> tier;
    tier.set_option("weight_tier_budget", stage_bytes);
    tier.set_option("weight_tier_reserve", (int)kStoryPlacement.tier_reserve);
    tier.set_option("weight_cache", true);
    tier.set_option("weight_tier_rank", rank);
    const Error e = executorch::runtime::set_option("Esp32Backend", tier.view());
    printf("  weight tier: budget %s, %s, cache on: %s\n", stage_budget, rank ? "ranked" : "load order",
           e == Error::Ok ? "OK" : "REFUSED");
  }
  printf("  PSRAM before model load: free %u B, largest %u B; i4 stage budget %s\n",
         (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
         (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM), stage_budget);
  // A temp tier of 0 is none: every scratch must then fit the 32 KB internal
  // tier (or be planned, plan_scratch).
  if (arena == nullptr || (temp_bytes > 0 && large_temp.data() == nullptr)) {
    printf("  method/temp allocation FAILED\n"); return;
  }
  printf("  temp: %u B internal + %u B PSRAM\n", (unsigned)sizeof(temp_arena), (unsigned)large_temp.size());
  // Every load reads the weights file through this: the small constants
  // outside the delegate come from internal SRAM, the rest as before.
  StoryHotConstants hot(&weights.get());
  StoryModel model = {&*program, prefill_spans, decode_spans, n_planned, &hot, &hot, arena, arena_size,
                      large_temp.data(), large_temp.size()};

  // The exporter's golden: greedy, with checksums.
  constexpr size_t kPromptN = sizeof(kStoryGoldenPromptTokens) / sizeof(kStoryGoldenPromptTokens[0]);
  constexpr size_t kGoldenN = sizeof(kStoryGoldenTokens) / sizeof(kStoryGoldenTokens[0]);
  constexpr size_t kMaxSums = 256;
  EXT_RAM_BSS_ATTR static double prefill_sums[kMaxSums];
  EXT_RAM_BSS_ATTR static double decode_sums[kMaxSums];
  EXT_RAM_BSS_ATTR static int64_t out[kStoryMaxNew];
  EXT_RAM_BSS_ATTR static char text[kStoryMaxText];
  EXT_RAM_BSS_ATTR static double rels[kMaxSums * 2];
  StoryStats stats;
  const int generated = story_generate(model, kStoryGoldenPromptTokens, kPromptN, kGoldenN, false, 0.0f, 1,
                                       out, &stats, prefill_sums, decode_sums, kMaxSums,
                                       kStoryGoldenTokens);
  // The paths the standard operators took over exactly the golden's
  // executions (the first story, written below, adds its own).
  espressif::report_override_dispatch();
  if (stats.error != nullptr) {
    printf("  STORY_GOLDEN result=ERROR error=%s\n", stats.error);
  } else {
    const size_t n_pref = sizeof(kStoryGoldenPrefillChecksums) / sizeof(double);
    const size_t n_dec = sizeof(kStoryGoldenDecodeChecksums) / sizeof(double);
    double max_rel = 0.0;
    size_t n_rels = 0;
    for (size_t i = 0; i < n_pref && i < (size_t)stats.prefill_blocks; ++i) {
      const double rel = fabs(prefill_sums[i] - kStoryGoldenPrefillChecksums[i]) / fmax(fabs(kStoryGoldenPrefillChecksums[i]), 1e-9);
      if (rel > max_rel) max_rel = rel;
      rels[n_rels++] = rel;
      printf("  prefill block %u: checksum %.6f golden %.6f rel %.3e\n", (unsigned)i, prefill_sums[i], kStoryGoldenPrefillChecksums[i], rel);
    }
    for (size_t i = 0; i < n_dec && i < (size_t)stats.decode_steps; ++i) {
      const double rel = fabs(decode_sums[i] - kStoryGoldenDecodeChecksums[i]) / fmax(fabs(kStoryGoldenDecodeChecksums[i]), 1e-9);
      if (rel > max_rel) max_rel = rel;
      rels[n_rels++] = rel;
      if (i < 4 || i + 2 >= n_dec) {  // first four and last two print; every step is in max_rel
        printf("  decode step %u: checksum %.6f golden %.6f rel %.3e\n", (unsigned)i, decode_sums[i], kStoryGoldenDecodeChecksums[i], rel);
      }
    }
    // Choices: the device's own pick at every forced step. A different pick
    // is tolerated only where the reference's top two were within
    // STORY_GOLDEN_NEAR_TIE of each other; anywhere else it is a failure.
#ifndef STORY_GOLDEN_NEAR_TIE
#define STORY_GOLDEN_NEAR_TIE 0.1
#endif
    int matched = 0, first_diff = -1, near_ties = 0, hard = 0;
    for (size_t i = 0; i < kGoldenN && i < (size_t)generated; ++i) {
      if (out[i] == kStoryGoldenTokens[i]) { ++matched; continue; }
      if (first_diff < 0) first_diff = (int)i;
      const bool tie = kStoryGoldenMargins[i] < STORY_GOLDEN_NEAR_TIE;
      if (tie) ++near_ties; else ++hard;
      printf("  choice %u: device %lld, reference %lld, reference lead %.4f logit (%s)\n",
             (unsigned)i, (long long)out[i], (long long)kStoryGoldenTokens[i],
             (double)kStoryGoldenMargins[i], tie ? "near-tie" : "FAIL");
    }
    story_detokenize(out, (size_t)generated, text, sizeof(text));
    printf("  device greedy: \"");
    story_print_escaped(text);
    printf("\"\n  golden greedy: \"");
    story_print_escaped(kStoryGoldenText);
    printf("\"\n");
    // Values, not only choices: each step's logit checksum against the
    // exporter's device-arithmetic reference, gated on the first steps only.
    // Step 0 runs every layer and the head, so a kernel error shows there.
    // Later steps are reported, not gated: the dynamic int8 activation
    // rounding turns ~1e-6 differences (rsqrt, libm) into an occasional
    // one-code flip that the KV cache carries forward, and the reference
    // nudged by 1e-6 against itself drifts up to 0.35 the same way.
#ifndef STORY_GOLDEN_TOL_EARLY
#define STORY_GOLDEN_TOL_EARLY 1e-5
#endif
#ifndef STORY_GOLDEN_EARLY_STEPS
#define STORY_GOLDEN_EARLY_STEPS 4
#endif
    double early_max = 0.0;
    int onset = -1;
    printf("  step drift:");
    for (size_t i = 0; i < n_rels; ++i) {
      printf(" %.0e", rels[i]);
      if (i < (size_t)STORY_GOLDEN_EARLY_STEPS && rels[i] > early_max) early_max = rels[i];
      if (onset < 0 && rels[i] > 1e-5) onset = (int)i;
    }
    printf("\n");
    std::sort(rels, rels + n_rels);
    const double median_rel = n_rels ? rels[n_rels / 2] : 0.0;
    const bool tokens_ok = hard == 0 && matched + near_ties == (int)kGoldenN;
    const bool values_ok = n_rels >= (size_t)STORY_GOLDEN_EARLY_STEPS &&
                           early_max <= STORY_GOLDEN_TOL_EARLY;
    printf("STORY_GOLDEN result=%s tokens_match=%d/%u near_ties=%d hard=%d first_diff=%d "
           "early_max=%.3e tol_early=%.1e early_steps=%d onset=%d median_rel=%.3e max_rel=%.3e values=%s "
           "prefill_blocks=%d prefill_us=%lld decode_steps=%d decode_us=%lld digest=0x%016llx\n",
           tokens_ok && values_ok ? "MATCH" : "MISMATCH", matched, (unsigned)kGoldenN, near_ties, hard,
           first_diff, early_max, (double)STORY_GOLDEN_TOL_EARLY, (int)STORY_GOLDEN_EARLY_STEPS, onset,
           median_rel, max_rel, values_ok ? "OK" : "DRIFT", stats.prefill_blocks,
           (long long)stats.prefill_us, stats.decode_steps, (long long)stats.decode_us,
           (unsigned long long)stats.digest);
    // The cold start beside decode's clock, as its own line (board comparisons
    // hold no time but decode's): the program and its data map loaded
    // (Program::load, FlatTensorDataMap::load), then each method's first
    // load_method and the ranked placement after it (story_place_weights;
    // with ranking off the copies are made inside the load). A method the
    // golden does not run shows 0: prefill, for a prompt shorter than a chunk.
    printf("STORY_COLD program_load_us=%lld prefill_load_us=%lld prefill_place_us=%lld "
           "decode_load_us=%lld decode_place_us=%lld decode_ms_per_token=%.2f\n",
           (long long)program_load_us, (long long)stats.prefill_load_us,
           (long long)stats.prefill_place_us, (long long)stats.decode_load_us,
           (long long)stats.decode_place_us,
           stats.decode_steps ? stats.decode_us / 1000.0 / stats.decode_steps : 0.0);
    // First story: none written yet on a first boot, so write one now; later
    // boots showed the saved one before the check (below, at the top).
    if (!g_story_shown_at_boot) {
      EXT_RAM_BSS_ATTR static StoryDraft first;
      StorySettings settings;
      story_show_writing();
      if (story_write_paged(model, settings, &first, nullptr, 1)) story_show(first);
    }
  }
  {
    uint32_t copies = 0, hits = 0;
    size_t resident = 0;
    esp32_backend_weight_cache_stats(&copies, &hits, &resident);
    printf("  weight cache: %u op copies, %u reuses, %u B resident\n",
           (unsigned)copies, (unsigned)hits, (unsigned)resident);
  }
  // The backend's counts since boot, as the parity firmware prints them:
  // where each delegate call's scratch came from (its planned buffer when the
  // program was exported with plan_scratch, else the temp allocator, and any
  // fallback), and PIE against scalar columns. Printed once, after the timed
  // check.
  esp32_backend_report();
  esp32_backend_report_timing();  // per-op totals when built with ESP32_BACKEND_TIMING
#ifdef ESP32_OP_TIMING
  g_story_op_tracer.report();
  g_story_op_tracer.reset();
#endif
#ifdef ET_PAPERCOLOR_DISPLAY
  printf("PAPERCOLOR_STATUS final=%s\n", et_papercolor::flush(60'000) ? "YES" : "NO");
#endif
  // The first story is written and on the panel (on a boot that showed a
  // saved story, only the golden ran): the floor once more, with the lowest
  // internal SRAM has been since boot.
  story_report_internal(g_story_shown_at_boot ? "after the golden" : "after the first story");
  printf("=== DONE ===\n");
  fflush(stdout);
  story_serve(model);
  printf("=== SERVE DONE ===\n");
  fflush(stdout);
  heap_caps_free(arena);
  for (uint8_t* allocation : planned) heap_caps_free(allocation);
}

}  // namespace
