/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// ESP32_HOT_CODE marks a function a transformer decode runs on every token:
// the delegate's dispatch and row loops, the activation quantiser, the split
// worker, and the custom ops and fast paths between delegates. The set comes
// from a profile of the story Glimmer's decode on the host runner (calls per
// token), kept to what GCC leaves out of line. With
// CONFIG_ESP32_BACKEND_HOT_IRAM (the backend component's Kconfig) the marked
// functions run from internal RAM (IRAM_ATTR), and esp32_hot.lf moves the PIE
// kernels and ExecuTorch's instruction loop there too. Otherwise, and on the
// host, the mark is empty.
//
// The mark moves code without changing it: GCC keeps a function's section on
// its clones (constprop, isra) and inlines across sections. The exception is
// partial inlining: GCC does not split a function that names its section, so
// a function whose fast path it inlines and whose cold rest it outlines
// (ensure_worker) stays unmarked, or its callers would change.
#pragma once

#if defined(ESP32_HOST_BUILD)
#define ESP32_HOT_CODE
#else
#include <esp_attr.h>
#include <sdkconfig.h>
#if defined(CONFIG_ESP32_BACKEND_HOT_IRAM)
#define ESP32_HOT_CODE IRAM_ATTR
#else
#define ESP32_HOT_CODE
#endif
#endif
