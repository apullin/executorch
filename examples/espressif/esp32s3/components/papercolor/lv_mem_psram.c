/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * LVGL's heap on the board (LV_STDLIB_CUSTOM in ui/lv_conf.h): LVGL's own
 * C-library glue (src/stdlib/clib/lv_mem_core_clib.c) with PSRAM in place of
 * malloc's internal-first policy. The UI's objects, styles and draw buffers
 * are touched only while a card renders, a few times a minute; internal SRAM
 * belongs to the model's KV cache and scratch and to the instruction cache.
 * Anything PSRAM cannot hold falls back to the rest of the heap.
 *
 * The example project's CMakeLists.txt compiles this file into the lvgl
 * library, next to the calls it answers.
 */
#include "lvgl.h"

#if LV_USE_STDLIB_MALLOC == LV_STDLIB_CUSTOM
#include "esp_heap_caps.h"

#define LV_MEM_PSRAM_CAPS (MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)

void lv_mem_init(void)
{
    return; /*Nothing to init*/
}

void lv_mem_deinit(void)
{
    return; /*Nothing to deinit*/
}

lv_mem_pool_t lv_mem_add_pool(void * mem, size_t bytes)
{
    /*Not supported*/
    LV_UNUSED(mem);
    LV_UNUSED(bytes);
    return NULL;
}

void lv_mem_remove_pool(lv_mem_pool_t pool)
{
    /*Not supported*/
    LV_UNUSED(pool);
}

void * lv_malloc_core(size_t size)
{
    void * p = heap_caps_malloc(size, LV_MEM_PSRAM_CAPS);
    return p != NULL ? p : heap_caps_malloc(size, MALLOC_CAP_8BIT);
}

void * lv_realloc_core(void * p, size_t new_size)
{
    /*lv_realloc() never asks for 0 bytes, and a failed realloc leaves p intact*/
    void * moved = heap_caps_realloc(p, new_size, LV_MEM_PSRAM_CAPS);
    return moved != NULL ? moved : heap_caps_realloc(p, new_size, MALLOC_CAP_8BIT);
}

void lv_free_core(void * p)
{
    heap_caps_free(p);
}

void lv_mem_monitor_core(lv_mem_monitor_t * mon_p)
{
    /*Not supported*/
    LV_UNUSED(mon_p);
}

lv_result_t lv_mem_test_core(void)
{
    /*Not supported*/
    return LV_RESULT_OK;
}

#endif /*LV_STDLIB_CUSTOM*/
