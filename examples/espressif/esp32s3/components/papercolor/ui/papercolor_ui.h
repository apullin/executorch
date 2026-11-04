/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * The PaperColor status screens, built on LVGL.
 *
 * The firmware binds an LVGL display to the panel driver
 * (papercolor_display.cpp) and calls these; they depend on LVGL alone.
 *
 * The screen is 400x600, one bit per pixel. Cards are static: nothing
 * animates, every call replaces the whole frame and the caller decides when
 * to present it (e-paper refreshes are deliberate, seconds-long events).
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "lvgl.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PC_UI_WIDTH 400
#define PC_UI_HEIGHT 600

/** Create the screen objects on the given display. Call once after
 *  lv_display_create(). */
void pc_ui_init(lv_display_t *display);

/** A text card: a title strip, a headline, and a body that wraps. Long
 *  bodies are clipped at the footer with an ellipsis rather than overflowing. */
void pc_ui_show_text(const char *title, const char *headline, const char *body);

/** A story card: a title strip and a long body in the small face, wrapped,
 *  clipped at the footer with an ellipsis. */
void pc_ui_show_story(const char *title, const char *body);

/** Story face size in pixels: 14, 16, 18 or 20 (anything else keeps the
 *  current one). Default 18. */
void pc_ui_set_story_font_px(int px);
int pc_ui_story_font_px(void);

/** Split a story into pages that fit the story card at the current face.
 *  Writes the byte offset where each page starts into starts[] (starts[0]
 *  is 0) and returns the page count, at most max_pages. A story that needs
 *  more than one page is split into pages of about equal height, each break
 *  at a paragraph end or else a sentence end near the target, so no page is
 *  a few stranded lines. Measures with the story face and the card's text
 *  width and touches no LVGL objects: any task may call it after
 *  pc_ui_init. Not reentrant (one measuring buffer). */
int pc_ui_story_paginate(const char *text, int *starts, int max_pages);

/** How full the last page of a paginated story is, in percent. */
int pc_ui_story_last_fill(const char *text, const int *starts, int pages);

/** One page of a story: `corner` sits at the right of the title strip (the
 *  page number "1/2", or "writing..."), `page_text` fills the body. */
void pc_ui_show_story_page(const char *title, const char *corner, const char *page_text);

/** A tool card: the generated action and its argument above a rule, the live
 *  result and its detail below it. */
void pc_ui_show_tool(const char *title, const char *action, const char *argument,
                     const char *result, const char *detail);

/** Force the pending screen to render now (LVGL otherwise renders on its
 *  timer). Returns true when the display's flush callback ran. */
bool pc_ui_render(void);

/** Height, in pixels, the body label of the last text card needed, and the
 *  height it was given: a fit report. */
void pc_ui_last_fit(int32_t *needed, int32_t *available);

/** Called by the display's flush callback so pc_ui_render() can report
 *  whether a frame actually reached the flush. */
void pc_ui_mark_flushed(void);

#ifdef __cplusplus
}
#endif
