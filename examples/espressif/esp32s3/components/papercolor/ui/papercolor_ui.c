/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "papercolor_ui.h"

#include <string.h>

/* Layout constants, in pixels on the 400x600 panel. One place, not per card. */
#define PC_MARGIN 20          /* frame inset from the panel edge */
#define PC_FRAME 3            /* frame stroke */
#define PC_PAD 22             /* text inset from the frame */
#define PC_RULE 3             /* rule thickness */
#define PC_TITLE_H 84         /* title strip height */
#define PC_FOOTER_H 56        /* footer strip height */

static lv_display_t *s_display;
static lv_obj_t *s_screen;
static lv_obj_t *s_title;
static lv_obj_t *s_corner;      /* right end of the title strip: page number */

/* The story screen: no frame, a one-line header (the story's title, the
 * page number), the text, and a footer saying what the keys do. About a
 * quarter more text per page than the framed card. */
#define ST_SIDE 18            /* text inset from the panel edge */
#define ST_HEAD_Y 12          /* title baseline row */
#define ST_HEAD_RULE_Y 46
#define ST_BODY_Y 58
#define ST_FOOT_RULE_Y 560
#define ST_FOOT_Y 570
static lv_obj_t *s_story_screen;
static lv_obj_t *s_story_title;
static lv_obj_t *s_story_page;
static lv_obj_t *s_story_body;
static int32_t s_story_w, s_story_body_h;
static lv_obj_t *s_title_rule;
static lv_obj_t *s_footer_rule;
static lv_obj_t *s_footer;
static lv_obj_t *s_body_area;   /* the region between the two rules */
static bool s_flushed;
static int32_t s_last_needed;
static int32_t s_last_available;
/* Geometry from the constants, not queried from objects: LVGL resolves
 * object sizes lazily, and a width read before layout comes back as zero,
 * which wraps a label into a one-glyph column. */
static int32_t s_text_x, s_text_w, s_body_y, s_body_h;

static const lv_font_t *font_title(void) { return &lv_font_montserrat_28; }
static const lv_font_t *font_headline(void) { return &lv_font_montserrat_24; }
static const lv_font_t *font_body(void) { return &lv_font_montserrat_24; }
static const lv_font_t *font_label(void) { return &lv_font_montserrat_20; }
static const lv_font_t *font_footer(void) { return &lv_font_unscii_16; }
static int s_story_px = 18;
static const lv_font_t *font_story(void) {
    switch (s_story_px) {
    case 14: return &lv_font_montserrat_14;
    case 18: return &lv_font_montserrat_18;
    case 20: return &lv_font_montserrat_20;
    case 22: return &lv_font_montserrat_22;
    default: return &lv_font_montserrat_16;
    }
}
static const lv_font_t *font_corner(void) { return &lv_font_montserrat_20; }

static void ink(lv_obj_t *obj) {
    lv_obj_set_style_text_color(obj, lv_color_black(), 0);
}

static lv_obj_t *rule(lv_obj_t *parent, int32_t y) {
    lv_obj_t *bar = lv_obj_create(parent);
    lv_obj_remove_style_all(bar);
    lv_obj_set_pos(bar, PC_MARGIN + PC_PAD, y);
    lv_obj_set_size(bar, PC_UI_WIDTH - 2 * (PC_MARGIN + PC_PAD), PC_RULE);
    lv_obj_set_style_bg_color(bar, lv_color_black(), 0);
    lv_obj_set_style_bg_opa(bar, LV_OPA_COVER, 0);
    return bar;
}

static lv_obj_t *label(lv_obj_t *parent, const lv_font_t *font, int32_t x,
                       int32_t y, int32_t width) {
    lv_obj_t *l = lv_label_create(parent);
    lv_obj_set_style_text_font(l, font, 0);
    ink(l);
    lv_obj_set_pos(l, x, y);
    lv_obj_set_width(l, width);
    lv_label_set_long_mode(l, LV_LABEL_LONG_WRAP);
    return l;
}

void pc_ui_init(lv_display_t *display) {
    s_display = display;
    s_screen = lv_display_get_screen_active(display);
    lv_obj_remove_style_all(s_screen);
    lv_obj_set_style_bg_color(s_screen, lv_color_white(), 0);
    lv_obj_set_style_bg_opa(s_screen, LV_OPA_COVER, 0);

    /* The frame is the screen's own border, inset by a padding object. */
    lv_obj_t *frame = lv_obj_create(s_screen);
    lv_obj_remove_style_all(frame);
    lv_obj_set_pos(frame, PC_MARGIN, PC_MARGIN);
    lv_obj_set_size(frame, PC_UI_WIDTH - 2 * PC_MARGIN, PC_UI_HEIGHT - 2 * PC_MARGIN);
    lv_obj_set_style_border_color(frame, lv_color_black(), 0);
    lv_obj_set_style_border_width(frame, PC_FRAME, 0);
    lv_obj_set_style_border_opa(frame, LV_OPA_COVER, 0);

    const int32_t text_x = PC_MARGIN + PC_PAD;
    const int32_t text_w = PC_UI_WIDTH - 2 * text_x;
    s_text_x = text_x;
    s_text_w = text_w;
    s_title = label(s_screen, font_title(), text_x, PC_MARGIN + 28, text_w);
    lv_label_set_long_mode(s_title, LV_LABEL_LONG_CLIP);
    s_corner = label(s_screen, font_corner(), text_x + text_w - 110, PC_MARGIN + 34, 110);
    lv_obj_set_style_text_align(s_corner, LV_TEXT_ALIGN_RIGHT, 0);
    lv_label_set_text(s_corner, "");
    s_title_rule = rule(s_screen, PC_MARGIN + PC_TITLE_H);

    const int32_t footer_y = PC_UI_HEIGHT - PC_MARGIN - PC_FOOTER_H;
    s_footer_rule = rule(s_screen, footer_y);
    s_footer = label(s_screen, font_footer(), text_x, footer_y + 20, text_w);
    lv_label_set_text(s_footer, "EXECUTORCH ESP32-S3");
    lv_obj_set_style_text_align(s_footer, LV_TEXT_ALIGN_CENTER, 0);

    s_body_y = PC_MARGIN + PC_TITLE_H + PC_RULE;
    s_body_h = footer_y - s_body_y;
    s_body_area = lv_obj_create(s_screen);
    lv_obj_remove_style_all(s_body_area);
    lv_obj_set_pos(s_body_area, text_x, s_body_y);
    lv_obj_set_size(s_body_area, text_w, s_body_h);
    lv_obj_set_style_pad_all(s_body_area, 0, 0);
    lv_obj_remove_flag(s_body_area, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_update_layout(s_screen);

    /* The story screen (loaded by pc_ui_show_story_page). */
    s_story_screen = lv_obj_create(NULL);
    lv_obj_remove_style_all(s_story_screen);
    lv_obj_set_style_bg_color(s_story_screen, lv_color_white(), 0);
    lv_obj_set_style_bg_opa(s_story_screen, LV_OPA_COVER, 0);
    s_story_w = PC_UI_WIDTH - 2 * ST_SIDE;
    /* One line of title beside a page number ("2/3" is under 30 px); a
     * title that is still too long ends in "..." rather than wrapping into
     * the rule. */
    s_story_title = label(s_story_screen, &lv_font_montserrat_20, ST_SIDE, ST_HEAD_Y, s_story_w - 42);
    lv_label_set_long_mode(s_story_title, LV_LABEL_LONG_DOT);
    lv_obj_set_height(s_story_title, lv_font_get_line_height(&lv_font_montserrat_20));
    s_story_page = label(s_story_screen, &lv_font_montserrat_16, ST_SIDE + s_story_w - 40, ST_HEAD_Y + 3, 40);
    lv_obj_set_style_text_align(s_story_page, LV_TEXT_ALIGN_RIGHT, 0);
    lv_obj_t *head_rule = lv_obj_create(s_story_screen);
    lv_obj_remove_style_all(head_rule);
    lv_obj_set_pos(head_rule, ST_SIDE, ST_HEAD_RULE_Y);
    lv_obj_set_size(head_rule, s_story_w, 2);
    lv_obj_set_style_bg_color(head_rule, lv_color_black(), 0);
    lv_obj_set_style_bg_opa(head_rule, LV_OPA_COVER, 0);
    lv_obj_t *foot_rule = lv_obj_create(s_story_screen);
    lv_obj_remove_style_all(foot_rule);
    lv_obj_set_pos(foot_rule, ST_SIDE, ST_FOOT_RULE_Y);
    lv_obj_set_size(foot_rule, s_story_w, 2);
    lv_obj_set_style_bg_color(foot_rule, lv_color_black(), 0);
    lv_obj_set_style_bg_opa(foot_rule, LV_OPA_COVER, 0);
    lv_obj_t *keys = label(s_story_screen, &lv_font_montserrat_14, ST_SIDE, ST_FOOT_Y, s_story_w);
    lv_label_set_text(keys, LV_SYMBOL_UP " back     " LV_SYMBOL_DOWN " next     "
                            LV_SYMBOL_REFRESH " new story");
    lv_obj_t *brand = label(s_story_screen, &lv_font_montserrat_14, ST_SIDE, ST_FOOT_Y, s_story_w);
    lv_obj_set_style_text_align(brand, LV_TEXT_ALIGN_RIGHT, 0);
    lv_label_set_text(brand, "nanoGlimmer");
    s_story_body_h = ST_FOOT_RULE_Y - 8 - ST_BODY_Y;
    s_story_body = label(s_story_screen, font_story(), ST_SIDE, ST_BODY_Y, s_story_w);
}

static void clear_body(void) {
    lv_obj_clean(s_body_area);
    if (s_corner) lv_label_set_text(s_corner, "");
}

void pc_ui_show_text(const char *title, const char *headline, const char *body) {
    lv_screen_load(s_screen);
    lv_label_set_text(s_title, title ? title : "");
    clear_body();
    const int32_t w = s_text_w;
    lv_obj_t *head = label(s_body_area, font_headline(), 0, 18, w);
    lv_label_set_text(head, headline ? headline : "");
    lv_obj_update_layout(head);
    const int32_t body_y = 18 + lv_obj_get_height(head) + 22;
    const int32_t body_h = s_body_h - body_y - 8;
    lv_obj_t *text = label(s_body_area, font_body(), 0, body_y, w);
    lv_label_set_text(text, body ? body : "");
    lv_obj_update_layout(text);
    s_last_needed = lv_obj_get_height(text);
    s_last_available = body_h;
    if (s_last_needed > body_h) {
        /* Clip with an ellipsis rather than run under the footer rule. */
        lv_obj_set_height(text, body_h);
        lv_label_set_long_mode(text, LV_LABEL_LONG_DOT);
    }
}

void pc_ui_set_story_font_px(int px) {
    if (px == 14 || px == 16 || px == 18 || px == 20 || px == 22) s_story_px = px;
}

int pc_ui_story_font_px(void) { return s_story_px; }

static int32_t story_body_h(void) { return s_story_body_h; }

void pc_ui_show_story_page(const char *title, const char *corner, const char *page_text) {
    lv_screen_load(s_story_screen);
    lv_label_set_text(s_story_title, title ? title : "");
    lv_label_set_text(s_story_page, corner ? corner : "");
    lv_obj_set_style_text_font(s_story_body, font_story(), 0);
    lv_obj_set_height(s_story_body, LV_SIZE_CONTENT);
    lv_label_set_long_mode(s_story_body, LV_LABEL_LONG_WRAP);
    lv_label_set_text(s_story_body, page_text ? page_text : "");
    lv_obj_update_layout(s_story_body);
    s_last_needed = lv_obj_get_height(s_story_body);
    s_last_available = s_story_body_h;
    if (s_last_needed > s_story_body_h) {
        lv_obj_set_height(s_story_body, s_story_body_h);
        lv_label_set_long_mode(s_story_body, LV_LABEL_LONG_DOT);
    }
}

void pc_ui_show_story(const char *title, const char *body) {
    pc_ui_show_story_page(title, "", body);
}

/* Height text[from, to) takes on the card in the story face. */
static int32_t story_measure(const char *text, int from, int to) {
    static char buf[2048];
    int n = to - from;
    if (n > (int)sizeof(buf) - 1) n = (int)sizeof(buf) - 1;
    if (n < 0) n = 0;
    memcpy(buf, text + from, (size_t)n);
    buf[n] = 0;
    lv_point_t size;
    lv_text_get_size(&size, buf, font_story(), 0, 0, s_story_w, LV_TEXT_FLAG_NONE);
    return size.y;
}

enum { BREAK_WORD = 1, BREAK_SENTENCE = 2, BREAK_PARAGRAPH = 3 };

/* How good a place to end a page is just before text[i]: after a newline,
 * after the space that follows a sentence end, after any space, or no. */
static int break_kind(const char *text, int i) {
    if (i <= 0) return 0;
    if (text[i - 1] == '\n') return BREAK_PARAGRAPH;
    if (text[i - 1] != ' ') return 0;
    int j = i - 2;
    while (j >= 0 && text[j] == ' ') --j;
    if (j >= 0 && (text[j] == '"' || text[j] == '\'')) --j;
    if (j >= 0 && (text[j] == '.' || text[j] == '!' || text[j] == '?')) return BREAK_SENTENCE;
    return BREAK_WORD;
}

static int skip_blank(const char *text, int i) {
    while (text[i] == ' ' || text[i] == '\n') ++i;
    return i;
}

/* One way to cut text into `pages`: balanced (each page aims at an equal
 * share of what is left) or filled (each page as full as a good break allows).
 * Writes starts[1..pages-1]; true when the last page fits. */
static bool story_split(const char *text, int len, int pages, bool balanced,
                        int *starts, int32_t body, int32_t line) {
    int from = 0;
    for (int page = 1; page < pages; ++page) {
        const int32_t left = story_measure(text, from, len);
        const int32_t target = balanced ? (left + (pages - page)) / (pages - page + 1) : body;
        const int32_t ceiling = target + line < body ? target + line : body;
        int best = -1, best_kind = 0;
        int32_t best_gap = 0;
        for (int i = from + 1; i <= len; ++i) {
            const int kind = break_kind(text, i);
            if (kind == 0 || kind < best_kind) continue;
            const int32_t h = story_measure(text, from, i);
            if (h > ceiling) break;
            /* A paragraph end within 3 lines of the target, else a sentence
             * end within 2, else the word that comes closest. */
            const int32_t gap = target > h ? target - h : h - target;
            const int32_t reach = kind == BREAK_PARAGRAPH ? 3 * line
                                : kind == BREAK_SENTENCE ? 2 * line : body;
            if (gap > reach) continue;
            if (kind > best_kind || gap <= best_gap) {
                best = i; best_kind = kind; best_gap = gap;
            }
        }
        if (best < 0) return false;
        from = skip_blank(text, best);
        starts[page] = from;
    }
    return story_measure(text, from, len) <= body;
}

int pc_ui_story_last_fill(const char *text, const int *starts, int pages) {
    const int len = (int)strlen(text);
    return (int)(story_measure(text, starts[pages - 1], len) * 100 / story_body_h());
}

int pc_ui_story_paginate(const char *text, int *starts, int max_pages) {
    const int len = (int)strlen(text);
    const int32_t body = story_body_h();
    const int32_t line = lv_font_get_line_height(font_story());
    starts[0] = 0;
    if (max_pages <= 1 || story_measure(text, 0, len) <= body) return 1;
    /* Fewest pages, each filled like a book's (an even split can leave page 1
     * half empty, which reads as a mistake). A story whose last page would
     * be a few lines is the caller's to avoid: pc_ui_story_last_fill. */
    for (int pages = 2; pages <= max_pages; ++pages) {
        if (story_split(text, len, pages, false, starts, body, line)) return pages;
    }
    return max_pages;   /* the last page clips; callers keep stories shorter */
}

void pc_ui_show_tool(const char *title, const char *action, const char *argument,
                     const char *result, const char *detail) {
    lv_screen_load(s_screen);
    lv_label_set_text(s_title, title ? title : "");
    clear_body();
    const int32_t w = s_text_w;
    int32_t y = 14;
    lv_obj_t *l;
    l = label(s_body_area, font_label(), 0, y, w);
    lv_label_set_text(l, "GENERATED ACTION");
    y += 30;
    l = label(s_body_area, font_body(), 0, y, w);
    lv_label_set_text(l, action ? action : "");
    lv_obj_update_layout(l);
    y += lv_obj_get_height(l) + 6;
    l = label(s_body_area, font_body(), 0, y, w);
    lv_label_set_text(l, argument ? argument : "");
    lv_obj_update_layout(l);
    y += lv_obj_get_height(l) + 22;
    lv_obj_t *bar = rule(s_body_area, 0);
    lv_obj_set_pos(bar, 0, y);
    y += PC_RULE + 22;
    l = label(s_body_area, font_label(), 0, y, w);
    lv_label_set_text(l, "LIVE TOOL RESULT");
    y += 30;
    l = label(s_body_area, font_body(), 0, y, w);
    lv_label_set_text(l, result ? result : "");
    lv_obj_update_layout(l);
    y += lv_obj_get_height(l) + 6;
    l = label(s_body_area, font_body(), 0, y, w);
    lv_label_set_text(l, detail ? detail : "");
    lv_obj_update_layout(l);
    s_last_needed = y + lv_obj_get_height(l);
    s_last_available = s_body_h;
}

void pc_ui_mark_flushed(void) { s_flushed = true; }

bool pc_ui_render(void) {
    s_flushed = false;
    lv_obj_invalidate(lv_display_get_screen_active(s_display));
    lv_refr_now(s_display);
    return s_flushed;
}

void pc_ui_last_fit(int32_t *needed, int32_t *available) {
    if (needed) *needed = s_last_needed;
    if (available) *available = s_last_available;
}
