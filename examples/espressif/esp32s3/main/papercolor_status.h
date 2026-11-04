/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stdint.h>

namespace et_papercolor {

// Starts the PaperColor renderer on core 1 and waits only for controller init.
// Refreshes remain asynchronous with respect to model execution.
bool start();

// Queue length is one by design: intermediate progress may coalesce, but the
// newest state (especially the final result) is never trapped behind stale UI.
bool post(const char* title, const char* line1, const char* line2);

// A sensor tool card: the status task performs the board I/O and renders the
// measured result; callers never format a pretend sensor response.
bool postHumidityReading();
// The same live read, labelled with the sensor the model asked for
// ("humidity" or "temperature"; the SHT40 reports both either way).
bool postSensorReading(const char* sensor);
// A tool card with no board I/O: the generated action and a caller-supplied
// result, for actions this board has no device for.
bool postTool(const char* title, const char* action, const char* argument,
              const char* result, const char* detail);

// Whether cards may use the interrupted fast waveform (0.57 s, black/yellow,
// and visibly worse with each consecutive use) or always the complete one
// (about 15 s, clean). Default: fast, within the balance policy.
void setFastRefresh(bool allowed);
// How long a fast refresh lets the panel's waveform run before cutting it
// short: 100 to 2000 ms, default 2000. Longer is cleaner and slower.
void setFastCutoffMs(uint16_t ms);
uint16_t fastCutoffMs();
bool fastRefresh();
// A story card: the whole generated text, wrapped in the small face.
bool postStory(const char* title, const char* text);
// One page of a story: `corner` sits at the right of the title strip (the
// page number "1/2", or "writing..."). `fast` lets this frame use the
// interrupted waveform (0.57 s, for writing progress and page turns); false
// asks for the complete one (about 15 s, clean). The balance policy still
// forces a complete refresh after eight fast ones.
bool postStoryPage(const char* title, const char* corner, const char* text, bool fast);
// Page breaks for a story at the current story face, measured with the
// display's own fonts (pc_ui_story_paginate). Call after start().
int paginateStory(const char* text, int* starts, int max_pages);
// How full the last page of that layout is, in percent.
int storyLastFill(const char* text, const int* starts, int pages);
// Story face size in pixels: 14, 16, 18 or 20.
void setStoryFontPx(int px);
int storyFontPx();
// Wait until the newest submitted state reaches the physical panel.
bool flush(uint32_t timeout_ms);

}  // namespace et_papercolor
