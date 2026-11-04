# M5Stack PaperColor display path

This is a native ESP-IDF driver for the 400x600 ED2208-DOA/EL040EF1 panel in
M5Stack PaperColor. It does not depend on Arduino or M5GFX.

## Data path

- Application framebuffer: 30,000 B, 1 bit/pixel in PSRAM.
- Controller format: 120,000 B, 4 bits/pixel.
- Expansion: a 256-entry byte-to-four-byte LUT. The emitted LX7 loop is one
  byte load, one indexed 32-bit load, and one 32-bit store.
- Upload: two 4 KiB internal DMA buffers. Expansion of the next chunk overlaps
  transfer of the current chunk, so no 120 KiB expanded framebuffer exists.
- SPI: 20 MHz. A frame is 960,000 wire bits, giving a 48 ms physical floor.

The ESP32-S3 SPI DMA length field is only 18 bits (32 KiB per transaction).
The streaming path stays below that limit and holds panel CS low across the
complete RAM-write operation.

## Measured on M5Stack PaperColor

ESP-IDF v5.5.5, 240 MHz CPU, 80 MHz octal PSRAM, release optimization:

| SPI | Pack CPU | Stream wall time |
| ---: | ---: | ---: |
| 4 MHz | 0.831 ms | 240.672 ms |
| 8 MHz | 0.818 ms | 120.639 ms |
| 12 MHz | 0.817 ms | 84.635 ms |
| 16 MHz | 0.816 ms | 60.633 ms |
| 20 MHz | 0.812 ms | 48.632 ms |

The 20 MHz limit is supported by the timing table for the related ED2208-GCA
family controller and has been characterized on this exact board. M5GFX's
PaperColor configuration conservatively uses 4 MHz.

Refresh timing on the same unit:

- Complete, DC-balanced color waveform: about 14.7 s end-to-end.
- Interrupted fast waveform: about 0.57 s end-to-end, including 340 ms panel
  drive, roughly 150 ms power-on, 48.6 ms upload, and 30 ms reset/recovery.

The fast mode deliberately resets the controller partway through its OTP
waveform. It leaves a black/yellow image, is not a true partial update, and is
not DC-balanced: after about eight consecutive fast frames the panel reads as
half black, and one complete refresh restores it. The story firmware therefore
uses the complete refresh by default. Use a complete refresh periodically in
any long-running product. The implementation refreshes the whole panel even
when little text changed so all pixels accrue the same interrupted-waveform
history.

The status service (`main/papercolor_status.cpp` in the example project)
persists its interrupted-refresh count in NVS and forces a complete balanced
waveform after eight fast frames (or one hour of continuous uptime). Persisting
the count matters during development because app reflashes must not reset the
panel-aging policy while the physical charge state survives. Display updates
run on their own task and are awaited outside model timing; a one-entry queue
coalesces stale states.

Handwritten LX7 assembly is intentionally not used for framebuffer expansion:
the compiler already emits the minimal lookup loop, and double-buffering hides
all 0.8 ms of CPU work under the 48.6 ms wire transfer. Assembly cannot reduce
the electrophoretic or SPI floors.

## Screens

The cards are LVGL 9.5 (the ESP Component Registry's `lvgl/lvgl` 9.5.0,
pinned in `idf_component.yml`), not hand-placed pixels.
`ui/lv_conf.h` configures it for this panel: one bit per pixel as the native
format, Montserrat 20/24/28 for text, unscii 16 for the footer, no demos,
widgets, images or file systems. `ui/papercolor_ui.c` builds the cards as
labels: a title strip, then a headline and a body that wraps, a story page, or
an action above a rule with its result below it. A body that would run under
the footer is cut with an ellipsis, and the fit is reported as
`body_px=needed/available` on the `PAPERCOLOR_STATUS seq=` line.

LVGL renders into a full frame in PSRAM whose layout is the driver's own (one
bit per pixel, most significant bit first, 1 = paper), so the flush is a copy
into `Display::loadFrame()`, and the driver refreshes it like any other frame. LVGL is
single-threaded and only the status task on core 1 touches it. The
firmware reads the same `lv_conf.h` through `LV_CONF_PATH` (set on the `lvgl`
component in the example project's `CMakeLists.txt`), not LVGL's Kconfig menu.

Primary references:

- [M5Stack PaperColor documentation](https://docs.m5stack.com/en/core/PaperColor)
- [M5GFX ED2208 driver](https://github.com/m5stack/M5GFX/blob/master/src/lgfx/v1/panel/Panel_ED2208.cpp)
- [FreeInk interrupted-waveform implementation](https://github.com/Free-Ink/freeink-sdk)
- [Related ED2208-GCA timing specification](https://files.seeedstudio.com/wiki/Other_Display/7.3-E6-color/7.3_E6_eInk_%20Display_%20module_%20Datasheet.pdf)
