# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# LVGL: idf_component.yml adds the registry's lvgl/lvgl (lvgl__lvgl); with the
# component manager off, the project's local LVGL checkout is the component lvgl.
idf_build_get_property(_component_manager IDF_COMPONENT_MANAGER)
if(_component_manager)
    set(_lvgl "")
else()
    set(_lvgl lvgl)
endif()
idf_component_register(
    SRCS "papercolor_display.cpp" "ui/papercolor_ui.c"
    INCLUDE_DIRS "include" "ui"
    REQUIRES esp_driver_gpio esp_driver_i2c esp_driver_spi esp_timer ${_lvgl}
)
