# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The flash split: the tracked default table, and tables sized per model."""

import csv
import os
from pathlib import Path
import sys
import tempfile

from executorch.examples.espressif.esp32s3.scripts.flash_layout import (
    DEFAULT_CSV,
    FLASH_END,
    PTE_LIMIT,
    PTE_OFFSET,
    PTD_LIMIT,
    PTD_OFFSET,
    layout_for,
    partition_csv,
    defaults_disagree,
    pin_defaults,
    use_layout,
)

FAILS = []


def check(label, ok):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    if not ok:
        FAILS.append(label)


def parse(text):
    rows = {}
    for row in csv.reader(line for line in text.splitlines()
                          if line.strip() and not line.lstrip().startswith("#")):
        rows[row[0].strip()] = tuple(int(value.strip(), 0) for value in row[3:5])
    return rows


def contiguous(rows):
    return (rows["pleprog"][0] + rows["pleprog"][1] == rows["pleweights"][0]
            and rows["pleweights"][0] + rows["pleweights"][1] == FLASH_END
            and rows["factory"][0] + rows["factory"][1] == PTE_OFFSET)


def main():
    tracked = parse((Path(__file__).resolve().parents[1] / DEFAULT_CSV).read_text())
    check("tracked table is the Python default",
          tracked.get("pleprog") == (PTE_OFFSET, PTE_LIMIT)
          and tracked.get("pleweights") == (PTD_OFFSET, PTD_LIMIT))
    check("tracked table is contiguous and ends at 16 MiB", contiguous(tracked))

    default = layout_for(PTE_LIMIT, PTD_LIMIT)
    check("a model that fits keeps the default split", default["default"])
    check("the default renders to the tracked rows",
          parse(partition_csv(default)) == {k: tracked[k] for k in tracked})
    small = layout_for(114_946, 14_898_480)
    check("a 114,946 B program with 14,898,480 B of weights uses the default", small["default"])

    story = layout_for(218_114, 15_033_216)
    check("story Glimmer (218,114 B program) gets 54 program pages",
          not story["default"] and story["pte_size"] == 0x36000)
    check("story Glimmer's weights fit its sized split", 15_033_216 <= story["ptd_size"])
    flash_lm = layout_for(19_202, 15_136_928)
    check("flash-scale speed check (15,136,928 B weights) fits a sized split",
          not flash_lm["default"] and 15_136_928 <= flash_lm["ptd_size"])
    for name, layout in (("story", story), ("flash-scale", flash_lm)):
        rows = parse(partition_csv(layout))
        check(f"{name} table is contiguous and ends at 16 MiB", contiguous(rows))
        check(f"{name} table offsets are 4 KiB aligned",
              all(offset % 0x1000 == 0 and size % 0x1000 == 0
                  for offset, size in rows.values()))
    try:
        layout_for(0x40000, PTD_LIMIT)
        check("an oversized model is refused", False)
    except ValueError:
        check("an oversized model is refused", True)

    with tempfile.TemporaryDirectory() as build:
        with open(os.path.join(build, "sdkconfig"), "w") as out:
            out.write('CONFIG_X=y\nCONFIG_PARTITION_TABLE_CUSTOM=y\n'
                      'CONFIG_PARTITION_TABLE_CUSTOM_FILENAME="partitions_ple.csv"\n'
                      'CONFIG_PARTITION_TABLE_FILENAME="partitions_ple.csv"\n')
        defaults = use_layout(build, flash_lm)
        table = os.path.join(os.path.abspath(build), "partitions_model.csv")
        sdk = open(os.path.join(build, "sdkconfig")).read()
        check("a sized split points the existing sdkconfig at the build's table",
              f'CONFIG_PARTITION_TABLE_CUSTOM_FILENAME="{table}"' in sdk
              and f'CONFIG_PARTITION_TABLE_FILENAME="{table}"' in sdk
              and "CONFIG_X=y" in sdk)
        check("and writes the same setting for a fresh build",
              f'CONFIG_PARTITION_TABLE_CUSTOM_FILENAME="{table}"' in open(defaults).read())
        check("the build's table is the sized one",
              parse(open(table).read())["pleweights"] == (flash_lm["ptd_offset"],
                                                          flash_lm["ptd_size"]))
        use_layout(build, default)
        sdk = open(os.path.join(build, "sdkconfig")).read()
        check("returning to the default points back at the tracked table",
              f'CONFIG_PARTITION_TABLE_CUSTOM_FILENAME="{DEFAULT_CSV}"' in sdk)

    with tempfile.TemporaryDirectory() as build:
        defaults = os.path.join(build, "story.defaults")
        with open(defaults, "w") as out:
            out.write("# cold .bss in PSRAM\nCONFIG_A=y\nCONFIG_B=4\n")
        pin_defaults(build, defaults)
        check("pinning leaves a build directory without an sdkconfig to its defaults",
              not os.path.exists(os.path.join(build, "sdkconfig")))
        with open(os.path.join(build, "sdkconfig"), "w") as out:
            out.write("# CONFIG_A is not set\nCONFIG_B=2\nCONFIG_C=y\n")
        pin_defaults(build, defaults)
        sdk = open(os.path.join(build, "sdkconfig")).read()
        check("pinning turns on an option the existing sdkconfig has off",
              "CONFIG_A=y" in sdk and "CONFIG_A is not set" not in sdk)
        check("and replaces a value, keeping every other line",
              "CONFIG_B=4" in sdk and "CONFIG_B=2" not in sdk and "CONFIG_C=y" in sdk)
        # Cache sizes are Kconfig choices, so the runner regenerates rather than
        # pins: defaults_disagree names the keys an existing sdkconfig has wrong.
        with open(os.path.join(build, "sdkconfig"), "w") as out:
            out.write("CONFIG_A=y\n# CONFIG_B is not set\nCONFIG_C=y\n")
        check("disagreement names an option off in the sdkconfig and on in the defaults",
              defaults_disagree(build, defaults) == ["CONFIG_B"])
        with open(os.path.join(build, "sdkconfig"), "w") as out:
            out.write("CONFIG_A=y\nCONFIG_B=4\n")
        check("and nothing when every default already holds",
              defaults_disagree(build, defaults) == [])
        check("and nothing when there is no sdkconfig yet",
              defaults_disagree(os.path.join(build, "absent"), defaults) == [])

    print(f"lean flash layout: default PTE {PTE_LIMIT:,} B @0x{PTE_OFFSET:x}; "
          f"PTD {PTD_LIMIT:,} B @0x{PTD_OFFSET:x}; sized per model when needed")
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
