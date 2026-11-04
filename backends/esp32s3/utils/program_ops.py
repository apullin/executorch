# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The operators an emitted ExecuTorch program looks up at load time."""


def ops_of_program(et):
    """The operator names an emitted ExecuTorch program will look up at load time."""
    names = set()
    for plan in et.executorch_program.execution_plan:
        for op in plan.operators:
            names.add(f"{op.name}.{op.overload}" if op.overload else op.name)
    return names
