# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Name and total the story firmware's ET_OP_PROFILE lines.

The firmware's EventTracer (built with run_story_hardware.py --op-timing)
reports decode time per instruction index. This reads the decode plan out of
the exported .pte, so each index gets its operator (or delegate) name, and
prints time per operator per token.

    story_op_profile.py <device log> [--pte main/ple.pte]
"""

import argparse
import collections
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROFILE = re.compile(r"ET_OP_PROFILE event=(\S+) debug=(\d+|all) calls=(\d+) total_us=(\d+)")


def instruction_names(pte_path, method="decode"):
    from executorch.exir._serialize._program import deserialize_pte_binary
    from executorch.exir.schema import DelegateCall, KernelCall

    with open(pte_path, "rb") as f:
        program = deserialize_pte_binary(f.read())
    program = getattr(program, "program", program)
    plan = next(p for p in program.execution_plan if p.name == method)
    names = {}
    for index, instruction in enumerate(plan.chains[0].instructions):
        args = instruction.instr_args
        if isinstance(args, KernelCall):
            op = plan.operators[args.op_index]
            names[index] = op.name + (f".{op.overload}" if op.overload else "")
        elif isinstance(args, DelegateCall):
            names[index] = "delegate:" + plan.delegates[args.delegate_index].id
        else:
            names[index] = type(args).__name__
    return names


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("log")
    parser.add_argument("--pte", default=os.path.join(os.path.dirname(HERE), "main", "ple.pte"))
    args = parser.parse_args(argv)
    names = instruction_names(args.pte)
    by_op = collections.defaultdict(lambda: [0, 0, 0])   # us, calls, instructions
    steps = None
    # A log can hold several reports (one per firmware run); an instruction
    # seen twice starts the next one. Use the LAST complete report only:
    # summing reports would double the times against one report's steps.
    reports, seen = [[]], set()
    for row in PROFILE.findall(open(args.log, errors="replace").read()):
        key = (row[0], row[1])
        if key in seen:
            reports.append([])
            seen = set()
        seen.add(key)
        reports[-1].append(row)
    for event, debug, calls, total in reports[-1]:
        if event != "OPERATOR_CALL" and event != "DELEGATE_CALL":
            continue
        name = names.get(int(debug), f"unknown_{debug}") if debug != "all" else event
        row = by_op[name]
        row[0] += int(total)
        row[1] += int(calls)
        row[2] += 1
        steps = int(calls) if steps is None else max(steps, int(calls))
    if not by_op:
        print("no ET_OP_PROFILE lines found")
        return 1
    total = sum(r[0] for r in by_op.values())
    print(f"{steps} decode steps; {total / steps / 1000:.1f} ms/token in instructions")
    for name, (us, calls, n) in sorted(by_op.items(), key=lambda kv: -kv[1][0]):
        print(f"  {name:42s} {us / steps / 1000:7.2f} ms/token  {n:3d} instr  "
              f"{us / calls:8.1f} us/call  {us / total:5.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
