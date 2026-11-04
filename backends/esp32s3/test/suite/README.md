# ExecuTorch's backend test suite on the ESP32-S3 backend

`backends/test/suite` is ExecuTorch's black-box battery for backends: 80
operator test files, 467 tests, each lowering a small model through every
registered flow, running the program and comparing its outputs with eager at
a fixed tolerance (atol 0.1, rtol 0.04). This directory registers two flows
for this backend and runs them on the host:

| Flow | Quantisation | Reference |
|---|---|---|
| `esp32s3` | none: Esp32Partitioner's defaults (a float weight the delegate takes is stored as int8, its activation quantised per row at execute) | the float model |
| `esp32s3_8da4w` | the default `Esp32Quantizer()`: int4 per-channel symmetric weights, dynamic int8 activations | the converted model |

Each case is lowered the README's way (`lower_esp32`, `apply_passes`,
`planning_pass("esp32")`, 16-byte alignment), serialized, and run by
`esp32_host_runner` (`backends/esp32s3/test/host_runner/`): the delegate's runtime and the
operator library built for the build machine, with C reference kernels in
place of the PIE assembly, and the portable kernels for whatever the delegate
does not take. Every case runs, delegated or not: refusing an op is a
supported outcome, but the program must still lower, load and give the right
numbers. The device's own bits are proven by QEMU and silicon parity
(`../parity/run_parity.py`), not here.

## Running it

Build the host runner first (`backends/esp32s3/test/host_runner/README.md`), then, from the root
of the ExecuTorch tree:

```bash
pytest -c /dev/null --confcutdir . \
  -p executorch.backends.esp32s3.test.suite.plugin \
  backends/test/suite/operators -m backend_esp32s3 \
  --esp32-report esp32s3_suite.json
```

- `-p ...plugin` registers the flows with no change to upstream's files: it
  wraps `all_flows()` in `backends/test/suite/flow.py` before the suite's
  conftest imports it (`plugin.py`). Without a built runner (or
  `ESP32_HOST_RUNNER` naming one) no flow is registered.
- `-m backend_esp32s3` runs both flows; `-m flow_esp32s3` or
  `-m flow_esp32s3_8da4w` one.
- `-c /dev/null` keeps the tree's `pytest.ini`, which ignores
  `backends/test`, out of it, as upstream's CI does; `--confcutdir .` keeps
  conftests above the tree out.
- `--esp32-seed N` changes the inputs (default 0, so a run repeats);
  `--esp32-report` writes every case's result, delegated ops and output error
  statistics in pytest-json-report's layout, which
  `python -m executorch.backends.test.suite.generate_markdown_summary_json`
  reads.
- One process runs every case: the plugin releases the upstream caches that
  would otherwise keep every lowered program alive.

The suite's model tests (`backends/test/suite/models`) need torchvision and
build full-size models; they are not run here.

## Expected failures and skips

`flow_esp32s3.py` lists, per flow, the cases skipped and the cases expected
to fail, each with its reason. An expected failure still runs, strictly: one
that starts passing fails the run until its entry goes. The reasons are
printed on each case.

`test_suite_flow.py` checks the host stage, the suite's `run_test` on each
flow, and the plugin on one upstream file.
