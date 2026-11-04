# Nano model zoo

53 Hugging Face model families at ESP32-S3 size, with random weights, for the
Espressif backend (`backends/esp32s3`). Each family lowers through the
backend, runs on the backend's host runner, and is checked against a reference
that computes with the device's arithmetic (int4 weights, int8 activations).
The zoo is a test bed for the backend, not part of it: nothing in
`backends/esp32s3` needs it. With the zoo present, the backend's tests that
use it run too: the quantizer census's transformer pairs and the nanoLlama
case of `test_lowering_compat.py`.

Each family is one module, named with its release (`llama3_2`, `gemma4`). Its
docstring says what the released model looks like and what was shrunk to fit
the board's flash; `CENSUS.md` lists how every family lowers.

## Requirements

- An ExecuTorch environment with the Espressif backend (this tree).
- The backend's host runner, built as `backends/esp32s3/test/host_runner/README.md`
  says; `ESP32_HOST_RUNNER` names it if it is not at its default path.
- Hugging Face `transformers` 5.19.0. The configs are written for that release
  (`TRANSFORMERS_VERSION` in `__init__.py`), because config fields move between
  releases, and `build.config` refuses any other version. It comes from PyPI:

  ```bash
  pip install -r examples/espressif/nano_zoo/requirements.txt
  ```

  ExecuTorch's `requirements-examples.txt` pins an older transformers for its
  other examples. To keep that one, install the zoo's into a directory of its
  own and put that on the path for the zoo's runs:

  ```bash
  pip install --target /path/to/zoo-site --no-deps transformers==5.19.0 safetensors
  export PYTHONPATH=/path/to/zoo-site
  ```

  `--no-deps` relies on the environment for transformers' other requirements
  (huggingface_hub, tokenizers, regex, ...), which ExecuTorch's example
  requirements install.

Nothing is downloaded: the configs are written out in full and the weights are
random and seeded.

## Running

From the root of the ExecuTorch tree:

```bash
# one family: lower, run on the host runner, compare with the device arithmetic
python -m executorch.examples.espressif.nano_zoo.run llama3_2 --size device
# every family (NANO_ZOO_SIZE=ci for the toy-width configs, seconds each)
python -m pytest examples/espressif/nano_zoo/test
# the census table from the records
python examples/espressif/nano_zoo/census.py
```

Each run writes `results/<family>.<size>.json` (and its lowered program under
`results/work/`) beside the package; `results/` is not tracked. A family that
is known not to lower or run yet says why in its `EXPECT`, and the test marks
it as a strict expected failure.

A run passes when the host runner's output is within cosine 0.99 of the
device-arithmetic reference and every q@kT a delegate computes (the zoo lowers
with `quantize_qk=True`) is exact. The reference keeps attention scores fp32,
and with random weights the scores barely move the output, so the cosine
cannot see a wrong q@kT. Instead a second program, `results/work/<family>.<size>/qk_probe/`,
adds each delegated q@kT's operands and scores to the outputs, and its scores
must equal `bmm_qk_i8` (`backends/esp32s3/test/reference.py`) of the operands,
bit for bit (the record's `qk`). The zoo's own `model.pte` is unchanged by it.
