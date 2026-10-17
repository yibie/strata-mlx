# Image understanding: setup and protocol

This PR implements the MLX image receiver. It does **not** bundle the external
HTTP server, image encoder, model weights or projector. The supported wiring is:

```text
OpenAI image request -> official Strata Python HTTP server
                     -> official strata-vision encoder + matching BF16 mmproj
                     -> SVE1 / GENI -> this MLX language-model engine
```

Only MLX loads the language model for generation. The standalone encoder loads
its vision/projector data; the Python server handles requests and tokenization.
Do not start the original Strata language-model engine alongside MLX. This is
image understanding (text answers about pictures), not image generation.
Both projects are MIT licensed; see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## Prerequisites

- Apple Silicon, Python3.12/3.13 and uv, with the MLX requirements from the main README.
- Apple Command Line Tools, CMake and Ninja to build the standalone encoder.
- Official Strata checkout pinned to `296b59d2f73c8f04a805df29c2fa97655adace78`.
  Other server/encoder revisions are not qualified by this contribution.
- Both shards of a supported Flash-Next GGUF, a **matching BF16 vision projector**,
  and the tokenizer exported from that text GGUF. Reuse existing files; configuration
  downloads nothing. Models/projectors retain their own licenses. File checks cannot
  establish that an arbitrary projector is semantically compatible.
- Optional raw MTP files from `tools/mtp_fetch.py`; vision also works without MTP.

## 1. Install the two Python environments

From this MLX checkout, run `uv sync` as in the main README. Set the following
absolute paths to your own checkouts; do not copy a machine-specific service config:

```sh
export MLX_ROOT="$(pwd)"
export STRATA_ROOT="$HOME/src/strata-official-vision"
git clone https://github.com/Niko1221/Strata.git "$STRATA_ROOT"
git -C "$STRATA_ROOT" checkout 296b59d2f73c8f04a805df29c2fa97655adace78
uv venv --python 3.12 "$STRATA_ROOT/.venv"
uv pip install --python "$STRATA_ROOT/.venv/bin/python" \
  pillow jinja2 regex jsonschema referencing cmake ninja
```

These server dependencies are separate from the MLX environment. `jsonschema` and
`referencing` are mandatory in the provided launcher; its preflight rejects the
official server's object-only fallback before any model starts. The helper also
checks Pillow, Jinja, regex and the official tokenizer imports.

## 2. Build only the standalone encoder

Install Apple's Command Line Tools first if `xcode-select -p` fails. The following
uses the llama.cpp revision pinned by that official Strata checkout, not its main branch:

```sh
git clone https://github.com/ggml-org/llama.cpp.git "$STRATA_ROOT/third_party/llama.cpp"
git -C "$STRATA_ROOT/third_party/llama.cpp" checkout 3cf03257f219afbe7334045ff7c6a06ac68c627d
"$STRATA_ROOT/.venv/bin/cmake" -S "$STRATA_ROOT/tools/vision" \
  -B "$STRATA_ROOT/build-vision" -G Ninja \
  -DCMAKE_MAKE_PROGRAM="$STRATA_ROOT/.venv/bin/ninja" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_DIR="$STRATA_ROOT/third_party/llama.cpp" \
  -DSTRATA_VISION_CUDA=OFF -DSTRATA_PORTABLE=OFF
"$STRATA_ROOT/.venv/bin/cmake" --build "$STRATA_ROOT/build-vision" \
  --target strata-vision -j 2
```

This builds `build-vision/bin/strata-vision`, **not** the original language-model
engine. If the pinned llama.cpp checkout/encoder already exists, reuse it rather
than cloning over it. Official `setup.py` is not needed for this manual Mac path.

## 3. Export the matching tokenizer and create a configuration

Choose existing model/projector files and a new output directory:

```sh
export MODEL="/absolute/path/to/model-00001-of-00002.gguf"
export MMPROJ="/absolute/path/to/matching-BF16-mmproj.gguf"
export TOKENIZER_PARENT="$HOME/.cache/strata-mlx/vision-tokenizer"
export VISION_PROFILE="$HOME/.config/strata-mlx/vision"
"$STRATA_ROOT/.venv/bin/python" "$STRATA_ROOT/tools/strata_tokenizer.py" \
  --gguf "$MODEL" --out "$TOKENIZER_PARENT" --check
python3 "$MLX_ROOT/tools/configure_vision.py" \
  --strata-root "$STRATA_ROOT" \
  --server-python "$STRATA_ROOT/.venv/bin/python" \
  --model "$MODEL" --mmproj "$MMPROJ" \
  --tokenizer "$TOKENIZER_PARENT/tokenizer" --output "$VISION_PROFILE"
```

Optional flags: `--mtp /absolute/path/to/raw-mtp`, `--engine-python /path/to/python`,
`--encoder /path/to/strata-vision`, `--max-context N` and `--port N`.
The defaults are this checkout's `.venv/bin/python`,16K context and loopback8095.
The helper validates required files/dependencies, refuses an existing output
folder, and creates `config.json`, `engine.sh`, `serve.sh` and `preflight.json`.
It starts no process with model weights and makes no network requests. Only its
CPU-side Python dependency check runs. Generated launchers quote paths with spaces
or apostrophes. No resident-mode/fusion controls or dependency on PR11 is introduced.

The example uses upstream memory defaults plus the receiver reservation below;
it does not reproduce our separate resident-tuned Mac profile. Budget the encoder
separately and qualify your machine before increasing context or cache budgets.

## 4. Check dependencies, start, and verify readiness

```sh
"$VISION_PROFILE/serve.sh" --preflight-only   # safe: no model/encoder/server starts
"$VISION_PROFILE/serve.sh"                  # explicitly loads encoder + MLX, then serves
```

In a second terminal, `curl -fsS http://127.0.0.1:8095/health` must report
`service: strata`, `loaded: true`, and `images: true`. HTTP200/status-ok alone
is insufficient. The OpenAI base is `http://127.0.0.1:8095/v1`, model alias
`qwen3.8-flash-next`. The encoder and language-model processes are children of
the server; Ctrl+C stops them. No automatic service or reboot setup is installed.
Normal prompts/outputs are unlogged (`log: null`, API monitor off); conversation
parking remains RAM-only. Keep the loopback default while qualifying the setup.

For a generated-image smoke check, run the following with the **server** Python:

```python
import base64, io, json, urllib.request
from PIL import Image
image = Image.new("RGB", (256, 256), "red")
buf = io.BytesIO(); image.save(buf, format="PNG")
url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
body = {"model": "qwen3.8-flash-next", "temperature": 0, "max_tokens": 32,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "What color is the image? Reply only the lowercase color."},
            {"type": "image_url", "image_url": {"url": url}}]}]}
req = urllib.request.Request("http://127.0.0.1:8095/v1/chat/completions",
    json.dumps(body).encode(), {"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=120) as response:
    answer = json.load(response)["choices"][0]["message"]["content"].strip()
print(answer)
assert answer.lower().strip(". ") == "red"
```

This checks basic wiring, not general image accuracy. On failure, first check the
paths/permissions, server Python dependencies, loaded readiness, projector match
and model/context/memory compatibility. Do not solve an admission failure by
raising OS wired limits or copying a larger machine's settings.

## Setup validation

The configuration helper and generated preflight were exercised on a Mac using a
fresh server Python environment and the pinned external source, with existing
encoder/model/tokenizer files. Paths containing spaces/apostrophes and venv Python
symlinks are covered. Eleven CPU-only setup regressions pass; the full upstream-only
Mac suite passes511 tests. The generated serving profile was not started as another
resident model. These checks do not claim a fresh operating-system install or new
full-model accuracy results; the receiver integration results below remain separate.

## Receiver protocol and limits

`GENI <max_new> [sampling/checkpoint keys] <sve-file> <id,id,...>` accepts
concatenated SVE1 little-endian float32 records from the official encoder. Width,
rectangular grids, payload length, finite values and exact pad coverage are
validated before mutating state. Encoded payloads are capped at 128MiB per request. `img_keys` is accepted for protocol compatibility;
identity is computed from actual embedding bytes instead of trusting supplied keys.
Prompt IDs remain unchanged for the PLE n-gram hashes. Image rows replace input
embeddings in plain/fused/chunked/layer-major reads and shifted draft-head inputs.
Sparse query and pooled-key rotary phases use physical cell to (t,h,w) mapping;
causal visibility continues to use physical token indices.

Caches remain in RAM. A request may reuse state only when the entire sequence of
(image position, grid, SHA256 of embedding rows) matches. Different images can be
parked independently, including when their text/pad token IDs match. This safely
forgoes shared text-prefix reuse and prior-image-prefix reuse when adding an image.
Image session SAVE is rejected; text session behavior is unchanged. Current-request
embeddings live only with the active engine; parked state retains small identities.
As in text serving, checkpoints/prompt chunks determine cancellation recovery.

Vision reserves 512MiB from the machine budget before expert allocation for
receiver input buffers/arrays. The separate local integration also accounts for
this reservation in its optional resident-mode admission; resident mode itself
is outside this upstream contribution. The standalone encoder's memory is
external to MLX: budget and measure it separately. HTTP encoder caches use temporary
SVE files under the official server's lifecycle; use synthetic fixtures during
qualification. Normal prompts/outputs must stay unlogged and this is not permission
to enable persistent conversation caches or process private photos.

The separate local resident integration passed12/12 synthetic image contents (10 strict),
live/parked430-token reuse and HTTP-disconnect image recovery on an M5 Pro64GiB.
The independent upstream-main tiny-model suite passed500 tests; the integrated
suite passed548 plus22 subtests. See `bench/results/e55-m5pro-20261017/REPORT.md`
for exact profiles, source hashes, raw outputs, failed attempts and restoration.

Validation status and machine measurements belong in the project report; source
unit tests alone do not establish image understanding on a real model. Open gates include broad visual tasks, long/concurrent serving, arbitrary16K/131K
multimodal contexts, and private-photo accuracy. The bounded synthetic cases above
cover multiple images, reuse, draft/STOP/disconnect recovery and combined memory. No universal MLX versus
Metal performance claim follows from this work.

After rebasing onto main295c965 (merged PR11), the upstream-only M5 Pro suite
passed548 tests plus22 subtests in41.40s, including portable setup and combined
vision/fusion/pressure option tests. The guarded test window restored the selected
production runtime and verified synthetic text/images and unchanged photo checkpoints.
This is source regression validation, not a new full-model accuracy measurement.
