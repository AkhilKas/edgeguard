# EdgeGuard

On-device prompt injection detection for edge LLM deployments.

## The problem

Small on-device language models are showing up in places that used to be simple
firmware: smart-home hubs, security cameras, IoT controllers. As soon as a
device accepts free-form natural-language input — typed, transcribed from
voice, or read from a log — it inherits prompt injection risk: text crafted to
override the assistant's real instructions ("ignore all previous instructions
and...", role-play jailbreaks, delimiter tricks that impersonate a system
prompt, or asks to leak the system prompt itself).

Cloud LLM providers can afford a heavyweight moderation layer in front of the
model. An edge device usually can't: it's offline-first, has no GPU, and often
has no reliable connection to a moderation API at all. Without a local
guardrail, a compromised prompt reaches the on-device model directly, and
whatever the model does next — controlling a lock, writing a log entry,
driving an actuator — happens with no filter in between.

## What EdgeGuard does

EdgeGuard sits between "prompt arrives" and "prompt reaches the model" as a
two-stage pipeline, running entirely on-device:

1. **Guardrail classifier** — screens the prompt first. The default is a fast
   rule-based heuristic (regex/pattern matching for known injection shapes:
   instruction overrides, role jailbreaks, prompt-leak requests, delimiter
   injection) that needs no model and runs in microseconds. A distilled ML
   classifier mode exists behind the same interface for a future, higher-recall
   version once one is trained.
2. **LLM inference** — only runs if the guardrail passes the prompt. Uses a
   small local GGUF model (`llama-cpp-python`) via chat completion, so a
   blocked prompt never reaches the model at all.

Every request returns a verdict (`clean` / `injected` / `uncertain`), a
confidence score, which rule fired (if any), the model's response (if the
prompt wasn't blocked), and per-stage latency — exposed over a small FastAPI
service (`/prompt`, `/health`).

## Why this design

- **Nothing leaves the device.** No prompt, verdict, or response is sent
  anywhere; the guardrail and the model both run locally.
- **The guardrail is intentionally cheap.** A regex pass costs microseconds,
  so most clean traffic pays almost nothing before reaching the model, and
  blocked traffic never pays LLM inference cost at all.
- **Swappable, not hardcoded.** Guardrail mode (`heuristic` | `ml`) and every
  model/runtime parameter live in one settings module and are read from
  `.env` — nothing is hardcoded in application code.
- **Built for the hardware it runs on.** The target device is an Arduino
  UNO Q — a Linux-capable Qualcomm QRB2210 paired with an STM32U585
  microcontroller. The Linux side runs this service; the MCU side talks to it
  over the Arduino Router Bridge (MessagePack-RPC), sending prompts in and
  receiving verdicts back, so the guardrail can gate a real physical action
  (e.g. a smart-home command) and not just a chat response.

## Project layout

```
src/edgeguard/
  guardrail/   heuristic + ML classifier implementations, behind one interface
  llm/         llama-cpp-python engine wrapper (chat completion, GGUF models)
  mcu/         Arduino Router Bridge interface to the STM32 MCU
  pipeline/    orchestrates guardrail -> LLM per request
  api/         FastAPI app (routes, schemas, startup/shutdown wiring)
  config/      all tunable settings, read from environment / .env
  training/    notebook(s) for training the ML guardrail classifier
               (not imported by the app; run in Colab/Jupyter, exported
               model artifacts are deployed separately, not committed)
```

## Quickstart

```bash
cp .env.example .env
bash scripts/download_model.sh   # fetches the default GGUF model into models/
make install                     # pip install -e ".[dev]"
make run                         # starts the API on API_HOST:API_PORT
```

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/prompt \
  -H "Content-Type: application/json" \
  -d '{"text": "Turn off the kitchen lights"}'
```

On a machine without the MCU hardware (local dev, CI), set `MCU_ENABLED=false`
in `.env` — the guardrail + LLM pipeline works standalone over HTTP.

## Development

```bash
make lint       # ruff check
make typecheck  # mypy --strict
make test       # pytest tests/unit (mocked LLM/MCU, no hardware or model needed)
make check      # all three
```

## Deploying to the UNO Q

```bash
bash scripts/setup_service.sh   # installs EdgeGuard as a systemd service
bash scripts/setup_runner.sh    # registers a self-hosted GitHub Actions runner
                                 # so pushes to main redeploy automatically
```

## Status

- Heuristic guardrail: implemented and tested.
- ML guardrail mode: implemented (ONNX Runtime + tokenizers, see
  `guardrail/ml_classifier.py`) and unit tested against mocks, but not yet
  tested against real trained weights — the training notebook hasn't been
  run end-to-end and no `.onnx`/`tokenizer.json` artifacts exist in the repo
  yet. Set `GUARDRAIL_MODEL_PATH` and `GUARDRAIL_TOKENIZER_PATH` once they do.
- MCU integration: Linux-side Router Bridge wrapper implemented; the
  corresponding MCU sketch (calling `Bridge.call("on_prompt", ...)` and
  providing `set_status`) is not written yet.
