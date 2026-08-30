# mikroelektronix

A desktop window over the local ASIC design model. Chat with the fine-tuned
model, watch it call the frozen tool contract, and see the results come back
from a real simulator.

```bash
# start the model and the window together
PYTHONPATH=src python -m mikroelektronix.app --serve

# or, if a model server is already running
PYTHONPATH=src python scripts/serve_local.py     # in one terminal
PYTHONPATH=src python -m mikroelektronix.app     # in another
```

## Why Python + pywebview

This machine has no Node and no Rust, and the backend is already Python.
pywebview renders through the system WebView2 -- the same Chromium family
Electron bundles -- instead of shipping its own copy, so the dependency is about
15 MB rather than several hundred.

There is no HTTP server and no port: the page talks to Python directly through
pywebview's `js_api` bridge. The one HTTP connection in the system is to
`llama-server`, which already existed.

## What it reuses, deliberately

| Piece | Module |
|---|---|
| The model on the iGPU | `asic_ai.inference.llama_server` |
| The canonical system message | `asic_ai.data.format.build_system_message` |
| Tool-call parsing and contract validation | `asic_ai.inference.parser` |
| Tool execution against a real simulator | `asic_ai.training.rl_env` |

Nothing about a design turn is reimplemented. That is a deliberate constraint,
not an aesthetic one: this repo has already carried the agent loop three
separate times, two of which produced nothing at all, and a tool-call parser
written against a format that appears nowhere in its own training data. A
desktop app -- its own UI, its own thread, nobody running its code in CI -- is
exactly where a fourth private copy would go unnoticed.
`tests/test_mikroelektronix.py` asserts it uses the shared pieces.

## What it will not do

- **Open a chat with no model behind it.** With no server reachable it says so,
  in the window, with the command that fixes it. A chat box that silently
  answers nothing is the same failure as a simulator that returns zeros.
- **Hide a rejected tool call.** A hallucinated tool or a missing required
  argument is shown as a rejected card, and the rejection is fed back to the
  model. Recovering from a bad call is the behaviour worth watching.
- **Show a result it did not get.** Simulator errors appear verbatim
  ("the netlist has no .ac card ... Refusing to invent a sweep").

## Layout

```
mikroelektronix/
  app.py            entry point: window, optional llama-server child process
  api.py            the js_api bridge and one design turn on a worker thread
  web/index.html    the whole UI, self-contained, no CDN
```

## Notes

A turn runs on a worker thread and pushes events into the page, so the window
stays responsive and the tool calls appear as they happen rather than all at the
end. `MAX_STEPS` in `api.py` bounds a turn.

The chat has no task specs until you state some, so `spec.check` reports what it
could measure rather than scoring against a target. Running the eval set is
`scripts/measure_baseline.py`.
