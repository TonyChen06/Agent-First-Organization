# Columbia Dining agent

An Arklex agent that recommends meals from Columbia Dining's menus and answers
dining questions.

## Setup

From the repository root:

```bash
cd examples/columbia_dining
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install --no-deps -e .
```

Set `OPENAI_API_KEY` in `.env`, then run:

```bash
.venv/bin/python -m columbia_dining.prepare_docs
.venv/bin/python -m columbia_dining.run
```

Use `/reset` to start over or `/quit` to exit. Re-run `prepare_docs` after editing
references. Chat and evaluation use paid model calls.

## Evaluation

```bash
.venv/bin/python -m pip install -e '.[evaluation]'
.venv/bin/python -m columbia_dining.evaluate
```

Generated transcripts and the HTML report are saved in `columbia_dining/data/evaluation/`.
