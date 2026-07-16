# Running Phase 2: DSPy prompt compilation

Phase 2 optimizes the Manager/Browser/Synthesizer prompts with DSPy's
MIPROv2 against the train-pool split (MuSiQue + HotpotQA) — never the
eval-only benchmarks (FRAMES/BrowseComp/ResearchRubrics). See
`docs/running-phase-1.md` first for environment setup; everything there
(inference endpoint, `TAVILY_API_KEY`, `.env`) is also required here, since
compilation runs the same retrieval tools live during optimization.

## 1. What it does

`scripts/compile_dspy.py`:
1. Pulls MuSiQue + HotpotQA questions, splits them into train/val by a
   stable hash of each question id (never touches the benchmarks).
2. Builds a DSPy program mirroring the MAF pipeline — same three roles,
   same retrieval tools, different mechanism (see the design note below).
3. Runs `dspy.MIPROv2` to search for better instructions + few-shot
   demonstrations for each role, scored by exact-match/F1 against the
   train pool's verifiable answers.
4. Renders the optimized instructions + demos back into plain text and
   writes `src/mas_deepr/prompts/templates/{role}.compiled.yaml`.

## 2. Running it

```bash
uv run python scripts/compile_dspy.py --model qwen3-8b --auto light
```

| Flag | Required | Notes |
|---|---|---|
| `--model` | yes | Registry key, e.g. `qwen3-8b` |
| `--auto` | no (default `light`) | MIPROv2 search budget: `light` \| `medium` \| `heavy` — higher = more trials, more LLM calls, better prompts, longer run |
| `--musique-limit` | no (default 200) | Cap on MuSiQue examples pulled into the pool |
| `--hotpot-limit` | no (default 200) | Cap on HotpotQA examples pulled into the pool |
| `--val-fraction` | no (default 0.2) | Train/val split fraction within the pool |
| `--version` | no (default: timestamp) | Tag written into the compiled YAML's `version` field |

Output:

```
Compiled prompts written:
  manager: src/mas_deepr/prompts/templates/manager.compiled.yaml
  browser: src/mas_deepr/prompts/templates/browser.compiled.yaml
  synthesizer: src/mas_deepr/prompts/templates/synthesizer.compiled.yaml
```

## 3. Evaluating the result

The compiled prompts don't do anything until a run explicitly asks for
them. That happens automatically at the `post-dspy` milestone:

```bash
uv run python scripts/run_milestone_eval.py \
  --milestone post-dspy \
  --models qwen3-8b
```

`run_milestone_eval.py` passes `prefer_compiled=True` for every milestone
except `baseline`, so this picks up whatever `*.compiled.yaml` files exist
in the registry with no other changes. To go back to the hand-written
prompts for comparison, just run `--milestone baseline` again — nothing to
toggle, `load_prompt()` reads from the registry file that matches the
milestone.

## 4. Design note: why the Browser isn't `dspy.ReAct`

The DSPy Browser role does **not** use `dspy.ReAct`. ReAct's compiled
instructions bake in DSPy's own tool-selection text protocol
(`next_thought`/`next_tool_name`/`next_tool_args`), which has no equivalent
in MAF's native function-calling `Agent` — porting that text into the
prompt registry would produce a MAF system prompt full of references to a
mechanism MAF doesn't use.

Instead, `optimize/modules.py`'s `make_retriever()` calls the same cached
`web_search`/`fetch_page` tools imperatively (blocking, via `asyncio.run`,
since MIPROv2's compile loop is synchronous), and DSPy only optimizes the
"reason over already-gathered evidence" signature — which *is* directly
portable, because it's the same reasoning task the MAF browser performs,
just with the LM's tool-calling decision replaced by a Python function
call. `optimize/render.py` renders each role's optimized instructions +
demos with human-readable field labels and prepends the hand-written
MAF-specific tool-use preamble that DSPy never sees.

## 5. Cost/time expectations

- `--auto light` is the cheapest search; use it for a first pass or a
  smoke test of the wiring.
- Every train-pool question the compile job touches makes real
  `web_search`/`fetch_page` calls (cached — repeated questions across
  trials are cheap after the first hit, but the first pass over N
  questions is N live searches).
- `--musique-limit`/`--hotpot-limit` are your main cost levers: smaller
  pools compile faster and cheaper at the risk of a less-optimized prompt.

## 6. Re-running / versioning

Each compile run overwrites the existing `*.compiled.yaml` per role — there
is no automatic history. If you want to keep a specific compiled version
around (e.g. to compare two `--auto` settings), pass `--version` and copy
the resulting file aside before recompiling; `save_compiled_prompt()`
always writes to the same path regardless of the version tag inside it.
