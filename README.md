# ObscureFacts

[![⭐ OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/ObscureFacts)

## Description

ObscureFacts is an environment for evaluating an agent's ability to find answers to obscure trivia questions using web search. Agents must use web search tools to research and answer 49 intentionally difficult factual questions spanning sports, technology, local history, and academia.

## Capabilities

- Web search and information retrieval
- Multi-step research and evidence synthesis
- Answering obscure factual questions across diverse domains
- Extracting specific facts from web content

## Compute Requirements

Agents are given a standard environment with no special compute requirements.

## License

[MIT](https://opensource.org/licenses/MIT)

## Tasks

There is one split in this environment:

- **Test**: 49 obscure trivia questions

Questions span diverse domains including sports statistics, technology history, local history, and academic trivia.

## Reward Structure

This is a multi-turn environment. The agent searches the web, gathers information, and submits a final answer. The reward is binary (1.0 or 0.0) based on LLM-based semantic grading using `gpt-5-mini`. The grader checks semantic equivalence, accepting synonyms and paraphrasing while requiring exact numeric values.

## Data

Task data consists of 49 curated trivia questions with reference answers stored in a JSON file. Task data is stored on the OpenReward platform.

## Tools

Search and fetch come from the OpenReward SDK's `WebToolset`
(`toolsets = [WebToolset]`) rather than being implemented in this environment.

| Tool | Description |
|------|-------------|
| `web_search` | Search the web. Takes a `query` and optional `allowed_domains` **or** `blocked_domains`; returns a `Links:` list of `{title, url}` sources. |
| `web_fetch` | Fetch the content of a URL. Takes a `url` and a `prompt` describing what to extract (truncated to 100 KB). |

Grading runs through a hidden `@terminal` tool rather than a tool the agent can
call: replying with a plain message ends the rollout, and that message text is
semantically graded against the reference answer.

### Choosing a search backend

Which provider answers those two tools is configuration, not code here — so
swapping it needs no change to this environment. Requires
`openreward >= 0.1.152`.

| `OPENREWARD_SEARCH_BACKEND` | Sources | `as_of` cutoff | Extra install |
|---|---|---|---|
| unset (default) | `backsearch` — GR's backdated corpus: CC-News, SEC filings, arXiv | **honoured** — results bounded to the cutoff | none |
| `tavily` | the live web | **ignored** (warned once) | `pip install 'openreward[search]'` |

```bash
# default — no configuration needed
python server.py

# route the same two tools to Tavily instead
export OPENREWARD_SEARCH_BACKEND=tavily
export TAVILY_API_KEY=tvly-...
python server.py
```

An unset or unrecognised value falls back to `backsearch`; a *typo* logs a
warning, an unset variable does not. Note that setting `TAVILY_API_KEY` does
**not** by itself select Tavily — only `OPENREWARD_SEARCH_BACKEND` does, so a
key sitting in a shared `.env` cannot silently reroute a run.

**Which one should this environment use?** Empirically, **Tavily** — see
[Backend trade-offs](#backend-trade-offs) below.

### Credentials

Both backends read their key from the session `secrets` mapping first, falling
back to the server's process environment. The environment exposes
`self.search_secrets = secrets` so the configured backend can pick out the name
it needs:

```python
async with environment.session(
    task=task,
    secrets={
        "openai_api_key": OPENAI_API_KEY,   # grader
        "api_key":        OPENREWARD_API_KEY,   # backsearch
        "tavily_api_key": TAVILY_API_KEY,       # tavily
    },
) as session:
    ...
```

Pass only the ones the configured backend needs; the others are ignored.

### Per-environment tuning

These are read live on every tool call, so a `@property` or callable works as
well as a plain attribute:

| Attribute | Env var | Default | Purpose |
|---|---|---|---|
| `web_as_of` | `OPENREWARD_WEB_AS_OF` | today | Cutoff date, `YYYY-MM-DD`. Honoured by backdated backends only |
| `web_max_fetch_chars` | `OPENREWARD_WEB_MAX_FETCH_CHARS` | 100,000 | How much page text `web_fetch` returns before truncating |
| `web_include_snippets` | `OPENREWARD_WEB_INCLUDE_SNIPPETS` | off | Add each hit's snippet to `web_search` output, so the agent can triage without a follow-up fetch |
| `search_backend` | `OPENREWARD_SEARCH_BACKEND` | `backsearch` | Pin the backend in code, for deployments where you cannot set process env |

```python
class ObscureFacts(Environment):
    toolsets = [WebToolset]
    web_include_snippets = True
    web_max_fetch_chars = 20_000
```

### Errors the agent sees

The toolset separates failures the agent can work around from ones it cannot,
because the difference decides whether a broken rollout scores 0.0 or is
discarded:

| Kind | Examples | Behaviour |
|---|---|---|
| Recoverable | empty results, a page that will not extract, blocked domain, bad URL | Returned as tool output with the code in `metadata["error"]` — the agent tries something else |
| Fatal | missing API key, exhausted quota, provider still failing after retries | Raises `SearchBackendUnavailable`, ending the rollout with a *blank* reward rather than a 0.0 that reads as a wrong answer |

Transient provider failures are retried three times with exponential backoff
before being treated as fatal; a dead credential is not retried at all.

### Backend trade-offs

Running the same two tasks on each backend, same model, same environment code:

| Backend | Result |
|---|---|
| `backsearch` | 1 of 2 answered — 14 turns and 15 searches on the other without converging |
| `tavily` | 2 of 2 answered, in 3 and 4 turns |

The questions here are reference-data lookups ("career Premier League
appearances") that live on Wikipedia, Transfermarkt and 11v11 — not in a
news/SEC/arXiv archive. `web_fetch` on a Wikipedia URL returns
`no CC-News capture of ...` under `backsearch`, which is correct behaviour for a
news corpus rather than a bug.

So **this environment suits live search**, and it is the inverse of a prediction
or forecasting task, where you would want `backsearch` precisely because it
cannot see past its cutoff. Two caveats if you switch:

- **`as_of` is silently ignored on Tavily.** Anything depending on a
  leakage-free cutoff must stay on `backsearch`.
- **`allowed_domains` is best-effort on Tavily.** It filters correctly for
  indexed domains, but for one Tavily has *not* indexed (`reuters.com`, for
  example) it silently returns unfiltered results instead of an empty set.

See [Web Tools](https://docs.openreward.ai/environments/web-tools) for the full
reference.

## Time Horizon

ObscureFacts is a multi-turn environment. Agents iteratively search the web, fetch page content, and synthesize information before submitting a final answer.

## Environment Difficulty

[Put environment difficulty here]

## Other Environment Requirements

- **OpenAI API key**: Required for LLM-based answer grading. Pass via `secrets={"openai_api_key": "..."}`. Grader failures are deliberately not swallowed — "could not grade" is not the same as "answered wrongly", so they raise rather than scoring 0.0.
- **Search credentials**: Whatever the configured search backend needs — see [Credentials](#credentials) above.
- **`openreward >= 0.1.152`**: earlier releases either lack `WebToolset` (< 0.1.150) or ship a transport that ignores `HTTP_PROXY`/`HTTPS_PROXY`, which makes the web tools hang in a proxied deployment.

## Safety

Agents in ObscureFacts search the web and answer trivia questions. The environment does not present direct safety risks.

## Citations

```bibtex
@dataset{GRObscureFacts,
  author    = {General Reasoning Inc. Team},
  title     = {ObscureFacts},
  year      = {2026},
  publisher = {OpenReward},
  url       = {https://openreward.ai/GeneralReasoning/ObscureFacts}
}
```
