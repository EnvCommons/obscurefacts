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

### Search backend

The environment pins its search backend to **Tavily** (the live web) in code, via the `search_backend = "tavily"` class attribute. `WebToolset` reads that attribute on every tool call and it takes precedence over the `OPENREWARD_SEARCH_BACKEND` process env var, so process configuration cannot swap this environment onto another backend. Tavily support requires `pip install 'openreward[search]'`. The pin is empirical, not incidental — see [Why Tavily](#why-tavily) below.

### Credentials

Tavily reads its key from the session `secrets` mapping first (`tavily_api_key`), falling back to the server's process environment (`TAVILY_API_KEY`). The environment exposes `self.search_secrets = secrets` so the toolset can pick the key out of the session:

```python
async with environment.session(
    task=task,
    secrets={
        "openai_api_key": OPENAI_API_KEY,   # grader
        "tavily_api_key": TAVILY_API_KEY,   # web_search / web_fetch
    },
) as session:
    ...
```

### Per-environment tuning

These are read live on every tool call, so a `@property` or callable works as
well as a plain attribute:

| Attribute | Env var | Default | Purpose |
|---|---|---|---|
| `web_as_of` | `OPENREWARD_WEB_AS_OF` | today | Cutoff date, `YYYY-MM-DD`. Honoured by backdated backends only |
| `web_max_fetch_chars` | `OPENREWARD_WEB_MAX_FETCH_CHARS` | 100,000 | How much page text `web_fetch` returns before truncating |
| `web_include_snippets` | `OPENREWARD_WEB_INCLUDE_SNIPPETS` | off | Add each hit's snippet to `web_search` output, so the agent can triage without a follow-up fetch |
| `search_backend` | `OPENREWARD_SEARCH_BACKEND` | `backsearch` | Pin the backend in code — this environment sets it to `tavily` |

```python
class ObscureFacts(Environment):
    toolsets = [WebToolset]
    search_backend = "tavily"
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

### Why Tavily

Running the same two tasks on each available backend, same model, same environment code:

| Backend | Result |
|---|---|
| `backsearch` (GR's backdated news/SEC/arXiv corpus) | 1 of 2 answered — 14 turns and 15 searches on the other without converging |
| `tavily` (live web) | 2 of 2 answered, in 3 and 4 turns |

The questions here are reference-data lookups ("career Premier League appearances") that live on Wikipedia, Transfermarkt and 11v11 — not in a news/SEC/arXiv archive, which is why the environment pins the live-web backend. Two caveats that come with Tavily:

- **`as_of` is silently ignored.** Nothing in this environment depends on a leakage-free cutoff, but a prediction or forecasting task would need a backdated backend instead.
- **`allowed_domains` is best-effort.** It filters correctly for indexed domains, but for one Tavily has *not* indexed (`reuters.com`, for example) it silently returns unfiltered results instead of an empty set.

See [Web Tools](https://docs.openreward.ai/environments/web-tools) for the full
reference.

## Time Horizon

ObscureFacts is a multi-turn environment. Agents iteratively search the web, fetch page content, and synthesize information before submitting a final answer.

## Environment Difficulty

[Put environment difficulty here]

## Other Environment Requirements

- **OpenAI API key**: Required for LLM-based answer grading. Pass via `secrets={"openai_api_key": "..."}`. Grader failures are deliberately not swallowed — "could not grade" is not the same as "answered wrongly", so they raise rather than scoring 0.0.
- **Tavily API key**: Required for the `web_search` and `web_fetch` tools. Pass via `secrets={"tavily_api_key": "..."}` — see [Credentials](#credentials) above.
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
