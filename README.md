# ObscureFacts

[![⭐ OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/ObscureFacts)

## Description

ObscureFacts is an environment for evaluating an agent's ability to find answers to obscure trivia questions using web search. Agents must use web search tools to research and answer 50 intentionally difficult factual questions spanning sports, technology, local history, and academia.

## Capabilities

- Backdated web search and information retrieval
- Multi-step research and evidence synthesis
- Answering obscure factual questions across diverse domains
- Extracting specific facts from web content

## Compute Requirements

Agents are given a standard environment with no special compute requirements.

## License

[MIT](https://opensource.org/licenses/MIT)

## Tasks

There is one split in this environment:

- **Train**: 50 obscure trivia questions

Questions span diverse domains including sports statistics, technology history, local history, and academic trivia.

## Reward Structure

This is a multi-turn environment. The agent searches the web, gathers information, and submits a final answer. The reward is binary (1.0 or 0.0) based on LLM-based semantic grading using `gpt-5-mini`. The grader checks semantic equivalence, accepting synonyms and paraphrasing while requiring exact numeric values.

## Data

Task data consists of 50 curated trivia questions with reference answers stored in a JSON file. Task data is stored on the OpenReward platform.

## Tools

Search and fetch come from the OpenReward SDK's backdated web toolset (a `BackSearchToolset` subclass declared via `toolsets = [ObscureFactsBackSearch]`) rather than being implemented in this environment. Both tools read OpenReward's backsearch corpus as it stood on the day the session started.

| Tool | Description |
|------|-------------|
| `web_search` | Search the backdated web corpus. Takes a `query` and optional `allowed_domains` **or** `blocked_domains`; returns a `Links:` list of up to 8 `{title, url, snippet}` hits fanned out over the backend's default corpora (news, SEC filings, Wikipedia, general web, live captures, arXiv). |
| `web_fetch` | Fetch the archived text of a URL as it existed on or before the session's start date. Takes a `url` and a `prompt` describing what to extract (truncated to 100 KB). A fetch that 404s on a percent-encoded URL is retried with the path decoded, because the archive stores Wikipedia titles in raw Unicode. |

Grading runs through a hidden `@terminal` tool rather than a tool the agent can call: replying with a plain message ends the rollout, and that message text is semantically graded against the reference answer.

### Search backend

The environment is pinned to **backsearch**, OpenReward's point-in-time web archive, in code. The cutoff (`web_as_of`) is set once per session in the environment's constructor to the UTC date the session starts, and the toolset reads it on every call, so it outranks the `OPENREWARD_WEB_AS_OF` env var. Unlike the SDK's swappable `WebToolset`, this toolset cannot be switched to a live-web provider by an environment variable. No corpus is pinned: naming corpora replaces the backend's default set rather than extending it.

### Errors the agent sees

The toolset separates failures the agent can work around from ones it cannot, because the difference decides whether a broken rollout scores 0.0 or is discarded:

| Kind | Examples | Behaviour |
|---|---|---|
| Recoverable | empty results, a page not in the archive, blocked domain, bad URL | Returned as tool output with the code in `metadata["error"]` — the agent tries something else |
| Fatal | missing API key, exhausted quota, provider still failing after retries | Raises `SearchBackendUnavailable`, ending the rollout with a *blank* reward rather than a 0.0 that reads as a wrong answer |

Transient backend failures are retried three times with exponential backoff before being treated as fatal. The environment also fails fast at session start if the backdated web service is not configured.

### Backsearch vs Tavily

This environment previously pinned Tavily (the live web) on the strength of a two-task comparison made when backsearch covered only news, SEC filings and arXiv. With the Wikipedia, general-web and live-capture corpora now in the archive, a paired A/B on 2026-09-21 (gpt-5.2, same environment code, only the backend flipped, 20 tasks per arm) found no difference:

| Backend | Mean reward (unanswered = 0) | Wins / ties / losses | Fetch failures |
|---|---|---|---|
| `backsearch` | 0.50 | 3 / 12 / 3 | 15% (pages not in the archive) |
| `tavily` | 0.50 | 3 / 12 / 3 | 18% (anti-bot blocks) |

Backsearch also returns far fewer social and video pages (2% of hits vs 31% for Tavily) and does not need a second vendor key. See [Backdated Web Tools](https://docs.openreward.ai/environments/backdated-web-tools) for the full reference.

## Time Horizon

ObscureFacts is a multi-turn environment. Agents iteratively search the web, fetch page content, and synthesize information before submitting a final answer.

## Environment Difficulty

[Put environment difficulty here]

## Other Environment Requirements

- **OpenAI API key**: Required for LLM-based answer grading. Pass via `secrets={"openai_api_key": "..."}`. Grader failures are deliberately not swallowed — "could not grade" is not the same as "answered wrongly", so they raise rather than scoring 0.0.
- **`openreward >= 0.1.158`**: the SDK release the backdated toolset and its corpus fan-out were verified against.

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
