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

Which provider answers those two tools is configuration on the environment
server, not code here — so swapping it needs no change to this environment:

| `OPENREWARD_SEARCH_BACKEND` | Backend | Needs |
|---|---|---|
| unset (default) | `backsearch` — GR's backdated corpus, bounded to an `as_of` cutoff | `OPENREWARD_API_KEY`, or `api_key` in session secrets |
| `tavily` | Tavily — live web | `TAVILY_API_KEY` (or `tavily_api_key` in session secrets) and `pip install 'openreward[search]'` |

Tavily searches the live web and cannot bound results to a cutoff date, so keep
the default `backsearch` backend wherever post-cutoff leakage would matter. See
[Web Tools](https://docs.openreward.ai/environments/web-tools).

## Time Horizon

ObscureFacts is a multi-turn environment. Agents iteratively search the web, fetch page content, and synthesize information before submitting a final answer.

## Environment Difficulty

[Put environment difficulty here]

## Other Environment Requirements

- **OpenAI API key**: Required for LLM-based answer grading. Pass via `secrets={"openai_api_key": "..."}`.
- **Search credentials**: Whatever the configured search backend needs — `secrets={"api_key": "..."}` for the default backsearch backend, or `secrets={"tavily_api_key": "..."}` when running with `OPENREWARD_SEARCH_BACKEND=tavily`. These fall back to the server process environment (`OPENREWARD_API_KEY` / `TAVILY_API_KEY`) if not passed. An unconfigured backend surfaces as a recoverable tool error rather than failing session creation.

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
