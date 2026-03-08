# ObscureFacts

[![⭐ OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/ObscureFacts)

## Description

ObscureFacts is an ORS environment for evaluating an agent's ability to find answers to obscure trivia questions using web search. Agents must use web search tools to research and answer 49 intentionally difficult factual questions spanning sports, technology, local history, and academia.

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

| Tool | Description |
|------|-------------|
| `web_search` | Search the web using Tavily. Returns up to 5 results with titles, URLs, and snippets. |
| `fetch_url` | Fetch and return full text content from a specific URL (truncated to 8,000 characters). |
| `submit_answer` | Submit a final answer for semantic grading against the reference answer. |

Note that the `fetch_url` and `web_search` tools require Tavily, but are optional. If you want to use a different provider for search you can exclude these tools and use external tools instead.

## Time Horizon

ObscureFacts is a multi-turn environment. Agents iteratively search the web, fetch page content, and synthesize information before submitting a final answer.

## Environment Difficulty

[Put environment difficulty here]

## Other Environment Requirements

- **OpenAI API key**: Required for LLM-based answer grading. Pass via `secrets={"openai_api_key": "..."}`.
- **Tavily API key**: Required for web search and URL extraction. Pass via `secrets={"tavily_api_key": "..."}`.

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
