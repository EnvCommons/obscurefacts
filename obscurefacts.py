from __future__ import annotations

import asyncio
import json
from pathlib import Path

import openai
from pydantic import BaseModel
from tavily import AsyncTavilyClient

from openreward.environments import Environment, JSONObject, TextBlock, ToolOutput, terminal, tool


# ============= Data Loading =============
# Check for /orwd_data first (production), then fall back to local path (development)
if Path("/orwd_data/").exists():
    DATA_PATH = Path("/orwd_data/")
else:
    DATA_PATH = Path(__file__).parent

with open(DATA_PATH / "tasks.json", "r") as f:
    TASKS = json.load(f)

# Build lookup for answers (private - not exposed to agent)
ANSWERS = {task["id"]: task["answer"] for task in TASKS}


# ============= Pydantic Models for Tool Inputs =============
class WebSearchInput(BaseModel):
    query: str


class FetchUrlInput(BaseModel):
    url: str


class SubmitAnswerInput(BaseModel):
    answer: str


# ============= Environment Class =============
class ObscureFacts(Environment):
    """
    ObscureFacts: A trivia question-answering environment requiring web search
    to find answers to obscure factual questions. Uses Tavily for web search
    and gpt-5-mini for semantic grading.
    """

    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec)

        # Extract task info
        self.task_id = str(task_spec["id"])
        self.question = str(task_spec["question"])

        # CRITICAL: Validate API keys from secrets (no env var fallback)
        openai_api_key = secrets.get("openai_api_key")
        if not openai_api_key:
            raise ValueError(
                "OpenAI API key required in secrets parameter. "
                "Pass secrets={'openai_api_key': 'your-key'} when creating session."
            )

        tavily_api_key = secrets.get("tavily_api_key")
        if not tavily_api_key:
            raise ValueError(
                "Tavily API key required in secrets parameter. "
                "Pass secrets={'tavily_api_key': 'your-key'} when creating session."
            )

        self.openai_client = openai.AsyncClient(api_key=openai_api_key)
        self.tavily_client = AsyncTavilyClient(api_key=tavily_api_key)

        # Load golden answer from backend storage
        self.answer = ANSWERS.get(self.task_id)
        if not self.answer:
            raise ValueError(f"Task {self.task_id} not found in dataset")

    @classmethod
    def list_splits(cls) -> list[str]:
        """Return available splits"""
        return ["train"]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        """Return task specifications for a given split (without answers)"""
        if split != "train":
            raise ValueError(f"Unknown split: {split}. Available splits: ['train']")
        # Return only id and question - NOT the answer
        return [{"id": task["id"], "question": task["question"]} for task in TASKS]

    async def get_prompt(self) -> list[TextBlock]:
        """Return the prompt shown to the agent"""
        prompt_text = f"""Answer the following trivia question. These questions are intentionally obscure and may require extensive web searching to find the correct answer.

You have access to the following tools:
- web_search: Search the web for information
- fetch_url: Get the full content of a specific URL

Question: {self.question}

Search thoroughly and verify your answer. When you have your answer, reply with it as an ordinary message (no tool call) — that message is graded."""

        return [TextBlock(text=prompt_text)]

    MAX_TAVILY_ATTEMPTS = 3

    async def _call_tavily(self, op, **kwargs):
        """
        Run a Tavily call with a bounded retry budget, then let the exception
        propagate.

        A transient blip is worth retrying; a persistent failure (plan/quota
        limit, bad key, outage) must reach the platform, which retries the tool
        call and terminates the rollout with a blank reward if it still fails.
        Returning the error as tool output instead would leave the agent
        re-issuing a dead call until the wall-clock cap, never reaching
        submit_answer, and score the rollout 0.0 as though it had answered
        wrongly.
        """
        for attempt in range(self.MAX_TAVILY_ATTEMPTS):
            try:
                return await op(**kwargs)
            except Exception:
                if attempt == self.MAX_TAVILY_ATTEMPTS - 1:
                    raise
                await asyncio.sleep(2 ** attempt)

    @tool
    async def web_search(self, params: WebSearchInput) -> ToolOutput:
        """
        Search the web using Tavily. Returns search results with titles, URLs, and snippets.
        Use fetch_url tool to get full content from specific URLs if needed.
        """
        response = await self._call_tavily(
            self.tavily_client.search,
            query=params.query,
            search_depth="basic",
            max_results=5
        )

        # Format results. An empty result set is a real answer to the query,
        # not a failure — the agent can act on it by searching differently.
        results = response.get("results", [])
        if not results:
            return ToolOutput(
                blocks=[TextBlock(text="No search results found.")],
                metadata={"query": params.query, "results": []},
                reward=0.0,
                finished=False
            )

        # Build display text
        display_parts = [f"Search results for: {params.query}\n"]
        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            url = result.get("url", "")
            snippet = result.get("content", "")
            display_parts.append(f"{i}. {title}\n   URL: {url}\n   {snippet}\n")

        display_text = "\n".join(display_parts)

        return ToolOutput(
            blocks=[TextBlock(text=display_text)],
            metadata={
                "query": params.query,
                "results": results,
                "count": len(results)
            },
            reward=0.0,
            finished=False
        )

    @tool
    async def fetch_url(self, params: FetchUrlInput) -> ToolOutput:
        """
        Fetch and return the full text content from a specific URL using Tavily's extract method.
        Use this after web_search to get complete information from a page.
        """
        response = await self._call_tavily(self.tavily_client.extract, urls=[params.url])

        # A page Tavily can't extract is useful feedback — the agent should try
        # another URL rather than have the rollout thrown away.
        results = response.get("results", [])
        if not results:
            return ToolOutput(
                blocks=[TextBlock(text=f"No content extracted from {params.url}")],
                metadata={"url": params.url, "results": []},
                reward=0.0,
                finished=False
            )

        # Get the first result (we only passed one URL)
        result = results[0]
        raw_content = result.get("raw_content", "")

        # Truncate if too long
        max_length = 8000
        if len(raw_content) > max_length:
            raw_content = raw_content[:max_length] + "...\n[Content truncated]"

        return ToolOutput(
            blocks=[TextBlock(text=f"Content from {params.url}:\n\n{raw_content}")],
            metadata={
                "url": params.url,
                "length": len(raw_content)
            },
            reward=0.0,
            finished=False
        )

    @terminal
    @tool
    async def submit_answer(self, params: SubmitAnswerInput) -> ToolOutput:
        """
        Grade the assistant's final message against the golden answer.

        Terminal tool: hidden from the agent, which replies with its answer as
        an ordinary message rather than calling a tool. The harness routes that
        message text here for semantic LLM grading, and the episode ends.
        """
        grader_result = await self._grade_answer(params.answer)

        reward = grader_result["reward"]
        is_correct = grader_result["is_correct"]
        justification = grader_result["justification"]

        # Format display
        result_text = "CORRECT" if is_correct else "INCORRECT"

        display_text = f"""{result_text}

Evaluation:
{justification}

Reference Answer: {self.answer}
"""

        return ToolOutput(
            blocks=[TextBlock(text=display_text)],
            metadata={
                "task_id": self.task_id,
                "submitted_answer": params.answer,
                "golden_answer": self.answer,
                "is_correct": is_correct,
                "justification": justification,
            },
            reward=reward,
            finished=True
        )

    async def _grade_answer(self, predicted_answer: str) -> dict:
        """
        Grade the answer using gpt-5-mini LLM grader.
        Compares submitted answer against golden answer for semantic equivalence.
        Returns: {is_correct: bool, justification: str, reward: float}
        """
        # Handle empty answers
        if not predicted_answer or len(predicted_answer.strip()) == 0:
            return {
                "is_correct": False,
                "justification": "Empty or whitespace-only answer provided.",
                "reward": 0.0
            }

        # Build grader prompt
        grader_prompt = f"""You are an expert trivia evaluator. Determine if the predicted answer is semantically equivalent to the golden answer.

Question: {self.question}

Golden Answer: {self.answer}

Predicted Answer: {predicted_answer}

Instructions:
1. Check if the predicted answer is semantically equivalent to the golden answer
2. Consider synonyms, paraphrasing, and equivalent expressions
3. For numeric answers, the values should match exactly
4. Ignore minor formatting differences (e.g., "443" vs "443 appearances")
5. Provide a brief justification (2-3 sentences)
6. End your response with EXACTLY one of these labels on a new line:
   - "CORRECT" if semantically equivalent
   - "INCORRECT" if not equivalent

Format:
[Your justification here]

CORRECT or INCORRECT"""

        # Grader failures are deliberately not caught. "Couldn't grade" is not
        # the same as "answered wrongly": scoring it 0.0 would penalise the
        # agent for infrastructure it can't control and skew reward stats.
        # Letting it raise hands the call to the platform's retry, and a
        # persistent failure ends the rollout with a blank reward.
        response = await self.openai_client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": grader_prompt}],
        )

        grading_response = response.choices[0].message.content or ""
        if not grading_response.strip():
            raise RuntimeError(
                f"Grader returned an empty response for task {self.task_id}"
            )

        # Parse CORRECT/INCORRECT
        upper_response = grading_response.upper()
        is_correct = "CORRECT" in upper_response and "INCORRECT" not in upper_response

        reward = 1.0 if is_correct else 0.0

        return {
            "is_correct": is_correct,
            "justification": grading_response,
            "reward": reward
        }
