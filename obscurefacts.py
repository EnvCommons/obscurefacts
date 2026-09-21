from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import unquote, urlsplit, urlunsplit

import openai
from pydantic import BaseModel

from openreward.environments import Environment, JSONObject, TextBlock, ToolOutput, terminal, tool
from openreward.toolsets import BackSearchToolset
from openreward.toolsets._web_common import WebFetchParams, WebSearchParams, to_tool_output
from openreward.tools.web import FETCH_DESCRIPTION, SEARCH_DESCRIPTION, WebToolResult, run_fetch, run_search
from openreward.web_service import WebServiceConfig


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
# Reward for a submission made after the task has already been graded. Negative
# so repeat submissions are actively discouraged, not merely left unscored.
REPEAT_SUBMISSION_PENALTY = -0.1


class SubmitAnswerInput(BaseModel):
    answer: str


# ============= Backdated web tools =============
def today_utc_iso() -> str:
    """Today's date in UTC as ISO ``YYYY-MM-DD`` — the backsearch cutoff.

    UTC rather than the server's local date so every replica of the env agrees
    on the cutoff regardless of the timezone it happens to run in.
    """
    return datetime.now(timezone.utc).date().isoformat()


def _url_variants(url: str) -> list[str]:
    """Cosmetic respellings of ``url`` worth retrying after a 404.

    The archive matches URLs byte-for-byte, so a page it holds under one
    spelling 404s under another. Measured on 84 real-but-mangled URLs: 89% of
    cosmetic variations 404, and this ladder recovered 100% of them
    (percent-encoding 9/9, trailing slash 41/41, ``www.`` 25/25).

    Percent-decoding is applied first and the toggles build on the decoded
    form, so a URL that is both re-encoded *and* slash-mangled is still
    repaired inside the three-attempt budget. Capped at three so a genuine miss
    costs at most three extra calls.
    """
    parts = urlsplit(url)
    decoded = unquote(parts.path)
    variants: list[str] = []
    base = url
    if decoded != parts.path:
        base = urlunsplit((parts.scheme, parts.netloc, decoded, parts.query, parts.fragment))
        variants.append(base)
    p = urlsplit(base)
    if p.path and p.path != "/":
        flipped = p.path[:-1] if p.path.endswith("/") else p.path + "/"
        variants.append(urlunsplit((p.scheme, p.netloc, flipped, p.query, p.fragment)))
    host = p.netloc[4:] if p.netloc.startswith("www.") else "www." + p.netloc
    variants.append(urlunsplit((p.scheme, host, p.path, p.query, p.fragment)))
    seen = {url}
    return [v for v in variants if not (v in seen or seen.add(v))]


def _is_not_archived(result: WebToolResult) -> bool:
    """True for "the archive has no capture of this URL", not for a real outage."""
    return (
        not result.ok
        and result.error_code == "web-service-error"
        and "HTTP 404" in (result.output or "")
    )


def _not_archived_result(url: str, as_of: Optional[str]) -> WebToolResult:
    """Replace the backend's raw 404 envelope with something the agent can act on.

    The stock message reads as an infrastructure failure and leaks internal
    corpus names, so agents re-fetch the same dead URL or guess neighbouring
    ones until the turn cap. Half the failed fetches observed in real rollouts
    were URLs the model invented rather than ones search returned, which is
    exactly the behaviour this wording is meant to stop.
    """
    return WebToolResult.error(
        "page-not-archived",
        f"No archived capture of {url} on or before {as_of or 'today'}. The archive "
        f"only serves pages it captured, so this URL may never have been captured, or "
        f"may not exist. Do not retry this URL or guess variations of it - choose a "
        f"different result from web_search instead.",
    )


def _drop_content_mirror(out: ToolOutput) -> ToolOutput:
    """Remove ``metadata["content"]``, a byte-identical copy of the text already
    in ``blocks``.

    The SDK's ``build_fetch_output`` puts the whole page into ``data["content"]``
    and ``to_tool_output`` copies that into the ToolOutput metadata, so every
    fetch ships the page twice. Nothing reads the copy. The training harness
    caps environment tool output at 32,768 bytes and replaces the *entire*
    result with "[env tool output exceeded cap before content rendering]" when
    it is exceeded, so the duplicate alone can cost the agent the page it just
    fetched. Measured on a real rollout: a 65,550-byte fetch was exactly half
    mirror, and dropping it brought the payload under the cap.
    """
    if not out.metadata or "content" not in out.metadata:
        return out
    slim = {k: v for k, v in out.metadata.items() if k != "content"}
    return ToolOutput(
        blocks=out.blocks,
        metadata=slim or None,
        reward=out.reward,
        finished=out.finished,
    )


class ObscureFactsBackSearch(BackSearchToolset):
    """BackSearchToolset with five adjustments, all backdating-preserving.

    First, the session's secrets (``api_key`` / ``openreward_api_key``) are
    consulted when building the web-service config, falling back to the process
    environment's ``OPENREWARD_API_KEY`` — the stock toolset reads the process
    env only. Second, ``web_search`` passes ``include_snippets=True`` so results
    carry text snippets instead of bare titles and URLs, letting the agent
    triage hits without a fetch per candidate. Third, fatal backend errors (a
    missing key, an exhausted quota) raise ``SearchBackendUnavailable`` instead
    of becoming tool output: handed back as text, the agent would re-issue a
    dead call until the turn cap and the rollout would score 0.0 as though the
    model had answered wrongly, rather than being discarded as an
    infrastructure failure. Fourth, a fetch that 404s is retried against
    cosmetic respellings of the URL (see ``_url_variants``), and a genuine miss
    returns an actionable ``page-not-archived`` message. Fifth, the redundant
    ``metadata["content"]`` mirror is dropped (see ``_drop_content_mirror``).

    The cutoff still resolves through the parent's ``_current_as_of``
    (``env.web_as_of``, the UTC date the session was created) on every call.
    No ``corpus`` is pinned, so the backend fans out over its default corpora
    (news, SEC filings, Wikipedia, general web, live captures, arXiv) — naming
    corpora *replaces* that set rather than extending it. Unlike the SDK's
    swappable web toolset, this one cannot be switched to a live-web provider
    by an environment variable.
    """

    def __init__(self, env: Optional[Any] = None, **kwargs: Any) -> None:
        if kwargs.get("config") is None:
            secrets = getattr(env, "search_secrets", None)
            kwargs["config"] = WebServiceConfig.from_env(secrets)
        super().__init__(env, **kwargs)

    @tool
    async def web_search(self, params: WebSearchParams) -> ToolOutput:
        result = await run_search(
            query=params.query,
            as_of=self._current_as_of(),
            allowed_domains=params.allowed_domains,
            blocked_domains=params.blocked_domains,
            config=self.config,
            include_snippets=True,
        )
        return to_tool_output(result, raise_on_fatal=True)

    @tool
    async def web_fetch(self, params: WebFetchParams) -> ToolOutput:
        as_of = self._current_as_of()
        result = await run_fetch(
            url=params.url, prompt=params.prompt, as_of=as_of, config=self.config
        )
        if _is_not_archived(result):
            for candidate in _url_variants(params.url):
                retry = await run_fetch(
                    url=candidate, prompt=params.prompt, as_of=as_of, config=self.config
                )
                if retry.ok:
                    result = retry
                    break
            else:
                result = _not_archived_result(params.url, as_of)
        return _drop_content_mirror(to_tool_output(result, raise_on_fatal=True))


# The environment framework reads ``fn.__doc__`` for each tool's description.
ObscureFactsBackSearch.web_search.__doc__ = SEARCH_DESCRIPTION
ObscureFactsBackSearch.web_fetch.__doc__ = FETCH_DESCRIPTION


# ============= Environment Class =============
class ObscureFacts(Environment):
    """
    ObscureFacts: A trivia question-answering environment requiring web search
    to find answers to obscure factual questions. Search and page fetch come
    from the SDK's backdated toolset (OpenReward's backsearch corpus, cutoff =
    the day the session starts); gpt-5-mini does the semantic grading.
    """

    # web_search / web_fetch come from the SDK's backdated toolset, pinned to
    # OpenReward's backsearch corpus. The cutoff (``web_as_of``) is set per
    # session in ``__init__`` to the UTC date the session was created, and the
    # toolset reads it live on every call.
    toolsets = [ObscureFactsBackSearch]

    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec, secrets)

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

        self.openai_client = openai.AsyncClient(api_key=openai_api_key)

        # Graded submissions this session. @terminal already hides this tool from
        # the model, so the harness normally invokes it once at the end of the
        # rollout -- but Environment._call_tool dispatches by name and does not
        # exclude terminal tools, so a direct second call would re-grade and pay
        # out again. Defence in depth.
        self.submitted = 0

        # Backsearch cutoff: the UTC date this session was created. Read live
        # by ObscureFactsBackSearch on every tool call (it outranks the
        # OPENREWARD_WEB_AS_OF env var), so the agent sees the web as it stood
        # on the day it started researching.
        self.web_as_of = today_utc_iso()

        # Read by ObscureFactsBackSearch when it builds its config, so the
        # backsearch key can come from the session (`api_key`) rather than the
        # server process.
        self.search_secrets = secrets

        # Fail fast if the backdated web service is unconfigured. Without this
        # the first search would raise mid-rollout instead of at session start.
        if WebServiceConfig.from_env(secrets) is None:
            raise ValueError(
                "Backdated web service is not configured: set OPENREWARD_API_KEY "
                "in the server process environment (or pass api_key in secrets)."
            )

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
- web_search: Search the web for information. Takes a query.
- web_fetch: Get the content of a specific URL. Takes a url and a prompt describing what to extract from the page.

Question: {self.question}

Search thoroughly and verify your answer. When you have your answer, reply with it as an ordinary message (no tool call) — that message is graded."""

        return [TextBlock(text=prompt_text)]

    @terminal
    @tool
    async def submit_answer(self, params: SubmitAnswerInput) -> ToolOutput:
        """
        Grade the assistant's final message against the golden answer.

        Terminal tool: hidden from the agent, which replies with its answer as
        an ordinary message rather than calling a tool. The harness routes that
        message text here for semantic LLM grading, and the episode ends.
        """
        if self.submitted > 0:
            return ToolOutput(
                blocks=[TextBlock(text="An answer has already been submitted for this task. "
                                       "This episode is over: it is not re-graded, and repeat "
                                       "submissions are penalised (reward -0.1).")],
                metadata={"already_submitted": True, "submission_count": self.submitted},
                reward=REPEAT_SUBMISSION_PENALTY,
                finished=True,
            )

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

        self.submitted += 1

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
