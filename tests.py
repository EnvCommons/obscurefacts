"""Unit tests for ObscureFacts grading.

The LLM grader is replaced by a fake client that returns a fixed reply, so the
tests check how a verdict becomes a reward and what the result shows the
agent. Run with:
    uv run --no-project --with-requirements requirements.txt --with pytest \
        --with pytest-asyncio python -m pytest tests.py
"""

import json
from types import SimpleNamespace

import pytest

import obscurefacts
from obscurefacts import ObscureFacts, SubmitAnswerInput

TASKS = obscurefacts.TASKS[:5]
SECRETS = {"openai_api_key": "test-openai-key", "api_key": "test-openreward-key"}


class FakeCompletions:
    def __init__(self, reply: str):
        self.reply = reply
        self.prompts: list[str] = []

    async def create(self, model, messages):
        self.prompts.append(messages[0]["content"])
        message = SimpleNamespace(content=self.reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def make_env(task: dict, grader_reply: str) -> tuple[ObscureFacts, FakeCompletions]:
    env = ObscureFacts(task_spec={"id": task["id"], "question": task["question"]}, secrets=SECRETS)
    completions = FakeCompletions(grader_reply)
    env.openai_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return env, completions


def correct_reply(task: dict) -> str:
    # Real grader replies restate the golden answer in their justification.
    return (
        f"The predicted answer matches the golden answer ({task['answer']}) "
        f"exactly.\n\nCORRECT"
    )


def incorrect_reply(task: dict) -> str:
    return (
        f"The golden answer is {task['answer']}; the predicted answer names "
        f"something else.\n\nINCORRECT"
    )


def visible_to_agent(result) -> str:
    """Everything the agent can see in a result, minus its own submitted answer."""
    metadata = {k: v for k, v in (result.metadata or {}).items() if k != "submitted_answer"}
    return "\n".join(b.text for b in result.blocks) + "\n" + json.dumps(metadata, ensure_ascii=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("task", TASKS, ids=lambda t: t["id"])
async def test_gold_answer_scores_one(task):
    env, completions = make_env(task, correct_reply(task))
    result = await env.submit_answer(SubmitAnswerInput(answer=task["answer"]))
    assert result.reward == 1.0
    assert result.finished is True
    assert result.metadata["is_correct"] is True
    assert result.metadata["submitted_answer"] == task["answer"]
    # The grader itself still sees the golden answer.
    assert task["answer"] in completions.prompts[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("task", TASKS, ids=lambda t: t["id"])
async def test_wrong_answer_scores_zero(task):
    env, _ = make_env(task, incorrect_reply(task))
    result = await env.submit_answer(SubmitAnswerInput(answer="definitely not it"))
    assert result.reward == 0.0
    assert result.finished is True
    assert result.metadata["is_correct"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("task", TASKS, ids=lambda t: t["id"])
async def test_empty_answer_scores_zero_without_grader(task):
    env, completions = make_env(task, correct_reply(task))
    result = await env.submit_answer(SubmitAnswerInput(answer="   "))
    assert result.reward == 0.0
    assert completions.prompts == []


@pytest.mark.asyncio
@pytest.mark.parametrize("task", TASKS, ids=lambda t: t["id"])
@pytest.mark.parametrize("kind", ["gold", "wrong"])
async def test_result_does_not_reveal_golden_answer(task, kind):
    if kind == "gold":
        env, _ = make_env(task, correct_reply(task))
        result = await env.submit_answer(SubmitAnswerInput(answer=task["answer"]))
    else:
        env, _ = make_env(task, incorrect_reply(task))
        result = await env.submit_answer(SubmitAnswerInput(answer="definitely not it"))

    assert set(result.metadata) == {"task_id", "submitted_answer", "is_correct"}
    assert task["answer"] not in visible_to_agent(result)


@pytest.mark.asyncio
async def test_repeat_submission_is_penalised_and_not_regraded():
    task = TASKS[0]
    env, completions = make_env(task, correct_reply(task))
    first = await env.submit_answer(SubmitAnswerInput(answer=task["answer"]))
    second = await env.submit_answer(SubmitAnswerInput(answer=task["answer"]))
    assert first.reward == 1.0
    assert second.reward == obscurefacts.REPEAT_SUBMISSION_PENALTY
    assert second.metadata == {"already_submitted": True, "submission_count": 1}
    assert len(completions.prompts) == 1


def test_canary_wharf_question_names_the_overground_direction():
    # One overground stop from Whitechapel is Shoreditch High Street northbound
    # and Shadwell southbound, so the question has to say which way.
    task = next(t for t in obscurefacts.TASKS if t["id"] == "obscure_028")
    assert "north" in task["question"]
    assert task["answer"] == "Shoreditch High Street"


def test_benoit_final_opponent_age_accepts_both_birth_years():
    # Benoit's last match was against Elijah Burke on 19 June 2007. Wikipedia
    # gives Burke's birth date as 24 May 1981, so he was 26; Cagematch and
    # Wikidata give 1978, which makes him 29. Sources disagree, so both ages
    # are accepted.
    task = next(t for t in obscurefacts.TASKS if t["id"] == "obscure_004")
    assert task["answer"] == "26 or 29 (either answer acceptable)"


def test_tasks_keep_their_positions():
    # Training runs pin the task count and address tasks by position
    # (task_train_<i> is TASKS[i]), so fixes are made in place: no task is
    # removed, added or reordered while a run depends on the list. obscure_029
    # (the ABC Stores count) stays until a restart re-pins the count.
    ids = [t["id"] for t in obscurefacts.TASKS]
    assert ids == [f"obscure_{i:03d}" for i in range(1, 51)]


def test_clough_league_cup_count():
    # Forest won the League Cup four times under Clough: 1978, 1979, 1989, 1990.
    task = next(t for t in obscurefacts.TASKS if t["id"] == "obscure_003")
    assert task["answer"] == "4 (1978, 1979, 1989, 1990)"


def test_trust_me_question_names_honolulu():
    # Sushi Nozawa in Los Angeles was also known for "Trust Me", and "in the
    # Pacific" read as the Pacific coast. The reference is the Honolulu Sasabune.
    task = next(t for t in obscurefacts.TASKS if t["id"] == "obscure_025")
    assert "Honolulu" in task["question"]
    assert task["answer"] == "Sushi Sasabune"


def test_general_reasoning_question_does_not_imply_an_ipo():
    # The company is private, so "first went public" read as asking about an IPO.
    task = next(t for t in obscurefacts.TASKS if t["id"] == "obscure_033")
    assert "went public" not in task["question"]
    assert "launched publicly" in task["question"]


def test_treasury_deputy_director_name_is_spelled_correctly():
    task = next(t for t in obscurefacts.TASKS if t["id"] == "obscure_040")
    assert task["answer"] == "Gwyneth Nurse"


def test_transformer_week_question_singles_out_selu():
    # arXiv v1 dates: SELU 8 June 2017, "Attention Is All You Need" 12 June.
    # "Deep RL from Human Preferences" shares the Transformer's day and several
    # other well-known papers fall within a week of it, so the question names
    # the lab's country: SELU came from JKU Linz, Austria.
    task = next(t for t in obscurefacts.TASKS if t["id"] == "obscure_048")
    assert "within a week" in task["question"]
    assert "Austrian" in task["question"]
    assert "SELU" in task["answer"]
