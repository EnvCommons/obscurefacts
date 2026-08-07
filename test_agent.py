"""Agent loop for ObscureFacts.

ObscureFacts uses a one-argument @terminal tool: the grader is hidden from the
model, which researches with web_search / web_fetch and then simply writes its
answer as an ordinary message. The harness sees a message with no tool calls
and routes its text to session.call_terminal_tool(), which grades it
semantically against the reference answer.

Runs against the deployed environment by default; set LOCAL=1 to point at a
local `python server.py` on port 8080.

Search and fetch come from the SDK's WebToolset, so the search backend is
configuration rather than code: the *environment server* picks it up from
OPENREWARD_SEARCH_BACKEND (default "backsearch"; "tavily" also needs
TAVILY_API_KEY and `pip install 'openreward[search]'`).

Records each task as an OpenReward rollout (visible at
https://openreward.ai/rollout/<id>) and also writes a local trajectory to
obscurefacts_trajectory.jsonl (one JSON object per line: config, each turn, and
a final summary).
"""

import asyncio
import json
import os
from datetime import datetime, timezone

from openai import AsyncOpenAI
from openreward import AsyncOpenReward

TRAJECTORY_PATH = "obscurefacts_trajectory.jsonl"


def _text_of(response) -> str:
    """Concatenate the assistant's output_text items."""
    parts = []
    for item in response.output:
        if item.type == "message":
            for block in item.content:
                if block.type == "output_text":
                    parts.append(block.text)
    return "\n".join(parts).strip()


async def main():
    or_client = AsyncOpenReward()
    oai_client = AsyncOpenAI()

    MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-5.2")
    ENV_NAME = "GeneralReasoning/ObscureFacts"
    SPLIT = os.environ.get("SPLIT", "train")
    NUM_TASKS = int(os.environ.get("NUM_TASKS", "2"))
    MAX_TURNS = int(os.environ.get("MAX_TURNS", "30"))
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    # Search credentials are forwarded to the environment's WebToolset, which
    # picks whichever the configured backend needs: `api_key` for the default
    # backsearch backend, `tavily_api_key` when the environment server runs with
    # OPENREWARD_SEARCH_BACKEND=tavily. Both are optional here.
    OPENREWARD_API_KEY = os.environ.get("OPENREWARD_API_KEY", "")
    TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
    # Labels the recorded rollouts so runs against different search backends are
    # distinguishable. The server is what actually resolves the backend; when
    # running LOCAL=1 both processes see the same environment.
    SEARCH_BACKEND = os.environ.get("OPENREWARD_SEARCH_BACKEND", "backsearch")
    RUN_NAME = os.environ.get("RUN_NAME", f"obscurefacts-{SEARCH_BACKEND}")

    # Deployed environment unless LOCAL=1 (or ENV_URL points somewhere else).
    base_url = os.environ.get("ENV_URL") or (
        "http://localhost:8080" if os.environ.get("LOCAL") else None
    )
    environment = or_client.environments.get(name=ENV_NAME, base_url=base_url)
    print(f"Environment: {ENV_NAME} ({base_url or 'deployed'})")

    tasks = await environment.list_tasks(split=SPLIT)
    tools = await environment.list_tools(format="openai")
    terminal_tool = await environment.terminal_tool()

    print(f"Found {len(tasks)} tasks")
    print(f"Tools visible to the model: {[t['name'] for t in tools]}")
    print(f"Terminal tool (hidden from model): {terminal_tool}")

    traj = open(TRAJECTORY_PATH, "w")

    def record(kind: str, **fields):
        traj.write(json.dumps({
            "kind": kind,
            "ts": datetime.now(timezone.utc).isoformat(),
            **fields,
        }) + "\n")
        traj.flush()

    record(
        "config",
        model=MODEL_NAME,
        env=ENV_NAME,
        base_url=base_url or "deployed",
        split=SPLIT,
        search_backend=SEARCH_BACKEND,
        run_name=RUN_NAME,
        visible_tools=[t["name"] for t in tools],
        terminal_tool=None if terminal_tool is None else {
            "name": terminal_tool.name,
            "arg": terminal_tool.arg,
            "description": terminal_tool.description,
        },
    )

    rewards = []

    rollout_urls = []

    for task in tasks[:NUM_TASKS]:
        print(f"\n=== Task {task.task_spec['id']} ===")
        print(f"Question: {task.task_spec['question']}")

        # One OpenReward rollout per task — the recorded trace shows the prompt,
        # every search/fetch call and its result, and the graded reward.
        rollout = or_client.rollout.create(
            run_name=RUN_NAME,
            rollout_name=f"task-{task.task_spec['id']}",
            environment=ENV_NAME,
            split=SPLIT,
            task_spec=task.task_spec,
            metadata={"model": MODEL_NAME, "search_backend": SEARCH_BACKEND},
            print_messages=True,
        )
        rollout_urls.append(f"https://openreward.ai/rollout/{rollout.event_id}")

        async with environment.session(
            task=task,
            secrets={
                "openai_api_key": OPENAI_API_KEY,
                **({"api_key": OPENREWARD_API_KEY} if OPENREWARD_API_KEY else {}),
                **({"tavily_api_key": TAVILY_API_KEY} if TAVILY_API_KEY else {}),
            },
        ) as session:
            # The whole point: ask the environment which convention it uses.
            assistant_ends_rollout = await session.is_assistant_message_final()
            session_tools = await session.list_tools()
            print(f"is_assistant_message_final() -> {assistant_ends_rollout}")
            print(f"session.list_tools() -> {[t.name for t in session_tools]}")
            assert "submit_answer" not in [t.name for t in session_tools], \
                "terminal tool leaked into the model's tool list"

            prompt = await session.get_prompt()
            input_list = [{"role": "user", "content": prompt[0].text}]
            rollout.log_openai_response(input_list[0])

            record(
                "task_start",
                task_id=task.task_spec["id"],
                question=task.task_spec["question"],
                is_assistant_message_final=assistant_ends_rollout,
                session_tools=[t.name for t in session_tools],
                prompt=prompt[0].text,
            )

            reward = None
            turn = 0

            while turn < MAX_TURNS:
                turn += 1
                print(f"\n--- Turn {turn} ---")

                response = await oai_client.responses.create(
                    model=MODEL_NAME,
                    tools=tools,
                    input=input_list,
                )
                input_list += response.output
                # Logs every output item (reasoning, message, tool calls).
                rollout.log_openai_response(response)

                calls = [i for i in response.output if i.type == "function_call"]

                if calls:
                    for item in calls:
                        args = json.loads(str(item.arguments))
                        tool_result = await session.call_tool(item.name, args)
                        input_list.append({
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": tool_result.blocks[0].text,
                        })
                        rollout.log_openai_response(
                            input_list[-1],
                            reward=tool_result.reward,
                            is_finished=tool_result.finished,
                            metadata=tool_result.metadata,
                        )
                        preview = tool_result.blocks[0].text[:200]
                        print(f"Tool: {item.name}({json.dumps(args)[:120]})")
                        print(f"  -> {preview}...")
                        record(
                            "tool_call",
                            task_id=task.task_spec["id"],
                            turn=turn,
                            tool=item.name,
                            arguments=args,
                            output=tool_result.blocks[0].text,
                            reward=tool_result.reward,
                            finished=tool_result.finished,
                        )
                    continue

                # No tool calls: this message is the answer.
                final_message = _text_of(response)
                print(f"Final message: {final_message[:300]}")
                record(
                    "assistant_final_message",
                    task_id=task.task_spec["id"],
                    turn=turn,
                    text=final_message,
                )

                if not assistant_ends_rollout:
                    print("Environment is not terminal-tool style; stopping.")
                    break

                out = await session.call_terminal_tool(final_message)
                reward = out.reward
                # The graded answer carries the episode reward in the trace.
                rollout.log_openai_response(
                    {"role": "assistant", "content": final_message},
                    reward=reward,
                    is_finished=True,
                    metadata=out.metadata,
                )
                print(f"\ncall_terminal_tool -> reward={reward} finished={out.finished}")
                print(out.blocks[0].text[:400])
                record(
                    "terminal_tool_result",
                    task_id=task.task_spec["id"],
                    turn=turn,
                    submitted=final_message,
                    reward=out.reward,
                    finished=out.finished,
                    output=out.blocks[0].text,
                    metadata=out.metadata,
                )
                break

            rewards.append(reward)
            record("task_end", task_id=task.task_spec["id"], turns=turn, reward=reward)

    scored = [r for r in rewards if r is not None]
    summary = {
        "num_tasks": len(rewards),
        "num_scored": len(scored),
        "mean_reward": (sum(scored) / len(scored)) if scored else None,
        "rewards": rewards,
        "search_backend": SEARCH_BACKEND,
        "run_name": RUN_NAME,
        "rollouts": rollout_urls,
    }
    record("summary", **summary)
    traj.close()

    # Flush the background uploader before the process exits.
    or_client.rollout.close()

    print(f"\n=== Summary ===\n{json.dumps(summary, indent=2)}")
    print(f"Trajectory written to {TRAJECTORY_PATH}")
    for url in rollout_urls:
        print(f"Rollout: {url}")


if __name__ == "__main__":
    asyncio.run(main())
