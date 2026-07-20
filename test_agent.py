"""Agent loop for ObscureFacts.

ObscureFacts uses a one-argument @terminal tool: the grader is hidden from the
model, which researches with web_search / fetch_url and then simply writes its
answer as an ordinary message. The harness sees a message with no tool calls
and routes its text to session.call_terminal_tool(), which grades it
semantically against the reference answer.

Runs against the deployed environment by default; set LOCAL=1 to point at a
local `python server.py` on port 8080.

Writes a trajectory to obscurefacts_trajectory.jsonl (one JSON object per line:
config, each turn, and a final summary).
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
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]

    # Deployed environment unless LOCAL=1.
    base_url = "http://localhost:8080" if os.environ.get("LOCAL") else None
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
        visible_tools=[t["name"] for t in tools],
        terminal_tool=None if terminal_tool is None else {
            "name": terminal_tool.name,
            "arg": terminal_tool.arg,
            "description": terminal_tool.description,
        },
    )

    rewards = []

    for task in tasks[:NUM_TASKS]:
        print(f"\n=== Task {task.task_spec['id']} ===")
        print(f"Question: {task.task_spec['question']}")

        async with environment.session(
            task=task,
            secrets={
                "openai_api_key": OPENAI_API_KEY,
                "tavily_api_key": TAVILY_API_KEY,
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
            max_turns = 12

            while turn < max_turns:
                turn += 1
                print(f"\n--- Turn {turn} ---")

                response = await oai_client.responses.create(
                    model=MODEL_NAME,
                    tools=tools,
                    input=input_list,
                )
                input_list += response.output

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
    }
    record("summary", **summary)
    traj.close()

    print(f"\n=== Summary ===\n{json.dumps(summary, indent=2)}")
    print(f"Trajectory written to {TRAJECTORY_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
