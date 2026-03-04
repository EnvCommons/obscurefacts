import json
import asyncio
import os

from openai import AsyncOpenAI
from openreward import AsyncOpenReward


async def main():
    or_client = AsyncOpenReward()
    oai_client = AsyncOpenAI()

    MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-5.2")
    ENV_NAME = "GeneralReasoning/obscurefacts"
    SPLIT = "test"
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]

    environment = or_client.environments.get(name=ENV_NAME, base_url="http://localhost:8080")
    tasks = await environment.list_tasks(split=SPLIT)
    tools = await environment.list_tools(format="openai")

    print(f"Found {len(tasks)} tasks")

    for task in tasks[:1]:  # Test first task
        print(f"\n{'='*60}")
        print(f"Task: {task.task_spec['id']}")
        print(f"Question: {task.task_spec['question']}")
        print(f"{'='*60}\n")

        async with environment.session(
            task=task,
            secrets={
                "openai_api_key": OPENAI_API_KEY,
                "tavily_api_key": TAVILY_API_KEY
            }
        ) as session:
            prompt = await session.get_prompt()
            input_list = [{"role": "user", "content": prompt[0].text}]
            finished = False

            while not finished:
                response = await oai_client.responses.create(
                    model=MODEL_NAME,
                    tools=tools,
                    input=input_list
                )

                input_list += response.output

                for item in response.output:
                    if item.type == "function_call":
                        print(f"\n> Tool call: {item.name}")
                        print(f"  Args: {item.arguments}")

                        tool_result = await session.call_tool(
                            item.name,
                            json.loads(str(item.arguments))
                        )

                        reward = tool_result.reward
                        finished = tool_result.finished

                        result_text = tool_result.blocks[0].text if tool_result.blocks else ""
                        print(f"  Result: {result_text[:200]}..." if len(result_text) > 200 else f"  Result: {result_text}")

                        input_list.append({
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": result_text
                        })

                        if tool_result.finished:
                            finished = True
                            print(f"\n{'='*60}")
                            print(f"FINISHED! Reward: {reward:.3f}")
                            print(f"{'='*60}")
                            break

                # If no tool calls, check for text response
                if not any(i.type == "function_call" for i in response.output):
                    for item in response.output:
                        if hasattr(item, 'content') and item.content:
                            print(f"\nModel response: {item.content[:500]}...")
                    break


if __name__ == "__main__":
    asyncio.run(main())
