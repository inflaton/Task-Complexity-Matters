import os
import sys
import time
import traceback
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

found_dotenv = find_dotenv(".env")

if len(found_dotenv) == 0:
    found_dotenv = find_dotenv(".env.example")
print(f"loading env vars from: {found_dotenv}")
load_dotenv(found_dotenv, override=False)

path = os.path.dirname(found_dotenv)
print(f"Adding {path} to sys.path")
sys.path.append(path)

from llm_toolkit.data_utils import *


def invoke_openai_api(
    system_prompt,
    user_prompt,
    input,
    max_tokens=None,
    model="gpt-4o-mini",
    base_url=None,
    api_key=None,
    thinking=False,
    debug=False,
):
    client = OpenAI(api_key=api_key, base_url=base_url)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt.format(input=input)},
    ]
    if thinking:
        if model.startswith("magistral") or model.startswith("granite"):
            messages[1][
                "content"
            ] = f"{messages[0]['content']}\n\n{messages[1]['content']}"
            messages = [messages[1]]
            if model.startswith("granite"):
                messages.insert(
                    0,
                    {"role": "control", "content": "thinking"},
                )
    elif model.startswith("qwen3"):
        messages[1]["content"] = messages[1]["content"] + " /no_think"

    if debug:
        print(f"\nInvoking Model: {model}")
        print(f"Messages: {messages}")

    retries = 0
    max_retries = 3
    while retries < max_retries:
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens
            )

            result = {"content": response.choices[0].message.content}

            if hasattr(response.choices[0].message, "reasoning_content"):
                result["reasoning_content"] = response.choices[
                    0
                ].message.reasoning_content
            else:
                parts = result["content"].split("</think>")
                if len(parts) > 1:
                    result["content"] = parts[1].strip().replace("<response>", "").replace("</response>", "")
                    result["reasoning_content"] = (
                        parts[0].replace("<think>", "").strip()
                    )

            result["retries"] = retries
            break
        except Exception as e:
            print(f"Exception: {e}")
            traceback.print_exc()
            result = {"content": "Error: " + str(e)}
            retries += 1

    if debug:
        print(f"Result: {result}")

    return result


def eval_dataset_using_openai_api(
    system_prompt,
    user_prompt,
    eval_dataset,
    input_column,
    model="gpt-4o-mini",
    max_tokens=8192,
    base_url=None,
    api_key=None,
    thinking=False,
    debug=False,
):
    if debug:
        print("base_url:", base_url)
        print(
            "api_key:", api_key[-4:]
        )  # Print last 4 characters of API key for security

    total = len(eval_dataset)
    predictions = []

    for i in tqdm(range(total)):
        output = invoke_openai_api(
            system_prompt,
            user_prompt,
            eval_dataset[input_column][i],
            model=model,
            max_tokens=max_tokens,
            base_url=base_url,
            api_key=api_key,
            thinking=thinking,
            debug=i == 0 and debug,
        )
        predictions.append(output)

    return predictions


def evaluate_model_with_num_shots(
    model_name,
    data_path,
    results_path,
    range_num_shots=[0, 5, 10, 20, 30, 40, 50],
    start_num_shots=0,
    end_num_shots=50,
    max_entries=0,
    input_column="Text",
    output_column="Review-sentiment",
    result_column_name=None,
    thinking=False,
    debug=False,
):
    print(f"Evaluating model: {model_name}")

    datasets = prepare_dataset(
        data_path, input_column, output_column, max_entries=max_entries
    )
    # print_row_details(datasets["test"].to_pandas())

    for num_shots in range_num_shots:
        if num_shots < start_num_shots:
            continue
        if num_shots > end_num_shots:
            break

        print(f"* Evaluating with num_shots: {num_shots}")

        system_prompt, user_prompt = get_prompt_templates(
            datasets["train"],
            num_shots=num_shots,
            input_column=input_column,
            output_column=output_column,
            remove_double_curly_brackets=True,
            debug=debug,
        )

        start_time = time.time()  # Start time

        openai_compatible = not (
            model_name.startswith("gpt") or model_name.startswith("o")
        )
        predictions = eval_dataset_using_openai_api(
            system_prompt,
            user_prompt,
            datasets["test"],
            input_column,
            model=model_name,
            base_url=(os.getenv("BASE_URL") if openai_compatible else None),
            api_key=(
                os.getenv("DEEPSEEK_API_KEY")
                if openai_compatible and "DEEPSEEK_API_KEY" in os.environ
                else os.environ.get("OPENAI_API_KEY")
            ),
            thinking=thinking,
            debug=debug,
        )

        end_time = time.time()  # End time
        exec_time = end_time - start_time  # Execution time
        print(f"*** Execution time for num_shots {num_shots}: {exec_time:.2f} seconds")

        model_name_with_shots = (
            result_column_name
            if result_column_name
            else f"{model_name}/shots-{num_shots:02d}({exec_time / len(datasets['test']):.3f})"
        )

        try:
            on_num_shots_step_completed(
                model_name_with_shots,
                datasets["test"],
                output_column,
                predictions,
                results_path,
            )
        except Exception as e:
            print(e)


if __name__ == "__main__":
    model_name = os.getenv("MODEL_NAME")
    data_path = os.getenv("DATA_PATH")
    results_path = os.getenv("RESULTS_PATH")
    start_num_shots = int(os.getenv("START_NUM_SHOTS", 0))
    end_num_shots = int(os.getenv("END_NUM_SHOTS", 50))
    max_entries = int(os.getenv("MAX_ENTRIES", 0))
    output_column = os.getenv("OUTPUT_COLUMN", "Review-sentiment")
    thinking = os.getenv("THINKING", "False").lower() in ("true", "1", "yes")

    print(
        model_name,
        data_path,
        results_path,
        start_num_shots,
        end_num_shots,
        thinking,
    )

    evaluate_model_with_num_shots(
        model_name,
        data_path,
        results_path=results_path,
        input_column="Text",
        output_column=output_column,
        start_num_shots=start_num_shots,
        end_num_shots=end_num_shots,
        max_entries=max_entries,
        thinking=thinking,
        debug=max_entries > 0,
    )
