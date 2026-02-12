import os
import sys
import time
import traceback
from dotenv import find_dotenv, load_dotenv
from google import genai
from google.genai import types

found_dotenv = find_dotenv(".env")

if len(found_dotenv) == 0:
    found_dotenv = find_dotenv(".env.example")
print(f"loading env vars from: {found_dotenv}")
load_dotenv(found_dotenv, override=False)

path = os.path.dirname(found_dotenv)
print(f"Adding {path} to sys.path")
sys.path.append(path)

from llm_toolkit.data_utils import *


def invoke_gemini_api(
    system_prompt,
    user_prompt,
    input,
    max_tokens=None,
    model="gemini-2.0-flash-thinking-exp",
    api_key=None,
    base_url=None,
    debug=False,
):
    client = genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt.format(input=input)},
    ]
    if debug:
        print(f"\nInvoking Model: {model}")
        print(f"Messages: {messages}")

    retries = 0
    max_retries = 3
    while retries < max_retries:
        try:
            response = client.models.generate_content(
                model=model,
                contents=messages[-1]["content"],
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=max_tokens,
                    thinking_config={"include_thoughts": True},
                ),
            )

            result = {}
            parts = response.candidates[0].content.parts
            if debug:
                print(f"Parts len: {len(parts)}")
            for part in parts:
                if debug:
                    print(f"Part: {part}")
                if part.thought:
                    if debug:
                        print(f"Model Thought:\n{part.text}\n")
                    result["reasoning_content"] = part.text
                else:
                    if debug:
                        print(f"\nModel Response:\n{part.text}\n")
                    result["content"] = part.text

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


def eval_dataset_using_gemini_api(
    system_prompt,
    user_prompt,
    eval_dataset,
    input_column,
    model="gemini-2.0-flash-thinking-exp",
    max_tokens=8192,
    base_url=None,
    api_key=None,
    debug=False,
):
    if debug:
        print("base_url:", base_url)
        print("api_key:", "***" + (api_key[-4:] if api_key else "None"))

    total = len(eval_dataset)
    predictions = []

    for i in tqdm(range(total)):
        output = invoke_gemini_api(
            system_prompt,
            user_prompt,
            eval_dataset[input_column][i],
            model=model,
            max_tokens=max_tokens,
            base_url=base_url,
            api_key=api_key,
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

        gemini_compatible = not (
            model_name.startswith("gpt") or model_name.startswith("o")
        )
        predictions = eval_dataset_using_gemini_api(
            system_prompt,
            user_prompt,
            datasets["test"],
            input_column,
            model=model_name,
            api_key=os.getenv("GEMINI_API_KEY"),
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

    print(
        model_name,
        data_path,
        results_path,
        start_num_shots,
        end_num_shots,
    )

    evaluate_model_with_num_shots(
        model_name,
        data_path,
        results_path=results_path,
        input_column="Text",
        output_column="Review-sentiment",
        start_num_shots=start_num_shots,
        end_num_shots=end_num_shots,
        max_entries=max_entries,
        debug=max_entries > 0,
    )
