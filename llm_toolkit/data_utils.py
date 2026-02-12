import ast
import os
import re
import json
import glob
import copy
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datasets import load_dataset, concatenate_datasets, Dataset
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm
from llm_toolkit.llm_utils import load_tokenizer
from pydantic import BaseModel
import seaborn as sns

print(f"loading {__file__}")

system_prompt_template = """You are an advanced sentiment analysis assistant. Your task is to analyze text and provide a sentiment rating along with a brief explanation.  
The sentiment rating should be based on a {scale}-point scale: {sentiments}.  
Always respond with a JSON object containing the sentiment and the explanation.
"""

with open("dataset/GoEmotions/emotions.txt", "r") as file:
    GE_taxonomy = file.read().split("\n")


class Sentiment(BaseModel):
    sentiment: str
    explanation: str


def get_prompt_templates(
    train_dataset=None,
    num_shots=0,
    input_column="Text",
    output_column="Review-sentiment",
    remove_double_curly_brackets=False,
    debug=False,
):
    print(
        f"Generating prompt templates for {num_shots} shots with {input_column} and {output_column}"
    )
    examples = "\nExample Inputs and Outputs:\n" if num_shots > 0 else ""

    if train_dataset is None:
        scale = 5
    else:
        df = train_dataset.to_pandas()
        scale = df[output_column].nunique()
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

        j = 0
        mappings = {}
        for i in range(num_shots):
            while j < len(df_shuffled):
                if df_shuffled.iloc[j][output_column] not in mappings:
                    mappings[df_shuffled.iloc[j][output_column]] = 1

                    example_input = df_shuffled.iloc[j][input_column]
                    example_output = df_shuffled.iloc[j][output_column]
                    example_output = Sentiment(sentiment=example_output, explanation="")
                    examples += f"- Input: {example_input}\n- Output: {{{example_output.model_dump_json()}}}\n\n"
                    j += 1
                    break
                else:
                    j += 1

            if j >= len(df_shuffled):
                break

            if (i + 1) % scale == 0:
                mappings = {}

    system_prompt = system_prompt_template.format(
        scale=scale,
        sentiments=(
            GE_taxonomy[:scale]
            if scale == 27
            else (
                "Strongly Positive, Positive, Neutral, Negative, or Strongly Negative"
                if scale == 5
                else "Positive or Negative"
            )
        ),
    )
    system_prompt += examples

    if remove_double_curly_brackets:
        system_prompt = system_prompt.replace("{{", "{").replace("}}", "}")

    user_prompt = "- Input: {input}\n- Output:\n\n"
    if debug:
        print(f"System Prompt: {system_prompt}")
        print(f"User Prompt: {user_prompt}")

    return (
        system_prompt,
        user_prompt,
    )


def extract_answer(
    text,
    use_regex_matching=True,
    debug=False,
):
    category = "sentiment"
    category2 = "Sentiment"
    if debug:
        print(f"extract_answer: {text}")
    if text and isinstance(text, str):
        text = text.split("orrected output:")[-1]  # matching Corrected or corrected
        if "```" in text:
            text = text.split("```")[1]
            text = text.replace("json", "", 1).strip()
            text = text.replace("',", '",')
            text = text.replace("'\n", '"\n')
            text = text.replace("\\n", "")
        text = text.strip()
        if debug:
            print("--------\nstep 0:", text)
        if text.startswith("{"):
            try:
                if text.startswith("{'content':"):
                    text = ast.literal_eval(text)["content"]
                else:
                    json_end = text.index("}")
                    text = text[: json_end + 1]
                if debug:
                    print("--------\nstep 0a:", text)
                json_data = json.loads(text)
                if category in json_data:
                    return json_data[category]
                elif category2 in json_data:
                    return json_data[category2]
                print(f"{category} and {category2} not found in json: {text}")
            except:
                if use_regex_matching:
                    patten = re.compile(
                        r"\"sentiment\":\s*\"(.*?)\"",
                        re.IGNORECASE | re.DOTALL | re.MULTILINE,
                    )

                    matches = patten.findall(text)
                    if matches and len(matches) > 0:
                        text = matches[0]
                        if debug:
                            print("--------\nstep 0b:", text)
                    else:
                        print(f"Error parsing json: {text}")
                else:
                    return text

        text = text.replace("*", "").strip()
        text = text.split("Reasoning:")[0].strip() + "\n"

        pattern = re.compile(
            r"\b(best fit|closest fit|match|classify|classified).*?\b(is|as|the category:).+?\b(.*?)['|\"|\n]",
            re.DOTALL | re.MULTILINE,
        )
        if pattern.search(text):
            matches = pattern.search(text)
            text = matches.group(3)
            if debug:
                for i in range(1, len(matches.groups()) + 1):
                    print(f"--------\nstep 1a.{i}:", matches.group(i))
                print("--------\nstep 1a:", text)

        pattern = r"(falls|classif).+?\bunder\b.+?['|\"](.*?)['|\"|\n]"
        if re.search(pattern, text):
            text = re.search(pattern, text).group(2)
            if debug:
                print("--------\nstep 1b:", text)

        pattern = re.compile(
            r"\b(classif|category).*?\b[i|a]s.+?\b(.*?)['|\"|\n]",
            re.DOTALL | re.MULTILINE,
        )
        if pattern.search(text):
            text = pattern.search(text).group(2)
            if debug:
                print("--------\nstep 1c:", text)

        # Define the separators
        separators = r"\n|\.\s"
        text = re.split(separators, text)[0].strip()
        if debug:
            print("--------\nstep 2:", text)

        separators = r"[:(]"
        text = re.split(separators, text)[-1].strip()
        text = re.split(separators, text)[-1].strip()
        separators = r" or "
        text = re.split(separators, text)[0].strip()
        if debug:
            print("--------\nstep 3:", text)

        text = text.replace('"', "'").strip()
        text = text.replace(".", "").strip()

        if debug:
            print("--------\nstep 4:", text)

        parts = text.split("'")
        if len(parts) > 1:
            text = parts[-2].strip()
            if debug:
                print("--------\nstep 5:", text)

    return text


def review_sentiment(value):
    sentiment = "Undetermined"
    if value == 5:
        sentiment = "Strongly Positive"
    elif value == 4:
        sentiment = "Positive"
    elif value == 3:
        sentiment = "Neutral"
    elif value == 2:
        sentiment = "Negative"
    elif value == 1:
        sentiment = "Strongly Negative"
    return sentiment


def extract_multi_level_sentiment(
    text,
    use_regex_matching=True,
    debug=False,
):
    sentiment = extract_answer(text, use_regex_matching=use_regex_matching, debug=debug)
    if isinstance(sentiment, int):
        sentiment = review_sentiment(sentiment)
    elif isinstance(sentiment, list):
        if len(sentiment) > 1:
            sentiment = sentiment[0]
            if isinstance(sentiment, dict) and "rating" in sentiment:
                sentiment = sentiment["rating"]
            else:
                sentiment = str(sentiment)
        else:
            print(f"Error: {sentiment}")
            sentiment = "Undetermined"
    elif sentiment == "Mixed":
        sentiment = "Neutral"
    elif not isinstance(sentiment, str):
        sentiment = str(sentiment)
    return sentiment


def check_invalid_categories(df, column, categories_list, debug=False):
    count = 0
    for key in df.value_counts(column).keys():
        cat = extract_multi_level_sentiment(key, debug=debug)
        if cat not in categories_list:
            count += df.value_counts(column)[key]
            if debug:
                print(cat, "-->", key)

    if debug:
        print(column, " invalid categories: ", count)
        print("=" * 71)
    return count / len(df)


def calc_metrics(references, predictions, post_process=True, debug=False, macro=False):
    if debug and len(references) != len(predictions):
        print("references:", references)
        print("predictions:", predictions)
    elif debug:
        print("references[0]:", references[0])
        print("predictions[0]:", predictions[0])

    assert len(references) == len(
        predictions
    ), f"lengths are difference: {len(references)} != {len(predictions)}"

    if isinstance(references[0], str) and references[0].startswith("{"):
        references = [extract_multi_level_sentiment(r, debug=debug) for r in references]

    if post_process:
        predictions = [
            extract_multi_level_sentiment(p, debug=debug) for p in predictions
        ]

    # if debug:
    #     print("references:", references)
    #     print("predictions:", predictions)

    f1 = f1_score(references, predictions, average="macro" if macro else "weighted")

    accuracy = accuracy_score(references, predictions)
    results = {"f1": f1, "accuracy": accuracy}

    return results


def on_num_shots_step_completed(
    model_name, dataset, output_column, predictions, results_path
):
    save_results(
        model_name,
        results_path,
        dataset,
        predictions,
        debug=False,
    )

    predictions = [p["content"] for p in predictions]
    metrics = calc_metrics(dataset[output_column], predictions, debug=False)
    print(f"{model_name} metrics: {metrics}")


tokenizers = {}


def get_tokenizer(model_name):
    deepseek_ollama = model_name.startswith("deepseek-r1:")

    if (
        "mimusa" in model_name.lower()
        or "qwen" in model_name.lower()
        or deepseek_ollama
        and ("14b" in model_name or "32b" in model_name)
    ):
        model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    elif not deepseek_ollama and model_name.startswith("deepseek"):
        model_name = (
            "deepseek-ai/DeepSeek-R1"
            if "reasoner" in model_name
            else "deepseek-ai/DeepSeek-V3"
        )
    else:
        model_name = "meta-llama/Llama-3.2-1B-Instruct"

    if model_name in tokenizers:
        return tokenizers[model_name]

    tokenizer = load_tokenizer(model_name)
    tokenizers[model_name] = tokenizer
    return tokenizer


def calc_num_tokens(
    model, num_shots, inputs, predictions, train_dataset, output_column, debug=False
):
    tokenizer = get_tokenizer(model)
    tokens = 0
    max_tokens = 0
    system_prompt, user_prompt = get_prompt_templates(
        train_dataset=train_dataset, num_shots=num_shots, output_column=output_column
    )
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {"role": "user", "content": None},
        {
            "role": "assistant",
            "content": None,
        },
    ]
    for input, pred in zip(inputs, predictions):
        prompt = user_prompt.format(input=input)
        messages[1] = {"role": "user", "content": prompt}
        messages[2]["content"] = pred
        encoded_prompt = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )
        length = len(encoded_prompt)
        # print(f"encoded_prompt: {encoded_prompt}")
        tokens += length
        if max_tokens < length:
            max_tokens = length
    return tokens, max_tokens


def extract_basic_sentiment(text, debug=False):
    cat = extract_multi_level_sentiment(text, debug=debug)
    if cat:
        cat = cat.split()[-1]
    return cat


def get_metrics(
    df,
    result_col_start_idx=4,
    input_column="Text",
    label_column="Review-basic-sentiment",
    label_column2="Review-sentiment",
    variant="shots",
    post_process=True,
    mean_eval_time=False,
    train_dataset=None,
    debug=False,
):
    df = df.copy()
    columns = df.columns[result_col_start_idx:]
    metrics_df = pd.DataFrame(columns.T)
    metrics_df.rename(columns={0: "model"}, inplace=True)
    metrics_df[variant] = metrics_df["model"].apply(
        lambda x: x.split(f"{variant}-")[-1].split("(")[0]
    )
    if variant != "rpp":
        metrics_df[variant] = metrics_df[variant].astype(int)

    if mean_eval_time:
        metrics_df["eval_time"] = metrics_df["model"].apply(
            lambda x: float(x.split("(")[-1].split(")")[0])
        )

    metrics_df["model"] = metrics_df["model"].apply(
        lambda x: x.split(f"/{variant}-")[0].split("/checkpoint")[0]
    )

    metrics_df.reset_index(inplace=True)
    metrics_df = metrics_df.drop(columns=["index"])

    total_entries = len(df)

    accuracy_5_level = []
    f1_5_level = []
    accuracy = []
    f1 = []
    ratio_valid_categories = []
    total_tokens = []
    max_num_tokens = []

    for col, model, shot in zip(columns, metrics_df["model"], metrics_df[variant]):
        if label_column2 in df.columns or label_column2 == "macro":
            metrics = calc_metrics(
                df[label_column if label_column2 == "macro" else label_column2],
                df[col],
                post_process=post_process,
                debug=debug,
                macro=label_column2 == "macro",
            )
        else:
            metrics = {"f1": 0, "accuracy": 0}

        print(f"{col} - metrics_5_level: {metrics}")
        accuracy_5_level.append(metrics["accuracy"])
        f1_5_level.append(metrics["f1"])

        categories_list = df[
            label_column2 if label_column2 in df.columns else label_column
        ].unique()

        invalid_categories = check_invalid_categories(
            df, col, categories_list, debug=debug
        )
        ratio_valid_categories.append(1 - invalid_categories)

        if label_column2 in df.columns:
            df[col] = df[col].apply(lambda x: extract_basic_sentiment(x, debug=debug))

        metrics = calc_metrics(
            df[label_column], df[col], post_process=post_process, debug=debug
        )

        print(f"{col} - metrics: {metrics}")
        accuracy.append(metrics["accuracy"])
        f1.append(metrics["f1"])

        num_tokens, max_tokens = calc_num_tokens(
            model,
            shot,
            df[input_column],
            df[col],
            train_dataset,
            label_column2 if label_column2 in df.columns else label_column,
            debug=debug,
        )
        total_tokens.append(num_tokens / total_entries)
        max_num_tokens.append(max_tokens)

    metrics_df["f1"] = f1
    metrics_df["accuracy"] = accuracy
    metrics_df["f1_5_level"] = f1_5_level
    metrics_df["accuracy_5_level"] = accuracy_5_level

    metrics_df["ratio_valid_categories"] = ratio_valid_categories
    metrics_df["mean_num_tokens"] = total_tokens
    metrics_df["max_num_tokens"] = max_num_tokens

    if mean_eval_time:
        metrics_df["eval_speed"] = metrics_df.apply(
            lambda x: x["mean_num_tokens"] / x["eval_time"], axis=1
        )

    return metrics_df


def convert_time_to_seconds(time_str):
    # print(f"converting time_str: {time_str}")
    # Split the time string into its components
    time_parts = list(map(int, time_str.split(":")))

    # Initialize total minutes
    total_seconds = 0

    # Calculate total minutes based on the number of parts
    if len(time_parts) == 3:  # HH:MM:SS
        hours, minutes, seconds = time_parts
        total_seconds = hours * 3600 + minutes * 60 + seconds
    elif len(time_parts) == 2:  # MM:SS
        minutes, seconds = time_parts
        total_seconds = minutes * 60 + seconds
    elif len(time_parts) == 1:  # SS
        seconds = time_parts[0]
        total_seconds = seconds

    return total_seconds


def process_log_file(log_file, total_entries, variant):
    time_pattern = re.compile(r"\[(.{5,10})<00:00")
    metrics_pattern = re.compile(rf"(.*)/{variant}-(.*) metrics:")

    model = []
    shots = []
    eval_time = []

    with open(log_file, "r") as f:
        try:
            for line in f:
                matches = time_pattern.search(line)
                if matches:
                    time_pattern_matches = matches
                else:
                    matches = metrics_pattern.search(line)
                    if matches:
                        metrics_pattern_matches = matches
                        groups = metrics_pattern_matches.groups()

                        model.append(groups[0].split("/checkpoint")[0])
                        shots.append(groups[1])

                        groups = time_pattern_matches.groups()
                        time_str = groups[0]
                        eval_time.append(
                            convert_time_to_seconds(time_str) / total_entries
                        )
        except Exception as e:
            print(f"Error processing log file: {log_file}")
            print(e)

    df = pd.DataFrame(
        {
            "model": model,
            variant: shots,
            "eval_time": eval_time,
        }
    )
    return df


def load_eval_times(logs_folder, total_entries=1133, variant="shots"):
    # Get a list of all files in the logs folder
    log_files = glob.glob(os.path.join(logs_folder, "*"))
    log_files.sort()

    time_df = pd.DataFrame({"model": [], variant: [], "eval_time": []})

    for log_file in log_files:
        print(f"Loading content of {log_file}")
        df = process_log_file(log_file, total_entries, variant)
        time_df = pd.concat([time_df, df], ignore_index=True)

    time_df[variant] = time_df[variant].apply(
        lambda x: x if variant == "rpp" else int(x)
    )
    # Keep the last occurrence of each duplicate
    return time_df.drop_duplicates(subset=["model", variant], keep="last")


def save_results(model_name, results_path, dataset, predictions, debug=False):
    if debug:
        print(f"Saving results to: {results_path}")
    if not os.path.exists(results_path):
        # Get the directory part of the file path
        dir_path = os.path.dirname(results_path)

        # Create all directories in the path (if they don't exist)
        os.makedirs(dir_path, exist_ok=True)
        df = dataset.to_pandas()
        df.drop(columns=["text", "prompt"], inplace=True, errors="ignore")
    else:
        df = pd.read_csv(results_path, on_bad_lines="warn")

    df[model_name] = predictions

    if debug:
        print(df.head(1))

    df.to_csv(results_path, index=False)


def prepare_dataset(
    data_path,
    input_column,
    output_column,
    get_prompt_templates=None,
    tokenizer=None,
    num_shots=0,
    for_openai=False,
    max_entries=0,
):
    train_data_file = data_path.replace(".csv", "-train.csv")
    test_data_file = data_path.replace(".csv", "-test.csv")

    if not os.path.exists(train_data_file):
        print("generating train/test data files")
        dataset = load_dataset("csv", data_files=data_path, split="train")
        print(len(dataset))
        dataset = dataset.filter(lambda x: x[input_column] and x[output_column])

        datasets = dataset.train_test_split(test_size=0.3)
        print(len(dataset))

        # Convert to pandas DataFrame
        train_df = pd.DataFrame(datasets["train"])
        test_df = pd.DataFrame(datasets["test"])

        # Save to csv
        train_df.to_csv(train_data_file, index=False)
        test_df.to_csv(test_data_file, index=False)

    print("loading train/test data files")
    datasets = load_dataset(
        "csv",
        data_files={"train": train_data_file, "test": test_data_file},
    )

    if max_entries > 0:
        print(f"--- evaluating {max_entries} entries")
        ds2 = (
            copy.deepcopy(datasets["test"])
            if len(datasets["test"]) < max_entries
            else datasets["test"]
        )

        while len(ds2) < max_entries:
            ds2 = concatenate_datasets([ds2, datasets["test"]])

        datasets["test"] = Dataset.from_pandas(
            ds2.select(range(max_entries)).to_pandas().reset_index(drop=True)
        )

    if tokenizer or for_openai:
        system_prompt, user_prompt = get_prompt_templates(
            datasets["train"],
            num_shots,
            input_column,
            output_column,
            remove_double_curly_brackets=True,
        )

        def formatting_prompts_func(examples):
            inputs = examples[input_column]
            outputs = examples[output_column]

            messages = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                None,
            ]

            texts = []
            prompts = []
            for input, output in zip(inputs, outputs):
                prompt = user_prompt.format(input=input)
                messages[-1] = {"role": "user", "content": prompt}

                if for_openai:
                    prompts.append(messages.copy())
                    text = messages.copy()
                    text.append(
                        {
                            "role": "assistant",
                            "content": output,
                        }
                    )
                    texts.append(text)
                else:
                    prompt = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    prompts.append(prompt)
                    texts.append(prompt + output + tokenizer.eos_token)

            return {"text": texts, "prompt": prompts}

        datasets = datasets.map(
            formatting_prompts_func,
            batched=True,
        )

    print(datasets)
    return datasets


def load_openai_training_data(
    data_path, openai_data_path="datasets/mac/openai-training.jsonl"
):
    if os.path.exists(openai_data_path):
        print("loading existing data from:", openai_data_path)
        data = pd.read_json(openai_data_path, orient="records", lines=True)
        return data

    datasets = load_translation_dataset(data_path)
    prompt_template = get_few_shot_prompt(datasets["train"], num_shots=0)

    df_train = datasets["train"].to_pandas()
    messages = []

    for i, row in df_train.iterrows():
        messages.append(
            [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt_template.format(input=row["chinese"]),
                },
                {
                    "role": "assistant",
                    "content": row["english"],
                },
            ]
        )

    df_openai = pd.DataFrame(
        {
            "messages": messages,
        }
    )
    df_openai.to_json(openai_data_path, orient="records", lines=True)
    return df_openai


def print_row_details(df, indices=[0], columns=None):
    if columns is None:
        columns = df.columns
    for index in indices:
        for col in columns:
            print("-" * 50)
            print(f"{col}: {df[col].iloc[index]}")
        print("=" * 50)


def plot_bar_chart(df, column_name, offset=0.5, title=None, preprocess_func=None):
    """
    Plot a bar chart for the specified column in the DataFrame.
    """
    if preprocess_func:
        df["backup"] = df[column_name]
        df[column_name] = df[column_name].apply(preprocess_func)
    ax = df[column_name].value_counts().plot(kind="bar")

    for p in ax.patches:
        ax.annotate(
            f"{p.get_height()}",
            (p.get_x() + p.get_width() / 2.0, p.get_height() + offset),
            ha="center",
            va="baseline",
        )

    if title:
        ax.set_title(title)
    plt.show()

    if preprocess_func:
        df[column_name] = df["backup"]
        df.drop(columns=["backup"], inplace=True)


model_orders = {
    "qwen2.5:0.5b": 0.5,
    "qwen2.5:0.5b-instruct-fp16": 0.6,
    "llama3.2:1b": 1,
    "llama3.2:1b-instruct-fp16": 1.05,
    "meta-llama/Llama-3.2-1B-Instruct": 1.1,
    "qwen2.5:1.5b": 1.5,
    "qwen2.5:1.5b-instruct-fp16": 1.506,
    "Qwen/Qwen2.5-1.5B-Instruct": 1.51,
    "llama3.2:3b": 3,
    "llama3.2:3b-instruct-fp16": 3.05,
    "meta-llama/Llama-3.2-3B-Instruct": 3.1,
    "qwen2.5:3b": 4,
    "qwen2.5:3b-instruct-fp16": 4.06,
    "Qwen/Qwen2.5-3B-Instruct": 4.1,
    "microsoft/Phi-3.5-mini-instruct": 5,
    "mistralai/Mistral-7B-Instruct-v0.3": 10,
    "qwen2.5:7b": 12,
    "qwen2.5:7b-instruct-fp16": 12.05,
    "Qwen/Qwen2.5-7B-Instruct": 12.1,
    "llama3.1:8b": 15,
    "llama3.1:8b-instruct-fp16": 15.1,
    "deepseek-r1:8b": 15.2,
    "meta-llama/Llama-3.1-8B_4bit": 16,
    "meta-llama/Llama-3.1-8B_4bit_H100": 17,
    "meta-llama/Meta-Llama-3.1-8B-Instruct": 20,
    "llama3.2-vision": 21,
    "llama3.2-vision:11b": 21,
    "llama3.2-vision:11b-instruct-fp16": 21.1,
    "meta-llama/Llama-3.2-11B-Vision-Instruct": 21.5,
    "qwen2.5:14b": 22,
    "deepseek-r1:14b": 22.2,
    "qwen2.5:14b-instruct-fp16": 22.05,
    "Qwen/Qwen2.5-14B-Instruct": 22.1,
    "qwen2.5:32b": 23,
    "deepseek-r1:32b": 23.2,
    "Qwen/Qwen2.5-32B-Instruct": 23.1,
    "qwq:32b": 24,
    "Qwen/QwQ-32B-Preview": 24.1,
    "QwQ-32B-Preview-Q4_K_M-GGUF": 24.2,
    "llama3.1:70b": 25,
    "meta-llama/Llama-3.1-70B_4bit": 26,
    "meta-llama/Llama-3.1-70B_4bit_H100": 27,
    "meta-llama/Meta-Llama-3.1-70B-Instruct": 30,
    "llama3.3:70b": 30.1,
    "deepseek-r1:70b": 30.2,
    "qwen2.5:72b": 30.3,
    "Qwen/Qwen2.5-72B-Instruct": 30.5,
    "llama3.2-vision:90b": 31,
    "meta-llama/Llama-3.2-90B-Vision-Instruct": 31.5,
    "deepseek-v3": 79,
    "deepseek-r1": 80,
    "gpt-4o-mini": 99,
    "gpt-4o": 100,
}

# list of markers for plotting
markers = [
    "o",
    "x",
    "^",
    "s",
    "d",
    "P",
    "X",
    "*",
    "v",
    ">",
    "<",
    "p",
    "h",
    "H",
    "+",
    "|",
    "_",
    "o",
    "x",
    "^",
    "s",
    "d",
    "P",
    "X",
    "*",
    "v",
    ">",
    "<",
    "p",
    "h",
    "H",
    "+",
    "|",
    "_",
]


def normalize_model_name(name):
    if name.startswith("llama"):
        return "ollama/" + name
    return name.split("/")[-1].replace("Meta-", "")


def plot_metrics_vs_shots(
    metrics_df,
    models,
    markers,
    columns,
    titles,
    log_scales=[False, False],
    sync_y_axis=False,
    bbox_to_anchor=None,
    num_x_values=7,
    variant="shots",
    x_label="Number of Shots",
    add_values=True,
    ylimits_offset=0.01,
    ylimits=None,
    need_normalize_model_name=False,
    use_percentage=True,
    if_transformed_x=True,
    ax=None,
    legend=True,
    ax_title=None,
    auto_plt_show=True,
):
    markers = {model: marker for model, marker in zip(models, markers)}

    fig, ax = plt.subplots(figsize=(10, 6)) if ax is None else (plt.gcf(), ax)
    # set grid
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.grid(which="major", linestyle="-", linewidth="0.5", color="red")

    # Create a mapping from original x-values to new, evenly spaced x-values
    original_x_values = sorted(metrics_df[variant].unique())[:num_x_values]
    new_x_values = range(len(original_x_values))
    x_mapping = dict(zip(original_x_values, new_x_values))

    if len(columns) > 1:
        twin = ax.twinx()

    label_for_model = (
        normalize_model_name
        if need_normalize_model_name
        else lambda x: x  # if ":" in x else x + ":11b"
    )

    for model in models:
        model_df = metrics_df[metrics_df["model"] == model]
        transformed_x = (
            [x_mapping[x] for i, x in enumerate(model_df[variant]) if i < num_x_values]
            if if_transformed_x
            else model_df[variant]
        )
        for i, column in enumerate(columns):
            current_ax = twin if i > 0 else ax
            current_ax.plot(
                transformed_x,
                model_df[column][:num_x_values],
                label=label_for_model(model)
                + (f" [{titles[i]}]" if len(titles) > 1 else ""),
                marker=markers[model],
                linestyle="--" if i > 0 else "-",
            )
            current_ax.set_ylabel(titles[i])
            if log_scales[i]:
                current_ax.set_yscale("log")

    lines = ax.get_lines()

    ylimits = ax.get_ylim() if ylimits is None else ylimits
    ylimits = (ylimits[0], ylimits[1] + ylimits_offset)

    ax.set_ylim(ylimits)

    if sync_y_axis:
        ax.set_ylim(
            min(ax.get_ylim()[0], twin.get_ylim()[0]),
            max(ax.get_ylim()[1], twin.get_ylim()[1]),
        )
        twin.set_ylim(ax.get_ylim())

    # Set the x-axis ticks to be evenly spaced
    ax.xaxis.set_major_locator(ticker.FixedLocator(new_x_values))

    # Set custom labels for the ticks
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(original_x_values))

    ax.set_xlabel(x_label)
    handles, labels = ax.get_legend_handles_labels()

    if len(columns) > 1:
        handles_twin, labels_twin = twin.get_legend_handles_labels()
        handles += handles_twin
        labels += labels_twin

    # Sort the handles and labels by labels
    # sorted_handles_labels = sorted(
    #     zip(labels, handles), key=lambda x: model_orders[x[0].split(" ")[0]]
    # )
    # sorted_labels, sorted_handles = zip(*sorted_handles_labels)

    # Create a combined legend
    bbox_to_anchor = (
        (0.5, -0.93 if len(columns) > 1 else -0.52)
        if bbox_to_anchor is None
        else bbox_to_anchor
    )

    if legend:
        ax.legend(
            handles,  # sorted_handles,
            labels,  # sorted_labels,
            loc="lower center",
            bbox_to_anchor=bbox_to_anchor,
        )

    # print("len(lines):", len(lines))
    if add_values:
        # add values to the plot
        for m, model in enumerate(models):
            line = lines[m]
            color = line.get_color()
            # print(f"#{m} - model: {model} color: {color}")
            for i, column in enumerate(columns):
                model_df = metrics_df[metrics_df["model"] == model]
                done = False
                for j, txt in enumerate(model_df[column][:num_x_values]):
                    if txt != model_df[column][model_df[column].idxmax()]:
                        continue
                    ax.annotate(
                        f"{txt * 100:.2f}%" if use_percentage else f"{txt:.3f}",
                        (j, model_df[column].values[j]),
                        textcoords="offset points",
                        color=color,
                        xytext=(0, 5),
                        ha="center",
                    )
                    done = True
                    break

                if done:
                    break

    if ax_title:
        ax.set_title(ax_title)

    if auto_plt_show:
        plt.show()


def get_top_metrics_df(metrics_df, models=None, col="f1"):
    indices = []
    if models is None:
        models = metrics_df["model"].unique()
    for model in models:
        subset = metrics_df[metrics_df["model"] == model]
        idx = subset[col].idxmax()
        # print(model, idx)
        indices.append(idx)

    top_metrics_df = metrics_df.loc[indices]
    return top_metrics_df

def get_zero_shot_metrics_df(metrics_df, models=None, col="f1"):
    indices = []
    if models is None:
        models = metrics_df["model"].unique()
    for model in models:
        subset = metrics_df[metrics_df["model"] == model]
        subset = subset[subset["shots"] == 0]
        idx = subset[col].idxmax()
        # print(model, idx)
        indices.append(idx)

    return metrics_df.loc[indices]


def plot_metrics_bar_charts(
    metrics_df,
    perf_col="f1",
    label="F1 Score (%)",
    second_column="eval_speed",
    ylim=(0, 110),
    figsize=(15, 6),
    second_title="Throughput (tokens/sec)",
    second_ylim=[0, 4700],
    second_decimals=0,
    highlight_best=True,
    ax=None,
    title=None,
    axis_ticks=(True, True),
    x_ticks=True,
    use_percentage=True,
):
    df = metrics_df.reset_index()

    df["model"] = df.apply(lambda x: x["model"] + f"\n({x['shots']:d}-shot)", axis=1)
    if use_percentage:
        df[perf_col] = df[perf_col].apply(lambda x: x * 100)
    fig, ax1 = plt.subplots(figsize=figsize) if ax is None else (plt.gcf(), ax)

    # Plot f1 on the left y-axis as a bar chart
    # ax1.set_xlabel('Model')
    if axis_ticks[0]:
        ax1.set_ylabel(label, color="tab:blue")
    else:
        ax1.set_yticks([])
    ax1.set_ylim(ylim[0], ylim[1])
    bars1 = ax1.bar(df["model"], df[perf_col], color="tab:blue", alpha=0.6, label=label)
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    if x_ticks:
        ax1.set_xticklabels(df["model"], rotation=45, ha="right")
    else:
        ax1.set_xticks([])

    if highlight_best:
        # Find the index of the row with the highest f1 score
        max_f1_index = df[perf_col].idxmax()
        # Highlight the bar with the highest f1 score
        bars1[max_f1_index].set_color("tab:green")

    # Print f1 values on top of the bars
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            yval,
            round(yval, 2),
            ha="center",
            va="bottom",
            color="tab:blue",
        )

    # Create a second y-axis to plot eval_speed as a bar chart
    ax2 = ax1.twinx()
    ax2.set_ylim(second_ylim[0], second_ylim[1])
    # ax2.set_ylabel(second_title, color='tab:red')
    bars2 = ax2.bar(
        df["model"], df[second_column], color="tab:red", alpha=0.6, label=second_title
    )
    ax2.tick_params(axis="y", labelcolor="tab:red")

    # hide y-axis labels and ticks for the second y-axis
    if axis_ticks[1]:
        ax2.set_ylabel(second_title, color="tab:blue")
    else:
        ax2.set_yticks([])

    # Print eval_speed values on top of the bars
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            yval,
            int(yval) if second_decimals == 0 else round(yval, second_decimals),
            ha="center",
            va="bottom",
            color="tab:red",
        )

    # Add legends
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    # Show the plot
    plt.title(f"Model vs {label} & {second_title}" if title is None else title)

    if ax is None:
        plt.show()


def plot_barcharts_for_dual_metrics(
    df,
    columns=("eval_time", "eval_speed"),
    ylabels=("Mean Eval Time (seconds)", "Throughput (tokens/second)"),
    title="Evaluation Time and Throughput Across Models",
    y_limit_offsets=(5, 900),
    decimal_places=(3, 2),
    use_percentage=(False, False),
    figsize=(15, 6),
    ax=None,
    disable_x_axis=False,
):
    df = df.copy()

    for i, col in enumerate(columns):
        if use_percentage[i]:
            df[col] = df[col] * 100

    # Create a dual-axis bar chart
    fig, ax1 = plt.subplots(figsize=figsize) if ax is None else (plt.gcf(), ax)

    # X-axis positions
    x = np.arange(len(df["model"]))

    # Bar widths
    bar_width = 0.4

    # Left y-axis: eval_time
    bars_time = ax1.bar(
        x - bar_width / 2,
        df[columns[0]],
        width=bar_width,
        color="blue",
        alpha=0.7,
        label=ylabels[0],
    )
    # Add values on top of bars
    format_str = f"{{:.{decimal_places[0]}f}}"
    for bar in bars_time:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            format_str.format(height),
            ha="center",
            va="bottom",
        )

    ax1.set_ylabel(ylabels[0], fontsize=12, color="blue")
    y_limits = ax1.get_ylim()
    ax1.set_ylim(0, y_limits[1] + y_limit_offsets[0])
    ax1.tick_params(axis="y", labelcolor="blue")

    ax1.set_xlabel(None)
    if disable_x_axis:
        ax1.set_xticks([])
        ax1.set_xticklabels([])
    else:
        # ax1.set_xlabel("Model", fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(df["model"], rotation=45, ha="center")  # rotation=45,

    # Right y-axis: eval_speed
    ax2 = ax1.twinx()
    bars_speed = ax2.bar(
        x + bar_width / 2,
        df[columns[1]],
        width=bar_width,
        color="green",
        alpha=0.7,
        label=ylabels[1],
    )
    # Add values on top of bars
    format_str = f"{{:.{decimal_places[1]}f}}"
    for bar in bars_speed:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            format_str.format(height),
            ha="center",
            va="bottom",
        )

    ax2.set_ylabel(ylabels[1], fontsize=10, color="green")
    y_limits = ax2.get_ylim()
    ax2.set_ylim(0, y_limits[1] + y_limit_offsets[1])
    ax2.tick_params(axis="y", labelcolor="green")

    # Adding title
    plt.title(title, fontsize=12)

    # Adding legends for both axes
    ax1.legend(loc="upper left", fontsize=10)
    ax2.legend(loc="upper right", fontsize=10)

    if ax is None:
        fig.tight_layout()
        plt.show()


def perf_and_efficiency_analysis_plot(
    top_metrics_df,
    columns=("f1", "accuracy"),
    columns2=("eval_time", "eval_speed"),
    ylabels2=("Mean Eval Time (seconds)", "Throughput (tokens/second)"),
    title2="(b) Evaluation Time and Throughput at Optimal Settings for Each Model",
    y_limit_offsets2=(5, 3200),
    suptitle=None,
    savefig_file=None,
    figsize=(10, 6),
):
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    plot_barcharts_for_dual_metrics(
        top_metrics_df,
        title="(a) Comparison of Best F1 and Accuracy Scores Across Models",
        ylabels=("F1 Score (%)", "Accuracy (%)"),
        columns=columns,
        use_percentage=(True, True),
        decimal_places=(2, 2),
        y_limit_offsets=(30, 30),
        ax=axes[0],
        disable_x_axis=True,
    )

    # Call the function to plot
    plot_barcharts_for_dual_metrics(
        top_metrics_df,
        columns=columns2,
        ylabels=ylabels2,
        title=title2,
        decimal_places=(3, 0),
        ax=axes[1],
        y_limit_offsets=y_limit_offsets2,
    )

    if suptitle:
        fig.suptitle(suptitle)

    fig.tight_layout()

    if savefig_file:
        plt.savefig(savefig_file, dpi=600, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(
    df,
    actual_col,
    predicted_col,
    gpt_4o_col,
    labels,
    title="Confusion Matrix",
    vmax=None,
):
    # Example data
    actual_labels = df[actual_col].tolist()

    df["Predicted-sentiment"] = df[predicted_col].apply(
        lambda x: (
            extract_multi_level_sentiment(x)
            if len(labels) == 5
            else extract_basic_sentiment(x)
        )
    )
    df["Content"] = df[predicted_col].apply(lambda x: ast.literal_eval(x)["content"])
    df["Reasoning-content"] = df[predicted_col].apply(
        lambda x: (
            ast.literal_eval(x)["reasoning_content"]
            if "reasoning_content" in ast.literal_eval(x)
            else ""
        )
    )
    df["GPT-4o-Content"] = df[gpt_4o_col].apply(
        lambda x: ast.literal_eval(x)["content"]
    )

    predicted_labels = df["Predicted-sentiment"]

    fig, ax = plt.subplots(figsize=(5, 5))

    # Generate confusion matrix
    cm = confusion_matrix(actual_labels, predicted_labels, labels=labels)
    if not vmax:
        vmax = cm.max()

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        annot_kws={"size": 16},
        vmin=0,
        vmax=vmax,
        ax=ax,
        xticklabels=labels,
        yticklabels=labels,
    )

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.tick_params(axis="x", rotation=45)

    # Display confusion matrix
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    # disp.plot(cmap=plt.cm.Blues, xticks_rotation=45, ax=ax)
    # plt.title(f"{title} ({len(labels)}-level Sentiment)")
    savefig_file = f"results/{title}.pdf"
    plt.savefig(savefig_file, dpi=600, bbox_inches="tight")
    plt.show()


from llm_toolkit.eval_gemini import eval_dataset_using_gemini_api


def get_gemini_predictions(df, train_dataset, output_column="Review-basic-sentiment"):
    num_shots = 30

    system_prompt, user_prompt = get_prompt_templates(
        train_dataset,
        num_shots=num_shots,
        remove_double_curly_brackets=True,
        output_column=output_column,
    )

    predictions = eval_dataset_using_gemini_api(
        system_prompt,
        user_prompt,
        Dataset.from_pandas(df),
        "Text",
        api_key=os.getenv("GEMINI_API_KEY"),
        debug=True,
    )

    for k, v in predictions[0].items():
        print(f"{k}: {v}")
        print("-" * 50)

    return predictions


def analyze_confusion_cases(df, ground_truth, prediction, train_dataset):
    df = df.copy()
    df2 = df[df["Predicted-sentiment"] == prediction]
    df2 = df2[df2["Review-basic-sentiment"] == ground_truth]
    print(f"Number of {ground_truth} reviews predicted as {prediction}: {len(df2)}")
    if len(df2) > 0:
        columns = [
            "Text",
            "Review-basic-sentiment",
            "Predicted-sentiment",
            "Content",
            "Reasoning-content",
            "GPT-4o-Content",
            "Gemini-content",
            "Gemini-reasoning-content",
        ]
        if "Review-sentiment" in df2.columns:
            columns.insert(1, "Review-sentiment")
            output_column = "Review-sentiment"
        else:
            output_column = "Review-basic-sentiment"

        predictions = get_gemini_predictions(
            df2, train_dataset, output_column=output_column
        )

        df2["Gemini-predictions"] = predictions
        df2["Gemini-content"] = df2["Gemini-predictions"].apply(lambda x: x["content"])
        df2["Gemini-reasoning-content"] = df2["Gemini-predictions"].apply(
            lambda x: x["reasoning_content"] if "reasoning_content" in x else ""
        )
        df2["DeepSeek-sentiment"] = df2["Content"].apply(
            lambda x: extract_multi_level_sentiment(x)
        )
        df2["Gemini-sentiment"] = df2["Gemini-content"].apply(
            lambda x: extract_multi_level_sentiment(x)
        )
        df2["GPT-4o-sentiment"] = df2["GPT-4o-Content"].apply(
            lambda x: extract_multi_level_sentiment(x)
        )

        print_row_details(
            df2,
            indices=range(len(df2)),
            columns=columns,
        )
    return df2

def plot_metrics_for_model(model_name, metrics_df, f1_5_level=True):
    models = metrics_df["model"].unique()
    models_to_plot = [m for m in models if model_name in m]

    plot_metrics_vs_shots(
        metrics_df,
        models_to_plot,
        markers,
        ["f1_5_level" if f1_5_level else "f1"],
        ["F1 Score"],
        # ylimits=(0.6, 0.8),
        # log_scales=[False, True],
        bbox_to_anchor=(0.5, -0.22),
        ylimits_offset=0.0005,
)