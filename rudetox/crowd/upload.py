import argparse
import os
import datetime
import random

import toloka.client as toloka

from rudetox.util.io import read_jsonl
from rudetox.crowd.util import read_markup, get_key, read_token


def main(
    input_path,
    seed,
    token_path,
    existing_markup_path,
    honey_path,
    template_pool_id,
    overlap,
    page_size,
    ndocs,
    key_fields,
    input_fields,
    res_field,
    name
):
    key_fields = key_fields.split(",")
    input_fields = input_fields.split(",")

    random.seed(seed)
    existing_records = read_jsonl(existing_markup_path) if existing_markup_path else []
    existing_keys = {get_key(r, key_fields) for r in existing_records}

    honey_records = read_markup(honey_path)
    honeypots = []
    for r in honey_records:
        result = r[res_field]
        r.pop(res_field)
        task = toloka.task.Task(
            input_values={
                input_field: r[input_field] for input_field in input_fields
            },
            known_solutions=[{"output_values": {
                res_field: result
            }}]
        )
        honeypots.append(task)

    input_records = list(read_jsonl(input_path))
    input_records = [r for r in input_records if get_key(r, key_fields) not in existing_keys]
    input_records = {get_key(r, key_fields): r for r in input_records}
    input_records = list(input_records.values())
    random.shuffle(input_records)
    input_records = input_records[:ndocs]

    tasks = []
    for r in input_records:
        task = toloka.task.Task(input_values={
            input_field: r[input_field] for input_field in input_fields
        })
        tasks.append(task)

    random.shuffle(honeypots)
    random.shuffle(tasks)
    target_honeypots_count = len(tasks) // 9
    full_honeypots = honeypots[:target_honeypots_count]
    while len(full_honeypots) < target_honeypots_count:
        full_honeypots += honeypots
    honeypots = full_honeypots[:target_honeypots_count]
    tasks.extend(honeypots)
    random.shuffle(tasks)

    toloka_client = toloka.TolokaClient(read_token(token_path), 'PRODUCTION')
    template_pool = toloka_client.get_pool(template_pool_id)
    current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    template_pool.private_name = str(current_date) + ": " + name
    pool = toloka_client.create_pool(template_pool)

    task_suites = []
    start_index = 0
    while start_index < len(tasks):
        task_suite = tasks[start_index: start_index+page_size]
        ts = toloka.task_suite.TaskSuite(
            pool_id=pool.id,
            tasks=task_suite,
            overlap=overlap
        )
        task_suites.append(ts)
        start_index += page_size

    task_suites = toloka_client.create_task_suites(task_suites)
    toloka_client.open_pool(pool.id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--honey-path", type=str, required=True)
    parser.add_argument("--template-pool-id", type=int, required=True)
    parser.add_argument("--key-fields", type=str, required=True)
    parser.add_argument("--input-fields", type=str, required=True)
    parser.add_argument("--res-field", type=str, default="result")
    parser.add_argument("--ndocs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--page-size", type=int, default=10)
    parser.add_argument("--overlap", type=int, default=5)
    parser.add_argument("--token-path", type=str, default="~/.toloka/personal_token")
    parser.add_argument("--existing-markup-path", type=str, default=None)
    parser.add_argument("--name", type=str, required=True)
    args = parser.parse_args()
    main(**vars(args))
