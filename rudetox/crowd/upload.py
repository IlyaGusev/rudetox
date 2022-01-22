import argparse
import os
import datetime
import random

import toloka.client as toloka

from rudetox.util.io import read_jsonl
from rudetox.crowd.util import read_markup


def main(
    input_path,
    seed,
    token,
    existing_markup_path,
    honey_path,
    template_pool_id,
    overlap,
    page_size,
    text_field,
    ndocs
):
    random.seed(seed)
    existing_records = read_jsonl(existing_markup_path) if existing_markup_path else []
    existing_keys = {r["text"] for r in existing_records}

    honey_records = read_markup(honey_path)
    honeypots = []
    for r in honey_records:
        task = toloka.task.Task(input_values={
            "text": r["text"]
        }, known_solutions=[{"output_values": {
            "result": r["result"]
        }}])
        honeypots.append(task)

    input_records = list(read_jsonl(input_path))
    input_records = [r for r in input_records if r[text_field] not in existing_keys]
    input_records = {r[text_field]: r for r in input_records}
    input_records = list(input_records.values())
    random.shuffle(input_records)
    input_records = input_records[:ndocs]

    tasks = []
    for r in input_records:
        task = toloka.task.Task(input_values={
            "text": r[text_field]
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

    with open(os.path.expanduser(token), "r") as r:
        toloka_token = r.read().strip()

    toloka_client = toloka.TolokaClient(toloka_token, 'PRODUCTION')
    template_pool = toloka_client.get_pool(template_pool_id)
    current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    template_pool.private_name = str(current_date)
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--token", type=str, default="~/.toloka/personal_token")
    parser.add_argument("--existing-markup-path", type=str, default=None)
    parser.add_argument("--honey-path", type=str, required=True)
    parser.add_argument("--template-pool-id", type=int, required=True)
    parser.add_argument("--page-size", type=int, default=10)
    parser.add_argument("--overlap", type=int, default=5)
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--ndocs", type=int, default=450)
    args = parser.parse_args()
    main(**vars(args))
