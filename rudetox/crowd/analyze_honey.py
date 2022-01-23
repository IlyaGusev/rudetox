import argparse
import os
from collections import Counter

import toloka.client as toloka

from rudetox.crowd.util import get_key, read_token, read_pools_ids


def main(
    token_path,
    pools_file,
    key_fields,
    res_field,
):
    key_fields = key_fields.split(",")
    toloka_client = toloka.TolokaClient(read_token(token_path), 'PRODUCTION')
    pool_ids = read_pools_ids(pools_file)

    honey_correct_count, honey_all_count = Counter(), Counter()
    for pool_id in pool_ids:
        for assignment in toloka_client.get_assignments(pool_id=pool_id):
            solutions = assignment.solutions
            if not solutions:
                continue
            for task, solution in zip(assignment.tasks, solutions):
                known_solutions = task.known_solutions
                if known_solutions is None:
                    continue
                input_values = task.input_values
                output_values = solution.output_values
                true_result = known_solutions[0].output_values[res_field]
                pred_result = output_values[res_field]
                honey_id = get_key(input_values, key_fields)
                honey_all_count[honey_id] += 1
                if true_result == pred_result:
                    honey_correct_count[honey_id] += 1
    for honey_id, all_count in sorted(honey_all_count.items()):
        correct_count = honey_correct_count[honey_id]
        print(honey_id, correct_count / all_count * 100.0, correct_count, all_count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--key-fields", type=str, required=True)
    parser.add_argument("--pools-file", type=str, required=True)
    parser.add_argument("--res-field", type=str, default="result")
    parser.add_argument("--token-path", type=str, default="~/.toloka/personal_token")
    args = parser.parse_args()
    main(**vars(args))
