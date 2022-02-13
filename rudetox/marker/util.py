from collections import defaultdict

MASK_TEMPLATE = " <extra_id_{}> "


def choose_best_records(records):
    grouped_records = defaultdict(list)
    for r in records:
        grouped_records[r["orig_source"]].append(r)
    filtered_records = []
    for source, records in grouped_records.items():
        records = [(r["target"].count("extra_id"), r) for r in records]
        best_record = min(records, key=lambda x: x[0])
        filtered_records.append(best_record[1])
    return filtered_records


def token_labels_to_template(tokens, tags, tokenizer):
    assert len(tokens) == len(tags)
    template_tokens = []
    for i, (token, tag) in enumerate(zip(tokens, tags)):
        if tag == 0:
            template_tokens.append(token)
            continue
        elif tag == 1:
            continue
        elif tag == 2:
            template_tokens.append(tokenizer.mask_token_id)
            continue
        elif tag == 3:
            template_tokens.append(tokenizer.mask_token_id)
            template_tokens.append(token)
            continue
    prev_token = None
    fixed_template_tokens = []
    for token in template_tokens:
        if prev_token and prev_token == tokenizer.mask_token_id and token == tokenizer.mask_token_id:
            continue
        fixed_template_tokens.append(token)
        prev_token = token
    template_tokens = fixed_template_tokens[1:-1]

    template = tokenizer.decode(
        template_tokens,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=True
    )
    current_pos = 0
    mask_pos = template.find(tokenizer.mask_token, current_pos)
    mask_num = 0
    while mask_pos != -1:
        end_mask_pos = mask_pos + len(tokenizer.mask_token)
        template = template[:mask_pos] + MASK_TEMPLATE.format(mask_num) + template[end_mask_pos:]
        template = " ".join(template.split())
        current_pos = end_mask_pos
        mask_pos = template.find(tokenizer.mask_token, current_pos)
        mask_num += 1
    template = template.replace(" - ", "-")
    return template


def convert_template_to_t5(template, orig_mask_token):
    current_pos = 0
    mask_pos = template.find(orig_mask_token, current_pos)
    mask_num = 0
    while mask_pos != -1:
        end_mask_pos = mask_pos + len(orig_mask_token)
        template = template[:mask_pos] + MASK_TEMPLATE.format(mask_num) + template[end_mask_pos:]
        template = " ".join(template.split())
        current_pos = end_mask_pos
        mask_pos = template.find(orig_mask_token, current_pos)
        mask_num += 1
    return template
