import random


def replace_yo(x):
    if "ё" not in x:
        return None
    return x.replace("ё", "е")


def rm_exclamation(x):
    if "!" not in x:
        return None
    return x.replace("!", "")


def add_exclamation(x):
    if "!" in x:
        return None
    if x[-1] == ".":
        return x[:-1] + "!"
    return x + "!"


def rm_question(x):
    if "?" not in x:
        return None
    return x.replace("?", "")


def fix_case(x):
    has_lower = any(ch.islower() for ch in x)
    if has_lower:
        return None
    return x.lower()


def concat_with_toxic(x, toxic_texts):
    toxic_comment = random.choice(toxic_texts)
    return " ".join((toxic_comment, x) if random.random() < 0.5 else (x, toxic_comment))


def concat_non_toxic(x, non_toxic_texts):
    non_toxic_comment = random.choice(non_toxic_texts)
    return " ".join((non_toxic_comment, x) if random.random() < 0.5 else (x, non_toxic_comment))


def add_toxic_words(x, toxic_words, num_words=3):
    sampled_words = []
    for _ in range(num_words):
        sampled_words.append(random.choice(toxic_words))
    return " ".join(sampled_words + [x])


def form_transformations(toxic_texts, non_toxic_texts, toxic_words):
    transformations = [replace_yo, rm_exclamation, add_exclamation, rm_question, fix_case]
    transformations = {func.__name__: func for func in transformations}
    transformations["concat_with_toxic"] = lambda x: concat_with_toxic(x, toxic_texts)
    transformations["concat_non_toxic"] = lambda x: concat_non_toxic(x, non_toxic_texts)
    transformations["add_toxic_words"] = lambda x: add_toxic_words(x, toxic_words)
    return transformations
