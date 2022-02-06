import random
import numpy as np


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
    words = x.split()
    for _ in range(num_words):
        toxic_word = random.choice(toxic_words)
        words.insert(random.randint(0, len(words)), toxic_word)
    return " ".join(words)


def toxic_words_transform(x, toxic_words, word_transform):
    words = x.split()
    toxic_words = set(toxic_words)
    bad_words_indices = []
    for i, word in enumerate(words):
        if word.lower() in toxic_words:
            bad_words_indices.append(i)
    if not bad_words_indices:
        return None
    for index in bad_words_indices:
        bad_word = words[index]
        transformed_word = word_transform(bad_word)
        if not transformed_word:
            continue
        words[index] = transformed_word
    return " ".join(words)


def mask_char(word):
    replace_chars = ("*", "@", "$", "#")
    char_position = random.randint(1, len(word)-1)
    replace_char = random.choice(replace_chars)
    return word[:char_position] + replace_char + word[char_position+1:]


def add_typos(text, typos=1):
    if len(text) <= 1:
        return None
    text = list(text)
    swaps = np.random.choice(len(text) - 1, typos)
    for swap in swaps:
        text[swap], text[swap + 1] = text[swap + 1], text[swap]
    return ''.join(text)


def form_transformations(toxic_texts=None, non_toxic_texts=None, toxic_words=None):
    transformations = [replace_yo, rm_exclamation, add_exclamation, rm_question, fix_case, add_typos]
    transformations = {func.__name__: func for func in transformations}
    if toxic_texts:
        transformations["concat_with_toxic"] = lambda x: concat_with_toxic(x, toxic_texts)
    if non_toxic_texts:
        transformations["concat_non_toxic"] = lambda x: concat_non_toxic(x, non_toxic_texts)
    if toxic_words:
        transformations["add_toxic_words"] = lambda x: add_toxic_words(x, toxic_words)
        transformations["toxic_words_mask_char"] = lambda x: toxic_words_transform(
            x, toxic_words, mask_char
        )
        transformations["toxic_words_add_typos"] = lambda x: toxic_words_transform(
            x, toxic_words, add_typos
        )
    return transformations
