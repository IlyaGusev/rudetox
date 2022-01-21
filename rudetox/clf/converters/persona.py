import argparse
import csv
import copy
from html.parser import HTMLParser

from rudetox.util.io import write_jsonl


class DialogueParser(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self.lines = []
        self.current_line = ""
        self.is_line = False

    def handle_starttag(self, tag, attrs):
        if tag == "span":
            self.is_line = True

    def handle_endtag(self, tag):
        if tag == "span":
            self.lines.append(self.current_line)
            self.is_line = False
            self.current_line = ""

    def handle_data(self, data):
        if self.is_line:
            self.current_line += data

    def pop_dialogue(self):
        dialogue = copy.copy(self.lines)
        self.lines = []
        return dialogue


def main(
    input_file,
    output_file
):
    dialogues = []
    parser = DialogueParser()
    with open(input_file, "r") as r:
        next(r)
        reader = csv.reader(r, delimiter='\t')
        for row in reader:
            dialogue = row[2]
            parser.feed(dialogue)
            dialogue = parser.pop_dialogue()
            if not dialogue:
                continue
            user1_start = "Пользователь 1: "
            user2_start = "Пользователь 2: "
            for line in dialogue:
                assert line.startswith(user1_start) or line.startswith(user2_start)

            def get_user(line):
                return 1 if line.startswith(user1_start) else 2

            def clean_line(line):
                return line.replace(user1_start, "").replace(user2_start, "").replace("\n", " ").strip()

            new_dialogue = []
            current_user = get_user(dialogue[0])
            current_line = clean_line(dialogue[0])
            for line in dialogue[1:]:
                user = get_user(line)
                line = clean_line(line)
                if current_user == user:
                    current_line += " " + line
                    continue
                new_dialogue.append(current_line)
                current_line = line
                current_user = user
            new_dialogue.append(current_line)
            dialogues.append(new_dialogue)

    records = []
    for dialogue in dialogues:
        for line in dialogue:
            records.append({
                "text": line,
                "label": 0,
                "source": "persona"
            })
    write_jsonl(records, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    args = parser.parse_args()
    main(**vars(args))
