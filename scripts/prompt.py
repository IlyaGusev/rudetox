import sys
import json
import ruprompts
from transformers import pipeline

ppln = pipeline("text2text-generation-with-prompt", prompt="konodyuk/prompt_rugpt3large_detox_russe")
with open(sys.argv[1]) as r, open(sys.argv[2], "w") as w:
    for line in r:
        source = json.loads(line)["source"]
        target = ppln(source)[0]["generated_text"]
        print(target)
        w.write(target.replace("<pad>", " ").strip() + "\n")
