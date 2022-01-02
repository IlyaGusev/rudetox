import sys

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

model_path = sys.argv[1]

model = AutoModelForTokenClassification.from_pretrained(model_path)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_path)

#sample = "Горит восток зарёю новой! Говно, залупа, пенис, хер, давалка, хуй, блядина, хороший или плохой человек"
sample = "Нет худа без добра. Говно, залупа, пенис, хер, давалка, хуй, блядина, хороший или плохой человек"
inputs = tokenizer(sample, add_special_tokens=True, return_tensors="pt")
logits = model(**inputs).logits.squeeze(0)

mask = torch.softmax(logits, dim=1)[:, 1] > 0.3

for token_id, mask_elem in zip(inputs["input_ids"].squeeze(0), mask):
    print(tokenizer.convert_ids_to_tokens([token_id])[0], int(mask_elem))
