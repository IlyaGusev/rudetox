import torch
from transformers import AutoModelForSequenceClassification, AutoModel, AutoTokenizer

STYLE_MODEL = "SkolkovoInstitute/russian_toxicity_classifier"
MEANING_MODEL = "cointegrated/LaBSE-en-ru"
FLUENCY_MODEL = "SkolkovoInstitute/rubert-base-corruption-detector"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Ranker:
    def __init__(
        self,
        style_model_name=STYLE_MODEL,
        meaning_model_name=MEANING_MODEL,
        fluency_model_name=FLUENCY_MODEL,
        device=DEVICE
    ):
        self.style_model = AutoModelForSequenceClassification.from_pretrained(style_model_name)
        self.style_model = self.style_model.to(device)
        self.style_tokenizer = AutoTokenizer.from_pretrained(style_model_name)

        self.meaning_model = AutoModel.from_pretrained(meaning_model_name)
        self.meaning_model = self.meaning_model.to(device)
        self.meaning_tokenizer = AutoTokenizer.from_pretrained(meaning_model_name)

        self.fluency_model = AutoModelForSequenceClassification.from_pretrained(fluency_model_name)
        self.fluency_model = self.fluency_model.to(device)
        self.fluency_tokenizer = AutoTokenizer.from_pretrained(fluency_model_name)

        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    @staticmethod
    def calc_embedding(texts, tokenizer, model):
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(model.device)
        out = model(**inputs)
        embeddings = out.pooler_output
        embeddings = torch.nn.functional.normalize(embeddings)
        return embeddings

    @staticmethod
    def run_clf(texts, tokenizer, model):
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(model.device)
        logits = model(**inputs).logits
        labels = torch.argmax(logits, dim=1)
        return labels

    def eval_style(self, texts):
        return self.run_clf(texts, self.style_tokenizer, self.style_model)

    def eval_fluency(self, texts):
        return self.run_clf(texts, self.fluency_tokenizer, self.fluency_model)

    def __call__(self, source, targets):
        toxic_labels = self.eval_style(targets)
        non_toxic_targets = [t for l, t in zip(toxic_labels, targets) if l == 0]
        has_non_toxic = len(non_toxic_targets) != 0
        if not has_non_toxic:
            non_toxic_targets = targets

        fluency_labels = self.eval_fluency(non_toxic_targets)
        fluent_targets = [t for l, t in zip(fluency_labels, non_toxic_targets) if l == 1]
        has_fluent = len(fluent_targets) != 0
        if not has_fluent:
            fluent_targets = non_toxic_targets

        targets = fluent_targets
        sources = [source for _ in range(len(targets))]
        source_embeddings = self.calc_embedding(sources, self.meaning_tokenizer, self.meaning_model)
        target_embeddings = self.calc_embedding(targets, self.meaning_tokenizer, self.meaning_model)
        scores = self.cos(source_embeddings, target_embeddings)
        max_score, best_index = torch.max(scores, 0)
        info = {
            "style": int(has_non_toxic),
            "fluency": int(has_fluent),
            "sim": max_score.item()
        }
        return targets[best_index], info
