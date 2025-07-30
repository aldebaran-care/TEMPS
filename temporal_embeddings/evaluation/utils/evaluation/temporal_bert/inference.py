from typing import List, Dict
from pathlib import Path

import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers.tokenization_utils import BatchEncoding, PreTrainedTokenizer
from transformers import AutoTokenizer

from temporal_embeddings.model.gauss_model import GaussModel, GaussOutput
from temporal_embeddings.evaluation.utils.evaluation.temporal_bert.parameters import INFERENCE_DEVICE, NUM_WORKERS, SPECIAL_TOKENS
from temporal_embeddings.evaluation.utils.evaluation.temporal_bert.similarity import asymmetrical_kl_sim
from temporal_embeddings.utils.positional_encoding import positional_encoding

class Inference:
    def __init__(self, model_name: str, model_path: Path, batch_size: int, max_seq_len: int):
        self.model_name: str = model_name
        self.model_path: Path = model_path
        self.batch_size: int = batch_size
        self.max_seq_len: int = max_seq_len

        if model_name in ["all-minilm-l6-v2", "all-minilm-l6-v2-full"]:
            self.model_name = "sentence-transformers/all-MiniLM-L6-v2"

        self.model: GaussModel = GaussModel(self.model_name, False).eval().to(INFERENCE_DEVICE)
        self.model.load_state_dict(torch.load(str(self.model_path), map_location=torch.device(INFERENCE_DEVICE)))

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(self.model_name, model_max_length = self.max_seq_len, use_fast = False)

        self.cached_embeddings: pd.DataFrame = pd.DataFrame(columns=['mu', 'std', 'dates'])

    def set_sentences(self, sentences1: List[str], sentences1_dates: List[str], sentences2: List[str], sentences2_dates: List[str], scores: List[float]):
        self.sentences1, self.sentences2, self.scores = sentences1, sentences2, scores
        self.sentences1_dates, self.sentences2_dates = sentences1_dates, sentences2_dates

    def tokenize(self, batch: List[str]) -> BatchEncoding:
        return self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=self.max_seq_len, add_special_tokens=SPECIAL_TOKENS)
    
    def data_loader(self, sentences: List[str]):
        return DataLoader(sentences, collate_fn=self.tokenize, batch_size=self.batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, drop_last=False)

    def sim_fn(self, sent1: List[str], sent1_dates: List[str], sent2: List[str], sent2_dates: List[str]) -> float:
        sentences_to_embed: List[List[str, str]] = []

        for i, sentence in enumerate(sent1):
            if sentence not in self.cached_embeddings.index:
                sentences_to_embed.append([sentence, sent1_dates[i]])

        for i, sentence in enumerate(sent2):
            if sentence not in self.cached_embeddings.index:
                sentences_to_embed.append([sentence, sent2_dates[i]])

        sentences_to_embed_emb: GaussOutput = self.encode_fn([s[0] for s in sentences_to_embed], [s[1] for s in sentences_to_embed])

        for i, sent in enumerate(sentences_to_embed):
            if sent[0] not in self.cached_embeddings.index:
                self.cached_embeddings.loc[sent[0]] = {
                    'mu': sentences_to_embed_emb.mu[i].cpu().tolist(),
                    'std': sentences_to_embed_emb.std[i].cpu().tolist(),
                    'dates': sent[1]
                }

        sent1_emb_mu = self.cached_embeddings.loc[sent1, 'mu']
        sent1_emb_std = self.cached_embeddings.loc[sent1, 'std']
        sent2_emb_mu = self.cached_embeddings.loc[sent2, 'mu']
        sent2_emb_std = self.cached_embeddings.loc[sent2, 'std']

        return asymmetrical_kl_sim(torch.FloatTensor(sent1_emb_mu), torch.FloatTensor(sent1_emb_std), torch.FloatTensor(sent2_emb_mu), torch.FloatTensor(sent2_emb_std))

    @torch.inference_mode()
    def encode_fn(self, sentences: List[str], dates: List[str], **_) -> GaussOutput:
        self.model.eval()

        output: GaussOutput = None

        for batch in self.data_loader(sentences):
            output = self.model.forward(**batch.to(INFERENCE_DEVICE), dates=positional_encoding(dates[:self.batch_size]).to(INFERENCE_DEVICE))
            break

        return output
    
    def evaluate(self) -> dict:
        similarities: List[float] = []
        
        similarities = [i.item() for i in list(self.sim_fn(self.sentences1, self.sentences1_dates, self.sentences2, self.sentences2_dates))]
        
        return {"sent1": self.sentences1, "sent2": self.sentences2, "similarity": similarities, "ground_truth": self.scores}