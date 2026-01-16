from typing import List, Dict
from pathlib import Path

import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers.tokenization_utils import BatchEncoding, PreTrainedTokenizer
from transformers import AutoTokenizer

from temporal_embeddings.model.gauss_model import GaussModel, GaussOutput
from temporal_embeddings.evaluation.utils.evaluation.temporal_model.parameters import INFERENCE_DEVICE, NUM_WORKERS, SPECIAL_TOKENS
from temporal_embeddings.evaluation.utils.evaluation.temporal_model.similarity import asymmetrical_kl_sim
from temporal_embeddings.utils.positional_encoding import positional_encoding

class Inference:
    def __init__(self, model_name: str, model_path: Path, batch_size: int, max_seq_len: int):
        self.model_name: str = model_name
        self.model_path: Path = model_path
        self.batch_size: int = batch_size
        self.max_seq_len: int = max_seq_len

        if model_name in ["all-minilm-l6-v2", "all-minilm-l6-v2-full"]:
            self.model_name = "sentence-transformers/all-MiniLM-L6-v2"

        elif model_name in ["prajjwal1/bert-tiny", "prajjwal1/bert-tiny-full"]:
            self.model_name = "prajjwal1/bert-tiny"

        self.model: GaussModel = GaussModel(self.model_name, False).eval().to(INFERENCE_DEVICE)

        checkpoint = torch.load(str(self.model_path), map_location=torch.device(INFERENCE_DEVICE))
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(self.model_name, model_max_length = self.max_seq_len, use_fast = False)

    def set_sentences(self, sentences1: List[str], sentences1_dates: List[str], sentences2: List[str], sentences2_dates: List[str], scores: List[float]):
        self.sentences1, self.sentences2, self.scores = sentences1, sentences2, scores
        self.sentences1_dates, self.sentences2_dates = sentences1_dates, sentences2_dates

    def tokenize(self, batch: List[str]) -> BatchEncoding:
        return self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=self.max_seq_len, add_special_tokens=SPECIAL_TOKENS)
    
    def data_loader(self, sentences: List[str]):
        return DataLoader(sentences, collate_fn=self.tokenize, batch_size=self.batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, drop_last=False)

    def sim_fn(self, sent1_emb: GaussOutput, sent2_emb: GaussOutput):
        return asymmetrical_kl_sim(sent1_emb.mu, sent1_emb.std, sent2_emb.mu, sent2_emb.std)

    @torch.inference_mode()
    def compute_embeddings(self, sentences: List[str], dates: List[str]) -> pd.DataFrame:
        self.model.eval()

        embeddings_df = pd.DataFrame(columns=['mu', 'std', 'dates'])
        
        batch_start = 0
        for batch in self.data_loader(sentences):
            batch_size = len(batch['input_ids'])
            batch_dates = dates[batch_start:batch_start + batch_size]
            new_embeddings = self.model.forward(**batch.to(INFERENCE_DEVICE), dates=positional_encoding(batch_dates).to(INFERENCE_DEVICE))
            
            for i, sentence in enumerate(sentences[batch_start:batch_start + batch_size]):
                embeddings_df.loc[sentence] = {
                    'mu': new_embeddings.mu[i].cpu().tolist(),
                    'std': new_embeddings.std[i].cpu().tolist(),
                    'dates': batch_dates[i]
                }
            batch_start += batch_size
        
        return embeddings_df

    def encode_fn(self, sentences: List[str], embedding_cache: pd.DataFrame) -> GaussOutput:
        """Retrieve embeddings from cache for given sentences."""
        mu_tensors = [torch.FloatTensor(embedding_cache.loc[sentence, 'mu']) for sentence in sentences]
        std_tensors = [torch.FloatTensor(embedding_cache.loc[sentence, 'std']) for sentence in sentences]
        
        return GaussOutput(mu=torch.stack(mu_tensors), std=torch.stack(std_tensors))
    
    def evaluate(self, embedding_cache: pd.DataFrame) -> dict:
        similarities: List[float] = []
        
        sent1_emb = self.encode_fn(self.sentences1, embedding_cache)
        sent2_emb = self.encode_fn(self.sentences2, embedding_cache)
        similarities = [i.item() for i in list(self.sim_fn(sent1_emb, sent2_emb))]
        
        return {"sent1": self.sentences1, "sent2": self.sentences2, "similarity": similarities, "ground_truth": self.scores}