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
    def __init__(self, model_name: str, model_path: Path, batch_size: int, max_seq_len: int, cache_file_path: Path = None):
        self.model_name: str = model_name
        self.model_path: Path = model_path
        self.batch_size: int = batch_size
        self.max_seq_len: int = max_seq_len
        self.cache_file_path: Path = cache_file_path

        if model_name in ["all-minilm-l6-v2", "all-minilm-l6-v2-full"]:
            self.model_name = "sentence-transformers/all-MiniLM-L6-v2"

        self.model: GaussModel = GaussModel(self.model_name, False).eval().to(INFERENCE_DEVICE)
        self.model.load_state_dict(torch.load(str(self.model_path), map_location=torch.device(INFERENCE_DEVICE)))

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(self.model_name, model_max_length = self.max_seq_len, use_fast = False)

        # Load embedding cache if exists
        self.cached_embeddings: pd.DataFrame = pd.DataFrame(columns=['mu', 'std', 'dates'])
        if self.cache_file_path and self.cache_file_path.exists():
            self.cached_embeddings = pd.read_pickle(self.cache_file_path)

    def set_sentences(self, sentences1: List[str], sentences1_dates: List[str], sentences2: List[str], sentences2_dates: List[str], scores: List[float]):
        self.sentences1, self.sentences2, self.scores = sentences1, sentences2, scores
        self.sentences1_dates, self.sentences2_dates = sentences1_dates, sentences2_dates

    def tokenize(self, batch: List[str]) -> BatchEncoding:
        return self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=self.max_seq_len, add_special_tokens=SPECIAL_TOKENS)
    
    def data_loader(self, sentences: List[str]):
        return DataLoader(sentences, collate_fn=self.tokenize, batch_size=self.batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, drop_last=False)

    def sim_fn(self, sent1: List[str], sent1_dates: List[str], sent2: List[str], sent2_dates: List[str]) -> float:
        # Get embeddings for all sentences (caching handled in encode_fn)
        sent1_emb = self.encode_fn(sent1, sent1_dates)
        sent2_emb = self.encode_fn(sent2, sent2_dates)

        return asymmetrical_kl_sim(sent1_emb.mu, sent1_emb.std, sent2_emb.mu, sent2_emb.std)

    @torch.inference_mode()
    def encode_fn(self, sentences: List[str], dates: List[str], **_) -> GaussOutput:
        self.model.eval()

        sentences_to_embed: List[str] = []
        sentence_indices: List[int] = []

        for i, sentence in enumerate(sentences):
            if sentence not in self.cached_embeddings.index:
                sentences_to_embed.append(sentence)
                sentence_indices.append(i)

        if sentences_to_embed:
            dates_to_embed = [dates[i] for i in sentence_indices]
            
            batch_start = 0
            for batch in self.data_loader(sentences_to_embed):
                batch_size = len(batch['input_ids'])
                batch_dates = dates_to_embed[batch_start:batch_start + batch_size]
                new_embeddings = self.model.forward(**batch.to(INFERENCE_DEVICE), dates=positional_encoding(batch_dates).to(INFERENCE_DEVICE))
                
                for i, sentence in enumerate(sentences_to_embed[batch_start:batch_start + batch_size]):
                    if sentence not in self.cached_embeddings.index:
                        self.cached_embeddings.loc[sentence] = {
                            'mu': new_embeddings.mu[i].cpu().tolist(),
                            'std': new_embeddings.std[i].cpu().tolist(),
                            'dates': batch_dates[i]
                        }
                batch_start += batch_size

        mu_tensors = [torch.FloatTensor(self.cached_embeddings.loc[sentence, 'mu']) for sentence in sentences]
        std_tensors = [torch.FloatTensor(self.cached_embeddings.loc[sentence, 'std']) for sentence in sentences]
        
        return GaussOutput(mu=torch.stack(mu_tensors), std=torch.stack(std_tensors))
    
    def evaluate(self) -> dict:
        similarities: List[float] = []
        
        similarities = [i.item() for i in list(self.sim_fn(self.sentences1, self.sentences1_dates, self.sentences2, self.sentences2_dates))]
        
        # Save embedding cache if cache file path is provided
        if self.cache_file_path:
            self.cached_embeddings.to_pickle(self.cache_file_path)
        
        return {"sent1": self.sentences1, "sent2": self.sentences2, "similarity": similarities, "ground_truth": self.scores}