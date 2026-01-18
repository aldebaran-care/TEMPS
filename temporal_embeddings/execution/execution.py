from tqdm import tqdm
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers.tokenization_utils import BatchEncoding, PreTrainedTokenizer
from transformers.optimization import get_linear_schedule_with_warmup
from scipy.stats import spearmanr

from temporal_embeddings.model.gauss_model import GaussModel, GaussOutput
from temporal_embeddings.parameters.parameters import (
    NUM_WORKERS, MAX_SEQ_LEN, DTYPE, SPECIAL_TOKENS
)
from temporal_embeddings.utils.gauss_data import GaussData
from temporal_embeddings.utils.log_info import log_info
from temporal_embeddings.utils.similarity import asymmetrical_kl_sim
from temporal_embeddings.utils.positional_encoding import positional_encoding

class Execution():
    def __init__(self, data_fraction: float, model_name: str, batch_size: int, lr: float, weight_decay: float, epochs: int, num_warmup_ratio: float, temperature: float, num_eval_steps: int, input_file_path: str, output_directory_path: str, continue_training: bool = False, checkpoint_data: dict = None, local_rank: int = 0, rank: int = 0, world_size: int = 1, resume_step: int = 0) -> None:
        self.parameters = {
            "model_name": model_name,
            "batch_size": batch_size,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "num_warmup_ratio": num_warmup_ratio,
            "temperature": temperature,
            "num_eval_steps": num_eval_steps,
            "input_file_path": input_file_path,
            "output_directory_path": output_directory_path,
        }

        self.local_rank = local_rank
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{local_rank}")

        self.model: GaussModel = GaussModel(self.parameters["model_name"], False)
        self.model = self.model.to(self.device)
        
        if continue_training and checkpoint_data and 'model_state_dict' in checkpoint_data:
            self.model.load_state_dict(checkpoint_data['model_state_dict'])
            if self.rank == 0:
                print("Loaded model state from checkpoint")
        
        # Wrap model with DDP
        self.model = DDP(self.model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        self.model.train()  # Set to train mode after DDP wrapping
        
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(self.parameters["model_name"], model_max_length=MAX_SEQ_LEN, use_fast=True)

        self.gauss_data: GaussData = GaussData(self.parameters["input_file_path"], self.tokenizer, self.parameters["batch_size"], data_fraction, rank=rank, world_size=world_size)

        # For continued training, create a fresh scheduler for the new epochs only
        self.optimizer, self.lr_scheduler = self.create_optimizer(
            model=self.model,
            train_steps_per_epoch=len(self.gauss_data.train_dataloader),
            last_epoch=-1,  # Always start fresh scheduler
        )
        
        # Restore only optimizer state from checkpoint (not scheduler)
        if continue_training and checkpoint_data:
            if 'optimizer_state_dict' in checkpoint_data:
                try:
                    self.optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
                    # Reset learning rate to the new specified value
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.parameters["learning_rate"]
                        param_group["initial_lr"] = self.parameters["learning_rate"]
                        
                    if self.rank == 0:
                        print("Loaded optimizer state from checkpoint and reset learning rate")
                except RuntimeError as e:
                    if self.rank == 0:
                        print(f"Warning: Could not load optimizer state: {e}")
                        print("Starting with fresh optimizer state")
            
            # Don't restore scheduler - use fresh scheduler for new training phase
            if self.rank == 0:
                print(f"Created fresh learning rate scheduler for {epochs} new epochs")

    def tokenize(self, batch: list[str]) -> BatchEncoding:
        return self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=MAX_SEQ_LEN, add_special_tokens=SPECIAL_TOKENS)
    
    def collate_fn(self, data_list: list[dict]) -> BatchEncoding:
        """
        Merges a list of samples to form a mini-batch of Tensor(s). Used when using batched loading from a map-style dataset.
        """

        return BatchEncoding(
            {
                "sent0": self.tokenize([d["sent0"] for d in data_list]),
                "sent0_date": positional_encoding([d["sent0_date"] for d in data_list]),
                "sent1": self.tokenize([d["sent1"] for d in data_list]),
                "sent1_date": positional_encoding([d["sent1_date"] for d in data_list]),
                "score": torch.FloatTensor([float(d["score"]) for d in data_list]),
            }
        )

    def create_optimizer(self, model: torch.nn.Module, train_steps_per_epoch: int, last_epoch: int = -1) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
            no_decay = {"bias", "LayerNorm.weight"}
            optimizer_grouped_parameters = [
                {
                    "params": [param for name, param in model.named_parameters() if name not in no_decay and param.requires_grad
                    ],
                    "weight_decay": self.parameters["weight_decay"],
                },
                {
                    "params": [param for name, param in model.named_parameters() if name in no_decay and param.requires_grad
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.parameters["learning_rate"])
            for param_group in optimizer.param_groups:
                param_group["initial_lr"] = self.parameters["learning_rate"]

            num_training_steps = train_steps_per_epoch * self.parameters["epochs"]
            num_warmup_steps = int(num_training_steps * self.parameters["num_warmup_ratio"])

            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                last_epoch=last_epoch,
            )

            return optimizer, lr_scheduler

    @torch.inference_mode()
    def evaluator(self, split: str) -> float:
        # Use the underlying model for evaluation (not DDP wrapper)
        model = self.model.module if isinstance(self.model, DDP) else self.model
        model.eval()

        sent0_output: list[GaussOutput] = []
        sent1_output: list[GaussOutput] = []

        scores: torch.FloatTensor = torch.FloatTensor()

        data_loader: DataLoader = self.gauss_data.val_dataloader
        
        if split == "train":
            data_loader: DataLoader = self.gauss_data.train_dataloader
        if split == "val":
            data_loader: DataLoader = self.gauss_data.val_dataloader
        elif split == "test":
            data_loader: DataLoader = self.gauss_data.test_dataloader

        for batch in tqdm(data_loader, desc=f"Evaluating {split} split", disable=(self.rank != 0)):
            sent0_out = model.forward(
                input_ids=batch.sent0.input_ids.to(self.device), 
                attention_mask=batch.sent0.attention_mask.to(self.device), 
                dates=batch.sent0_date.to(self.device)
            )
            
            sent1_out = model.forward(
                input_ids=batch.sent1.input_ids.to(self.device), 
                attention_mask=batch.sent1.attention_mask.to(self.device), 
                dates=batch.sent1_date.to(self.device)
            )
            
            scores = torch.cat([scores.to(self.device), batch.score.to(self.device)], dim=0)

            sent0_output.append(sent0_out)
            sent1_output.append(sent1_out)

        output0: GaussOutput = GaussOutput(mu=torch.cat([out.mu for out in sent0_output], dim=0), std=torch.cat([out.std for out in sent0_output], dim=0))
        output1: GaussOutput = GaussOutput(mu=torch.cat([out.mu for out in sent1_output], dim=0), std=torch.cat([out.std for out in sent1_output], dim=0))

        similarities = asymmetrical_kl_sim(output0.mu, output0.std, output1.mu, output1.std)

        spearman = float(spearmanr(scores.cpu().abs(), similarities.cpu())[0]) * 100

        self.model.train()  # Set back to train mode after evaluation

        return spearman
    
    @torch.inference_mode()
    def encode_fn(self, sentences: list[str], **_) -> GaussOutput:
        model = self.model.module if isinstance(self.model, DDP) else self.model
        model.eval()

        data_loader = DataLoader(sentences, collate_fn=self.tokenize, batch_size=self.parameters["batch_size"], shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, drop_last=False)

        output: list[GaussOutput] = []
        for batch in data_loader:
            batch_device = {k: v.to(self.device) for k, v in batch.items()}
            out = model.forward(**batch_device)
            output.append(out)

        output = GaussOutput(
            mu=torch.cat([out.mu for out in output], dim=0),
            std=torch.cat([out.std for out in output], dim=0),
        )

        return output

    def log(self, metrics: dict) -> None:
        log_info(metrics, f"{self.parameters['output_directory_path']}/log.csv")
        tqdm.write(
            f"epoch: {metrics['epoch']} \t"
            f"step: {metrics['step']} \t"
            f"loss: {metrics['loss']:2.6f}       \t"
            f"dev_score: {metrics['dev_score']:.4f}"
        )
    
    def clone_state_dict(self) -> dict:
        model = self.model.module if isinstance(self.model, DDP) else self.model
        return {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
    
    def sim_fn(self, sent0: list[str], sent1: list[str]) -> list[float]:
        sent0: GaussOutput = self.encode_fn(sent0)
        sent1: GaussOutput = self.encode_fn(sent1)
        return asymmetrical_kl_sim(sent0.mu, sent0.std, sent1.mu, sent1.std).tolist()