import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from tqdm import trange, tqdm
from torch.utils.tensorboard import SummaryWriter

from temporal_embeddings.parameters.parameters import (
    EPOCHS, DEVICE, NUM_EVAL_STEPS, OUTPUT_DIRECTORY_PATH, MODEL_NAME, LR,
    WEIGHT_DECAY, NUM_WARMUP_RATIO, TEMPERATURE, INPUT_FILE_PATH, BATCH_SIZE, SEED
)
from temporal_embeddings.model.gauss_model import GaussOutput
from temporal_embeddings.utils.similarity import asymmetrical_kl_sim
from temporal_embeddings.utils.set_seed import set_seed
from temporal_embeddings.execution.execution import Execution
from temporal_embeddings.utils.save import save_json
from temporal_embeddings.utils.loss.cosent_loss import CoSentLoss
from temporal_embeddings.utils.os.folder_management import create_folders

def main(data_fraction: float,
         model_name: str,
         batch_size: int,
         lr: float,
         weight_decay: float,
         epochs: int,
         num_warmup_ratio: float,
         temperature: float,
         num_eval_steps: int,
         input_file_path: str,
         output_directory_path: str,
         continue_training: bool,
         model_path: str) -> None:
    
    set_seed(seed=SEED)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir_path: str = f"logs/runs/{model_name}_{current_time}"
    writer = SummaryWriter(log_dir=log_dir_path)
    
    parameters = {
        "model_name": model_name,
        "batch_size": batch_size,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "num_warmup_ratio": num_warmup_ratio,
        "temperature": temperature,
        "num_eval_steps": num_eval_steps,
        "input_file_path": str(input_file_path),
        "output_directory_path": str(output_directory_path),
        "data_fraction": data_fraction,
        "continue_training": continue_training,
        "model_path": str(model_path) if continue_training else None,
    }
    
    with open(f"{log_dir_path}/parameters.json", "w") as param_file:
        json.dump(parameters, param_file, indent=4)
    
    print("Parameters logged in:", f"{log_dir_path}/parameters.json")
    
    print("Load the execution object")
    execution = Execution(data_fraction=data_fraction, 
                          model_name=model_name, 
                          batch_size=batch_size, 
                          lr=lr, 
                          weight_decay=weight_decay, 
                          epochs=epochs, 
                          num_warmup_ratio=num_warmup_ratio, 
                          temperature=temperature, 
                          num_eval_steps=num_eval_steps, 
                          input_file_path=input_file_path, 
                          output_directory_path=output_directory_path,
                          continue_training=continue_training,
                          model_path=model_path)

    print("Compute the first dev score")
    best_dev_score = execution.evaluator("val")
    best_epoch, best_step = 0, 0
    val_metrics = {
        "epoch": best_epoch,
        "step": best_step,
        "loss": float("inf"),
        "dev_score": best_dev_score,
    }
    execution.log(val_metrics)
    best_state_dict = execution.clone_state_dict()
    
    current_step = 0

    for epoch in trange(epochs, leave=False, dynamic_ncols=True, desc="Epoch"):
        train_losses = []
        execution.model.train()

        for batch in tqdm(execution.gauss_data.train_dataloader, total=len(execution.gauss_data.train_dataloader), dynamic_ncols=True, leave=False, desc="Step"):
            current_step += 1

            with torch.autograd.set_detect_anomaly(True):
            
                # Move tensors to device without unnecessary clone/detach operations
                sent0_out: GaussOutput = execution.model.forward(
                    input_ids=batch.sent0.input_ids.to(DEVICE), 
                    attention_mask=batch.sent0.attention_mask.to(DEVICE), 
                    dates=batch.sent0_date.to(DEVICE)
                )

                sent1_out: GaussOutput = execution.model.forward(
                    input_ids=batch.sent1.input_ids.to(DEVICE), 
                    attention_mask=batch.sent1.attention_mask.to(DEVICE), 
                    dates=batch.sent1_date.to(DEVICE)
                )

                sim_mat: torch.Tensor = asymmetrical_kl_sim(sent0_out.mu, sent0_out.std, sent1_out.mu, sent1_out.std)
                
                loss_func = CoSentLoss()
                loss = loss_func(sim_mat, batch.score.to(DEVICE))

                train_losses.append(loss.item())

                execution.optimizer.zero_grad()
                loss.backward()
                execution.optimizer.step()
                execution.lr_scheduler.step()
            
            if current_step % num_eval_steps == 0:
                execution.model.eval()

                checkpoint_path: Path = Path(output_directory_path) / Path(f"checkpoint_step_{current_step}.pth")
                
                torch.save({
                    "step": current_step,
                    "epoch": epoch,
                    "model_state_dict": execution.model.state_dict(),
                    "optimizer_state_dict": execution.optimizer.state_dict(),
                    "lr_scheduler_state_dict": execution.lr_scheduler.state_dict(),
                    "best_dev_score": best_dev_score,
                    "best_state_dict": best_state_dict,
                    "val_metrics": val_metrics,
                }, checkpoint_path)
                print(f"Checkpoint saved at step {current_step} -> {checkpoint_path}")

                dev_score = execution.evaluator("val")

                if best_dev_score < dev_score:
                    best_dev_score = dev_score
                    best_epoch, best_step = epoch, current_step
                    best_state_dict = execution.clone_state_dict()

                val_metrics = {
                    "epoch": epoch,
                    "step": current_step,
                    "loss": sum(train_losses) / len(train_losses),
                    "dev_score": dev_score,
                }

                print(f"Writing to TensorBoard at step {current_step}:")
                
                for key, value in val_metrics.items():
                    print(f"  Metrics/{key}: {value}")
                    writer.add_scalar(f"Metrics/{key}", value, current_step)

                print(f"  Loss/train: {sum(train_losses) / len(train_losses)}")
                writer.add_scalar("Loss/train", sum(train_losses) / len(train_losses), current_step)

                print(f"  Score/dev: {dev_score}")
                writer.add_scalar("Score/dev", dev_score, current_step)

                print(f"  Learning_Rate: {execution.optimizer.param_groups[0]['lr']}")
                writer.add_scalar("Learning_Rate", execution.optimizer.param_groups[0]["lr"], current_step)

                execution.log(val_metrics)
                train_losses = []

                execution.model.train()

    dev_metrics = {
        "best-epoch": best_epoch,
        "best-step": best_step,
        "best-dev-auc": best_dev_score,
    }
    dev_metrics_path: Path = Path(f"{output_directory_path}/metrics/dev_metrics_{model_name.replace('/', '_')}_{current_time}.json")
    create_folders([dev_metrics_path.parent])
    save_json(dev_metrics, dev_metrics_path)
    print("Dev metrics saved in:", dev_metrics_path)

    execution.model.load_state_dict(best_state_dict)
    model_path = f"{output_directory_path}/trained_models/model_{model_name.replace('/', '_')}_{current_time}.pth"
    create_folders([Path(model_path).parent])
    torch.save(execution.model.state_dict(), model_path)
    print("Model saved in:", model_path)
    execution.model.eval().to(DEVICE)

    metrics = execution.evaluator(split="test")
    metrics_path: Path = Path(f"{output_directory_path}/metrics/metrics_{model_name.replace('/', '_')}_{current_time}.json")
    create_folders([metrics_path.parent])
    save_json(metrics, metrics_path)
    print("Train metrics saved in:", metrics_path)

    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the temporal embeddings model.")
    parser.add_argument("--data_fraction", type=float, default=1.0, help="Fraction of data to use for training.")
    parser.add_argument("--model_name", type=str, default=MODEL_NAME, help="Name of the model.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Size of the batch.")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY, help="Weight decay.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs.")
    parser.add_argument("--num_warmup_ratio", type=float, default=NUM_WARMUP_RATIO, help="Warmup ratio.")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE, help="Temperature for loss scaling.")
    parser.add_argument("--num_eval_steps", type=int, default=NUM_EVAL_STEPS, help="Number of evaluation steps.")
    parser.add_argument("--input_file_path", type=str, default=str(INPUT_FILE_PATH), help="Path to the input file.")
    parser.add_argument("--output_directory_path", type=str, default=str(OUTPUT_DIRECTORY_PATH), help="Path to the output directory.")
    parser.add_argument("--continue_training", action="store_true", help="Flag to indicate whether to continue training a previous model.")
    parser.add_argument("--model_path", type=str, default=None, help="Path to a pre-trained model to continue training.")
    args = parser.parse_args()

    main(
        data_fraction=args.data_fraction,
        model_name=args.model_name,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        num_warmup_ratio=args.num_warmup_ratio,
        temperature=args.temperature,
        num_eval_steps=args.num_eval_steps,
        input_file_path=args.input_file_path,
        output_directory_path=args.output_directory_path,
        continue_training=args.continue_training,
        model_path=args.model_path,
    )