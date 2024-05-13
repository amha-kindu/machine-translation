import json
import torch
from config import *
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from tokenizers import Tokenizer
from model import MtTransformerModel
from dataset import ParallelTextDataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split, DataLoader



def get_tokenizer(lang: str, basename: str=TOKENIZER_BASENAME) -> Tokenizer:
    tokenizer_filename = f"{basename.format(lang)}"
    tokenizer_path = Path(TOKENIZER_FOLDER) / tokenizer_filename

    tokenizer: Tokenizer = Tokenizer.from_file(str(tokenizer_path))
        
    tokenizer.enable_truncation(max_length=SEQ_LEN - 2)
    
    return tokenizer

def get_dataset() -> tuple[ParallelTextDataset, ParallelTextDataset, ParallelTextDataset]:
    with open(DATASET_PATH, 'r', encoding='utf-8') as data:
        dataset = json.load(data)
    
    train_size = int(0.8 * len(dataset))
    test_size = int(0.15 * len(dataset))
    val_size = len(dataset) - train_size - test_size
    
    train_test_raw, val_raw = random_split(dataset, (train_size+test_size, val_size))
    train_raw, test_raw = random_split(train_test_raw, (train_size, test_size))
    
    src_tokenizer = get_tokenizer(SRC_LANG)
    tgt_tokenizer = get_tokenizer(TGT_LANG)

    train_dataset = ParallelTextDataset(train_raw, src_tokenizer, tgt_tokenizer)
    val_dataset = ParallelTextDataset(val_raw, src_tokenizer, tgt_tokenizer)
    test_dataset = ParallelTextDataset(test_raw, src_tokenizer, tgt_tokenizer)
    
    return train_dataset, val_dataset, test_dataset
    

def get_model(src_vocab_size: int, tgt_vocab_size):
    return MtTransformerModel.build(
        src_vocab_size=src_vocab_size, 
        tgt_vocab_size=tgt_vocab_size
    )

@torch.no_grad()
def validate(model: MtTransformerModel, val_batch_iterator: DataLoader, loss_func: nn.CrossEntropyLoss):    
    """
        Set the transformer module(the model) to evaluation mode
    """
    model.eval()
    
    val_losses = []    
    # Evaluate model with `num_examples` number of random examples
    for batch in val_batch_iterator:
        # Retrieve the data points from the current batch
        encoder_input = batch["encoder_input"].to(DEVICE)       # (batch, seq_len) 
        decoder_input = batch["decoder_input"].to(DEVICE)       # (batch, seq_len) 
        encoder_mask = batch["encoder_mask"].to(DEVICE)         # (batch, 1, 1, seq_len) 
        decoder_mask = batch["decoder_mask"].to(DEVICE)         # (batch, 1, seq_len, seq_len) 
        label: torch.Tensor = batch['label'].to(DEVICE)         # (batch, seq_len)
        
        # Perform the forward pass according to the operations defined in 
        # the transformer model in order to build the computation graph of the model
        encoder_output = model.encode(encoder_input, encoder_mask)                                  # (batch, seq_len, d_model)
        decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)    # (batch, seq_len, d_model)
        proj_output: torch.Tensor = model.project(decoder_output)                                   # (batch, seq_len, tgt_vocab_size)
                    
        # Compute the cross entropy loss
        loss: torch.Tensor = loss_func.forward(
            proj_output.view(-1, val_dataset.tgt_tokenizer.get_vocab_size()),     # (batch, seq_len, tgt_vocab_size) --> (batch*seq_len, tgt_vocab_size)
            label.view(-1)                                                          # (batch, seq_len) --> (batch * seq_len, )
        )
        
        val_losses.append(loss.item())
        
        if len(val_losses) > 1:
            break

    return sum(val_losses) / len(val_losses)

    
def train(model: MtTransformerModel, train_dataset: ParallelTextDataset, val_dataset: ParallelTextDataset) -> None:   
    # Configure Tensorboard
    writer = SummaryWriter(TB_LOG_DIR)
    
    # Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=INIT_LR, eps=1e-09)
    
    initial_epoch = 0
    global_step = 0
    if PRELOAD_MODEL_SUFFIX:
        model_filename = get_weights_file_path(PRELOAD_MODEL_SUFFIX)
        print(f"Preloading model {model_filename}")
        
        state = torch.load(model_filename)
        initial_epoch = state["epoch"] + 1
        global_step = state["global_step"]
        
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        
    loss_func = nn.CrossEntropyLoss(ignore_index=train_dataset.src_tokenizer.token_to_id('[PAD]'), label_smoothing=0.1).to(DEVICE)
    
    batch_iterator = train_dataset.batch_iterator(BATCH_SIZE)
    val_batch_iterator = val_dataset.batch_iterator(BATCH_SIZE)
    
    prev_loss = float('inf')
    val_loss = 0
    for epoch in range(initial_epoch, EPOCHS):
        # Wrap train_dataloader with tqdm to show a progress bar to show
        # how much of the batches have been processed on the current epoch
        batch_iterator = tqdm(batch_iterator, desc=f"Processing epoch {epoch: 02d}", colour="BLUE")
        
        train_losses = []
        val_losses = []
        # Iterate through the batches
        for batch in batch_iterator:    
            """
                Set the transformer module(the model) to back to training mode
            """
            model.train() 
                 
            # Retrieve the data points from the current batch
            encoder_input = batch["encoder_input"].to(DEVICE)       # (batch, seq_len) 
            decoder_input = batch["decoder_input"].to(DEVICE)       # (batch, seq_len) 
            encoder_mask = batch["encoder_mask"].to(DEVICE)         # (batch, 1, 1, seq_len) 
            decoder_mask = batch["decoder_mask"].to(DEVICE)         # (batch, 1, seq_len, seq_len) 
            label: torch.Tensor = batch['label'].to(DEVICE)         # (batch, seq_len)
            
            # Perform the forward pass according to the operations defined in 
            # the transformer model in order to build the computation graph of the model
            encoder_output = model.encode(encoder_input, encoder_mask)                                  # (batch, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)    # (batch, seq_len, d_model)
            proj_output: torch.Tensor = model.project(decoder_output)                                   # (batch, seq_len, tgt_vocab_size)
                        
            # Compute the training loss
            train_loss: torch.Tensor = loss_func.forward(
                proj_output.view(-1, train_dataset.tgt_tokenizer.get_vocab_size()),     # (batch, seq_len, tgt_vocab_size) --> (batch*seq_len, tgt_vocab_size)
                label.view(-1)                                                          # (batch, seq_len) --> (batch * seq_len, )
            )
            
            if global_step % 200 == 0:
                # Evaluate the model on the validation dataset(aka unseen data)
                val_loss = validate(model, val_batch_iterator, loss_func)
                
                # Log the training and validation loss on tensorboard
                writer.add_scalars("Cross-Entropy-Loss", { "Training": train_loss.item(), "Validation": val_loss }, global_step)
            else:
                writer.add_scalars("Cross-Entropy-Loss", { "Training": train_loss.item() }, global_step)
                
            writer.flush()
            
            # Add the calculated training loss and validation loss as a postfix to the progress bar shown by tqdm
            batch_iterator.set_postfix({"train_loss": f"{train_loss.item():6.3f}"})

            # Perform the backward pass on the computation graph built during the forward pass, 
            # in order to calculate the grad for each of the intermediate and leaf tensors on the computation graph
            train_loss.backward()
            
            # Update the model parameters
            optimizer.step()
            
            # Zero the gradients of the model parameters to prevent gradient accumulation 
            optimizer.zero_grad()
            
            train_losses.append(train_loss.item())
            val_losses.append(val_loss)
            
            global_step += 1
                    
        current_avg_train_loss = sum(train_losses) / len(train_losses)
        current_avg_val_loss = sum(val_losses) / len(val_losses)
        
        if current_avg_train_loss < prev_loss:
            prev_loss = current_avg_train_loss
            
            # Save the model at the end of every epoch
            model_filename = get_weights_file_path(f"epoch-{epoch:02d}_avgTrainLoss-{current_avg_train_loss:6.3f}_avgValLoss-{current_avg_val_loss:6.3f}_batch-{BATCH_SIZE}_init_lr-{INIT_LR:.0e}")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step
            }, model_filename)


if __name__ == "__main__":
    print(f"Training started on `{DEVICE}` device")
    train_dataset, val_dataset, test_dataset = get_dataset()
    
    model = get_model(train_dataset.src_tokenizer.get_vocab_size(), train_dataset.tgt_tokenizer.get_vocab_size()).to(DEVICE)    
    
    train(model, train_dataset, val_dataset)