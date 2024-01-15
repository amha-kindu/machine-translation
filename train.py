import json
import torch
import torch.nn as nn
from tqdm import tqdm
from tokenizers import Tokenizer
from model import MtTransformerModel
from dataset import BilingualDataset
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.tensorboard import SummaryWriter
from config import DEVICE, get_weights_file_path, get_config
from torch.utils.data import Dataset, DataLoader, random_split

from pathlib import Path

def get_all_sentences(dataset: dict, lang: str) -> list[str]:
    for item in dataset:
        yield item[lang]

def get_or_build_tokenizer(config: dict, dataset: list[dict], lang: str) -> Tokenizer:
    tokenizer_filename = f"{config['tokenizer_basename'].format(lang)}"
    tokenizer_path = Path(config["tokenizer_folder"]) / tokenizer_filename
    if not Path.exists(tokenizer_path):
        Path(config["tokenizer_folder"]).mkdir(parents=True, exist_ok=True)
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer: WordLevelTrainer = WordLevelTrainer(special_tokens=["[UNK]", "[SOS]", "[EOS]", "[PAD]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(dataset, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        
    return tokenizer

def get_dataset(config: dict) -> tuple[DataLoader, DataLoader, Tokenizer, Tokenizer]:
    with open("data/languages.json", 'r', encoding='utf-8') as dataset:
        dataset = json.load(dataset)
        
    src_tokenizer = get_or_build_tokenizer(config, dataset, config["src_lang"])
    tgt_tokenizer = get_or_build_tokenizer(config, dataset, config["tgt_lang"])
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_raw, val_raw = random_split(dataset, (train_size, val_size))
    
    train_dataset = BilingualDataset(train_raw, src_tokenizer, tgt_tokenizer, config["src_lang"], config["tgt_lang"], config["seq_len"])
    val_dataset = BilingualDataset(val_raw, src_tokenizer, tgt_tokenizer, config["src_lang"], config["tgt_lang"], config["seq_len"])
    
    max_src_len = 0
    max_tgt_len = 0
    for data in dataset:
        max_src_len = max(max_src_len, len(src_tokenizer.encode(data[config["src_lang"]]).ids))
        max_tgt_len = max(max_tgt_len, len(tgt_tokenizer.encode(data[config["tgt_lang"]]).ids))
        
    print(f"Max length of source sentence: {max_src_len}")
    print(f"Max length of target sentence: {max_tgt_len}")
    
    train_dataloader = DataLoader(train_dataset, config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_dataset, 1, shuffle=True)
    
    return train_dataloader, val_dataloader, src_tokenizer, tgt_tokenizer
    

def get_model(config: dict, src_vocab_size: int, tgt_vocab_size):
    return MtTransformerModel.build(
        src_vocab_size=src_vocab_size, 
        tgt_vocab_size=tgt_vocab_size, 
        src_seq_len=config["seq_len"], 
        tgt_seq_len=config["seq_len"], 
        d_model=config["d_model"],
        n_blocks=config["n_blocks"],
        heads=config["heads"],
        dropout=config["dropout"],
        dff=config["dff"]
    )
    
    
def train_model(config):
    print(f"Using the DEVICE: {DEVICE}")
    
    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)
    train_dataloader, val_dataloader, src_tokenizer, tgt_tokenizer = get_dataset(config)
    model = get_model(config, src_tokenizer.get_vocab_size(), tgt_tokenizer.get_vocab_size()).to(DEVICE)
    
    # Configure Tensorboard
    writer = SummaryWriter(config["experiment_name"])
    
    # Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-09)
    
    initial_epoch = 0
    global_step = 0
    if config["preload"]:
        model_filename = get_weights_file_path(config, config["preload"])
        print(f"Preloading model {model_filename}")
        
        state = torch.load(model_filename)
        initial_epoch = state["epoch"] + 1
        global_step = state["global_step"]
        
        optimizer.load_state_dict(state["optimizer_state_dict"])
        
    loss_func = nn.CrossEntropyLoss(ignore_index=src_tokenizer.token_to_id('[PAD]'), label_smoothing=0.1).to(DEVICE)
    
    for epoch in range(initial_epoch, config["epochs"]):
        # Set the module in training mode
        model.train()
        
        # Wrap train_dataloader with tqdm to show a progress bar to show
        # how much of the batches have been processed on the current epoch
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch: 02d}", colour="BLUE")
        
        # Iterate through the batches
        for batch in batch_iterator:
            # Retrieve the data points from the current batch
            encoder_input = batch["encoder_input"].to(DEVICE)       # (batches, seq_len) 
            decoder_input = batch["decoder_input"].to(DEVICE)       # (batches, seq_len) 
            encoder_mask = batch["encoder_mask"].to(DEVICE)         # (bathes, 1, 1, seq_len) 
            decoder_mask = batch["decoder_mask"].to(DEVICE)         # (bathes, 1, seq_len, seq_len) 
            label = batch['label'].to(DEVICE)                       # (batches, seq_len)
            
            # Perform the forward pass according to the operations defined in 
            # the transformer model in order to build the computation graph of the model
            encoder_output = model.encode(encoder_input, encoder_mask)                                  # (batches, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)    # (batches, seq_len, d_model)
            proj_output: torch.Tensor = model.project(decoder_output)                                   # (batches, seq_len, tgt_vocab_size)
                        
            # Compute the loss
            loss: torch.Tensor = loss_func(
                proj_output.view(-1, tgt_tokenizer.get_vocab_size()),     # (batches, seq_len, tgt_vocab_size) --> (batches*seq_len, tgt_vocab_size)
                label.view(-1)                                            # (batches, seq_len) --> (batches * seq_len, )
            )
            
            # Add the calculated loss as a postfix to the progress bar
            # shown by tqdm
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            
            # Log the loss on tensorboard
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Perform the backward pass on the computation graph built during the forward pass, 
            # in order to calculate the grad for each of the intermediate and leaf tensors on the computation graph
            loss.backward()
            
            # Update the model parameters
            optimizer.step()
            
            # Zero the gradients of the model parameters to prevent gradient accumulation 
            optimizer.zero_grad()
            
            global_step += 1
            
        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step
        }, model_filename)


if __name__ == "__main__":
    config=get_config()
    train_model(config)