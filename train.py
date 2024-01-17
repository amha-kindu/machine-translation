import json
import torch
import random
from torchmetrics.text import BLEUScore, WordErrorRate, CharErrorRate
import torch.nn as nn
from tqdm import tqdm
from tokenizers import Tokenizer
from inference import MtInferenceEngine, SamplingStrategy
from model import MtTransformerModel
from dataset import BilingualDataset
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.tensorboard import SummaryWriter
from config import DEVICE, get_weights_file_path, get_config
from torch.utils.data import Dataset, DataLoader, random_split

from pathlib import Path

from preprocessor import AmharicPreprocessor, EnglishPreprocessor


def get_or_build_tokenizer(dataset: list[dict], config: dict, lang: str) -> Tokenizer:
    tokenizer_filename = f"{config['tokenizer_basename'].format(lang)}"
    tokenizer_path = Path(config["tokenizer_folder"]) / tokenizer_filename
    all_sentences = [item[lang] for item in dataset]
    if not Path.exists(tokenizer_path):
        Path(config["tokenizer_folder"]).mkdir(parents=True, exist_ok=True)
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer: WordLevelTrainer = WordLevelTrainer(special_tokens=["[UNK]", "[SOS]", "[EOS]", "[PAD]"], min_frequency=2)
        tokenizer.train_from_iterator(all_sentences, trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        
    return tokenizer

def get_dataset(config: dict) -> tuple[Dataset, Dataset]:
    with open("data/languages.json", 'r', encoding='utf-8') as data:
        raw_data = json.load(data)
    
    eng_pcr = EnglishPreprocessor()
    amharic_pcr = AmharicPreprocessor()
        
    # Clean up data
    dataset = []
    for i in range(len(raw_data)):
        x, y = eng_pcr.preprocess(raw_data[i][config["src_lang"]]), amharic_pcr.preprocess(raw_data[i][config["tgt_lang"]])
        if len(x.split()) > 50 or len(y.split()) > 50:
            continue
        dataset.append({ config["src_lang"]: x, config["tgt_lang"]: y })
    
    train_size = int(0.8 * len(dataset))
    test_size = int(0.1 * len(dataset))
    val_size = len(dataset) - train_size - test_size
    
    train_test_raw, val_raw = random_split(dataset, (train_size+test_size, val_size))
    train_raw, test_raw = random_split(train_test_raw, (train_size, test_size))
    
    src_tokenizer = get_or_build_tokenizer(dataset, config, config["src_lang"])
    tgt_tokenizer = get_or_build_tokenizer(dataset, config, config["tgt_lang"])
    
    train_dataset = BilingualDataset(train_raw, config, src_tokenizer, tgt_tokenizer)
    val_dataset = BilingualDataset(val_raw, config, src_tokenizer, tgt_tokenizer)
    test_dataset = BilingualDataset(test_raw, config, src_tokenizer, tgt_tokenizer)
    
    max_src_len = max_tgt_len = 0
    for data in dataset:
        max_src_len = max(max_src_len, len(src_tokenizer.encode(data[config["src_lang"]]).ids))
        max_tgt_len = max(max_tgt_len, len(tgt_tokenizer.encode(data[config["tgt_lang"]]).ids))
        
    print(f"Max length of source sentence: {max_src_len}")
    print(f"Max length of target sentence: {max_tgt_len}")
    
    return train_dataset, val_dataset, test_dataset
    

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

def validate(
    inference_engine: MtInferenceEngine, val_dataset_iterator: DataLoader, 
    max_len: int, global_step: int, writer: SummaryWriter, num_examples=2
):
    source_texts = []
    expected = []
    predicted = []
    for batch in random.sample(list(val_dataset_iterator), k=num_examples):
        # Retrieve the data points from the current batch
        encoder_input = batch["encoder_input"].to(DEVICE)       # (batches, seq_len) 
        encoder_mask = batch["encoder_mask"].to(DEVICE)         # (bathes, 1, 1, seq_len) 
        decoder_mask = batch["decoder_mask"].to(DEVICE)         # (bathes, 1, seq_len, seq_len) 
        
        model_output = inference_engine.translate_raw(encoder_input, encoder_mask, decoder_mask, max_len, SamplingStrategy.GREEDY)
            
        source_texts.append(batch["src_text"][0])
        expected.append(batch["tgt_text"][0])
        predicted.append(model_output)
    
    """
        Evaluate the model's prediction on various standard metrics
    """    
    # Compute the char error rate 
    metric = CharErrorRate()
    writer.add_scalar('Validation CER', metric(predicted, expected), global_step)
    writer.flush()

    # Compute the word error rate
    metric = WordErrorRate()
    writer.add_scalar('Validation WER', metric(predicted, expected), global_step)
    writer.flush()

    # Compute the BLEU metric
    metric = BLEUScore()
    writer.add_scalar('Validation BLEU', metric(predicted, expected), global_step)
    writer.flush()
    
    
def train(model: MtTransformerModel, train_dataset: BilingualDataset, val_dataset: BilingualDataset, config: dict) -> None:
    TRAINING_INSTANCE = 1
    
    # Configure Tensorboard
    writer = SummaryWriter(config["tb_log_dir"])
    
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
        
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        
    loss_func = nn.CrossEntropyLoss(ignore_index=train_dataset.src_tokenizer.token_to_id('[PAD]'), label_smoothing=0.1).to(DEVICE)
    
    prev_loss = float('inf')
    for epoch in range(initial_epoch, config["epochs"]):
        """
            Set the transformer module(the model) to training mode
        """
        model.train()
        
        # Wrap train_dataloader with tqdm to show a progress bar to show
        # how much of the batches have been processed on the current epoch
        batch_iterator = tqdm(train_dataset.batch_iterator(config["batch_size"]), desc=f"Processing epoch {epoch: 02d}", colour="BLUE")
        
        loss = 0 
        # Iterate through the batches
        for batch in batch_iterator:        
            # Retrieve the data points from the current batch
            encoder_input = batch["encoder_input"].to(DEVICE)       # (batches, seq_len) 
            decoder_input = batch["decoder_input"].to(DEVICE)       # (batches, seq_len) 
            encoder_mask = batch["encoder_mask"].to(DEVICE)         # (bathes, 1, 1, seq_len) 
            decoder_mask = batch["decoder_mask"].to(DEVICE)         # (bathes, 1, seq_len, seq_len) 
            label: torch.Tensor = batch['label'].to(DEVICE)         # (batches, seq_len)
            
            # Perform the forward pass according to the operations defined in 
            # the transformer model in order to build the computation graph of the model
            encoder_output = model.encode(encoder_input, encoder_mask)                                  # (batches, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)    # (batches, seq_len, d_model)
            proj_output: torch.Tensor = model.project(decoder_output)                                   # (batches, seq_len, tgt_vocab_size)
                        
            # Compute the loss
            loss: torch.Tensor = loss_func(
                proj_output.view(-1, train_dataset.tgt_tokenizer.get_vocab_size()),     # (batches, seq_len, tgt_vocab_size) --> (batches*seq_len, tgt_vocab_size)
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
            
            loss += loss.item()
            global_step += 1
        
        current_avg_loss = loss / len(batch_iterator)
        """
            Set the transformer module(the model) to evaluation mode and validate the model's performance
        """
        infr_engine = MtInferenceEngine(model, train_dataset.src_tokenizer, train_dataset.tgt_tokenizer)
        validate(infr_engine, val_dataset.batch_iterator(1), config["seq_len"], global_step, writer)
        
        if current_avg_loss < prev_loss:
            prev_loss = current_avg_loss
            
            # Save the model at the end of every epoch
            model_filename = get_weights_file_path(config, f"ti-{TRAINING_INSTANCE:02d}_epoch-{epoch:02d}_avgLoss-{current_avg_loss:6.3f}_batch-{config['batch_size']}_lr-{config['lr']:.0e}")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step
            }, model_filename)


if __name__ == "__main__":
    config=get_config()
    print(f"Training started on `{DEVICE}` device")
    
    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)
    train_dataset, val_dataset, test_dataset = get_dataset(config)
    
    model = get_model(config, train_dataset.src_tokenizer.get_vocab_size(), train_dataset.tgt_tokenizer.get_vocab_size()).to(DEVICE)    
    
    train(model, train_dataset, val_dataset, config)