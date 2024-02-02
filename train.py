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
from preprocessor import AmharicPreprocessor, EnglishPreprocessor
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors



def get_or_build_tokenizer(dataset: list[dict], lang: str) -> Tokenizer:
    tokenizer_filename = f"{TOKENIZER_BASENAME.format(lang)}"
    tokenizer_path = Path(TOKENIZER_FOLDER) / tokenizer_filename
    if not Path.exists(tokenizer_path):
        preprocessor = AmharicPreprocessor(None) if lang == "am" else EnglishPreprocessor(None)
        all_sentences = [preprocessor.preprocess(item[lang], encode=False) for item in dataset]
        Path(TOKENIZER_FOLDER).mkdir(parents=True, exist_ok=True)
        
        # Initialize a tokenizer
        tokenizer = Tokenizer(models.BPE(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
        
        # Customize pre-tokenization and decoding
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
        
        if lang == "en":
            alphabet = pre_tokenizers.ByteLevel.alphabet()
        elif lang == "am":
            # Amharic chars and the arabic numerals
            alphabet = ["ሀ", "ሁ", "ሂ", "ሃ", "ሄ", "ህ", "ሆ", "ሇ", "ለ", "ሉ", "ሊ", "ላ", "ሌ", "ል", "ሎ", "ሏ", "ሐ", "ሑ", "ሒ", "ሓ", "ሔ", "ሕ", "ሖ", "ሗ", "መ", "ሙ", "ሚ", "ማ", "ሜ", "ም", "ሞ", "ሟ", "ሠ", "ሡ", "ሢ", "ሣ", "ሤ", "ሥ", "ሦ", "ሧ", "ረ", "ሩ", "ሪ", "ራ", "ሬ", "ር", "ሮ", "ሯ", "ሰ", "ሱ", "ሲ", "ሳ", "ሴ", "ስ", "ሶ", "ሷ", "ሸ", "ሹ", "ሺ", "ሻ", "ሼ", "ሽ", "ሾ", "ሿ", "ቀ", "ቁ", "ቂ", "ቃ", "ቄ", "ቅ", "ቆ", "ቇ", "ቈ", "ቊ", "ቋ", "ቌ", "ቍ", "በ", "ቡ", "ቢ", "ባ", "ቤ", "ብ", "ቦ", "ቧ", "ቨ", "ቩ", "ቪ", "ቫ", "ቬ", "ቭ", "ቮ", "ቯ", "ተ", "ቱ", "ቲ", "ታ", "ቴ", "ት", "ቶ", "ቷ", "ቸ", "ቹ", "ቺ", "ቻ", "ቼ", "ች", "ቾ", "ቿ", "ኀ", "ኁ", "ኂ", "ኃ", "ኄ", "ኅ", "ኆ", "ኇ", "ኈ", "ኊ", "ኋ", "ኌ", "ኍ", "ነ", "ኑ", "ኒ", "ና", "ኔ", "ን", "ኖ", "ኗ", "ኘ", "ኙ", "ኚ", "ኛ", "ኜ", "ኝ", "ኞ", "ኟ", "አ", "ኡ", "ኢ", "ኣ", "ኤ", "እ", "ኦ", "ኧ", "ከ", "ኩ", "ኪ", "ካ", "ኬ", "ክ", "ኮ", "ኯ", "ኰ", "ኲ", "ኳ", "ኴ", "ኵ", "ኸ", "ኹ", "ኺ", "ኻ", "ኼ", "ኽ", "ኾ", "ወ", "ዉ", "ዊ", "ዋ", "ዌ", "ው", "ዎ", "ዐ", "ዑ", "ዒ", "ዓ", "ዔ", "ዕ", "ዖ", "ዘ", "ዙ", "ዚ", "ዛ", "ዜ", "ዝ", "ዞ", "ዟ", "ዠ", "ዡ", "ዢ", "ዣ", "ዤ", "ዥ", "ዦ", "ዧ", "የ", "ዩ", "ዪ", "ያ", "ዬ", "ይ", "ዮ", "ደ", "ዱ", "ዲ", "ዳ", "ዴ", "ድ", "ዶ", "ዷ", "ጀ", "ጁ", "ጂ", "ጃ", "ጄ", "ጅ", "ጆ", "ጇ", "ገ", "ጉ", "ጊ", "ጋ", "ጌ", "ግ", "ጎ", "ጏ", "ጠ", "ጡ", "ጢ", "ጣ", "ጤ", "ጥ", "ጦ", "ጧ", "ጨ", "ጩ", "ጪ", "ጫ", "ጬ", "ጭ", "ጮ", "ጯ", "ጰ", "ጱ", "ጲ", "ጳ", "ጴ", "ጵ", "ጶ", "ጷ", "ጸ", "ጹ", "ጺ", "ጻ", "ጼ", "ጽ", "ጾ", "ጿ", "ፀ", "ፁ", "ፂ", "ፃ", "ፄ", "ፅ", "ፆ", "ፇ", "ፈ", "ፉ", "ፊ", "ፋ", "ፌ", "ፍ", "ፎ", "ፏ", "ፐ", "ፑ", "ፒ", "ፓ", "ፔ", "ፕ", "ፖ", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
            
            # Add punctuations and special symbols
            alphabet += ["!", "@", "#", "$", "%", "^", "«", "»", "&", "?", "*", "(", ")", "…", "[", "]", "{", "}", ";", "“", "”", "›", "’", "‘", '"', "'", ":", ",", ".", "‹", "/", "<", ">", "\\", "\\", "|", "`", "´", "~", "-", "=", "+", "፡", "።", "፤", ";", "፦", "፥", "፧", "፨", "፠", "፣"]
        else:
            raise UnicodeError("Unrecognized language")
           
        # Train on dataset
        trainer = trainers.BpeTrainer(
            vocab_size=15000, 
            special_tokens=["[UNK]", "[SOS]", "[EOS]", "[PAD]"], 
            initial_alphabet=alphabet,
            min_frequency=2,
            show_progress=True
        )        
        tokenizer.train_from_iterator(all_sentences, trainer=trainer)
        
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        
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
    
    src_tokenizer = get_or_build_tokenizer(dataset, SRC_LANG)
    tgt_tokenizer = get_or_build_tokenizer(dataset, TGT_LANG)

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
                    
        # Compute the cross entropy loss
        loss: torch.Tensor = loss_func(
            proj_output.view(-1, val_dataset.tgt_tokenizer.get_vocab_size()),     # (batches, seq_len, tgt_vocab_size) --> (batches*seq_len, tgt_vocab_size)
            label.view(-1)                                                          # (batches, seq_len) --> (batches * seq_len, )
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
                        
            # Compute the training loss
            train_loss: torch.Tensor = loss_func(
                proj_output.view(-1, train_dataset.tgt_tokenizer.get_vocab_size()),     # (batches, seq_len, tgt_vocab_size) --> (batches*seq_len, tgt_vocab_size)
                label.view(-1)                                                          # (batches, seq_len) --> (batches * seq_len, )
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
    
    Path(MODEL_FOLDER).mkdir(parents=True, exist_ok=True)
    train_dataset, val_dataset, test_dataset = get_dataset()
    
    model = get_model(train_dataset.src_tokenizer.get_vocab_size(), train_dataset.tgt_tokenizer.get_vocab_size()).to(DEVICE)    
    
    train(model, train_dataset, val_dataset)