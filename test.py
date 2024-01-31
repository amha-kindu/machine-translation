import tqdm
import torch
from config import *
import torch.nn as nn
from inference import MtInferenceEngine
from train import get_dataset, get_model
from config import DEVICE, get_weights_file_path


# def validate(
#     inference_engine: MtInferenceEngine, val_dataset_iterator: DataLoader, 
#     max_len: int, global_step: int, writer: SummaryWriter, num_examples=2
# ):
#     source_texts = []
#     expected = []
#     predicted = []
#     for batch in random.sample(list(val_dataset_iterator), k=num_examples):
#         # Retrieve the data points from the current batch
#         encoder_input = batch["encoder_input"].to(DEVICE)       # (batches, seq_len) 
#         encoder_mask = batch["encoder_mask"].to(DEVICE)         # (bathes, 1, 1, seq_len) 
#         decoder_mask = batch["decoder_mask"].to(DEVICE)         # (bathes, 1, seq_len, seq_len) 
        
#         model_output = inference_engine.translate_raw(encoder_input, encoder_mask, decoder_mask, max_len, SamplingStrategy.GREEDY)
            
#         source_texts.append(batch["src_text"][0])
#         expected.append(batch["tgt_text"][0])
#         predicted.append(model_output)
    
#     """
#         Evaluate the model's prediction on various standard metrics
#     """    
#     # Compute the char error rate 
#     metric = CharErrorRate()
#     writer.add_scalar('Validation CER', metric(predicted, expected), global_step)
#     writer.flush()

#     # Compute the word error rate
#     metric = WordErrorRate()
#     writer.add_scalar('Validation WER', metric(predicted, expected), global_step)
#     writer.flush()

#     # Compute the BLEU metric
#     metric = BLEUScore()
#     writer.add_scalar('Validation BLEU', metric(predicted, expected), global_step)
#     writer.flush()


if __name__ == "__main__":
    print(f"Testing started on `{DEVICE}` device")
    train_dataset, val_dataset, test_dataset = get_dataset()
    
    model = get_model(train_dataset.src_tokenizer.get_vocab_size(), train_dataset.tgt_tokenizer.get_vocab_size()).to(DEVICE)    
    
    model_filename = get_weights_file_path(PRELOAD_MODEL_SUFFIX)
    print(f"Preloading model {model_filename}")
    state = torch.load(model_filename)    
    model.load_state_dict(state["model_state_dict"])
    
    inference_engine = MtInferenceEngine(model, train_dataset.src_tokenizer, train_dataset.tgt_tokenizer)
    
    loss_func = nn.CrossEntropyLoss(ignore_index=train_dataset.src_tokenizer.token_to_id('[PAD]'), label_smoothing=0.1).to(DEVICE)
    
    batch_iterator = tqdm(val_dataset.batch_iterator(BATCH_SIZE), desc=f"Evaluating model on test dataset", colour="GREEN")

    losses = []
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
                    
        # Compute the training loss
        test_loss: torch.Tensor = loss_func(
            proj_output.view(-1, train_dataset.tgt_tokenizer.get_vocab_size()),     # (batches, seq_len, tgt_vocab_size) --> (batches*seq_len, tgt_vocab_size)
            label.view(-1)                                                          # (batches, seq_len) --> (batches * seq_len, )
        )
        
        losses.append(test_loss)
    
    avg_loss = sum(losses) / len(losses)
    print(f"Testing finished with an average cross entropy of {avg_loss}")
    
    user_input = input("Try is yourself. Enter a short english sentence")
    expected, predicted = inference_engine.translate(user_input, SEQ_LEN)
    print(f"\n Predicted: {predicted}")    