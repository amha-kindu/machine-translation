import sys
import torch
import random
from config import *
from enum import Enum
from PyQt5.QtGui import QFont
from tokenizers import Tokenizer
from model import MtTransformerModel
from dataset import ParallelTextDataset
from train import get_model, get_tokenizer
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QGridLayout


class SamplingStrategy(Enum):
    GREEDY=0,
    TOP_K_RANDOM=1,
    NUCLEUS=2

class MtInferenceEngine:
    
    def __init__(self, model: MtTransformerModel, src_tokenizer: Tokenizer, tgt_tokenizer: Tokenizer, top_k: int= 5, nucleus_threshold=10) -> None:
        self.model = model
        self.top_k = top_k
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.nucleus_threshold = nucleus_threshold
        self.sos_token = torch.tensor([self.src_tokenizer.token_to_id("[SOS]")], dtype=torch.int64, device=DEVICE)  # (1,)
        self.eos_token = torch.tensor([self.src_tokenizer.token_to_id("[EOS]")], dtype=torch.int64, device=DEVICE)  # (1,)
        self.pad_token = torch.tensor([self.src_tokenizer.token_to_id("[PAD]")], dtype=torch.int64, device=DEVICE)  # (1,)
        self.model.eval()
        
    def translate(self, source_text: str, max_len: int, strategy=SamplingStrategy.GREEDY) -> tuple[str, str]:
        dataset = ParallelTextDataset(
            dataset=[{"en": source_text, "am":"" }], 
            src_tokenizer=self.src_tokenizer,
            tgt_tokenizer=self.tgt_tokenizer
        )
        batch_iterator = iter( dataset.batch_iterator(1))
        batch = next(batch_iterator)
        
        encoder_input = batch["encoder_input"].to(DEVICE)       # (1, seq_len) 
        encoder_mask = batch["encoder_mask"].to(DEVICE)         # (1, 1, 1, seq_len) 
        decoder_mask = batch["decoder_mask"].to(DEVICE)         # (1, 1, seq_len, seq_len) 
                        
        return self.translate_raw(encoder_input, encoder_mask, decoder_mask, max_len, strategy)

    @torch.no_grad()
    def translate_raw(self, encoder_input: torch.Tensor, encoder_mask: torch.Tensor, decoder_mask: torch.Tensor, max_len: int, strategy=SamplingStrategy.GREEDY) -> str:        
        # Reuse the encoder output to generate subsequent tokens using the decoder
        encoder_output = self.model.encode(encoder_input, encoder_mask)
        
        # Initialize the decoder input with the start of sentence token
        decoder_input = torch.concat([
            self.sos_token,                                                                      # (1,)
            torch.tensor([self.pad_token] * (SEQ_LEN - 1), dtype=torch.int64, device=DEVICE)     # (max_seq_len - 1,)
        ]).unsqueeze(0)   
        
        count = 1
                
        # While length of the generated text is less than max_len and the 
        # last generated token is different from eos_token
        # keep generating the next token
        while count < max_len and decoder_input[:, -1] != self.eos_token:
            # Calculate the output of the decoder
            decoder_output = self.model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            
            # Retrieve the current probability distribution of the tokens in the vocabulary
            token_probab = self.model.project(decoder_output)                                   # (1, seq_len, tgt_vocab_size)
            
            # Retrieve the probability distribution over vocab_size on the `count`th  
            # position in the second dimension(`seq_len`)
            # (1, seq_len, tgt_vocab_size) --> (1, tgt_vocab_size)
            next_token_prob = token_probab[:, count]                                            # (1, tgt_vocab_size)
            
            """
                Retrieve the next token based on its probability and the strategy selected
            """
            if strategy == SamplingStrategy.GREEDY:
                # Retrieve the token with the highest probability
                _, next_token = torch.max(next_token_prob, dim=1)                   # value=tensor([max_prob]), indice=tensor([index{token_id}])
            elif strategy == SamplingStrategy.TOP_K_RANDOM:
                _, topk_indices = torch.topk(token_probab, k=self.top_k)
                next_token = random.choice(topk_indices.squeeze().tolist())
            else:
                raise Exception("Unknown sampling strategy passed")
            
            # Modify to the decoder input for the next iteration
            decoder_input[0, count] = next_token.item()
            count += 1
        
        # Remove the batch dimension 
        decoder_input = decoder_input.squeeze(0)                                    # torch.tensor([...]) with shape tensor.Size([max_len])
    
        return self.tgt_tokenizer.decode(decoder_input.detach().cpu().tolist())
    

class TranslationApp(QWidget):
    def __init__(self, inference_engine: MtInferenceEngine):
        super().__init__()
        self.init_ui()
        self.inference_engine = inference_engine

    def init_ui(self):
        self.setWindowTitle('Translation App')
        self.setGeometry(100, 100, 600, 400)

        self.input_label = QLabel('Input(English):')
        self.input_label.setFont(QFont('Arial', 12))  
        self.input_textbox = QLineEdit(self)
        self.input_textbox.setFont(QFont('Nyala', 14))  
        self.input_textbox
        self.input_textbox.returnPressed.connect(self.on_translate_button_clicked)

        self.output_label = QLabel('Output(Amharic):')
        self.output_label.setFont(QFont('Arial', 12))  
        self.output_textbox = QLineEdit(self)
        self.output_textbox.setReadOnly(True)
        self.output_textbox.setFont(QFont('Nyala', 14)) 

        self.translate_button = QPushButton('Translate', self)
        self.translate_button.setFont(QFont('Arial', 12))
        self.translate_button.setFixedWidth(self.width() // 2)
        self.translate_button.clicked.connect(self.on_translate_button_clicked)

        layout = QGridLayout()
        layout.addWidget(self.input_label, 0, 0)
        layout.addWidget(self.input_textbox, 0, 1)
        layout.addWidget(self.output_label, 1, 0)
        layout.addWidget(self.output_textbox, 1, 1)
        layout.addWidget(self.translate_button, 2, 1, 1, 2)

        self.setLayout(layout)

    def on_translate_button_clicked(self):
        input_text = self.input_textbox.text()
        prediction: str = self.inference_engine.translate(input_text, 10)
        self.output_textbox.setText(prediction)

if __name__ == '__main__':
    vocab_size = 6000
    app = QApplication(sys.argv)
    state = torch.load("./models/tmodel_v1.pt", map_location=DEVICE)
    
    src_tokenizer: Tokenizer = get_tokenizer(SRC_LANG, "tokenizer-en-v3.5-6k.json")
    tgt_tokenizer: Tokenizer = get_tokenizer(TGT_LANG, "tokenizer-am-v3.5-6k.json")

    print(src_tokenizer.decode(src_tokenizer.encode("What is your name?").ids))
    
    model = get_model(vocab_size, vocab_size).to(DEVICE)
    model.load_state_dict(state["model_state_dict"])
    
    model.eval()
    inference_engine = MtInferenceEngine(model, src_tokenizer, tgt_tokenizer)
    
    translation_app = TranslationApp(inference_engine)
    translation_app.show()
    sys.exit(app.exec_())