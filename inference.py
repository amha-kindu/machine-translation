import sys
import torch
from config import *
from PyQt5.QtGui import QFont
from tokenizers import Tokenizer
from model import MtTransformerModel
from dataset import ParallelTextDataset
from train import get_tokenizer
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QGridLayout


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
       
    @torch.no_grad() 
    def translate(self, source_text: str, max_len: int) -> str:
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
                        
        # yield self.translate_raw(encoder_input, encoder_mask, decoder_mask, max_len)
        sos_idx = self.tgt_tokenizer.token_to_id('[SOS]')
        eos_idx = self.tgt_tokenizer.token_to_id('[EOS]')

        # Precompute the encoder output and reuse it for every step
        encoder_output = model.encode(encoder_input, encoder_mask)
        
        # Initialize the decoder input with the sos token
        next_token = sos_idx
        decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(encoder_input).to(DEVICE)
        while decoder_input.size(1) < max_len and next_token != eos_idx:
            # Build required masking for decoder input
            decoder_mask = ParallelTextDataset.lookback_mask(decoder_input.size(1)).type_as(encoder_mask).to(DEVICE)

            # Calculate output of decoder
            decoder_out = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)       # (1, seq_len, d_model)
            
            # Retrieve the embedded vector form of the last token
            last_token_vec = decoder_out[:, -1]                         # (1, d_model)
            
            # Get the model's raw output(logits)
            last_token_logits = model.project(last_token_vec)           # (1, d_model) --> (1, tgt_vocab_size)
            
            # Evaluate the probability distribution across the vocab_size 
            # dimension using softmax
            last_token_prob = torch.softmax(last_token_logits, dim=1)
            
            # Greedily pick the one with the highest probability
            _, next_token = torch.max(last_token_prob, dim=1)
            
            # Append to the decoder input for the subsequent iterations
            decoder_input = torch.cat([
                decoder_input, 
                torch.empty(1, 1).type_as(encoder_input).fill_(next_token.item()).to(DEVICE)
            ], dim=1)

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
    vocab_size = 20000
    app = QApplication(sys.argv)

    state = torch.load("./models/tmodel-en-am-v1-20k.pt", map_location=DEVICE)
    model = MtTransformerModel.build(vocab_size, vocab_size, state).to(DEVICE)

    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {DEVICE}")
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"Model Size(MB): {total_params * 4 / (1024 ** 2):.2f}MB")
    
    src_tokenizer: Tokenizer = get_tokenizer(SRC_LANG, "tokenizer-en-v3.5-20k.json")
    tgt_tokenizer: Tokenizer = get_tokenizer(TGT_LANG, "tokenizer-am-v3.5-20k.json")
    inference_engine = MtInferenceEngine(model, src_tokenizer, tgt_tokenizer)

    translation_app = TranslationApp(inference_engine)
    translation_app.show()
    sys.exit(app.exec_())