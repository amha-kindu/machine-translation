import torch

torch.manual_seed(3000)
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    torch.cuda.manual_seed_all(3000) 
else:
    DEVICE = torch.device('cpu')

BATCH_SIZE = 32
EPOCHS = 50
LR = 2e-04
SEQ_LEN=52
D_MODEL = 512
N_BLOCKS = 6
HEADS = 8
DROPOUT = 0.1
DFF = 2048
SRC_LANG = "en"
TGT_LANG = "am"
MODEL_FOLDER = "models"
MODEL_BASENAME = "tmodel_"
PRELOAD_MODEL_SUFFIX = ""
TOKENIZER_FOLDER = "tokenizers"
TOKENIZER_BASENAME = "tokenizer_{0}.json"
TB_LOG_DIR = "logs/tmodel"
DATASET_PATH = "data/languages.json"
    
def get_weights_file_path(suffix: str):
    return f"{MODEL_FOLDER}/{MODEL_BASENAME}{suffix}.pt"