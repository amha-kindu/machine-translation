import torch
from pathlib import Path

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.manual_seed(3000)
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    torch.cuda.manual_seed_all(3000) 
else:
    DEVICE = torch.device('cpu')

def get_config():
    return {
        "batch_size": 8,
        "epochs": 20,
        "lr": 1e-04,
        "seq_len": 260,
        "d_model": 512,
        "n_blocks": 6,
        "heads": 8,
        "dropout": 0.1,
        "dff": 2048,
        "src_lang": "en",
        "tgt_lang": "am",
        "model_folder":"models",
        "model_basename": "tmodel_",
        "preload": False,
        "tokenizer_folder":"tokenizers",
        "tokenizer_basename": "tokenizer_{0}.json",
        "tb_log_dir": "logs/tmodel"
    }
    
def get_weights_file_path(config: dict, epoch: str):
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path(".")) / config["model_folder"] / model_filename