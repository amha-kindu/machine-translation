import json
from config import DATASET_PATH, SRC_LANG, TGT_LANG
from preprocessor import AmharicPreprocessor, EnglishPreprocessor
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors



def train(dataset: list[dict], lang: str, vocab_size: int) -> Tokenizer:
    preprocessor = AmharicPreprocessor(None) if lang == "am" else EnglishPreprocessor(None)
    all_sentences = [preprocessor.preprocess(item[lang], encode=False) for item in dataset]
    
    # Initialize a tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token='[UNK]'))
    
    # Customize pre-tokenization and decoding
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(use_regex=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel()
    
    if lang == "en":
        alphabet = list(filter(lambda x: x in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" ,pre_tokenizers.ByteLevel.alphabet()))
        print(alphabet)
    elif lang == "am":
        # Amharic chars and the arabic numerals
        alphabet = ["ሀ", "ሁ", "ሂ", "ሃ", "ሄ", "ህ", "ሆ", "ሇ", "ለ", "ሉ", "ሊ", "ላ", "ሌ", "ል", "ሎ", "ሏ", "ሐ", "ሑ", "ሒ", "ሓ", "ሔ", "ሕ", "ሖ", "ሗ", "መ", "ሙ", "ሚ", "ማ", "ሜ", "ም", "ሞ", "ሟ", "ሠ", "ሡ", "ሢ", "ሣ", "ሤ", "ሥ", "ሦ", "ሧ", "ረ", "ሩ", "ሪ", "ራ", "ሬ", "ር", "ሮ", "ሯ", "ሰ", "ሱ", "ሲ", "ሳ", "ሴ", "ስ", "ሶ", "ሷ", "ሸ", "ሹ", "ሺ", "ሻ", "ሼ", "ሽ", "ሾ", "ሿ", "ቀ", "ቁ", "ቂ", "ቃ", "ቄ", "ቅ", "ቆ", "ቇ", "ቈ", "ቊ", "ቋ", "ቌ", "ቍ", "በ", "ቡ", "ቢ", "ባ", "ቤ", "ብ", "ቦ", "ቧ", "ቨ", "ቩ", "ቪ", "ቫ", "ቬ", "ቭ", "ቮ", "ቯ", "ተ", "ቱ", "ቲ", "ታ", "ቴ", "ት", "ቶ", "ቷ", "ቸ", "ቹ", "ቺ", "ቻ", "ቼ", "ች", "ቾ", "ቿ", "ኀ", "ኁ", "ኂ", "ኃ", "ኄ", "ኅ", "ኆ", "ኇ", "ኈ", "ኊ", "ኋ", "ኌ", "ኍ", "ነ", "ኑ", "ኒ", "ና", "ኔ", "ን", "ኖ", "ኗ", "ኘ", "ኙ", "ኚ", "ኛ", "ኜ", "ኝ", "ኞ", "ኟ", "አ", "ኡ", "ኢ", "ኣ", "ኤ", "እ", "ኦ", "ኧ", "ከ", "ኩ", "ኪ", "ካ", "ኬ", "ክ", "ኮ", "ኯ", "ኰ", "ኲ", "ኳ", "ኴ", "ኵ", "ኸ", "ኹ", "ኺ", "ኻ", "ኼ", "ኽ", "ኾ", "ወ", "ዉ", "ዊ", "ዋ", "ዌ", "ው", "ዎ", "ዐ", "ዑ", "ዒ", "ዓ", "ዔ", "ዕ", "ዖ", "ዘ", "ዙ", "ዚ", "ዛ", "ዜ", "ዝ", "ዞ", "ዟ", "ዠ", "ዡ", "ዢ", "ዣ", "ዤ", "ዥ", "ዦ", "ዧ", "የ", "ዩ", "ዪ", "ያ", "ዬ", "ይ", "ዮ", "ደ", "ዱ", "ዲ", "ዳ", "ዴ", "ድ", "ዶ", "ዷ", "ጀ", "ጁ", "ጂ", "ጃ", "ጄ", "ጅ", "ጆ", "ጇ", "ገ", "ጉ", "ጊ", "ጋ", "ጌ", "ግ", "ጎ", "ጏ", "ጠ", "ጡ", "ጢ", "ጣ", "ጤ", "ጥ", "ጦ", "ጧ", "ጨ", "ጩ", "ጪ", "ጫ", "ጬ", "ጭ", "ጮ", "ጯ", "ጰ", "ጱ", "ጲ", "ጳ", "ጴ", "ጵ", "ጶ", "ጷ", "ጸ", "ጹ", "ጺ", "ጻ", "ጼ", "ጽ", "ጾ", "ጿ", "ፀ", "ፁ", "ፂ", "ፃ", "ፄ", "ፅ", "ፆ", "ፇ", "ፈ", "ፉ", "ፊ", "ፋ", "ፌ", "ፍ", "ፎ", "ፏ", "ፐ", "ፑ", "ፒ", "ፓ", "ፔ", "ፕ", "ፖ"] #, "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        
        # Add punctuations and special symbols
        # alphabet += ["!", "@", "#", "$", "%", "^", "«", "»", "&", "?", "*", "(", ")", "…", "[", "]", "{", "}", ";", "“", "”", "›", "’", "‘", '"', "'", ":", ",", ".", "‹", "/", "<", ">", "\\", "\\", "|", "`", "´", "~", "-", "=", "+", "፡", "።", "፤", ";", "፦", "፥", "፧", "፨", "፠", "፣"]
    else:
        raise UnicodeError("Unrecognized language")
        
    # Train on dataset
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, 
        special_tokens=["[UNK]", "[SOS]", "[EOS]", "[PAD]"], 
        initial_alphabet=alphabet,
        min_frequency=2,
        show_progress=True
    )        
    tokenizer.train_from_iterator(all_sentences, trainer=trainer)
    
    return tokenizer


if __name__ == "__main__":
    VOCAB_SIZE = 20000
    with open(DATASET_PATH, 'r', encoding='utf-8') as data:
        dataset = json.load(data)
    
    src_tokenizer = train(dataset, SRC_LANG, VOCAB_SIZE)
    tgt_tokenizer = train(dataset, TGT_LANG, VOCAB_SIZE)
    
    src_tokenizer.save(f"./tokenizers/tokenizer-en-v3.5-{VOCAB_SIZE // 1000}k.json")
    tgt_tokenizer.save(f"./tokenizers/tokenizer-am-v3.5-{VOCAB_SIZE // 1000}k.json")