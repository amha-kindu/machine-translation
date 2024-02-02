from easygoogletranslate import EasyGoogleTranslate


class ParallelTextAugmenter:
    def __init__(self, src_lang: str, tgt_lang: str) -> None:
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.timeout = 10000
        self.translator = EasyGoogleTranslate(
            source_language=src_lang,
            target_language=tgt_lang,
            timeout=self.timeout
        )    
        self.back_translator = EasyGoogleTranslate(
            source_language=tgt_lang,
            target_language=src_lang,
            timeout=self.timeout
        )    
        
    def augment(self, src_text: str, tgt_text: str) -> list[dict]:
        tgt_trans = self.translator.translate(src_text, timeout=self.timeout)
        src_trans = self.back_translator.translate(tgt_text, timeout=self.timeout)
        return [
            {   f"{self.src_lang}": src_text,   f"{self.tgt_lang}": tgt_trans   },
            {   f"{self.src_lang}": src_trans,  f"{self.tgt_lang}": tgt_text    },
            {   f"{self.src_lang}": src_trans,  f"{self.tgt_lang}": tgt_trans   },
        ]
        

if __name__ == "__main__":
    import os
    import json
    
    augmenter = ParallelTextAugmenter(src_lang='en', tgt_lang='am')

    with open("data/languages.json", 'r', encoding='utf-8') as data:
        dataset = json.load(data)
    
    original_size = len(dataset)
    
    i = 1
    augmented_text_pairs = []
    for data in dataset:
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"Processing Text Pair-{i} ...")
        augmented_text_pairs.append(data)
        augmented_text_pairs.extend(
            augmenter.augment(src_text=data['en'], tgt_text=data['am'])
        )
        i += 1
        
    augmented_size = len(augmented_text_pairs)
    
    print(f"Original Dataset Size:  {original_size}\n")
    print(f"Augmented Dataset Size: {augmented_size}\n")
    print(f"Dataset Increase: {augmented_size - original_size}")
        
    json.dump(augmented_text_pairs, open("data/languages_augmented.json", "w"), indent=4)
