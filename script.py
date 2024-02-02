import json, re

with open(".\copus00615.1.txt", 'r', encoding='utf-8') as file:
    data = file.readlines()

sent_pairs = []
for i in range(0, len(data) - 1, 2):
    sent_pairs.append({
        "en": data[i].strip(),
        "am": data[i+1].strip()
    })

json.dump(sent_pairs, open("data/parallel-corpus-en-am-3-part-II.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

# def merge_parallel_corpus(en_file_path, am_file_path, output_json_path):
#     # Read English and Amharic sentences from the respective files
#     with open(en_file_path, 'r', encoding='utf-8') as en_file, \
#          open(am_file_path, 'r', encoding='utf-8') as am_file:

#         en_sentences = en_file.readlines()
#         am_sentences = am_file.readlines()

#     # Check if the number of sentences in both files is the same
#     if len(en_sentences) != len(am_sentences):
#         raise ValueError("Number of sentences in English and Amharic files do not match.")

#     # Create a list of dictionaries, each containing an English and Amharic sentence pair
#     merged_corpus = [{"en": en.strip(), "am": am.strip()} for en, am in zip(en_sentences, am_sentences)]

#     # Write the merged corpus to a JSON file
#     with open(output_json_path, 'w', encoding='utf-8') as json_file:
#         json.dump(merged_corpus, json_file, ensure_ascii=False, indent=4)

# # Example usage:
# en_file_path = r'C:\Users\dell\Downloads\English\English.txt'
# am_file_path = r'C:\Users\dell\Downloads\Amharic\Amharic.txt'
# output_json_path = 'data\languages_new.json'

# merge_parallel_corpus(en_file_path, am_file_path, output_json_path)
