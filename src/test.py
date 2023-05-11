# from src.paraphrase import get_para_dataset_sent_level
from transformers import GPT2TokenizerFast, AutoTokenizer
import json
import nltk
import string
import pandas as pd
nltk.download('punkt')

# tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

# paraphrase_data = get_para_dataset_sent_level(
#     orig_dataset_path = "data/combined_data_base.json",
#     control_length = False,
#     n = 1,
#     dummy = False,
#     tokenizer = tokenizer,
#     sleep = 0
# )
# json.dump(paraphrase_data, open("data/temporary_dataset_files/combined_data_base_paraphrase_sent_level.json", "w"))

# original = "This all suite hotel is flawless. Around the holidays season, the hotel was well decorated for X-mas with lovely Palm trees and Christmas lights. This hotel is so close to the beach and is perfectly located for peace and quiet. The airport and Duval street is only a short shuttle away, but downtown is quite far away. Morden rooms with great size and a lot of storage space. The beds are super comfortable. The bathroom is also pretty large with a nice vanity area. The staff were even willing to help with last minute late checkout request. The restaurant provides great food, but it closes too early for us. The hotel actually has bikes to rent. Parking close-by can be very pricey per day. The elevators feel not safe."

# paraphrase = "  This hotel, which has only suites, is perfect. At the time of the festive season, the hotel was beautifully adorned with Christmas decorations, including attractive palm trees and twinkling lights. This hotel is situated in a tranquil spot, just a short distance away from the beach. The airport and Duval Street are within a brief shuttle ride, however downtown is quite distant. Modern rooms with ample dimensions and plentiful storage. The beds are extremely cozy. The restroom is quite spacious with a pleasant vanity area. The personnel were even accommodating to assist with a last minute request for a late checkout. The restaurant has delicious food, but it shuts down before we would like. The hotel provides bicycles for rent. Parking in the vicinity can be quite expensive on a daily basis. The elevators do not seem secure."


# print(f"Original length = {len(tokenizer(original)['input_ids'])}")
# print(f"para length = {len(tokenizer(paraphrase)['input_ids'])}")

records = []

indexed_data = json.load(open("data/temporary_dataset_files/base_split/indexed_split_sent_dataset_complete.json"))

for example_no, example in enumerate(indexed_data):
    split = example['split']
    entity_a = example['entity_a']
    entity_a_name = example['entity_a_name']
    entity_b = example['entity_b']
    entity_b_name = example['entity_b_name']
    
    for summ_type in ['refs_a', 'refs_b', 'refs_comm']:
        sent_dict = example[summ_type][0]
        
        for sent_id in sent_dict:
            complex_sent = sent_dict[sent_id]['sentence']
            simple_sent_dict = sent_dict[sent_id]['simple_sents'] if len(sent_dict[sent_id]['simple_sents']) > 0 else {"":""}
            
            for simple_sent_id in simple_sent_dict:
                simple_sent = simple_sent_dict[simple_sent_id]
                
                records.append((example_no, split, entity_a, entity_a_name, entity_b, entity_b_name, summ_type, 0, sent_id, complex_sent, simple_sent_id, simple_sent))

df = pd.DataFrame.from_records(
    data=records,
    columns=[
        'example_no', 'split', 'entity_a', 'entity_a_name', 'entity_b', 'entity_b_name', 'summ_type', 'summ_no', 'sent_id', 'complex_sent', 'simple_sent_id', 'simple_sent'
    ]
)

df.to_csv("data/temporary_dataset_files/negation_annotation.csv")
                
    
                
            