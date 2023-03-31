import json
import en_core_web_lg
import pandas as pd


# doc = nlp("The United Nations is headquartered in the Place d'Armes in Switzerland, where five of its member nations - the United States, Russia, China, France, and the United Kingdom - have diplomatic missions. The Secretary General is Armando Gutierrez.")


# print([(w.text, w.pos_) for w in doc])
# matcher = "l'Hotel de Geneva"

# for ent in doc.ents:
#     print(ent.text, ent.start_char, ent.end_char, ent.label_)
    

# region Entity Replacement | NER Library: Spacy | Similarity Library: Spacy
# nlp = en_core_web_lg.load()

# combined_data = json.load(open("data/combined_data.json", 'r'))
# new_combined_data = []


# summary_types = {'refs_a', 'refs_b', 'refs_comm', 'gen_a', 'gen_b', 'gen_comm', 'source_reviews_a', 'source_reviews_b'}

# entity_mentions = dict()
# idx = 0
# threshold = 0.629
# for pair_no, entity_pair in enumerate(combined_data):
#     hotel_a_name = entity_pair['entity_a_name']
#     hotel_b_name = entity_pair['entity_b_name']
#     for field in entity_pair:
#         if field in summary_types:
#             summaries = entity_pair[field]
#             if type(summaries) is not list:
#                 summaries = [summaries]
            
            
            
#             for summary_no, summary in enumerate(summaries):
#                 doc_summ = nlp(summary)
#                 for ent in reversed(doc_summ.ents:
#                     entity_mention = dict()
#                     entity_mention['entity_pair'] = pair_no
#                     entity_mention['split'] = entity_pair['split']
#                     entity_mention['entity_a_name'] = hotel_a_name
#                     entity_mention['entity_b_name'] = hotel_b_name
#                     entity_mention['summary_type'] = field
#                     entity_mention['summary_no'] = summary_no
#                     entity_mention['mention'] = ent.text
#                     entity_mention['start_char'] = ent.start_char
#                     entity_mention['end_char'] = ent.end_char
#                     entity_mention['ent_type'] = ent.label_
                    
#                     if '_a' in field:
#                         doc_hotel = nlp(hotel_a_name)
#                         entity_mention['similarity_a'] = ent.similarity(doc_hotel)
#                         entity_mention['similarity_b'] = -1.0
#                     elif '_b' in field:
#                         doc_hotel = nlp(hotel_b_name)
#                         entity_mention['similarity_b'] = ent.similarity(doc_hotel)
#                         entity_mention['similarity_a'] = -1.0
#                     else:
#                         doc_hotel = nlp(hotel_a_name)
#                         entity_mention['similarity_a'] = ent.similarity(doc_hotel)
#                         doc_hotel = nlp(hotel_b_name)
#                         entity_mention['similarity_b'] = ent.similarity(doc_hotel)
                    
#                     entity_mentions[idx] = entity_mention
#                     idx += 1

# df = pd.DataFrame.from_dict(entity_mentions, orient = 'index')

# df.to_csv("data/entity_mentions.csv")
                    
# endregion 
            

# region Entity Replacement | NER Library: Spacy | Similarity Library: Spacy | Token Recostruction: Merge Entities

nlp = en_core_web_lg.load()
nlp.add_pipe("merge_entities")
                                    
combined_data = json.load(open("data/combined_data.json", 'r'))
new_combined_data = []


summary_types = {'refs_a', 'refs_b', 'refs_comm', 'gen_a', 'gen_b', 'gen_comm', 'source_reviews_a', 'source_reviews_b'}

entity_mentions = dict()
idx = 0
threshold = 0.629
for pair_no, entity_pair in enumerate(combined_data):
    hotel_a_name = entity_pair['entity_a_name']
    hotel_b_name = entity_pair['entity_b_name']
    for field in entity_pair:
        if field in summary_types:
            summaries = entity_pair[field]
            if type(summaries) is not list:
                summaries = [summaries]
            new_summaries = []
            for summary_no, summary in enumerate(summaries):
                new_summ_tokens = []
                doc_summ = nlp(summary)
                for ent in doc_summ:
                    if not ent.ent_type_:
                        new_summ_tokens.append(ent.text)
                        continue
                        
                    entity_mention = dict()
                    entity_mention['entity_pair'] = pair_no
                    entity_mention['split'] = entity_pair['split']
                    entity_mention['entity_a_name'] = hotel_a_name
                    entity_mention['entity_b_name'] = hotel_b_name
                    entity_mention['summary_type'] = field
                    entity_mention['summary_no'] = summary_no
                    entity_mention['mention'] = ent.text
                    # entity_mention['start_char'] = ent.start_char
                    # entity_mention['end_char'] = ent.end_char
                    entity_mention['ent_type'] = ent.ent_type_
                    
                    if '_a' in field:
                        doc_hotel = nlp(hotel_a_name)
                        entity_mention['similarity_a'] = ent.similarity(doc_hotel)
                        entity_mention['similarity_b'] = -1.0
                        similarity = entity_mention['similarity_a']
                    elif '_b' in field:
                        doc_hotel = nlp(hotel_b_name)
                        entity_mention['similarity_b'] = ent.similarity(doc_hotel)
                        entity_mention['similarity_a'] = -1.0
                        similarity = entity_mention['similarity_b']
                    else:
                        doc_hotel = nlp(hotel_a_name)
                        entity_mention['similarity_a'] = ent.similarity(doc_hotel)
                        doc_hotel = nlp(hotel_b_name)
                        entity_mention['similarity_b'] = ent.similarity(doc_hotel)
                        similarity = max(entity_mention['similarity_a'], entity_mention['similarity_b'])
                    
                    if similarity > threshold:
                        new_summ_tokens.append("this hotel")
                    else:
                        new_summ_tokens.append(ent.text)
                    
                    entity_mentions[idx] = entity_mention
                    idx += 1
                new_summ = " ".join(new_summ_tokens)
                new_summaries.append(new_summ)
            
            if len(new_summaries) == 1:
                new_summaries = new_summaries[0]
            entity_pair[field] = new_summaries
    new_combined_data.append(entity_pair)
            

json.dump(new_combined_data, open("data/combined_data_masked_spacyner_spacysim", 'w'))
df = pd.DataFrame.from_dict(entity_mentions, orient = 'index')

df.to_csv("data/entity_mentions.csv")
                    

# endregion
            

# region Entity Replace
                                    
                                    
