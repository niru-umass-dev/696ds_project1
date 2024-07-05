import numpy as np
import pandas as pd
import torch
import json
# import nltk
from transformers import pipeline


######################################################################################################################

# RUN ON A PARTICULAR DATASET

class SummNLI:

    def __init__(self, data_path: str = 'data/combined_data.json', data_type: str = 'base',
                negation: bool = False):

        # ROBERTA-LARGE-MNLI
        self.pipe = pipeline("text-classification", model='roberta-large-mnli', return_all_scores=True, device= 0 if torch.cuda.is_available() else 'cpu')
        self.label_mapping = ['contradiction', 'neutral', 'entailment']

        self.data = json.load(open(data_path, 'r'))
        self.data_type = data_type
        self.negation = negation

    def compute(self):
        data = self.data
        label_mapping = self.label_mapping

        # RUN NLI FOR CONTRAST
        cont_labels = []
        cont_probs = []
        cont_labels_rev = []
        cont_probs_rev = []
        sent_pair_1 = []
        sent_pair_2 = []
        sent1_ent = []
        sent2_ent = []
        sample = []

        if self.negation == True:
            for d in range(len(data)):
                # ref summaries just first reference
                ref_a_sum = data[d]['refs_a'][0]
                ref_a_sum_neg = data[d]['refs_b'][0]

                # ref aggregations
                ref_sent_p1, ref_sent_p2, ref_sent1_ent, ref_sent2_ent = self.cont_helper(ref_a_sum,
                                                                                            ref_a_sum_neg)

                sent_pair_1 += ref_sent_p1
                sent_pair_2 += ref_sent_p2
                sent1_ent += ref_sent1_ent
                sent2_ent += ref_sent2_ent

                sample += [d] * len(sent1_ent)

                cont_label, cont_prob = self.compute_NLI(ref_sent_p1, ref_sent_p2)

                cont_label_rev, cont_prob_rev = self.compute_NLI(ref_sent_p1, ref_sent_p2, rev=True)

                cont_labels += cont_label
                cont_probs.append(cont_prob)

                cont_labels_rev += cont_label_rev
                cont_probs_rev.append(cont_prob_rev)

                print(d)

        else:
            for d in range(len(data)):
                # ref summaries just first reference
                ref_a_sum = data[d]['refs_a'][0]
                ref_b_sum = data[d]['refs_b'][0]
                ref_comm_sum = data[d]['refs_comm'][0]

                # ref aggregations
                ref_sent_p1, ref_sent_p2, ref_sent1_ent, ref_sent2_ent = self.cont_helper(ref_a_sum,
                                                                                            ref_b_sum,
                                                                                            ref_comm_sum)

                sent_pair_1 += ref_sent_p1
                sent_pair_2 += ref_sent_p2
                sent1_ent += ref_sent1_ent
                sent2_ent += ref_sent2_ent

                sample += [d] * len(sent1_ent)

                cont_label, cont_prob = self.compute_NLI(ref_sent_p1, ref_sent_p2)

                cont_label_rev, cont_prob_rev = self.compute_NLI(ref_sent_p1, ref_sent_p2, rev=True)

                cont_labels += cont_label
                cont_probs.append(cont_prob)

                cont_labels_rev += cont_label_rev
                cont_probs_rev.append(cont_prob_rev)

                print(d)

        cont = [j for i in cont_probs for j in np.array(i)[:, label_mapping.index("contradiction")]]
        neut = [j for i in cont_probs for j in np.array(i)[:, label_mapping.index("neutral")]]
        ent = [j for i in cont_probs for j in np.array(i)[:, label_mapping.index("entailment")]]
        cont_rev = [j for i in cont_probs_rev for j in np.array(i)[:, label_mapping.index("contradiction")]]
        neut_rev = [j for i in cont_probs_rev for j in np.array(i)[:, label_mapping.index("neutral")]]
        ent_rev = [j for i in cont_probs_rev for j in np.array(i)[:, label_mapping.index("entailment")]]

        cont_df = pd.DataFrame(
            {'Sentence1': sent_pair_1,
                'Sentence2': sent_pair_2,
                'Sent1_entity': sent1_ent,
                'Sent2_entity': sent2_ent,
                'Sample': sample,
                '1_2_neut': neut,
                '1_2_cont': cont,
                '1_2_ent': ent,
                '1_2_Label': cont_labels,
                '2_1_neut': neut_rev,
                '2_1_cont': cont_rev,
                '2_1_ent': ent_rev,
                '2_1_Label': cont_labels_rev,
                })

        # SAVE CSV FILE

        if self.negation == True:
            cont_df.to_csv(f'data/results/sentpairs_nli_contrast_{self.data_type}.csv', index=None, header=True)

        else:
            cont_df.to_csv(f'data/results/sentpairs_nli_contrast_{self.data_type}.csv', index=None, header=True)

        ##############################################################################################################

    # NLI to get labels and scores

    def compute_NLI(self, sent1, sent2, rev=False):

        labels = []
        prob = []
        combined_list = []

        if rev == True:
            for s1, s2 in zip(sent2, sent1):
                combined_list.append(s1 + "<SEP>" + s2)
        else:
            for s1, s2 in zip(sent1, sent2):
                combined_list.append(s1 + "<SEP>" + s2)

        results = self.pipe(combined_list)

        for i in results:
            result = {d['label']: d['score'] for d in i}
            labels.append(max(result, key=result.get))
            prob.append(list(result.values()))

        return labels, prob

    # HELPER METHOD FOR MAIN FUNCTION HELPFUL CONTRAST
    def cont_helper(self, a_sum, b_sum, comm_sum=None):

        sum_types = []
        sent_p1 = []
        sent_p2 = []
        sent1_ent = []
        sent2_ent = []

        count = 0
        for i in a_sum:
            for j in b_sum:
                sent_p1.append(i)
                sent_p2.append(j)
                sent1_ent.append('a')
                sent2_ent.append('b')
                count += 1

        # if only give 2 summaires
        if comm_sum == None:
            return sent_p1, sent_p2, sent1_ent, sent2_ent

        for i in a_sum:
            for j in comm_sum:
                sent_p1.append(i)
                sent_p2.append(j)
                sent1_ent.append('a')
                sent2_ent.append('comm')
                count += 1

        for i in b_sum:
            for j in comm_sum:
                sent_p1.append(i)
                sent_p2.append(j)
                sent1_ent.append('b')
                sent2_ent.append('comm')
                count += 1

        return sent_p1, sent_p2, sent1_ent, sent2_ent