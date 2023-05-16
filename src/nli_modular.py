import numpy as np
import pandas as pd
import torch
import json
import nltk
from transformers import pipeline


######################################################################################################################

# RUN ON A PARTICULAR DATASET

class SummNLI:

    def __init__(self, data_path: str = 'data/combined_data.json', data_type: str = 'base', summ_type: str = 'ref',
                 summ_sent: bool = False, factuality: bool = False, negation: bool = False):

        # ROBERTA-LARGE-MNLI
        # self.pipe = pipeline("text-classification", model='roberta-large-mnli', return_all_scores=True, device=0)
        # self.label_mapping = ['contradiction', 'neutral', 'entailment']

        # ROBERTA-LARGE-SNLI-MNLI-FEVER-ANLI...
        # https://huggingface.co/ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli
        self.pipe = pipeline("text-classification", model='ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli', return_all_scores=True, device=0)
        self.label_mapping = ['entailment', 'neutral', 'contradiction']

        self.data = json.load(open(data_path, 'r'))
        self.data_type = data_type
        self.summ_type = summ_type
        self.summ_sent = summ_sent
        self.factuality = factuality
        self.negation = negation

    def compute(self):
        data = self.data
        label_mapping = self.label_mapping
        summ_to_sent = self.summ_sent

        # RUN NLI FOR CONTRAST

        if summ_to_sent == True:

            cont_labels = []
            cont_probs = []
            summ_1 = []
            sent_pair_2 = []
            summ1_ent = []
            sent2_ent = []
            sample = []
            sum_type = []

            if self.summ_type == 'ref':

                for d in range(len(data)):
                    # ref summaries just first reference
                    ref_a_sum = nltk.sent_tokenize(data[d]['refs_a'][0])
                    ref_b_sum = nltk.sent_tokenize(data[d]['refs_b'][0])
                    ref_comm_sum = nltk.sent_tokenize(data[d]['refs_comm'][0])

                    # ref aggregations
                    ref_sent_p1, ref_sent_p2, ref_sent1_ent, ref_sent2_ent, ref_count = self.summ_sent_cont_helper(
                        data[d]['refs_a'][0], data[d]['refs_b'][0], data[d]['refs_comm'][0], ref_a_sum, ref_b_sum,
                        ref_comm_sum, 'ref')

                    summ_1 += ref_sent_p1
                    sent_pair_2 += ref_sent_p2
                    summ1_ent += ref_sent1_ent
                    sent2_ent += ref_sent2_ent
                    sum_type += ref_count

                    sample += [d] * len(ref_count)

                    cont_label, cont_prob = self.compute_NLI(ref_sent_p1, ref_sent_p2)

                    cont_labels += cont_label
                    cont_probs.append(cont_prob)

                    print(d)

            elif self.summ_type == 'gen':

                for d in range(20, len(data)):
                    # gen summaries
                    gen_a_sum = nltk.sent_tokenize(data[d]['gen_a'])
                    gen_b_sum = nltk.sent_tokenize(data[d]['gen_b'])
                    gen_comm_sum = nltk.sent_tokenize(data[d]['gen_comm'])

                    # gen aggregations
                    gen_sent_p1, gen_sent_p2, gen_sent1_ent, gen_sent2_ent, gen_count = self.summ_sent_cont_helper(
                        data[d]['gen_a'], data[d]['gen_b'], data[d]['gen_comm'], gen_a_sum, gen_b_sum, gen_comm_sum,
                        'gen')

                    summ_1 += gen_sent_p1
                    sent_pair_2 += gen_sent_p2
                    summ1_ent += gen_sent1_ent
                    sent2_ent += gen_sent2_ent
                    sum_type += gen_count

                    sample += [d] * len(gen_count)

                    cont_label, cont_prob = self.compute_NLI(gen_sent_p1, gen_sent_p2)

                    cont_labels += cont_label
                    cont_probs.append(cont_prob)

                    print(d - 20)

            cont = [j for i in cont_probs for j in np.array(i)[:, label_mapping.index("contradiction")]]
            neut = [j for i in cont_probs for j in np.array(i)[:, label_mapping.index("neutral")]]
            ent = [j for i in cont_probs for j in np.array(i)[:, label_mapping.index("entailment")]]

            cont_df = pd.DataFrame(
                {'Summary1': summ_1,
                 'Sentence2': sent_pair_2,
                 'Summary1_entity': summ1_ent,
                 'Sent2_entity': sent2_ent,
                 'Sample': sample,
                 '1_2_neut': neut,
                 '1_2_cont': cont,
                 '1_2_ent': ent,
                 '1_2_Label': cont_labels,
                 'Type': sum_type
                 })

        elif summ_to_sent == False:

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
            sum_type = []

            if self.summ_type == 'ref':

                if self.negation == True:
                    for d in range(len(data)):
                        # ref summaries just first reference
                        ref_a_sum = nltk.sent_tokenize(data[d]['refs_a'][0])
                        ref_a_sum_neg = nltk.sent_tokenize(data[d]['refs_a_neg'])

                        # ref aggregations
                        ref_sent_p1, ref_sent_p2, ref_sent1_ent, ref_sent2_ent, ref_count = self.cont_helper(ref_a_sum,
                                                                                                             ref_a_sum_neg,
                                                                                                             'ref')

                        sent_pair_1 += ref_sent_p1
                        sent_pair_2 += ref_sent_p2
                        sent1_ent += ref_sent1_ent
                        sent2_ent += ref_sent2_ent
                        sum_type += ref_count

                        sample += [d] * len(ref_count)

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
                        ref_a_sum = nltk.sent_tokenize(data[d]['refs_a'][0])
                        ref_b_sum = nltk.sent_tokenize(data[d]['refs_b'][0])
                        ref_comm_sum = nltk.sent_tokenize(data[d]['refs_comm'][0])

                        # ref aggregations
                        ref_sent_p1, ref_sent_p2, ref_sent1_ent, ref_sent2_ent, ref_count = self.cont_helper(ref_a_sum,
                                                                                                             ref_b_sum,
                                                                                                             'ref',
                                                                                                             ref_comm_sum)

                        sent_pair_1 += ref_sent_p1
                        sent_pair_2 += ref_sent_p2
                        sent1_ent += ref_sent1_ent
                        sent2_ent += ref_sent2_ent
                        sum_type += ref_count

                        sample += [d] * len(ref_count)

                        cont_label, cont_prob = self.compute_NLI(ref_sent_p1, ref_sent_p2)

                        cont_label_rev, cont_prob_rev = self.compute_NLI(ref_sent_p1, ref_sent_p2, rev=True)

                        cont_labels += cont_label
                        cont_probs.append(cont_prob)

                        cont_labels_rev += cont_label_rev
                        cont_probs_rev.append(cont_prob_rev)

                        print(d)

            elif self.summ_type == 'gen':

                for d in range(20, len(data)):
                    # gen summaries
                    gen_a_sum = nltk.sent_tokenize(data[d]['gen_a'])
                    gen_b_sum = nltk.sent_tokenize(data[d]['gen_b'])
                    gen_comm_sum = nltk.sent_tokenize(data[d]['gen_comm'])

                    # gen aggregations
                    gen_sent_p1, gen_sent_p2, gen_sent1_ent, gen_sent2_ent, gen_count = self.cont_helper(gen_a_sum,
                                                                                                         gen_b_sum,
                                                                                                         'gen',
                                                                                                         gen_comm_sum)

                    sent_pair_1 += gen_sent_p1
                    sent_pair_2 += gen_sent_p2
                    sent1_ent += gen_sent1_ent
                    sent2_ent += gen_sent2_ent
                    sum_type += gen_count

                    sample += [d] * len(gen_count)

                    cont_label, cont_prob = self.compute_NLI(gen_sent_p1, gen_sent_p2)

                    cont_label_rev, cont_prob_rev = self.compute_NLI(gen_sent_p1, gen_sent_p2, rev=True)

                    cont_labels += cont_label
                    cont_probs.append(cont_prob)

                    cont_labels_rev += cont_label_rev
                    cont_probs_rev.append(cont_prob_rev)

                    print(d - 20)

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
                 'Type': sum_type
                 })

        # SAVE CSV FILE

        if self.negation == True:
            if self.summ_sent == True:
                cont_df.to_csv(f'data/results/sumsent_nli_contrast_{self.summ_type}_{self.data_type}.csv', index=None, header=True)
            else:
                cont_df.to_csv(f'data/results/sentpairs_nli_contrast_{self.summ_type}_{self.data_type}.csv', index=None, header=True)

        else:
            if self.summ_sent == True:
                cont_df.to_csv(f'data/results/sumsent_nli_contrast_{self.summ_type}_{self.data_type}.csv', index=None, header=True)
            else:
                cont_df.to_csv(f'data/results/sentpairs_nli_contrast_{self.summ_type}_{self.data_type}.csv', index=None, header=True)

            # IF THE DATA_TYPE IS PARAPHRASES, SKIP FACTUALITY SINCE WE DON'T HAVE SOURCE DATA PARAPHRASES
        if self.data_type == 'paraphrase' or self.data_type == 'selfparaphrase':
            return

        ##############################################################################################################

        if self.factuality == False:
            return

        else:

            # RUN NLI FOR FACTUAL CONSISTENCY AND POPULAR OPINION

            if summ_to_sent == True:

                fact_labels = []
                fact_probs = []
                source_1 = []
                sent_pair_2 = []
                sent1_source = []
                sent1_source_ent = []
                sent2_ent = []
                sample = []
                sum_type = []

                if self.summ_type == 'ref':

                    for d in range(len(data)):
                        # source reviews
                        source_a = data[d]['source_reviews_a']
                        source_b = data[d]['source_reviews_b']

                        # ref summaries just first reference
                        ref_a_sum = nltk.sent_tokenize(data[d]['refs_a'][0])
                        ref_b_sum = nltk.sent_tokenize(data[d]['refs_b'][0])
                        ref_comm_sum = nltk.sent_tokenize(data[d]['refs_comm'][0])

                        # ref aggregations
                        ref_sent_p1, ref_sent_p2, ref_sent1_source, ref_sent1_source_ent, ref_sent2_ent, ref_count = self.source_sent_fact_helper(
                            source_a, source_b, ref_a_sum, ref_b_sum, ref_comm_sum, 'ref')

                        source_1 += ref_sent_p1
                        sent_pair_2 += ref_sent_p2
                        sent1_source += ref_sent1_source
                        sent1_source_ent += ref_sent1_source_ent
                        sent2_ent += ref_sent2_ent
                        sum_type += ref_count

                        sample += [d] * len(ref_count)

                        fact_label, fact_prob = self.compute_NLI(ref_sent_p1, ref_sent_p2)

                        fact_labels += fact_label
                        fact_probs.append(fact_prob)

                        print(d)

                elif self.summ_type == 'gen':

                    for d in range(20, len(data)):
                        # source reviews
                        source_a = data[d]['source_reviews_a']
                        source_b = data[d]['source_reviews_b']

                        # gen summaries
                        gen_a_sum = nltk.sent_tokenize(data[d]['gen_a'])
                        gen_b_sum = nltk.sent_tokenize(data[d]['gen_b'])
                        gen_comm_sum = nltk.sent_tokenize(data[d]['gen_comm'])

                        # gen aggregations
                        gen_sent_p1, gen_sent_p2, gen_sent1_source, gen_sent1_source_ent, gen_sent2_ent, gen_count = self.source_sent_fact_helper(
                            source_a, source_b, gen_a_sum, gen_b_sum, gen_comm_sum, 'gen')

                        source_1 += gen_sent_p1
                        sent_pair_2 += gen_sent_p2
                        sent1_source += gen_sent1_source
                        sent1_source_ent += gen_sent1_source_ent
                        sent2_ent += gen_sent2_ent
                        sum_type += gen_count

                        sample += [d] * len(gen_count)

                        fact_label_gen, fact_prob_gen = self.compute_NLI(gen_sent_p1, gen_sent_p2)

                        fact_labels += fact_label_gen
                        fact_probs.append(fact_prob_gen)

                        print(d - 20)

                cont = [j for i in fact_probs for j in np.array(i)[:, label_mapping.index("contradiction")]]
                neut = [j for i in fact_probs for j in np.array(i)[:, label_mapping.index("neutral")]]
                ent = [j for i in fact_probs for j in np.array(i)[:, label_mapping.index("entailment")]]

                fact_pop_df = pd.DataFrame({'Source1': source_1,
                                            'Sentence2': sent_pair_2,
                                            'Sample': sample,
                                            'Source1_source': sent1_source,
                                            'Sent1_source_entity': sent1_source_ent,
                                            'Sent2_entity': sent2_ent,
                                            '1_2_neut': neut,
                                            '1_2_cont': cont,
                                            '1_2_ent': ent,
                                            '1_2_Label': fact_labels,
                                            'Type': sum_type
                                            })

            else:

                fact_labels = []
                fact_probs = []
                fact_labels_rev = []
                fact_probs_rev = []
                sent_pair_1 = []
                sent_pair_2 = []
                sent1_source = []
                sent1_source_ent = []
                sent2_ent = []
                sample = []
                sum_type = []

                if self.summ_type == 'ref':

                    for d in range(len(data)):
                        # source reviews
                        source_a = [nltk.sent_tokenize(i) for i in data[d]['source_reviews_a']]
                        source_b = [nltk.sent_tokenize(i) for i in data[d]['source_reviews_b']]

                        # ref summaries just first reference
                        ref_a_sum = nltk.sent_tokenize(data[d]['refs_a'][0])
                        ref_b_sum = nltk.sent_tokenize(data[d]['refs_b'][0])
                        ref_comm_sum = nltk.sent_tokenize(data[d]['refs_comm'][0])

                        # ref aggregations
                        ref_sent_p1, ref_sent_p2, ref_sent1_source, ref_sent1_source_ent, ref_sent2_ent, ref_count = self.fact_helper(
                            source_a, source_b, ref_a_sum, ref_b_sum, ref_comm_sum, 'ref')

                        sent_pair_1 += ref_sent_p1
                        sent_pair_2 += ref_sent_p2
                        sent1_source += ref_sent1_source
                        sent1_source_ent += ref_sent1_source_ent
                        sent2_ent += ref_sent2_ent
                        sum_type += ref_count

                        sample += [d] * len(ref_count)

                        fact_label, fact_prob = self.compute_NLI(ref_sent_p1, ref_sent_p2)

                        fact_labels += fact_label
                        fact_probs.append(fact_prob)

                        fact_label_rev, fact_prob_rev = self.compute_NLI(ref_sent_p1, ref_sent_p2, rev=True)

                        fact_labels_rev += fact_label_rev
                        fact_probs_rev.append(fact_prob_rev)

                        print(d)

                elif self.summ_type == 'gen':

                    for d in range(20, len(data)):
                        # source reviews
                        source_a = [nltk.sent_tokenize(i) for i in data[d]['source_reviews_a']]
                        source_b = [nltk.sent_tokenize(i) for i in data[d]['source_reviews_b']]

                        # gen summaries
                        gen_a_sum = nltk.sent_tokenize(data[d]['gen_a'])
                        gen_b_sum = nltk.sent_tokenize(data[d]['gen_b'])
                        gen_comm_sum = nltk.sent_tokenize(data[d]['gen_comm'])

                        # gen aggregations
                        gen_sent_p1, gen_sent_p2, gen_sent1_source, gen_sent1_source_ent, gen_sent2_ent, gen_count = self.fact_helper(
                            source_a, source_b, gen_a_sum, gen_b_sum, gen_comm_sum, 'gen')

                        sent_pair_1 += gen_sent_p1
                        sent_pair_2 += gen_sent_p2
                        sent1_source += gen_sent1_source
                        sent1_source_ent += gen_sent1_source_ent
                        sent2_ent += gen_sent2_ent
                        sum_type += gen_count

                        sample += [d] * len(gen_count)

                        fact_label_gen, fact_prob_gen = self.compute_NLI(gen_sent_p1, gen_sent_p2)

                        fact_labels += fact_label_gen
                        fact_probs.append(fact_prob_gen)

                        fact_label_rev, fact_prob_rev = self.compute_NLI(gen_sent_p1, gen_sent_p2, rev=True)

                        fact_labels_rev += fact_label_rev
                        fact_probs_rev.append(fact_prob_rev)

                        print(d - 20)

                cont = [j for i in fact_probs for j in np.array(i)[:, label_mapping.index("contradiction")]]
                neut = [j for i in fact_probs for j in np.array(i)[:, label_mapping.index("neutral")]]
                ent = [j for i in fact_probs for j in np.array(i)[:, label_mapping.index("entailment")]]

                cont_rev = [j for i in fact_probs_rev for j in np.array(i)[:, label_mapping.index("contradiction")]]
                neut_rev = [j for i in fact_probs_rev for j in np.array(i)[:, label_mapping.index("neutral")]]
                ent_rev = [j for i in fact_probs_rev for j in np.array(i)[:, label_mapping.index("entailment")]]

                fact_pop_df = pd.DataFrame({'Sentence1': sent_pair_1,
                                            'Sentence2': sent_pair_2,
                                            'Sample': sample,
                                            'Sent1_source': sent1_source,
                                            'Sent1_source_entity': sent1_source_ent,
                                            'Sent2_entity': sent2_ent,
                                            '1_2_neut': neut,
                                            '1_2_cont': cont,
                                            '1_2_ent': ent,
                                            '1_2_Label': fact_labels,
                                            '2_1_neut': neut_rev,
                                            '2_1_cont': cont_rev,
                                            '2_1_ent': ent_rev,
                                            '2_1_Label': fact_labels_rev,
                                            'Type': sum_type
                                            })

        ####################################################################################################################

        # SAVE FACTUALITY POPULAR OPINION CSV FILE

        if self.summ_sent == True:

            fact_pop_df.to_csv(f'data/results/sumsent_nli_factuality_{self.summ_type}_{self.data_type}.csv',
                               index=None, header=True)
            fact_pop_df.to_csv(f'data/results/sumsent_nli_popular_{self.summ_type}_{self.data_type}.csv',
                               index=None, header=True)

        else:

            fact_pop_df.to_csv(f'data/results/sentpairs_nli_factuality_{self.summ_type}_{self.data_type}.csv',
                               index=None, header=True)
            fact_pop_df.to_csv(f'data/results/sentpairs_nli_popular_{self.summ_type}_{self.data_type}.csv', index=None,
                               header=True)

            ####################################################################################################################

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

        #         if self.summ_sent == True:

        #             if rev == True:
        #                 for s1, s2 in zip(sent2, sent1):
        #                     combined_list.append(s1 + "<SEP>" + s2)
        #             else:
        #                 for s1, s2 in zip(sent1, sent2):
        #                     combined_list.append(s1 + "<SEP>" + s2)

        #         else:

        #             if rev == True:
        #                 for s1, s2 in zip(sent2, sent1):
        #                     combined_list.append(s1 + ' ' + s2)
        #             else:
        #                 for s1, s2 in zip(sent1, sent2):
        #                     combined_list.append(s1 + ' ' + s2)

        results = self.pipe(combined_list)

        for i in results:
            result = {d['label']: d['score'] for d in i}
            labels.append(max(result, key=result.get))
            prob.append(list(result.values()))

        return labels, prob

    # HELPER METHOD FOR MAIN FUNCTION HELPFUL CONTRAST
    def cont_helper(self, a_sum, b_sum, sum_type, comm_sum=None):

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
            sum_types += [sum_type] * count
            return sent_p1, sent_p2, sent1_ent, sent2_ent, sum_types

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

        sum_types += [sum_type] * count

        return sent_p1, sent_p2, sent1_ent, sent2_ent, sum_types

    # HELPER METHOD FOR HELPFUL CONTRAST IF SUMMARY SENTENCE PAIR
    def summ_sent_cont_helper(self, whole_a, whole_b, whole_comm, a_sum, b_sum, comm_sum, sum_type):

        sum_types = []
        p1 = []
        sent_p2 = []
        summ1_ent = []
        sent2_ent = []

        count = 0

        # summ b -- sent a
        for i in a_sum:
            p1.append(whole_b)
            sent_p2.append(i)
            summ1_ent.append('b')
            sent2_ent.append('a')
            count += 1

        # summ comm -- sent a
        for i in a_sum:
            p1.append(whole_comm)
            sent_p2.append(i)
            summ1_ent.append('comm')
            sent2_ent.append('a')
            count += 1

        # summ a -- sent b
        for i in b_sum:
            p1.append(whole_a)
            sent_p2.append(i)
            summ1_ent.append('a')
            sent2_ent.append('b')
            count += 1

        # summ comm -- sent b
        for i in b_sum:
            p1.append(whole_comm)
            sent_p2.append(i)
            summ1_ent.append('comm')
            sent2_ent.append('b')
            count += 1

        # summ a -- sent comm
        for i in comm_sum:
            p1.append(whole_a)
            sent_p2.append(i)
            summ1_ent.append('a')
            sent2_ent.append('comm')
            count += 1

        # summ b -- sent comm
        for i in comm_sum:
            p1.append(whole_b)
            sent_p2.append(i)
            summ1_ent.append('b')
            sent2_ent.append('comm')
            count += 1

        sum_types += [sum_type] * count

        return p1, sent_p2, summ1_ent, sent2_ent, sum_types

    # HELPER METHOD FOR MAIN FUNCTION POPULAR OPINION FACTUAL CONSISTENCY
    def fact_helper(self, source_a, source_b, a_sum, b_sum, comm_sum, sum_type):
        sum_types = []

        sent_p1 = []
        sent_p2 = []
        sent1_source = []
        sent1_source_ent = []
        sent2_ent = []
        count = 0
        for i in range(len(source_a)):
            for j in source_a[i]:
                for k in a_sum:
                    sent_p1.append(j)
                    sent_p2.append(k)
                    sent1_source.append(i)
                    sent1_source_ent.append('a')
                    sent2_ent.append('a')
                    count += 1

        for i in range(len(source_a)):
            for j in source_a[i]:
                for k in comm_sum:
                    sent_p1.append(j)
                    sent_p2.append(k)
                    sent1_source.append(i)
                    sent1_source_ent.append('a')
                    sent2_ent.append('comm')
                    count += 1

        for i in range(len(source_b)):
            for j in source_b[i]:
                for k in b_sum:
                    sent_p1.append(j)
                    sent_p2.append(k)
                    sent1_source.append(i)
                    sent1_source_ent.append('b')
                    sent2_ent.append('b')
                    count += 1

        for i in range(len(source_b)):
            for j in source_b[i]:
                for k in comm_sum:
                    sent_p1.append(j)
                    sent_p2.append(k)
                    sent1_source.append(i)
                    sent1_source_ent.append('b')
                    sent2_ent.append('comm')
                    count += 1

        sum_types += [sum_type] * count

        return sent_p1, sent_p2, sent1_source, sent1_source_ent, sent2_ent, sum_types

    # HELPER METHOD FOR POPULAR OPINION FACTUAL CONSISTENCY IF SOURCE - SENTENCE PAIR
    def source_sent_fact_helper(self, source_a, source_b, a_sum, b_sum, comm_sum, sum_type):
        sum_types = []
        summ_p1 = []
        sent_p2 = []
        sent1_source = []
        sent1_source_ent = []
        sent2_ent = []

        count = 0

        # source a - sent a
        for i in range(len(source_a)):
            for k in a_sum:
                summ_p1.append(source_a[i])
                sent_p2.append(k)
                sent1_source.append(i)
                sent1_source_ent.append('a')
                sent2_ent.append('a')
                count += 1

        # source a - sent comm
        for i in range(len(source_a)):
            for k in comm_sum:
                summ_p1.append(source_a[i])
                sent_p2.append(k)
                sent1_source.append(i)
                sent1_source_ent.append('a')
                sent2_ent.append('comm')
                count += 1

        # source b - sent b
        for i in range(len(source_b)):
            for k in b_sum:
                summ_p1.append(source_b[i])
                sent_p2.append(k)
                sent1_source.append(i)
                sent1_source_ent.append('b')
                sent2_ent.append('b')
                count += 1

        # source b - sent comm
        for i in range(len(source_b)):
            for k in comm_sum:
                summ_p1.append(source_b[i])
                sent_p2.append(k)
                sent1_source.append(i)
                sent1_source_ent.append('b')
                sent2_ent.append('comm')
                count += 1

        sum_types += [sum_type] * count

        return summ_p1, sent_p2, sent1_source, sent1_source_ent, sent2_ent, sum_types
