from Aligner import Aligner
from utils import *
import pandas as pd



class annotation2MRPC_Aligner(Aligner):


    """
    gets gold crowdsourced alignments in a csv format
     and convert them to a .tsv file that fits the huggingface 'transformers' MRPC format (a classification paraphrasing model)

    """


    def __init__(self, data_path='.', mode='dev',
                 log_file='results/dev_log.txt', metric_precompute=True, output_file = './dev.tsv',
                 database='duc2004,duc2007,MultiNews'):
        super().__init__(data_path=data_path, mode=mode,
                 log_file=log_file, metric_precompute=metric_precompute, output_file = output_file,
                 database=database)
        self.filter_data = False
        self.use_stored_alignment_database = False
        self.alignment_database_list = []
        self.docSentsOIE = True
        self.alignment_database = pd.DataFrame(columns=['Quality', '#1 ID', '#2 ID', '#1 String', '#2 String','database', 'topic',
                                                        'summaryFile', 'scuSentCharIdx', 'scuSentence', 'documentFile',	'docSentCharIdx',
                                                        'docSentText', 'docSpanOffsets', 'summarySpanOffsets', 'docSpanText', 'summarySpanText'])


        

    def add_scu_doc_span_pairs(self, scu, doc_spans):

        scu_offset_str = offset_list2str(scu['scuOffsets'])
        id_scu = scu['topic'] + '_' + scu_offset_str

        for doc_span in doc_spans:
            doc_offset_str = offset_list2str(doc_span['docScuOffsets'])
            id_doc_sent = scu['topic'] + '_' + doc_span['documentFile'] + '_' + doc_offset_str
            label = 0 #label =0 for all. positive samples' label would be changed later


            self.alignment_database_list.append([label, id_scu, id_doc_sent,
                                                 scu['scuText'],
                                                 doc_span['docScuText'], scu['database'],
                                                 scu['topic'], scu['summaryFile'],
                                                 scu['scuSentCharIdx'],
                                                 scu['scuSentence'],
                                                 doc_span['documentFile'],
                                                 doc_span['docSentCharIdx'],
                                                 doc_span['docSentText'],
                                                 offset_list2str(
                                                     doc_span['docScuOffsets']),
                                                 offset_list2str(scu['scuOffsets']),
                                                 doc_span['docScuText'], scu['scuText']])


    def metric_filter(self, scu):
        if self.filter_data:
            return super().metric_filter(scu)
        return self.doc_sents

    def scu_span_aligner(self):
        if self.use_stored_alignment_database:
            if self.mode == 'dev':
                self.alignment_database = pd.read_pickle("./span_alignment_database_dev.pkl")
            else:
                self.alignment_database = pd.read_pickle("./span_alignment_database_test.pkl")
        else:
            super().scu_span_aligner()
            self.add_scu_doc_span_pairs(scu, doc_spans)







    def update_positive_labels(self):
        if self.mode == 'dev':
            self.annotation_file = pd.read_csv('SCUdataGenerator/finalAlignmentDataset_dev_cleaned_wo_duplications.csv')
        else:
            self.annotation_file = pd.read_csv('SCUdataGenerator/finalAlignmentDataset_test_cleaned_wo_duplications.csv')

        for index, row in self.annotation_file.iterrows():
            # row = self.annotation_file.sample().iloc[0]  # random row for debug.
            documentFile = row['documentFile']
            topic = row['topic']
            summarySpanOffsets = offset_str2list(row['summarySpanOffsets'])
            docSpanOffsets = offset_str2list(row['docSpanOffsets'])
            cands_df = self.alignment_database[(self.alignment_database['documentFile']==documentFile).values &
                                               (self.alignment_database['topic']==topic).values]
            scu_cands_offset = np.unique(cands_df['summarySpanOffsets'])
            doc_cands_offset = np.unique(cands_df['docSpanOffsets'])
            self.updateAlignment(summarySpanOffsets, scu_cands_offset, docSpanOffsets, doc_cands_offset, documentFile, topic)

            # print(row['summarySpanText'])
            # print(row['scuText'])
            # DEBUG_print_k_max_match(summarySpanOffsets, scu_cands_offset, documentFile, 3, self.alignment_database,topic)





    def save_predictions(self):
        if self.metric_precompute:
            super().save_predictions()

        self.alignment_database = pd.DataFrame(self.alignment_database_list,
                                               columns=['Quality', '#1 ID', '#2 ID', '#1 String', '#2 String',
                                                        'database', 'topic',
                                                        'summaryFile', 'scuSentCharIdx', 'scuSentence', 'documentFile',
                                                        'docSentCharIdx',
                                                        'docSentText', 'docSpanOffsets', 'summarySpanOffsets',
                                                        'docSpanText', 'summarySpanText'])
        # self.alignment_database.to_pickle("./span_alignment_database_test_filtered.pkl")
        self.update_positive_labels()
        self.alignment_database.to_csv(os.path.join(self.output_file,'dev.tsv'), index=False, sep='\t')





    def updateAlignment(self, summarySpanOffsets, scu_cands_offset, docSpanOffsets, doc_cands_offset, documentFile, topic):
        INTERSECTION_RATIO_THRESH = 0.25

        summary_match_arr = np.array(
            [intersectionOverUnion(summarySpanOffsets, offset_str2list(scu_cand_offset)) for scu_cand_offset in scu_cands_offset])

        matches_summary_scus = np.argwhere(summary_match_arr > INTERSECTION_RATIO_THRESH)#[[np.argmax(summary_match_arr)]]#

        doc_match_arr = np.array(
            [intersectionOverUnion(docSpanOffsets, offset_str2list(doc_cand_offset)) for doc_cand_offset in doc_cands_offset])

        matches_doc_spans = np.argwhere(doc_match_arr > INTERSECTION_RATIO_THRESH)#[[np.argmax(doc_match_arr)]]#



        for summ_cand_idx in matches_summary_scus:
            for doc_cand_idx in matches_doc_spans:
                matched_row = self.alignment_database[(self.alignment_database['documentFile'] == documentFile).values &
                                        (self.alignment_database['topic'] == topic).values &
                                        (self.alignment_database['summarySpanOffsets'] ==
                                         scu_cands_offset[summ_cand_idx[0]]).values  &
                                        (self.alignment_database['docSpanOffsets'] ==
                                         doc_cands_offset[doc_cand_idx[0]]).values]

                if len(matched_row.index) > 0:
                    assert(len(matched_row.index) == 1)
                    self.alignment_database.iloc[matched_row.index[0]]['Quality'] = 1





def DEBUG_print_k_max_match(summarySpanOffsets, scu_cands_offset, documentFile, k, alignment_database,topic):
    match_arr = np.array([intersectionOverUnion(summarySpanOffsets, offset_str2list(scu_cand_offset)) for scu_cand_offset in scu_cands_offset])

    max_index = match_arr.argsort()[-k:][::-1]

    for i in range(k):
        string = alignment_database[(alignment_database['documentFile'] == documentFile).values &
                                    (alignment_database['topic'] == topic).values &
                (alignment_database['summarySpanOffsets'] ==
                 scu_cands_offset[max_index[i]]).values].iloc[0]['#1 String']
        score = match_arr[max_index[i]]

        print(score, string)



