from annotation2MRPC_Aligner import annotation2MRPC_Aligner
from utils import *
import pandas as pd



class docSum2MRPC_Aligner(annotation2MRPC_Aligner):
    """
    finds document-summary alignment pair candidates
     and write them in a .tsv file that fits the huggingface 'transformers' MRPC format (a classification paraphrasing model)

    """


    def read_and_split(self, dataset, sfile):
        ### process the summary
        logging.info(f"Evaluating the Alignment for Summary: {sfile}")
        sfile_basename = os.path.basename(sfile)
        doc_name = sfile_basename.split('.')[0]#.lower()+'t'

        summary = read_generic_file(sfile)
        s_sents = tokenize.sent_tokenize(" ".join(summary))
        self.summ_sents = []
        idx_start = 0
        for sent in s_sents:
            self.summ_sents.append({'summaryFile': sfile_basename, 'scuSentCharIdx': idx_start,
                               'scuSentence': sent, 'database': dataset, 'topic': doc_name})
            idx_start = idx_start + len(sent) + 1  # 1 for the space character

        ## process all the documents files
        doc_files = glob.glob(f"{self.data_path}/{doc_name}/*")

        logging.info(f"Following documents have been found for them:")
        logging.info("\n".join(doc_files))
        self.doc_sents = []
        for df in doc_files:
            doc_id = os.path.basename(df)
            document = read_generic_file(df)
            dsents = tokenize.sent_tokenize(" ".join(document))
            idx_start = 0
            for dsent in dsents:
                if dsent != "...":  # this is an exception
                    self.doc_sents.append({'documentFile': doc_id, 'docSentCharIdx': idx_start,
                                      'docSentText': dsent})

                idx_start = idx_start + len(dsent) + 1  # 1 for the space charater between sentences




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

        self.alignment_database.to_csv(os.path.join(self.output_file,'dev.tsv'), index=False, sep='\t')





