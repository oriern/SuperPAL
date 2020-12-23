from utils import *


class Aligner(object):
    """
        finds document-summary alignment pair candidates

    """

    def __init__(self, data_path='.', mode='dev',
                 log_file='results/dev_log.txt', metric_precompute=True, output_file = './prediction_dev.csv',
                 database='duc2004,duc2007,MultiNews'):

        self.data_path = data_path
        self.mode = mode
        self.log_file = log_file
        self.metric_precompute = metric_precompute
        self.output_file = output_file
        if ',' in database:
            self.database = database.split(',')
        else:
            self.database = [database]


        self.summ_sents = []
        self.doc_sents = []
        self.final_alignments = []


        # set up logger
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
            handlers=[
                logging.FileHandler(f"{self.log_file}"),
                logging.StreamHandler()
            ])

        if self.metric_precompute:
            self.metrics_data = {}
        else:
            metric_precompute_path = f'data/final_data/metric_compute_{self.mode}_oie.json'
            if os.path.isfile(metric_precompute_path):
                with open(metric_precompute_path, 'r') as f:
                    self.metrics_data = json.load(f)
            else:
                logging.warning(f"WARNING: No metric_precompute data was found")
                self.metrics_data = None


                # self.scus_list = []
        # self.scu_sent_pairs = []
        self.docSentsOIE = True     #whether need to generate OIE for doc sents. Depends on aligner.




    def read_and_split(self, dataset, sfile):
        ### process the summary
        logging.info(f"Evaluating the Alignment for Summary: {sfile}")
        sfile_basename = os.path.basename(sfile)
        if dataset in ['duc2007', 'duc2004']:
            doc_name = sfile_basename.split('.')[0]
        elif dataset == 'MultiNews':
            doc_name = "MultiNews_" + sfile_basename.split('_')[0]
        else:
            doc_name = sfile_basename.split('.')[0]

        summary = read_generic_file(sfile)
        s_sents = []
        # for line in summary:
        #     s_sents.extend(tokenize.sent_tokenize(line))
        s_sents = tokenize.sent_tokenize(" ".join(summary))
        self.summ_sents = []
        idx_start = 0
        for sent in s_sents:
            self.summ_sents.append({'summaryFile': sfile_basename, 'scuSentCharIdx': idx_start,
                               'scuSentence': sent, 'database': dataset, 'topic': doc_name})
            idx_start = idx_start + len(sent) + 1  # 1 for the space character

        ## process all the documents files
        doc_files = glob.glob(f"{self.data_path}/{self.mode}_data/documents/{dataset}/{doc_name}/*")

        logging.info(f"Following documents have been found for them:")
        logging.info("\n".join(doc_files))
        self.doc_sents = []
        for df in doc_files:
            doc_id = os.path.basename(df)
            document = read_generic_file(df)
            dsents = []
            # for line in document:
            #     dsents.extend(tokenize.sent_tokenize(line))
            dsents = tokenize.sent_tokenize(" ".join(document))
            idx_start = 0
            for dsent in dsents:
                if dsent != "...":  # this is a exception
                    self.doc_sents.append({'documentFile': doc_id, 'docSentCharIdx': idx_start,
                                      'docSentText': dsent})

                idx_start = idx_start + len(dsent) + 1  # 1 for the space charater between sentences


    def calc_metric_precompute(self):
        scus = []
        # for s in self.summ_sents:
        #     scus.extend(generate_scu(s, max_scus=100))
        scus.extend(generate_scu_oie_multiSent(self.summ_sents, doc_summ='summ'))
        refs = []
        cands = []
        ids = []
        for s in scus:
            refs.extend([x['docSentText'] for x in self.doc_sents])
            cands.extend([s['scuText'] for _ in range(len(self.doc_sents))])
            ids.extend([s['summaryFile'] + s['scuText'] + x['documentFile'] + x['docSentText'] for x in self.doc_sents])
        rouge1_p, bert_p, ent = calculate_metric_scores(cands, refs)

        for idx, key in enumerate(ids):
            self.metrics_data[hashhex(key)] = {'rouge1_p': rouge1_p[idx], 'bert_p': bert_p[idx], 'ent': ent[idx]}





    def save_predictions(self):
        if self.metric_precompute:
            with open(f'data/final_data/metric_compute_{self.mode}_oie.json', 'w') as f:
                json.dump(self.metrics_data, f)

        else:
            ## save the predictions into a csv file
            with open(os.path.join(self.output_file,'dev.csv'), 'w') as f:
                csvwriter = csv.writer(f, delimiter=',')
                header = ['database', 'topic', 'summaryFile', 'scuSentCharIdx', 'scuSentence', 'scuOffsets', 'scuText',
                          'documentFile', 'docSentCharIdx', 'docSentText', 'docSpanOffsets', 'summarySpanOffsets',
                          'docSpanText', 'summarySpanText']
                csvwriter.writerow(header)
                for ind, row in enumerate(self.final_alignments):
                    data = []
                    # from lists to string format for csv
                    row['scuOffsets'] = ';'.join(', '.join(map(str, offset)) for offset in row['scuOffsets'])
                    row['docSpanOffsets'] = ';'.join(', '.join(map(str, offset)) for offset in row['docSpanOffsets'])
                    row['summarySpanOffsets'] = ';'.join(', '.join(map(str, offset)) for offset in row['summarySpanOffsets'])
                    for key in header:
                        if type(row[key]) is tuple:
                            data.append(f"{row[key][0]}, {row[key][1]}")
                        else:
                            data.append(row[key])
                    csvwriter.writerow(data)






    def main_filter(self, scu, cand_doc_sents):
        return

    def scu_span_aligner(self):
        """ Module which align scu and sentence
        in the document given a summary and document
        """
        ## generate SCUs
        scus = []
        scus.extend(generate_scu_oie_multiSent(self.summ_sents, doc_summ='summ'))

        if self.docSentsOIE:
            doc_spans = []
            doc_spans.extend(generate_scu_oie_multiSent(self.doc_sents, doc_summ='doc'))


        ## create candidate pool for sentences in
        ## the document for each scu
        for scu in scus:
            cand_doc_sents = self.metric_filter(scu)

            if self.docSentsOIE:
                if len(cand_doc_sents)==len(self.doc_sents):
                    self.main_filter(scu, doc_spans)
                else:
                    doc_spans = []
                    doc_spans.extend(generate_scu_oie_multiSent(cand_doc_sents, doc_summ='doc'))
                    self.main_filter(scu, doc_spans)
            else:
                self.main_filter(scu, cand_doc_sents)

    def metric_filter(self, scu, use_precompute_metrics=True):
        """ this module finds the candidate sentences
        that are close to the given scu using metric based
        filtering
        """
        refs = [x['docSentText'] for x in self.doc_sents]
        cands = [scu['scuText'] for _ in range(len(self.doc_sents))]
        ids = [scu['summaryFile'] + scu['scuText'] + x['documentFile'] + x['docSentText'] for x in self.doc_sents]
        if use_precompute_metrics:
            # global metrics_data
            rouge1_p = []
            bert_p = []
            ent = []
            for idx, key in enumerate(ids):
                rouge1_p.append(self.metrics_data[hashhex(key)]['rouge1_p'])
                bert_p.append(self.metrics_data[hashhex(key)]['bert_p'])
                ent.append(self.metrics_data[hashhex(key)]['ent'])
        else:
            rouge1_p, bert_p, ent = calculate_metric_scores(cands, refs)
        cands = []
        scores = []
        for ind in range(len(refs)):
            preds = 0
            if rouge1_p[ind] > 0.2:  # rouge1-p
                preds = 1
            if bert_p[ind] > 0.88:  # BERT-p
                preds = 1
            if ent[ind] < 0.001:
                preds = 0
            if rouge1_p[ind] < 0.2:
                preds = 0
            if rouge1_p[ind] > 0.25 and bert_p[ind] < 0.85:
                preds = 0
            if preds == 1:
                tmp = copy.deepcopy(self.doc_sents[ind])
                tmp['score'] = rouge1_p[ind] * bert_p[ind] * ent[ind]
                cands.append(tmp)

        cands = sorted(cands, key=lambda x: x['score'], reverse=True)

        return cands [:3]


