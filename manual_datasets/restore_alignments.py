import pandas as pd
import argparse
from utils import read_generic_file, offset_str2list
import os
from nltk import sent_tokenize
import numpy as np


def clean_duc_documents(doc_text_lines):
    #removes special tokens from DUC documents

    readLines = False
    textExtracts = []
    docId = ''
    datetime = ''
    for line in doc_text_lines:
        lineStripped = line.strip()
        if not readLines:
            if lineStripped.startswith('<DOCNO>'):
                docId = lineStripped[7:-8].strip()  # example: <DOCNO> APW19980818.0980 </DOCNO>
            elif lineStripped.startswith('<DATE_TIME>'):
                datetime = lineStripped[11:-12].strip()  # example: <DATE_TIME> 08/18/1998 15:32:00 </DATE_TIME>
            elif lineStripped.startswith('<TEXT>'):
                readLines = True
        else:
            if lineStripped.startswith('</TEXT>') or lineStripped.startswith('<ANNOTATION>'):
                break
            elif lineStripped.startswith('<P>'):
                continue
            elif lineStripped.startswith('</P>'):
                textExtracts.append('\n\n')  # skip line for new paragraph
                continue
            else:
                textExtracts.append(lineStripped)
    allText = ' '.join(textExtracts)
    return allText

def add_doc_sent_in_file_idx(alignments, data_path):
    doc_sent_idx = np.zeros(len(alignments), dtype=int)

    alignments['original_idx'] = range(len(alignments))
    for topic_dir in os.listdir(data_path):
        print(topic_dir)
        if topic_dir == 'summaries':
            continue

        topic_files = os.listdir(os.path.join(data_path, topic_dir))
        for file_idx, file in enumerate(topic_files):
            alignments_file = alignments[alignments['documentFile']==file]
            text = read_generic_file(os.path.join(data_path, topic_dir, file))
            document = " ".join(text)
            doc_sents = sent_tokenize(document)
            doc_sent_char_idx = 0
            for sent_idx, doc_sent in enumerate(doc_sents):
                alignments_topic_file_sent_original_idx = (alignments_file['original_idx'][alignments_file['docSentCharIdx'] == doc_sent_char_idx]).values
                doc_sent_idx[alignments_topic_file_sent_original_idx] = sent_idx
                doc_sent_char_idx += len(doc_sent) + 1  # 1 for space between sents

    alignments['inFile_doc_sentIdx'] = doc_sent_idx
    return alignments


def add_summ_sent_in_file_idx(alignments, data_path):
    doc_sent_idx = np.zeros(len(alignments), dtype=int)

    alignments['original_idx'] = range(len(alignments))
    for summaryFile in os.listdir(data_path):
        print(summaryFile)

        alignments_file = alignments[alignments['summaryFile']==summaryFile]
        text = read_generic_file(os.path.join(data_path, summaryFile))
        document = " ".join(text)
        doc_sents = sent_tokenize(document)
        doc_sent_char_idx = 0
        for sent_idx, doc_sent in enumerate(doc_sents):
            alignments_topic_file_sent_original_idx = (alignments_file['original_idx'][alignments_file['scuSentCharIdx'] == doc_sent_char_idx]).values
            doc_sent_idx[alignments_topic_file_sent_original_idx] = sent_idx
            doc_sent_char_idx += len(doc_sent) + 1  # 1 for space between sents

    alignments['inFile_summ_sentIdx'] = doc_sent_idx
    return alignments

def add_sentence(text, indx_csv_summaryFile, indx_csv, mode='summ'):
    if mode=='summ':
        KEY_SENT = 'scuSentence'
        KEY_SENT_IDX = 'inFile_summ_sentIdx'
    else:
        KEY_SENT = 'docSentText'
        KEY_SENT_IDX = 'inFile_doc_sentIdx'

    sents = sent_tokenize(text)
    idx2sent = {idx: sent for idx, sent in enumerate(sents)}
    indx_csv_summaryFile[KEY_SENT] = indx_csv_summaryFile[KEY_SENT_IDX].apply(lambda x: idx2sent[x])
    summaryFile_index_list = indx_csv_summaryFile.index.to_list()
    indx_csv[KEY_SENT].loc[summaryFile_index_list] = indx_csv_summaryFile[KEY_SENT].to_list()

    return indx_csv


def add_span(text, indx_csv_summaryFile, indx_csv, mode='summ'):
    def read_span(text, offset_list):
        span = text[offset_list[0][0]: offset_list[0][1]]
        for offset_pair in offset_list[1:]:
            span += '...'+text[offset_pair[0]: offset_pair[1]]
        return span




    if mode == 'summ':
        KEY_SPAN = 'summarySpanText'
        KEY_OFFSETS = 'summarySpanOffsets'
    else:
        KEY_SPAN = 'docSpanText'
        KEY_OFFSETS = 'docSpanOffsets'




    indx_csv_summaryFile[KEY_OFFSETS] = indx_csv_summaryFile[KEY_OFFSETS].apply(offset_str2list)
    indx_csv_summaryFile[KEY_SPAN] = indx_csv_summaryFile[KEY_OFFSETS].apply(lambda x: read_span(text, x))

    summaryFile_index_list = indx_csv_summaryFile.index.to_list()
    indx_csv[KEY_SPAN].loc[summaryFile_index_list] = indx_csv_summaryFile[KEY_SPAN].to_list()

    return indx_csv


parser = argparse.ArgumentParser()
parser.add_argument('-indx_csv_path', type=str, required=True)
parser.add_argument('-documents_path', type=str, required=True)
parser.add_argument('-summaries_path', type=str, required=True)

parser.add_argument('-output_file', type=str, required=True)

args = parser.parse_args()

if __name__ == "__main__":
    indx_csv = pd.read_csv(args.indx_csv_path)
    indx_csv = indx_csv[indx_csv['database']=='duc2004']

    #initialize columns
    indx_csv['scuSentence'] = None
    indx_csv['summarySpanText'] = None
    indx_csv['docSentText'] = None
    indx_csv['docSpanText'] = None


    #handle summaries
    indx_csv = add_summ_sent_in_file_idx(indx_csv, args.summaries_path)
    for summaryFile in indx_csv['summaryFile'].drop_duplicates():
        indx_csv_summaryFile = indx_csv[indx_csv['summaryFile']==summaryFile]



        summary = ' '.join(read_generic_file(os.path.join(args.summaries_path, summaryFile)))

        indx_csv = add_sentence(summary, indx_csv_summaryFile, indx_csv, mode='summ')
        indx_csv = add_span(summary, indx_csv_summaryFile, indx_csv, mode='summ')


    # handle documents
    indx_csv = add_doc_sent_in_file_idx(indx_csv, args.documents_path)
    for topic in indx_csv['topic'].drop_duplicates():
        indx_csv_topic = indx_csv[indx_csv['topic']==topic]
        for documentFile in indx_csv_topic['documentFile'].drop_duplicates():
            indx_csv_documentFile = indx_csv_topic[indx_csv_topic['documentFile'] == documentFile]

            doc = read_generic_file(os.path.join(args.documents_path,topic.lower(), documentFile))
            doc = clean_duc_documents(doc)

            indx_csv = add_sentence(doc, indx_csv_documentFile, indx_csv, mode='doc')
            indx_csv = add_span(doc, indx_csv_documentFile, indx_csv, mode='doc')






    indx_csv.to_csv(args.output_file)
