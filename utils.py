import sys
sys.path.append('/home/nlp/ernstor1/rouge/SummEval_referenceSubsets/code_score_extraction/')

from allennlp.predictors.predictor import Predictor
import csv
import argparse
import subprocess
from nltk import tokenize
from nltk.parse import CoreNLPParser
from rouge import Rouge
from bert_score import score
import requests
# import ipdb
import ast
import glob
import os
import logging
import copy
import hashlib
import json
from supervised_oie_wrapper.run_oie import run_oie
# import createRougeDataset
# import calculateRouge
import numpy as np
import shutil
from filterContained import *
from tqdm import tqdm
from itertools import chain
from collections import defaultdict


def str2bool(v):
    return v.lower() in ('true')


def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    s = s.encode('utf-8')
    h = hashlib.sha1()
    h.update(s)
    return h.hexdigest()


# metrics_data = {}
#
predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/bert-base-srl-2019.06.17.tar.gz")
nlp_parser = CoreNLPParser()  # (url='http://nlp3.cs.unc.edu:9000')
# rouge = Rouge()
#
# DATASETS = ['duc2004', 'duc2007', 'MultiNews']



def read_csv_data(csv_file):
    """ Reader to parse the csv file"""
    data = []
    with open(args.input_file_path, encoding='utf-8', errors='ignore') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for ind, row in enumerate(csv_reader):
            if ind == 0:
                header = row
            else:
                data.append(row)
    return header, data


def read_generic_file(filepath):
    """ reads any generic text file into
    list containing one line as element
    """
    text = []
    with open(filepath, 'r') as f:
        for line in f.read().splitlines():
            text.append(line.strip())
    return text


def calculate_metric_scores(cands, refs):
    """ calculate Rouge-1 precision, Bert precision
    and Entailment Scores
    """
    # calculate rouge-1 precision
    rouge = Rouge()
    rouge1_p = []
    for r, c in tqdm(zip(refs, cands)):
        r = " ".join(list(nlp_parser.tokenize(r))).lower()
        c = " ".join(list(nlp_parser.tokenize(c))).lower()
        scores = rouge.get_scores(c, r)[0]
        rouge1_p.append(round(scores['rouge-1']['p'], 4))
    # calculate bert precision
    P, _, _ = score(cands, refs, lang='en', verbose=True)
    P = [round(x, 4) for x in P.tolist()]
    ## calculate entaiment score
    url = 'http://localhost:5003/roberta_mnli_classifier'  # 'http://nlp1.cs.unc.edu:5003/roberta_mnli_classifier'
    mnli_data = []
    for p, h in zip(refs, cands):
        mnli_data.append({'premise': p, 'hypo': h})
    r = requests.post(url, json=mnli_data)
    results = r.json()
    ent_scores = []
    for ind, d in enumerate(results):
        ent_scores.append(float(d['entailment']))

    return rouge1_p, P, ent_scores





def generate_scu(sentence, max_scus=5):
    """ Given a scu sentence retrieve SCUs"""

    srl = predictor.predict(sentence=sentence['scuSentence'])
    # ipdb.set_trace()
    scus = srl['verbs']
    scu_list = []
    tokens = srl['words']
    for scu in scus:
        tags = scu['tags']
        words = []
        if not ("B-ARG1" in tags or "B-ARG2" in tags or "B-ARG0" in tags):
            continue
        scu_start_offset = None
        for ind, tag in enumerate(tags):
            # if "ARG0" in tag or "ARG1" in tag or "V" in tag:
            if "O" not in tag:
                if scu_start_offset is None:
                    if ind == 0:
                        scu_start_offset = sentence['scuSentCharIdx'] + ind
                    else:
                        scu_start_offset = sentence['scuSentCharIdx'] + len(" ".join(tokens[:ind]))
                else:
                    scu_end_offset = sentence['scuSentCharIdx'] + len(" ".join(tokens[:ind + 1]))
                words.append(tokens[ind])

        if len(words) <= 4:
            continue
        tmp = copy.deepcopy(sentence)
        tmp['scuText'] = " ".join(words)
        tmp['scuOffsets'] = (scu_start_offset, scu_end_offset)
        scu_list.append(tmp)
    # select the best SCU
    # sort SCUs based on their length and select middle one
    scu_list = sorted(scu_list, key=lambda x: len(x['scuText'].split()), reverse=True)
    # print(f"Best SCU:::{scu_list[int(len(scu_list)/2)]}")
    # return scu_list[int(len(scu_list)/2)]
    return scu_list[:max_scus]


def generate_scu_oie(sentence, max_scus=5, doc_summ='summ'):
    """ Given a scu sentence retrieve SCUs"""

    if doc_summ=='summ':
        KEY_sent = 'scuSentence'
        KEY_sent_char_idx = 'scuSentCharIdx'
        KEY_scu_text = 'scuText'
        KEY_scu_offset = 'scuOffsets'
    else:
        KEY_sent = 'docSentText'
        KEY_sent_char_idx = 'docSentCharIdx'
        KEY_scu_text = 'docScuText'
        KEY_scu_offset = 'docScuOffsets'

    _, oie = run_oie([sentence[KEY_sent]])

    # ipdb.set_trace()
    if not oie: #if list is empty
        return oie
    else:
        oie = oie[0]
    scus = oie['verbs']
    scu_list = []
    tokens = oie['words']
    for scu in scus:
        tags = scu['tags']
        words = []
        if not ("B-ARG1" in tags or "B-ARG2" in tags or "B-ARG0" in tags):
            continue
        scu_start_offset = None
        for ind, tag in enumerate(tags):
            # if "ARG0" in tag or "ARG1" in tag or "V" in tag:
            if "O" not in tag:
                if scu_start_offset is None:
                    if ind == 0:
                        scu_start_offset = sentence[KEY_sent_char_idx] + ind
                    else:
                        scu_start_offset = sentence[KEY_sent_char_idx] + len(" ".join(tokens[:ind]))
                else:
                    scu_end_offset = sentence[KEY_sent_char_idx] + len(" ".join(tokens[:ind + 1]))
                words.append(tokens[ind])

        # if len(words) <= 3:
        #     continue
        tmp = copy.deepcopy(sentence)
        tmp[KEY_scu_text] = " ".join(words)
        tmp[KEY_scu_offset] = (scu_start_offset, scu_end_offset)
        scu_list.append(tmp)
    # select the best SCU
    # sort SCUs based on their length and select middle one
    scu_list = sorted(scu_list, key=lambda x: len(x[KEY_scu_text].split()), reverse=True)
    # print(f"Best SCU:::{scu_list[int(len(scu_list)/2)]}")
    # return scu_list[int(len(scu_list)/2)]
    return scu_list[:max_scus]

def generate_scu_oie_multiSent(sentences, doc_summ='summ'):
    """ Given a scu sentence retrieve SCUs"""

    if doc_summ=='summ':
        KEY_sent = 'scuSentence'
        KEY_sent_char_idx = 'scuSentCharIdx'
        KEY_scu_text = 'scuText'
        KEY_scu_offset = 'scuOffsets'
    else:
        KEY_sent = 'docSentText'
        KEY_sent_char_idx = 'docSentCharIdx'
        KEY_scu_text = 'docScuText'
        KEY_scu_offset = 'docScuOffsets'

    _, oies = run_oie([sentence[KEY_sent] for sentence in sentences], cuda_device = 0)
    #adaptation for srl
    # oies = []
    # for sentence in sentences:
    #     oies.append(predictor.predict(sentence = sentence[KEY_sent] ))


    scu_list = []
    assert(len(sentences) == len(oies))
    for sentence ,oie in zip(sentences,oies):
        sentence[KEY_sent] = sentence[KEY_sent].replace(u'\u00a0', ' ')
        # ipdb.set_trace()
        if not oie:  # if list is empty
            continue

        # if  sentence[KEY_sent] =='Johnson\'s new TV show, ``The Magic Hour,\'\' is just one aspect of a busy life:  -- HIS HEALTH: While by no means cured, he owes the appearance of remarkable health to a Spartan lifestyle and modern medicine.':
        #     print('here')
        scus = oie['verbs']
        in_sentence_scu_dict = {}
        tokens = oie['words']
        for scu in scus:
            tags = scu['tags']
            words = []
            if not ("B-ARG1" in tags or "B-ARG2" in tags or "B-ARG0" in tags):
                continue
            sub_scu_offsets = []
            scu_start_offset = None
            offset = 0
            initialSpace = 0
            while sentence[KEY_sent][offset + initialSpace] == ' ':
                initialSpace += 1  ## add space if exists, so 'offset' would start from next token and not from space
            offset += initialSpace
            for ind, tag in enumerate(tags):
                # if "ARG0" in tag or "ARG1" in tag or "V" in tag:
                assert (sentence[KEY_sent][offset] == tokens[ind][0])
                if "O" not in tag:
                    if scu_start_offset is None:
                        scu_start_offset = sentence[KEY_sent_char_idx] + offset

                        assert(sentence[KEY_sent][offset] == tokens[ind][0])

                    words.append(tokens[ind])
                else:
                    if scu_start_offset is not None:
                        spaceBeforeToken = 0
                        while sentence[KEY_sent][offset-1-spaceBeforeToken] == ' ':
                            spaceBeforeToken += 1## add space if exists
                        if sentence[KEY_sent][offset] == '.' or sentence[KEY_sent][offset] == '?':
                            dotAfter = 1 + spaceAfterToken
                            dotTest = 1
                        else:
                            dotAfter = 0
                            dotTest = 0
                        scu_end_offset = sentence[KEY_sent_char_idx] + offset - spaceBeforeToken + dotAfter

                        if dotTest:
                            assert (sentence[KEY_sent][offset - spaceBeforeToken + dotAfter -1] == tokens[ind-1+ dotTest][0]) #check only the dot, the start of the token
                        else:
                            assert (sentence[KEY_sent][offset - spaceBeforeToken + dotAfter - 1] == tokens[ind - 1 + dotTest][-1])  #check end of token
                        sub_scu_offsets.append([scu_start_offset, scu_end_offset])
                        scu_start_offset = None


                ## update offset

                offset += len(tokens[ind])
                if ind < len(tags) - 1: #if not last token
                    spaceAfterToken = 0
                    while sentence[KEY_sent][offset + spaceAfterToken] == ' ':
                        spaceAfterToken += 1## add space after token if exists, so 'offset' would start from next token and not from space
                    offset += spaceAfterToken

            if scu_start_offset is not None: #end of sentence
                scu_end_offset = sentence[KEY_sent_char_idx] + offset
                sub_scu_offsets.append([scu_start_offset, scu_end_offset])
                scu_start_offset = None



            # if len(words) <= 3:
            #     continue
            scuText = "...".join([sentence[KEY_sent][strt_end_indx[0] - sentence[KEY_sent_char_idx]:strt_end_indx[1] - sentence[KEY_sent_char_idx]] for strt_end_indx in sub_scu_offsets])
            #assert(scuText==" ".join([sentence[KEY_sent][strt_end_indx[0]:strt_end_indx[1]] for strt_end_indx in sub_scu_offsets]))
            in_sentence_scu_dict[scuText] = sub_scu_offsets

        notContainedDict = checkContained(in_sentence_scu_dict, sentence[KEY_sent], sentence[KEY_sent_char_idx])


        for scuText, binaryNotContained in notContainedDict.items():
            scu_offsets = in_sentence_scu_dict[scuText]
            if binaryNotContained:
                tmp = copy.deepcopy(sentence)
                tmp[KEY_scu_text] = scuText
                tmp[KEY_scu_offset] = scu_offsets
                scu_list.append(tmp)
    # select the best SCU
    # sort SCUs based on their length and select middle one
    # scu_list = sorted(scu_list, key=lambda x: len(x[KEY_scu_text].split()), reverse=True)
    # print(f"Best SCU:::{scu_list[int(len(scu_list)/2)]}")
    # return scu_list[int(len(scu_list)/2)]
    return scu_list




def word_aligner(sent1, sent2):
    """ wrapper which calls the monolingual
    word aligner and gives the alignment scores between
    sent1 and sent2
    """
    ## tokenize
    sent1_tok = " ".join(list(nlp_parser.tokenize(sent1)))
    sent2_tok = " ".join(list(nlp_parser.tokenize(sent2)))

    ## create a subprocess to call the word aligner
    process = subprocess.Popen(['python2', 'predict_align.py', '--s1', sent1_tok,
                                '--s2', sent2_tok], stdout=subprocess.PIPE,
                               cwd='/ssd-playpen/home/ram/monolingual-word-aligner')
    output, error = process.communicate()
    ## parse the output
    output = output.decode('utf-8')
    output = output.split('\n')
    return ast.literal_eval(output[0]), ast.literal_eval(output[1]), sent1_tok, sent2_tok



def write_doc_scus(doc_sents, doc_sent_dir):


    if not os.path.exists(doc_sent_dir):
        os.makedirs(doc_sent_dir)
    for sent_idx, sentence in enumerate(doc_sents):
        html_path = os.path.join(doc_sent_dir, 'D061.M.250.J.' + str(sent_idx)+'.html')
        with open(html_path, 'w') as f:
            f.write(sentence)

    return len(doc_sents)

def write_summ_scus(summ_sents, summ_sent_dir):
    for sent_idx, sentence in enumerate(summ_sents):
        sent_dir = os.path.join(summ_sent_dir, str(sent_idx))
        if not os.path.exists(sent_dir):
            os.makedirs(sent_dir)
        html_path = os.path.join(sent_dir, 'D061.M.250.J.A' + '.html')
        with open(html_path, 'w') as f:
            f.write(sentence)
    return len(summ_sents)




# def calc_rouge_mat(summ_scus, doc_scus):
#     DOC_SENT_DIR = '/home/nlp/ernstor1/tmp/doc_sent_dir'
#     SUMM_SENT_DIR = '/home/nlp/ernstor1/tmp/summ_sent_dir'
#
#     num_doc_scus = write_doc_scus(doc_scus, DOC_SENT_DIR)
#     num_summ_scus = write_summ_scus(summ_scus, SUMM_SENT_DIR)
#     rouge_mat = np.zeros((num_summ_scus, num_doc_scus))
#
#     for summ_dir in os.listdir(SUMM_SENT_DIR):
#         INPUTS = [(calculateRouge.COMPARE_SAME_LEN, os.path.join(SUMM_SENT_DIR,summ_dir),DOC_SENT_DIR,
#                    None, None, calculateRouge.REMOVE_STOP_WORDS)]
#
#         compareType, refFolder, sysFolder, outputPath, ducVersion, stopWordsRemoval = INPUTS[0]
#
#         # get the different options:
#         taskNames, systemNames, summaryLengths = calculateRouge.getComparisonOptions(sysFolder, refFolder)
#         # get ROUGE scores:
#         allData = calculateRouge.runRougeCombinations(compareType, sysFolder, refFolder, systemNames,
#                                                           summaryLengths,
#                                                           ducVersion, stopWordsRemoval)
#         # calculate R1,R2,RL average
#         rouge_vec = createRougeDataset.extractRouge(allData, systemNames, summaryLengths)
#         rouge_mat[int(summ_dir),:] = rouge_vec
#
#
#     # remove tmp dirs
#     shutil.rmtree(DOC_SENT_DIR)
#     shutil.rmtree(SUMM_SENT_DIR)
#
#     return rouge_mat

def saveSCUsToCsv(scus, outputFilePath):
    # Outputs the selected SCUs to the output CSV path specified
    with open(outputFilePath, mode='w', newline='') as outFile:
        csvWriter = csv.writer(outFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csvWriter.writerow(['db', 'topic', 'sentCharIdx', 'sentence', 'offsets', 'scu'])
        # print in order of hitID (so that the sentence order is kept in the output CSV):
        for scu in scus:
            db = scu['database']
            topic = scu['summaryFile']
            sentCharIdx = scu['scuSentCharIdx']
            # sentId= annoPerHIT[hitId]['sentCharIdx']
            sentence = scu['scuSentence']
            scu_text = scu['scuText']
            offsetsStr = ';'.join(
                ', '.join(map(str, offset)) for offset in [scu['scuOffsets']])
            csvWriter.writerow([db, topic, sentCharIdx, sentence, offsetsStr, scu_text])


def saveSCU_SentFilteredPairsToCsv(scu_sent_pairs, outputCsvFilepath):
    # output fields:
    # db, topic, summaryFile, scuSentCharIdx, scuSentence, scuOffsets, documentFile, docSentCharIdx, scuText, docSentText, isAligned

    with open(outputCsvFilepath, 'w', newline='') as fOut:
        csvWriter = csv.writer(fOut, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csvWriter.writerow(
            ['db', 'topic', 'summaryFile', 'scuSentCharIdx', 'scuSentence', 'scuOffsets', 'documentFile', 'docSentCharIdx',
             'scuText', 'docSentText', 'isAligned'])

        for scu, doc_sent in scu_sent_pairs:
            db = scu['database']
            topic = scu['topic']
            summaryFile = scu['summaryFile']
            scuSentCharIdx = scu['scuSentCharIdx']
            # sentId= annoPerHIT[hitId]['sentCharIdx']
            scuSentence = scu['scuSentence']
            scuText = scu['scuText']
            scuOffsets = ';'.join(
                ', '.join(map(str, offset)) for offset in [scu['scuOffsets']])

            documentFile = doc_sent['documentFile']
            docSentCharIdx = doc_sent['docSentCharIdx']
            docSentText = doc_sent['docSentText']
            answer = 1

            csvWriter.writerow([db, topic, summaryFile, scuSentCharIdx, scuSentence, scuOffsets, documentFile,
                                 docSentCharIdx, scuText, docSentText, answer])


def intersectionOverUnion(offset1, offset2):
    ranges1 = [range(marking[0], marking[1]) for marking in offset1]
    ranges1 = set(chain(*ranges1))
    ranges2 = [range(marking[0], marking[1]) for marking in offset2]
    ranges2 = set(chain(*ranges2))
    return len(ranges1 & ranges2) / len(ranges1 | ranges2)



def Union(offsets, sentOffsets):
    ranges_tmp = set([])
    for offset, sentOffset in zip(offsets, sentOffsets):
        offset = offset_str2list(offset)
        offset = offset_decreaseSentOffset(sentOffset, offset)
        ranges = [range(marking[0], marking[1]) for marking in offset]
        ranges = set(chain(*ranges))
        ranges_tmp = ranges_tmp | ranges
    return  ranges_tmp






def offset_str2list(offset):
    return [[int(start_end) for start_end in offset.split(',')] for offset in offset.split(';')]

def offset_list2str(list):
    return ';'.join(', '.join(map(str, offset)) for offset in list)

def offset_decreaseSentOffset(sentOffset, scu_offsets):
    return [[start_end[0] - sentOffset, start_end[1] - sentOffset] for start_end in scu_offsets]

def chunks_new(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]




