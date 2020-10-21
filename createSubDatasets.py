import pandas as pd
import numpy as np
import json
import argparse
from os.path import join


def extract_salienceSpans(alignments):

    print('number of alignments: ', len(alignments))

    doc_alignments = alignments[['topic', 'documentFile', 'docSentCharIdx',
                                'docSentText', 'docSpanOffsets','docSpanText']]
    doc_alignments.to_csv(join(args.out_dir_path,"salience.csv"), index=False)


    print('number of salient IUs: ',len(doc_alignments.drop_duplicates()))


def extract_clusters(alignments):
    if 'scuText' in list(alignments.columns):   #if annotation data
        summSpansLabel = 'scuText'
        summSpansOffsetsLabel = 'scuOffsets'
    elif 'summarySpanOieText' in list(alignments.columns):  #if train data that uses openIE
        summSpansLabel = 'summarySpanOieText'
        summSpansOffsetsLabel = 'summarySpanOieOffsets'
    else:
        summSpansLabel = 'summarySpanText'
        summSpansOffsetsLabel = 'summarySpanOffsets'

    clusters_num  = 0
    clusters_dict = {'data':[]}
    scu2clusterIdx = {}
    alignmentsPerClusterList = []
    for topic in set(alignments['topic'].values):
        df_topic = alignments[alignments['topic'] == topic]
        clusters_dict['data'].append({'topic':str(topic), 'clusters':[]})
        for scuText in set(df_topic[summSpansLabel].values):
            clusters_num += 1
            df_topic_scu = df_topic[df_topic[summSpansLabel] == scuText]
            alignmentsPerClusterList.append(len(df_topic_scu))
            scu2clusterIdx[scuText] = clusters_num
            clusters_dict['data'][-1]['clusters'].append({'title':scuText, 'clusterID': clusters_num,
                                                            'scuSentCharIdx': str(df_topic_scu.iloc[0]['scuSentCharIdx']),
                                                          'scuSentence': df_topic_scu.iloc[0]['scuSentence'],
                                                          summSpansOffsetsLabel: df_topic_scu.iloc[0][summSpansOffsetsLabel], 'spans':[]})

            for index, row in df_topic_scu.iterrows():
                clusters_dict['data'][-1]['clusters'][-1]['spans'].append({'documentFile': str(row['documentFile']),
                                                                           'docSentCharIdx': str(row['docSentCharIdx']),
                                                                           'docSentText': row['docSentText'],
                                                                           'docSpanOffsets': row['docSpanOffsets'],
                                                                           'docSpanText': row['docSpanText']})



    print ('clusters number: ', clusters_num)
    print('Num of alignments per cluster: ', np.mean(alignmentsPerClusterList), '(',np.std(alignmentsPerClusterList),')')

    with open(join(args.out_dir_path,"clustering.json"), "w") as outfile:
        json.dump(clusters_dict, outfile)

    return clusters_dict, scu2clusterIdx




def extract_textPlanning(alignments, scu2clusterIdx):
    if 'scuText' in list(alignments.columns):   #if annotation data
        summSpansLabel = 'scuText'
    elif 'summarySpanOieText' in list(alignments.columns):  #if train data that uses openIE
        summSpansLabel = 'summarySpanOieText'
    else:
        summSpansLabel = 'summarySpanText'
    sentGeneration_dict = {'data': []}
    sentences_num = 0
    clustersPerSentenceList = []
    for topic in set(alignments['topic'].values):
        df_topic = alignments[alignments['topic'] == topic]
        sentGeneration_dict['data'].append({'topic': str(topic), 'sentences': []})
        scuSentenceList = list(set(zip(df_topic['scuSentence'].values, df_topic['scuSentCharIdx'].values)))
        scuSentenceList.sort(key=lambda x: x[1])
        sent_order = 0
        for scuSentence, scuSentCharIdx in scuSentenceList:
            df_topic_sent = df_topic[df_topic['scuSentence']==scuSentence]
            cluster_idx_list = [scu2clusterIdx[scuText] for scuText in set(df_topic_sent[summSpansLabel].values)]
            clustersPerSentenceList.append(len(cluster_idx_list))
            sentGeneration_dict['data'][-1]['sentences'].append({'clusters': cluster_idx_list, 'sentence': scuSentence, 'order': sent_order})
            sent_order += 1
            sentences_num += 1

    print('Num of sentence generation samples: ', sentences_num)
    print('Num of clusters per sentence: ',np.mean(clustersPerSentenceList), '(',np.std(clustersPerSentenceList),')')
    with open(join(args.out_dir_path,"generation.json"), "w") as outfile:
        json.dump(sentGeneration_dict, outfile)




parser = argparse.ArgumentParser()
parser.add_argument('-alignments_path', type=str, required=True)
parser.add_argument('-out_dir_path', type=str, default='.')
args = parser.parse_args()

if __name__ == "__main__":


    alignments = pd.read_csv(args.alignments_path)
    extract_salienceSpans(alignments)
    clusters_dict, scu2clusterIdx = extract_clusters(alignments)
    extract_textPlanning(alignments, scu2clusterIdx)
