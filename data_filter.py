#encoding=utf-8

import json
import jieba
from numpy.core.function_base import add_newdoc
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import joblib
from sklearn import preprocessing
import numpy as np
import SinglePassCluster

def clustering(tf_idf_dim, threshold=0.6):
    _, tf_idf = joblib.load('./data/tfidf_{}.pkl'.format(tf_idf_dim))
    tf_idf_array = preprocessing.normalize(tf_idf.toarray(), norm='l2')

    single_pass_cluster = SinglePassCluster.SinglePassCluster(tf_idf_array, t=threshold)
    single_pass_cluster = single_pass_cluster.cluster_by_cosine_similarity(src_file = './data/corpus.txt', tgt_file = './data/clustering_t_{}.txt'.format(threshold))
    joblib.dump(single_pass_cluster, './data/tfidf_{}_clustering.pkl'.format(if_idf_dim))

def gen_filtered_rumor():
    single_pass_cluster = joblib.load('./data/tfidf_4000_clustering.pkl')
    cluster_list = single_pass_cluster.cluster_list

    out = open('./data/filtered.json', 'w')
    out_pretty = open('./data/filtered_pretty.json', 'w')
    with open('./data/data_origin.json', 'r', encoding='utf-8') as src:
        items = json.load(src)
        for cluster in cluster_list:
            data_cluster = []
            for index in cluster.node_list:
                # print(index)
                item = items[index]
                data_cluster.append(item)
            if(data_cluster == []):
                print("该簇为空")
            weibos = [r['reportedWeibo'] for r in rumor_cluster]
            
            chosen_data = data_cluster[0]     # 这里简单使用该簇中的第一个元素作为该簇的代表，根据自己的需求可以改变优先选择方式
            if(len(data_cluster) != 0):
                out.write('{}\n'.format(json.dumps(chosen_data, ensure_ascii = False))   #生成的文件一行是一个json数据条目
                out_pretty.write('{}\n'.format(json.dumps(chosen_data, ensure_ascii = False, indent = 4, seperators = (',', ':'))))    # 生成一个便于阅读的文件
                out.flush()
                out_pretty.flush() 
    out.close()
    out_pretty.close()

clustering(category=4000)
gen_filtered_rumor()
