import json
import jieba
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import joblib
import re

def gen_corpus(src_file, out_file):
    corpus = []
    with open(src_file, 'r', encoding='utf-8') as src_f:
        json_file = json.load(src_f)
        print(len(json_file))
        # print(json_file[0])
        num = 0
        past_num = 0
        for dict_item in json_file:
            content = dict_item['content']
            corpus.append(content)
    
    with open(out_file, 'w', encoding='utf-8') as out_f:
        for c in corpus:
            c = re.sub('\n',' ',c)
            out_f.write('{}\n'.format(c))

def cut_words(corpus_file, cut_corpus_file):
    cut_corpus = []
    with open(corpus_file, 'r', encoding='utf-8') as src:
        lines = src.readlines()
        for line in lines:
            seg_list = jieba.cut(line)
            result = ' '.join(seg_list)
            cut_corpus.append(result)
    with open(cut_corpus_file, 'w', encoding='utf-8') as out:
        for c in cut_corpus:
            out.write('{}'.format(c))


def get_tfidf(cut_corpus_file, features_num = 4000):
    cut_corpus = []
    with open(cut_corpus_file, 'r', encoding='utf-8') as src:
        lines = src.readlines()
        for line in lines:
            cut_.append(line)
        print('The size of corpus is {}.'.format(len(cut_corpus)))
    
    vectorizer = CountVectorizer(max_features = features_num)
    transformer = TfidfTransformer()
    tf_idf = transformer.fit_transform(vectorizer.fit_transform(cut_corpus))
    vocabulary = vectorizer.get_feature_names()

    joblib.dump((vocabulary, tf_idf), '../data/corpus/rumor_tfidf_{}.pkl'.format(features_num))

gen_corpus(src_file='../data/data_origin.json', out_file = '../data/corpus.txt')
cut_words(corpus_file='../data/corpus.txt', cut_corpus_file='../data/cut_corpus.txt')
get_tfidf(cut_corpus_file='../data/cut_corpus.txt')
