#encoding:utf-8
import numpy as np
import time
import math
import joblib

def cosine_similarity(vec_a, vec_b):
    return float(vec_a.dot(vec_b))/(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

class ClusterUnit:
    def __init__(self):
        self.node_list = []
        self.node_num = 0
        self.centroid = None
    
    def add_node(self, node, node_vec):
        self.node_list.append(node)
        try:
            self.centroid = (self.node_num * self.centroid)/(self.node_num + 1)
        except TypeError:
            self.centroid = np.array(node_vec)*1
        self.node_num += 1
    

class SinglePassCluster:
    def __init__(self, vector_list, cluster_list = [], cluster_num = 0, t = 0.6):
        """
        param t: float, 一趟聚类的阈值
        param vector_list: (samples_num, features_size)
        """
        self.threshold = t
        self.vectors = np.array(vector_list)
        self.cluster_list = []
        self.cluster_num = len(self.cluster_list)
    
    def cluster_by_cosine_similarity(self, src_file = None, tgt_file = None):
        self.cluster_list.append(ClusterUnit())
        self.cluster_list[0].add_node(0, self.vectors[0])
        if(src_file != None):
            src = open(src_file, 'r', encoding = 'utf-8')
            lines = src.readlines()
        if(tgt_file != None):
            out = open(tgt_file, 'w', encoding = 'utf-8')

        for index in range(1, len(self.vectors)):
            start_time = time.time()

            max_similarity = cosine_similarity(vec_a = self.vectors[index], vec_b = self.cluster_list[0].centroid)
            max_cluster_index = 0

            for cluster_index, cluster in enumerate(self.cluster_list[1:]):
                similarity = cosine_similarity(vec_a = self.vectors[index], vec_b = self.cluster_list[cluster_index].centroid)
                if(similarity > max_similarity):
                    max_similarity = similarity
                    max_cluster_index = cluster_index + 1
            if(max_similarity > self.threshold):
                self.cluster_list[max_cluster_index].add_node(index, self.vectors[index])
            else:
                new_cluster = ClusterUnit()
                new_cluster.add_node(index, self.vectors[index])
                self.cluster_list.append(new_cluster)
                del new_cluster
            
            if(index % 50 == 0):
                print('[{}] 第 {}/{} 个vector处理成功, 耗时{:.1f} s, 目前共有{}个簇.'.format(
                    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                    index,
                    len(self.vectors),
                    time.time()-start_time,
                    len(self.cluster_list)
                ))
        
        print('聚类结束，共有{}个簇.'.format(len(self.cluster_list)))
        for cluster in self.cluster_list :
            for i in cluster.node_list:
                out.write('{}'.format(lines[i]))
            out.write('--------------------\n')
            out.flush()

        return SinglePassCluster(vector_list=self.vectors, cluster_list=self.cluster_list, cluster_num = len(self.cluster_list), t = 0.6)