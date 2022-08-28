from functools import reduce
import jieba.analyse

import pandas as pd
from math import sqrt
import math

import jieba
import numpy as np



class Similarity():
    def __init__(self, target1, target2, topK=10):
        self.target1 = target1
        self.target2 = target2
        self.topK = topK

    def vector(self):
        self.vdict1 = {}
        self.vdict2 = {}
        top_keywords1 = jieba.analyse.extract_tags(self.target1, topK=self.topK, withWeight=True)
        top_keywords2 = jieba.analyse.extract_tags(self.target2, topK=self.topK, withWeight=True)
        for k, v in top_keywords1:
            self.vdict1[k] = v
        for k, v in top_keywords2:
            self.vdict2[k] = v

    def mix(self):
        for key in self.vdict1:
            self.vdict2[key] = self.vdict2.get(key, 0)
        for key in self.vdict2:
            self.vdict1[key] = self.vdict1.get(key, 0)

        def mapminmax(vdict):
            """计算相对词频"""
            _min = min(vdict.values())
            _max = max(vdict.values())
            _mid = _max - _min
            #print _min, _max, _mid
            for key in vdict:
                vdict[key] = (vdict[key] - _min)/_mid
            return vdict

        self.vdict1 = mapminmax(self.vdict1)
        self.vdict2 = mapminmax(self.vdict2)

    def similar(self):
        self.vector()
        self.mix()
        sum = 0
        for key in self.vdict1:
            sum += self.vdict1[key] * self.vdict2[key]
        A = sqrt(reduce(lambda x,y: x+y, map(lambda x: x*x, self.vdict1.values())))
        B = sqrt(reduce(lambda x,y: x+y, map(lambda x: x*x, self.vdict2.values())))
        return sum/(A*B)



if __name__ == '__main__':
    
    topK = 20
    output_file =  'output/result.csv'

    test = pd.read_csv('data/test.csv')
    df2 = pd.DataFrame(columns=['id','answer'])

    for i in range( len(test) ):

        sc = test.loc[i,'sc']

        sc = sc.strip()

        candidate = []
        candidate.append(test.loc[i,'A'].strip())
        candidate.append(test.loc[i,'B'].strip())
        candidate.append(test.loc[i,'C'].strip())
        candidate.append(test.loc[i,'D'].strip())
        candidate.append(test.loc[i,'E'].strip())

        score = []
        for j in range(5):
            si = Similarity(sc, candidate[j], topK)
            score.append(si.similar())

        # print(np.argmax(score))
        df2.loc[i,'id'] = test.loc[i,'id']
        df2.loc[i,'answer'] = np.argmax(score) + 1


    df2['id'] = df2['id'].astype('int')
    df2['answer'] = df2['answer'].astype('int')
    df2.to_csv(output_file, encoding='utf-8', index=False)

