#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import codecs
import jieba
import re
import random
import math
from scipy.special import psi

# Corpus
class Document(object):
    def __init__(self, itemIdList, itemCountList, wordCount):
        # the list of distinct terms in the document
        self.itemIdList = itemIdList
        # the list of number of the existence of corresponding terms
        self.itemCountList = itemCountList
        # the number of total words (not terms)
        self.wordCount = wordCount
        
# LDA       
class lda(object):
    def __init__(self, *, K = 10):
        # K: number of topic
        self.K = K
        
    def preprocessing(self, *, corpus = './data/corpus.txt', stopwords = './data/stopwords.dic'):
        # segmentation, stopwords filtering, collect documents as objects of class Document
        
        # read the list of stopwords
        file = codecs.open(stopwords,'r','utf-8')
        stopwords = [line.strip() for line in file]
        file.close()
        
        # read the corpus for training
        file = codecs.open(corpus,'r','utf-8')
        documents = [document.strip() for document in file] 
        file.close()
        
        docs = []
        word2id = {}
        id2word = {}
        
        currentWordId = 0
        for document in documents:
            word2Count = {}
            # segmentation
            segList = jieba.cut(document)
            for word in segList: 
                word = word.lower().strip()
                # filter the stopwords
                if len(word) > 1 and not re.search('[0-9]', word) and word not in stopwords:
                    if word not in word2id:
                        word2id[word] = currentWordId
                        id2word[currentWordId] = word
                        currentWordId += 1
                    if word in word2Count:
                        word2Count[word] += 1
                    else:
                        word2Count[word] = 1
            itemIdList = []
            itemCountList = []
            wordCount = 0
    
            for word in word2Count.keys():
                itemIdList.append(word2id[word])
                itemCountList.append(word2Count[word])
                wordCount += word2Count[word]
    
            docs.append(Document(itemIdList, itemCountList, wordCount))

        self.docs = docs
        self.word2id = word2id
        self.id2word = id2word
        self.N = len(docs) # number of documents for training
        self.M = len(word2id)# number of distinct terms
        
        
    def EM(self, *, iterInference = 20, iterEM = 20, alpha = 5):
        # iteration times of variational inference
        self.iterInference = iterInference
        # iteration times of variational EM algorithm
        self.iterEM = iterEM
        
        # initial value of hyperparameter alpha
        self.alpha = alpha
        # sufficient statistic of alpha
        self.alphaSS = 0
        # the topic-word distribution (beta in D. Blei's paper)
        self.varphi = np.zeros([self.K, self.M])
        # topic-word count, this is a sufficient statistic to calculate varphi
        self.nzw = np.zeros([self.K, self.M])
        # topic count, sum of nzw with w ranging from [0, M-1], for calculating varphi
        self.nz = np.zeros([self.K])
        
        # inference parameter gamma
        self.gamma = np.zeros([self.N, self.K])
        # inference parameter phi
        self.phi = np.zeros([self.__maxItemNum(), self.K])
       
        # initialization of the model parameter varphi, the update of alpha is ommited
        self.__initialLdaModel()
        
        # variational EM Algorithm
        for iteration in range(0, self.iterEM): 
            self.nz = np.zeros([self.K])
            self.nzw = np.zeros([self.K, self.M])
            self.alphaSS = 0
            # E-Step
            for d in range(0, self.N):
                self.__variationalInference(d, self.docs, self.gamma, self.phi)
                gammaSum = 0
                for z in range(0, self.K):
                    gammaSum += self.gamma[d, z]
                    self.alphaSS += psi(self.gamma[d, z])
                self.alphaSS -= self.K * psi(gammaSum)
        
                for w in range(0, len(self.docs[d].itemIdList)):
                    for z in range(0, self.K):
                        self.nzw[z][self.docs[d].itemIdList[w]] += self.docs[d].itemCountList[w] * self.phi[w, z]
                        self.nz[z] += self.docs[d].itemCountList[w] * self.phi[w, z]
        
            # M-Step
            self.__updateVarphi()
    
    def topicWords(self, *, maxTopicWordsNum = 10):
        # maxTopicWordsNum: the number of the words realted to each topic
        topicwords = []
        for z in range(0, self.K):
            	ids = self.varphi[z, :].argsort()
            	topicword = []
            	for j in ids:
            		topicword.insert(0, self.id2word[j])
            	topicwords.append(topicword[0 : min(maxTopicWordsNum, len(topicword))])
        return topicwords
    
    def topicInference(self, documents = './data/held_out_sentences.txt'):
        # read the corpus to be inferred
        testDocs = []
        file = codecs.open(documents,'r','utf-8')
        testDocuments = [document.strip() for document in file] 
        file.close()
        
        for d in range(0, len(testDocuments)):
            document = testDocuments[d]
            word2Count = {}
            # segmentation
            segList = jieba.cut(document)
            for word in segList: 
                word = word.lower().strip()
                if word in self.word2id:
                    if word in word2Count:
                        word2Count[word] += 1
                    else:
                        word2Count[word] = 1
                          
            itemIdList = []
            itemCountList = []
            wordCount = 0
    
            for word in word2Count.keys():
                itemIdList.append(self.word2id[word])
                itemCountList.append(word2Count[word])
                wordCount += word2Count[word]
    
            testDocs.append(Document(itemIdList, itemCountList, wordCount))
        
        # topic inference
        gamma = np.zeros([len(testDocuments), self.K])
        for d in range(0, len(testDocs)):
            phi = np.zeros([len(testDocs[d].itemIdList), self.K])
            self.__variationalInference(d, testDocs, gamma, phi)
            inferZ = []
            for i in range(0, len(gamma)):
                inferZ.append(gamma[i, :].argmax())
        return inferZ
        
    
    def __initialLdaModel(self):
        for z in range(0, self.K):
            for w in range(0, self.M):
                self.nzw[z, w] += 1.0/self.M + random.random()
                self.nz[z] += self.nzw[z, w]
        self.__updateVarphi()
    
    def __maxItemNum(self):
        num = 0
        for d in range(0, self.N):
            if len(self.docs[d].itemIdList) > num:
                num = len(self.docs[d].itemIdList)
        return num
    
    def __updateVarphi(self):
        for z in range(0, self.K):
            for w in range(0, self.M):
                if(self.nzw[z, w] > 0):
                    self.varphi[z, w] = math.log(self.nzw[z, w]) - math.log(self.nz[z])
                else:
                    self.varphi[z, w] = -100
                    
    def __variationalInference(self, d, docs, gamma, phi):
        phisum = 0
        oldphi = np.zeros([self.K])
        digamma_gamma = np.zeros([self.K])
        
        for z in range(0, self.K):
            gamma[d][z] = self.alpha + docs[d].wordCount * 1.0 / self.K
            digamma_gamma[z] = psi(gamma[d][z])
            for w in range(0, len(docs[d].itemIdList)):
                phi[w, z] = 1.0 / self.K
    
        for iteration in range(0, self.iterInference):
            for w in range(0, len(docs[d].itemIdList)):
                phisum = 0
                for z in range(0, self.K):
                    oldphi[z] = phi[w, z]
                    phi[w, z] = digamma_gamma[z] + self.varphi[z, docs[d].itemIdList[w]]
                    if z > 0:
                        phisum = math.log(math.exp(phisum) + math.exp(phi[w, z]))
                    else:
                        phisum = phi[w, z]
                for z in range(0, self.K):
                    phi[w, z] = math.exp(phi[w, z] - phisum)
                    gamma[d][z] =  gamma[d][z] + docs[d].itemCountList[w] * (phi[w, z] - oldphi[z])
                    digamma_gamma[z] = psi(gamma[d][z])

        