#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from lda import lda

# model intialization
ldaModel = lda(K = 10)
# load training corpus
ldaModel.preprocessing(corpus = './data/corpus.txt')
# variational EM algorithm
ldaModel.EM()
# topic words
topicWords = ldaModel.topicWords(maxTopicWordsNum = 10)
# topic inference given documents
inference = ldaModel.topicInference(documents = './data/held_out_sentences.txt')