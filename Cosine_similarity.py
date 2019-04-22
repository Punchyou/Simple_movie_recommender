# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 19:11:54 2019

@author: Maria
"""
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


text = ['London Paris London', 'Paris Paris London']
cv = CountVectorizer()

#count the words
count_matrix = cv.fit_transform(text)
count_matrix.toarray()

#find similarity
similarity_scores = cosine_similarity(count_matrix)
similarity_scores

