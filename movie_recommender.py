# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 19:49:17 2019

@author: Maria
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


#helper functions
def get_title_from_index(index):
    return df[df.index == index]['title'].values[0]

def get_index_from_title(title):
    return df[df.title == title]['index'].values[0]


#Read csv file
df = pd.read_csv('C:/Users/Maria/Documents/MyCodes/FirstRecommenderSystem/movie_dataset.csv')
df.head()

#features
df.columns

#select features
features = ['keywords', 'cast', 'genres', 'director']

#avoid NaN values
for feature in features:
    df[feature] = df[feature].fillna('') #fill Nas with empty string

#create a column that combines the features
def combine_features(row):
    try:
        return row['keywords'] +' '+ row['cast'] +' '+ row['genres'] +' '+ row['director']
    except:
        print('Error: ', row)
        
#pass rows from dataframe
df['combine_features'] = df.apply(combine_features, axis=1)

print(df['combine_features'].head())

#craete a matrix of counts
cv = CountVectorizer()
count_matrix = cv.fit_transform(df['combine_features'])

#calculate cosine similatiry of features
cosine_sim = cosine_similarity(count_matrix)
movie_user_likes = 'Avatar'

#get movie index from its title
movie_index = get_index_from_title(movie_user_likes)

#find similar movies
similar_movies = list(enumerate(cosine_sim[movie_index]))
sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)

#print titles of similar movies
i = 0
for movie in sorted_similar_movies:
    print (get_title_from_index(movie[0]))
    i += 1
    if i > 5:
        break
