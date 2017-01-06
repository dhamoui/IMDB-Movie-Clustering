# This program uses a set of 5000 movies on IMDB. For each movie in the dataset, it identifies the most similar
# movie. Further versions will increase the dataset and the number
# of recommended movies.

import pandas as pd
import numpy as np

#Read in data

data = pd.read_csv("movie_metadata.csv", header=0)

#Keep only these columns

data_use = data.ix[:,['genres','plot_keywords','movie_title','actor_1_name',
                      'actor_2_name','actor_3_name','director_name','imdb_score']]

#Edit the names of the movie titles to remove \xa0

data_use['movie_title'] = [i.replace("\xa0","") for i in list(data_use['movie_title'])]

#Remove missing values
clean_data = data_use.dropna(axis = 0)

#Drop duplicate movies, reset the index of the movies
clean_data = clean_data.drop_duplicates(['movie_title'])
clean_data = clean_data.reset_index(drop=True)

#Combine actors and directors into one feature called 'people'
people_list = []
for i in range(clean_data.shape[0]):
    name1 = clean_data.ix[i,'actor_1_name'].replace(" ","_")
    name2 = clean_data.ix[i,'actor_2_name'].replace(" ","_")
    name3 = clean_data.ix[i,'actor_3_name'].replace(" ","_")
    name4 = clean_data.ix[i,'director_name'].replace(" ","_")
    people_list.append("|".join([name1,name2,name3,name4]))
clean_data['people'] = people_list

#Splits up keywords, genres and people by delimiter "|"

from sklearn.feature_extraction.text import CountVectorizer

def token(text):
    return(text.split("|"))

#Creates two variables for each feature, categorical values

cv_kw=CountVectorizer(max_features=100,tokenizer=token)
keywords = cv_kw.fit_transform(clean_data["plot_keywords"])
keywords_list = ["kw_" + i for i in cv_kw.get_feature_names()]

cv_ge=CountVectorizer(tokenizer=token)
genres = cv_ge.fit_transform(clean_data["genres"])
genres_list = ["genres_"+ i for i in cv_ge.get_feature_names()]

cv_pp=CountVectorizer(max_features=100,tokenizer=token )
people = cv_pp.fit_transform(clean_data["people"])
people_list = ["pp_"+ i for i in cv_pp.get_feature_names()]

#print(keywords_list)
#print("-----------------------------------------")
#print(keywords)

genres.todense()

#Creates the data to apply the clustering to, and the feature list
cluster_data = np.hstack([keywords.todense(),genres.todense(),people.todense()*3])
criterion_list = keywords_list+genres_list+people_list

from sklearn.cluster import KMeans

#Apply KMeans to the data
#Create variable category that categorizes the data
#Also create variable dataframe of 'category' variable

mod = KMeans(n_clusters=200)
category = mod.fit_predict(cluster_data)
category_dataframe = pd.DataFrame({"category":category},index = clean_data['movie_title'])

#From clean_data dataset, list genres, movie_title, and people for movies in category 1
clean_data.ix[list(category_dataframe['category'] == 1),['genres','movie_title','people']]

# If a movie is in the list clean_data
# set movie_cluster = to the movie_name and its category
# For score: Find a list in clean_data of movies that are in same cluster as given movie cluster
# sort list by descending scores
# Keep only movies with different names


def recommend(movie_name,recommend_number = 5):
    if movie_name in list(clean_data['movie_title']):
        movie_cluster = category_dataframe.ix[movie_name,'category']
        score = clean_data.ix[list(category_dataframe['category'] == movie_cluster),['imdb_score','movie_title']]
        sort_score = score.sort_values(['imdb_score'],ascending=[0])
        sort_score = sort_score[sort_score['movie_title'] != movie_name]
        recommend_number = min(sort_score.shape[0],recommend_number)
        recommend_movie = list(sort_score.iloc[range(recommend_number),1])
        print(recommend_movie)
    else:
        print("Can't find this movie!")

# Asks for 15 movie recommendations similar to Spider-Man 3
# Movies are sorted by highest IMDB score
recommend('Spider-Man 3',10)

clean_data.ix[(clean_data['movie_title']=='Spider-Man 3') | (clean_data['movie_title']=='Spider-Man')]

mod2 = KMeans(n_clusters=100)
category2 = mod2.fit_transform(cluster_data)

print(mod.cluster_centers_.shape)
print(mod.labels_.shape)
print(category.shape)
print('----------')
print(mod2.cluster_centers_.shape)
print(mod2.labels_.shape)
print(category2.shape)

temp_dist_list = []
index1_list = []
index2_list = []
for i in range(0,4659):
    temp_dist = 100
    for j in range(0,4659):
        if i != j:
            dist=0
            dist = (sum((category2[i]-category2[j])**2))**0.5
            if (dist<temp_dist):
                temp_dist = dist
                index2 = j
            
    temp_dist_list.append(temp_dist)
    index1_list.append(i)
    index2_list.append(index2)

movie_list = clean_data[['movie_title']]
movie_list['nearest_neighbor'] = index2_list
movie_list['distance'] = temp_dist_list
movie_list.sort('distance')
movie_list['most_similar_movie'] = movie_list['nearest_neighbor']

for i in range(0,4659):
    movie_list['most_similar_movie'][i] = clean_data['movie_title'][movie_list['nearest_neighbor'][i]]

movie_list[['movie_title','most_similar_movie']]
