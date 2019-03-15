import warnings 
warnings.filterwarnings("ignore")

#################################### IMPORT DATA ######################################
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from nltk.corpus import stopwords 

training_data = pd.read_csv("apps/classification/movies_metadata_2.csv")

raw_genres = training_data.genres
genre = []
synopsis = []

for i in range (0,len(raw_genres)):
    x = eval(raw_genres[i])
    if len(x) > 0:
        g = x[0]['name']
        if g != "TV Movie" and g != "War" and g != "Foreign" and g != "History" and g != "Music" and g != "Western":
            genre.append(g)
            synopsis.append(training_data.overview[i])

set_genre = list(set(genre))
count_genre = []
fix_genre = []
fix_syn = []

for i in range(0,len(set_genre)):
    count_genre.append([set_genre[i],0])
    
for i in range(0,len(genre)):
    for j in range(0,len(count_genre)):
        if genre[i] in count_genre[j][0]:
            if count_genre[j][1] < 700:
                fix_genre.append(genre[i])
                fix_syn.append(synopsis[i])
                count_genre[j][1] = count_genre[j][1] + 1

synopsis = fix_syn
genre = fix_genre

################################ DATA PREPROCESSING ##################################
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Clean stopwords function
def clean_stopwords(syn): 
    stop_words = set(stopwords.words('english'))
    clean = []
    for sent in syn:
        new = []
        token = word_tokenize(sent)
        for word in token:
            if word not in stop_words:
                new.append(word)
        clean.append(' '.join(new))
    
    return(clean)


import re

# Create empty array to store processed synopsis
syn = []

# Data preprocessing for synopsis
for sen in range(0,len(synopsis)):
#     Remove special characters and punctuation
    document = re.sub(r'\W', ' ', str(synopsis[sen]))
    
#     Set synopsis into lower case
    document = document.lower()
    
#     remove single character
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    
#     remove double space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    
#     save processed synopsis into array
    syn.append(document)
    
# clean stopwords
clean = clean_stopwords(syn)

############################### CONVERT TEXT INTO NUMBERS ######################################
# Also remove english stopwords
count_vect = CountVectorizer(min_df=5, max_df=0.7)
X_train_counts = count_vect.fit_transform(clean)

# save vector into model
pickle.dump(count_vect.vocabulary_, open("count_vector.pkl","wb"))

######################################### TF-IDF ##############################################
from sklearn.feature_extraction.text import TfidfTransformer

#transform vector into tf-idf
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

#save tf-idf
pickle.dump(tfidf_transformer, open("tfidf.pkl","wb"))

############################ SPLIT DATA INTO TRAIN AND TEST ####################################
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, genre, test_size=0.25, random_state=42)


########################## TRAINING CLASSIFICATION MODEL - KNN ###################################
from sklearn.naive_bayes import MultinomialNB

# Training model with Naive Bayes Algorithm
classifierNB = MultinomialNB().fit(X_train, y_train)
classifierNB.fit(X_train, y_train) 

# save model
pickle.dump(classifierNB, open("nb.pkl", "wb"))

############################## PREDICTION FUNCTION #################################################
def main(input_text):
    import pickle 
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer

    docs_new = input_text
    docs_new = [docs_new]

	#LOAD MODEL
    loaded_vec = CountVectorizer(vocabulary=pickle.load(open("count_vector.pkl", "rb")))
    loaded_tfidf = pickle.load(open("tfidf.pkl","rb"))
    loaded_model = pickle.load(open("nb.pkl","rb"))

    X_new_counts = loaded_vec.transform(docs_new)
    X_new_tfidf = loaded_tfidf.transform(X_new_counts)
    predicted = loaded_model.predict(X_new_tfidf)

    return predicted[0]