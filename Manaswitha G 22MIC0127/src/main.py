import pandas as pd
from preprocessing import clean_text
from vectorization import get_tfidf
from clustering import *

# load dataset
data = pd.read_csv("../data/dataset.csv")

# select text column
data = data[['review']]
data.columns = ['text']

# reduce size (for speed)
data = data.sample(5000, random_state=0)

# preprocessing
data['clean_text'] = data['text'].apply(clean_text)

# vectorization
X, vectorizer = get_tfidf(data['clean_text'])

# elbow method
elbow_method(X)

# clustering
k = 5
model, labels = apply_kmeans(X, k)

# evaluation
evaluate(X, labels)

# keywords
get_keywords(model, vectorizer, k)

# wordclouds
generate_wordclouds(data['clean_text'], labels, k)