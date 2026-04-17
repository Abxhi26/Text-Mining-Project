from sklearn.feature_extraction.text import TfidfVectorizer

def get_tfidf(data):
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(data)
    return X, vectorizer