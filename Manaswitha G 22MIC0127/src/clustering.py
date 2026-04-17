import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from wordcloud import WordCloud
import os

def elbow_method(X):
    wcss = []
    for i in range(1, 8):
        km = KMeans(n_clusters=i, random_state=0)
        km.fit(X)
        wcss.append(km.inertia_)

    plt.plot(range(1, 8), wcss)
    plt.title("Elbow Method")
    plt.xlabel("Clusters")
    plt.ylabel("WCSS")

    os.makedirs("../results", exist_ok=True)
    plt.savefig("../results/graphs.png")
    plt.close()

def apply_kmeans(X, k):
    model = KMeans(n_clusters=k, random_state=0)
    labels = model.fit_predict(X)
    return model, labels

def evaluate(X, labels):
    score = silhouette_score(X, labels)
    print("Silhouette Score:", score)

def get_keywords(model, vectorizer, k):
    terms = vectorizer.get_feature_names_out()

    for i in range(k):
        print("\nCluster", i)
        center = model.cluster_centers_[i]
        indices = center.argsort()[-10:]
        for ind in indices:
            print(terms[ind])

def generate_wordclouds(data, labels, k):
    os.makedirs("../results/wordclouds", exist_ok=True)

    for i in range(k):
        text = " ".join(data[labels == i])

        wc = WordCloud(width=500, height=250).generate(text)

        plt.imshow(wc)
        plt.axis("off")
        plt.title("Cluster " + str(i))

        plt.savefig(f"../results/wordclouds/cluster_{i}.png")
        plt.close()