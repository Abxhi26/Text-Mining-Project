📊 Text Clustering (IMDB Dataset)
📌 Project Overview
This project performs text mining and clustering on movie reviews using Natural Language Processing (NLP) techniques. The goal is to group similar reviews into clusters without using predefined labels.

The project uses:

Text preprocessing
TF-IDF vectorization
K-Means clustering
Evaluation using Silhouette Score
Visualization using WordCloud
📂 Project Structure
Text-Clustering-IMDB/
│
├── data/
│   └── dataset.csv   (not included)
│
├── src/
│   ├── preprocessing.py
│   ├── vectorization.py
│   ├── clustering.py
│   └── main.py
│
├── results/
│   ├── graphs.png
│   └── wordclouds/
│
├── README.md
└── 
📊 Dataset
This project uses the IMDB Movie Reviews Dataset.

⚠️ Note: The dataset is not uploaded in this repository due to size limitations.

👉 Download it from here: https://raw.githubusercontent.com/laxmimerit/All-CSV-ML-Data-Files-Download/master/IMDB-Dataset.csv

📥 Steps:
Download the dataset from the link above
Create a folder named data/ in the project
Save the file as:
data/dataset.csv
⚙️ Technologies Used
Python
Pandas
Scikit-learn
NLTK
Matplotlib
WordCloud
How to Run
Run the project
cd src
python main.py
📈 Output
📉 Elbow Method Graph → results/graphs.png
☁️ WordCloud Images → results/wordclouds/
📊 Cluster Keywords printed in terminal
🧠 Algorithms Used
🔹 TF-IDF (Term Frequency - Inverse Document Frequency)
Converts text into numerical form by assigning importance to words.

🔹 K-Means Clustering
Groups similar text data into clusters based on similarity.

🔹 Evaluation
Elbow Method (to find optimal clusters)
Silhouette Score (to measure clustering quality)
📌 Results
Reviews are grouped into meaningful clusters
Clusters represent patterns like positive/negative sentiment
WordCloud helps visualize dominant words in each cluster
👩‍💻 Author
Manaswitha G
