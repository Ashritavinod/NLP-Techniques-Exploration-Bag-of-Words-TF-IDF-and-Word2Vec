#  NLP Techniques Exploration: Bag-of-Words, TF-IDF, and Word2Vec

This repository contains a Jupyter Notebook that explores fundamental **Natural Language Processing (NLP)** techniques using popular Python libraries like **scikit-learn** and **Gensim**.

The notebook provides a hands-on walkthrough of transforming text data into numerical representations suitable for tasks like text classification, similarity analysis, and semantic exploration.

---

##  Techniques Covered

###  Part 1: Bag-of-Words (BoW) & N-Grams

- Uses `CountVectorizer` from `scikit-learn` to convert raw text into a **Bag-of-Words** matrix.
- Explores **n-gram modeling**, specifically **bigrams (2-word combinations)**, using the `ngram_range` parameter.
- Captures basic syntactic patterns and frequency-based representation.

---

###  Part 2: TF-IDF

- Introduces **TF-IDF (Term Frequency-Inverse Document Frequency)** using `TfidfVectorizer`.
- Assigns importance-based weights to words based on how uniquely they appear in documents.
- Helps reduce noise from overly common words and improves relevance-based modeling.

---

###  Part 3: Word2Vec Embeddings

- Utilizes `Gensim` to train a **Word2Vec** model on raw text.
- Learns **dense vector representations** that capture **semantic relationships** between words.
- Preprocessing includes:
  - Tokenization with `nltk.sent_tokenize`
  - `gensim.utils.simple_preprocess`
- Explores similarity queries:
  - Similar words
  - Word-to-word cosine similarity
- Visualizes embeddings in **3D using PCA and Plotly**.

---

##  Tools & Libraries

- `scikit-learn` – for CountVectorizer and TfidfVectorizer
- `Gensim` – for Word2Vec and preprocessing
- `NLTK` – for sentence tokenization
- `Plotly` – for 3D visualization
- `Pandas`, `NumPy`, `Matplotlib` – for support and analysis

---


