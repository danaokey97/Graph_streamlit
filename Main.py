import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import networkx as nx
import matplotlib.pyplot as plt

# Download stopwords for Indonesian
nltk.download('stopwords')
stop_words = stopwords.words('indonesian')

# Custom preprocessing functions
def remove_url(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)

def remove_html(text):
    return re.sub(r'<.*?>', '', text)

def remove_emoji(text):
    emoji_pattern = re.compile("[" 
                               u"\U0001F600-\U0001F64F" 
                               u"\U0001F300-\U0001F5FF" 
                               u"\U0001F680-\U0001F6FF" 
                               u"\U0001F1E0-\U0001F1FF" 
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_numbers(text):
    return re.sub(r'\d+', '', text)

def remove_symbols(text):
    return re.sub(r'[^a-zA-Z\s]', '', text)

def case_folding(text):
    return text.lower()

def tokenize(text):
    return text.split()

def remove_stopwords(text):
    return [word for word in text if word not in stop_words]

# Input news
st.title("Proses Berita dan Analisis Similarity")
user_input = st.text_area("Masukkan Berita Baru", "", key="input_text", on_change=None)

# Button to show results after data entry
if st.button("Tampilkan Hasil"):
    if user_input:
        # 2. Split sentences by period
        sentences = [s.strip() for s in user_input.split('.') if s.strip()]
        result_list = [{'kalimat ke n': f"Kalimat ke {i+1}", 'kalimat': sentence} for i, sentence in enumerate(sentences)]
        result_df = pd.DataFrame(result_list)

        # 3. Preprocessing
        result_df['clean'] = result_df['kalimat'].apply(remove_url).apply(remove_html).apply(remove_emoji).apply(remove_symbols).apply(remove_numbers).apply(case_folding)
        result_df['tokenize'] = result_df['clean'].apply(tokenize)
        result_df['stopword removal'] = result_df['tokenize'].apply(remove_stopwords)
        result_df['final'] = result_df['stopword removal'].apply(lambda x: ' '.join(x))

        # 4. TF-IDF
        documents = result_df['final'].tolist()
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
        feature_names = tfidf_vectorizer.get_feature_names_out()
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
        tfidf_df.insert(0, 'kalimat ke n', result_df['kalimat ke n'])

        # Store TF-IDF matrix in session state
        st.session_state['tfidf_df'] = tfidf_df

        # 5. Cosine Similarity
        cosine_sim = cosine_similarity(tfidf_matrix)
        cosine_sim_df = pd.DataFrame(cosine_sim, index=result_df['kalimat ke n'], columns=result_df['kalimat ke n'])

        # Store Cosine Similarity matrix in session state
        st.session_state['cosine_sim_df'] = cosine_sim_df

        # 6. Threshold 0.01 and Adjacency Matrix
        threshold = 0.01
        adjacency_matrix = np.where(cosine_sim >= threshold, 1, 0)
        adjacency_df = pd.DataFrame(adjacency_matrix, index=result_df['kalimat ke n'], columns=result_df['kalimat ke n'])

        # Store Adjacency Matrix in session state
        st.session_state['adjacency_df'] = adjacency_df

        # 8. Graph Adjacency
        G = nx.from_numpy_array(adjacency_matrix)
        mapping = {i: f"Kalimat ke {i+1}" for i in range(len(result_df))}
        G = nx.relabel_nodes(G, mapping)

        plt.figure(figsize=(10, 10))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, font_size=10, font_color='black')
        st.session_state['graph_fig'] = plt

        # 9. Centrality Measures
        betweenness_centrality = nx.betweenness_centrality(G)
        degree_centrality = nx.degree_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)

        centrality_df = pd.DataFrame({
            'Kalimat': list(betweenness_centrality.keys()),
            'Betweenness Centrality': list(betweenness_centrality.values()),
            'Degree Centrality': list(degree_centrality.values()),
            'Closeness Centrality': list(closeness_centrality.values())
        }).sort_values(by=['Degree Centrality'], ascending=False)

        # Store centrality data and result_df in session state
        st.session_state['centrality_df'] = centrality_df
        st.session_state['result_df'] = result_df

# Display stored matrices and centrality data if available
if 'tfidf_df' in st.session_state:
    st.subheader("TF-IDF Matrix")
    st.write(st.session_state['tfidf_df'])

if 'cosine_sim_df' in st.session_state:
    st.subheader("Cosine Similarity Matrix")
    st.write(st.session_state['cosine_sim_df'])

if 'adjacency_df' in st.session_state:
    st.subheader("Adjacency Matrix")
    st.write(st.session_state['adjacency_df'])

if 'graph_fig' in st.session_state:
    st.pyplot(st.session_state['graph_fig'])

if 'centrality_df' in st.session_state:
    st.subheader("Centrality Measures")
    st.write(st.session_state['centrality_df'])

# Graph Adjacency (graph of all sentences)
if 'graph_fig_main' not in st.session_state:
    # Graph for the entire dataset, created only once
    G_main = nx.from_numpy_array(adjacency_matrix)
    mapping_main = {i: f"Kalimat ke {i+1}" for i in range(len(result_df))}
    G_main = nx.relabel_nodes(G_main, mapping_main)

    plt.figure(figsize=(10, 10))
    pos_main = nx.spring_layout(G_main)
    nx.draw(G_main, pos_main, with_labels=True, node_color='skyblue', node_size=2000, font_size=10, font_color='black')
    st.session_state['graph_fig_main'] = plt  # Store the main graph figure

# Display the main graph (entire sentences)
if 'graph_fig_main' in st.session_state:
    st.pyplot(st.session_state['graph_fig_main'])

# Only refresh ranking and graph without recalculating full process
if 'centrality_df' in st.session_state and 'result_df' in st.session_state:
    centrality_df = st.session_state['centrality_df']
    result_df = st.session_state['result_df']
    top_n = st.selectbox("Pilih top N berdasarkan Degree Centrality", [3, 5, 10])

    # Get the top N rows based on Degree Centrality
    top_n_df = centrality_df.nlargest(top_n, 'Degree Centrality')

    # Merge top N with result_df to get final sentences
    top_n_final_df = pd.merge(top_n_df[['Kalimat']], result_df, left_on='Kalimat', right_on='kalimat ke n')

    # Display top N results
    st.subheader(f"Top {top_n} Kalimat berdasarkan Degree Centrality")
    st.write(top_n_final_df[['kalimat ke n', 'final']])

    # Generate the graph based on the selected top N sentences
    top_n_sentences = top_n_final_df['kalimat ke n'].tolist()
    top_n_indices = [i for i, sentence in enumerate(result_df['kalimat ke n']) if sentence in top_n_sentences]

    # Filter the cosine similarity matrix for the selected top N sentences
    filtered_cosine_sim = cosine_sim[top_n_indices, :][:, top_n_indices]

    # Generate the adjacency matrix for the selected top N
    threshold = 0.01
    filtered_adjacency_matrix = np.where(filtered_cosine_sim >= threshold, 1, 0)

    # Create a graph based on the filtered adjacency matrix
    G_top_n = nx.from_numpy_array(filtered_adjacency_matrix)
    mapping_top_n = {i: f"Kalimat ke {top_n_indices[i]+1}" for i in range(len(top_n_indices))}
    G_top_n = nx.relabel_nodes(G_top_n, mapping_top_n)

    # Plot the top N graph
    plt.figure(figsize=(10, 10))
    pos_top_n = nx.spring_layout(G_top_n)
    nx.draw(G_top_n, pos_top_n, with_labels=True, node_color='skyblue', node_size=2000, font_size=10, font_color='black')
    st.pyplot(plt)  # Plot the updated graph for top N
