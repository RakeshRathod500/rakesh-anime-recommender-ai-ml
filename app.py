import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Anime Recommendation System",
    page_icon="🎌",
    layout="wide"
)

# -------------------- THEME TOGGLE --------------------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

st.button("🌗 Toggle Dark / Light Theme", on_click=toggle_theme)

if st.session_state.theme == "dark":
    st.markdown("""
        <style>
        body { background-color: #0f172a; color: #e5e7eb; }
        .kpi { background: #020617; }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        body { background-color: #f8fafc; color: #020617; }
        .kpi { background: #ffffff; }
        </style>
    """, unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown("""
<h1 style='text-align:center;'>🎌 Anime Recommendation System</h1>
<p style='text-align:center; font-size:18px;'>
A production-grade content-based recommender using cosine similarity, genre intelligence,
popularity awareness, and explainable insights.
</p>
""", unsafe_allow_html=True)

# -------------------- FILE UPLOAD --------------------
st.sidebar.header("📂 Data Input")
file = st.sidebar.file_uploader(
    "Upload Anime CSV / Excel",
    type=["csv", "xlsx"]
)

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data(file):
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    df = df.copy()
    df['genre'] = df['genre'].fillna("Unknown")
    df['type'] = df['type'].fillna("Unknown")
    df['rating'] = df['rating'].fillna(df['rating'].mean())
    df['episodes'] = pd.to_numeric(df['episodes'], errors='coerce')
    df['episodes'] = df['episodes'].fillna(df['episodes'].median())
    df['members_log'] = np.log1p(df['members'])

    return df

if file is None:
    st.warning("📌 Upload a dataset to begin")
    st.stop()

df = load_data(file)

# -------------------- SIDEBAR CONTROLS --------------------
st.sidebar.header("⚙ Recommendation Controls")

top_n = st.sidebar.slider("🔢 Number of Recommendations", 5, 20, 10)
similarity_threshold = st.sidebar.slider("🎯 Similarity Threshold", 0.2, 0.95, 0.45)
genre_weight = st.sidebar.slider("🧠 Genre Weight", 0.5, 3.0, 1.5, 0.1)

# -------------------- FEATURE PIPELINE --------------------
@st.cache_resource
def build_similarity(df, genre_weight):
    genre_vec = TfidfVectorizer(stop_words="english", max_features=5000)
    genre_matrix = genre_vec.fit_transform(df['genre']) * genre_weight

    cat = OneHotEncoder(handle_unknown="ignore")
    type_matrix = cat.fit_transform(df[['type']])

    scaler = StandardScaler()
    num_matrix = scaler.fit_transform(df[['rating', 'episodes', 'members_log']])

    X = np.hstack([
        genre_matrix.toarray(),
        type_matrix.toarray(),
        num_matrix
    ])

    return cosine_similarity(X)

cosine_sim = build_similarity(df, genre_weight)

# -------------------- RECOMMENDER --------------------
def recommend(anime, top_n, threshold):
    idx = df[df['name'] == anime].index[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    recs = [(df.iloc[i]['name'], s) for i, s in scores[1:] if s >= threshold]
    recs = recs[:top_n]

    return pd.DataFrame(recs, columns=["Anime", "Similarity"])

# -------------------- KPI METRICS --------------------
def diversity_score(recs):
    idxs = df[df['name'].isin(recs['Anime'])].index
    sim = cosine_sim[np.ix_(idxs, idxs)]
    return round(1 - sim.mean(), 3)

# -------------------- MAIN UI --------------------
anime_list = sorted(df['name'].unique())

st.subheader("🎯 Select Anime")
anime = st.selectbox("Choose an anime", anime_list)

recs = recommend(anime, top_n, similarity_threshold)

# -------------------- KPI CARDS --------------------
col1, col2 = st.columns(2)

with col1:
    st.metric("⭐ Avg Similarity", round(recs['Similarity'].mean(), 3))

with col2:
    st.metric("🌈 Diversity Score", diversity_score(recs))

# -------------------- RECOMMENDATIONS TABLE --------------------
st.subheader("📋 Recommended Anime")
st.dataframe(recs, use_container_width=True)

# -------------------- VISUAL 1: SIMILARITY --------------------
st.subheader("📊 Similarity Distribution")
fig, ax = plt.subplots()
ax.barh(recs['Anime'], recs['Similarity'])
ax.set_xlim(recs['Similarity'].min() - 0.01, 1)
ax.invert_yaxis()
st.pyplot(fig)

# -------------------- VISUAL 2: POPULARITY --------------------
st.subheader("🔥 Popularity of Recommendations")
pop = df[df['name'].isin(recs['Anime'])]['members']
st.bar_chart(pop)

# -------------------- COMPARISON MODE --------------------
st.markdown("---")
st.subheader("🔄 Compare Two Anime")

c1, c2 = st.columns(2)

with c1:
    anime_a = st.selectbox("Anime A", anime_list, key="a")

with c2:
    anime_b = st.selectbox("Anime B", anime_list, key="b")

if anime_a and anime_b:
    ra = recommend(anime_a, 5, similarity_threshold)
    rb = recommend(anime_b, 5, similarity_threshold)

    c1.metric("Avg Similarity (A)", round(ra['Similarity'].mean(), 3))
    c2.metric("Avg Similarity (B)", round(rb['Similarity'].mean(), 3))

    c1.dataframe(ra)
    c2.dataframe(rb)

# -------------------- FOOTER --------------------
st.markdown("""
<hr>
<p style='text-align:center;'>
Built by <b>Rakesh Rathod</b> | Content-Based Recommendation System
</p>
""", unsafe_allow_html=True)
