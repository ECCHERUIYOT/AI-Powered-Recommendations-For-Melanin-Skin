import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import numpy as np

# Load the data
data = pd.read_csv('data.csv')  # Adjust file path as necessary
user_product_interactions = pd.read_csv('user_product_interactions.csv')  # Adjust file path as necessary

# Content-Based Filtering with TF-IDF on ingredients and product features
tfidf = TfidfVectorizer(stop_words='english')
ingredients_matrix = tfidf.fit_transform(data['ingredients'])
cosine_sim = cosine_similarity(ingredients_matrix, ingredients_matrix)

# Collaborative Filtering with SVD
interaction_matrix = user_product_interactions.pivot(index='user_id', columns='product_id', values='interaction')
interaction_matrix.fillna(0, inplace=True)
svd = TruncatedSVD(n_components=20, random_state=42)
user_factors = svd.fit_transform(interaction_matrix)
product_factors = svd.components_.T

# Content-Based Recommendation function using user input criteria
def content_based_recommendations(user_filters, top_n=5):
    # Apply filters
    recommendations = data.copy()
    for feature, value in user_filters.items():
        if value:
            recommendations = recommendations[recommendations[feature] == value]
    
    # Calculate similarity on filtered data and get top N results
    filtered_indices = recommendations.index.tolist()
    sim_matrix_filtered = cosine_sim[filtered_indices, :][:, filtered_indices]
    sim_scores = sim_matrix_filtered.mean(axis=0)
    top_indices = np.argsort(sim_scores)[-top_n:][::-1]
    
    return recommendations.iloc[top_indices][['product_name', 'brand_name', 'price_usd']]

# Collaborative Filtering Recommendation function
def collaborative_recommendations(user_id, top_n=5):
    user_index = interaction_matrix.index.get_loc(user_id)
    user_vector = user_factors[user_index]
    scores = np.dot(product_factors, user_vector)
    product_indices = np.argsort(scores)[-top_n:][::-1]  # Top recommendations
    return data.iloc[product_indices][['product_name', 'brand_name', 'price_usd']]

# Streamlit UI
st.title("AI-Powered Skincare Recommendations for Melanin-Rich Skin")
st.write("Get personalized skincare suggestions tailored for your unique skin tone and concerns.")

# User inputs for recommendation criteria
skin_tone = st.selectbox("Select Skin Tone:", data['skin_tone'].unique())
skin_type = st.selectbox("Select Skin Type:", data['skin_type'].unique())
price_ksh = st.slider("Maximum Price (Ksh):", int(data['price_ksh'].min()), int(data['price_ksh'].max()))
primary_category = st.selectbox("Primary Category:", data['primary_category'].unique())
secondary_category = st.selectbox("Secondary Category:", data['secondary_category'].unique())
user_id = st.number_input("Enter User ID for Collaborative Recommendations (optional)", min_value=1, step=1, value=0)

# Combine user filters into a dictionary
user_filters = {
    'skin_tone': skin_tone,
    'skin_type': skin_type,
    'price_ksh': price_ksh,
    'primary_category': primary_category,
    'secondary_category': secondary_category
}

# Content-Based Filtering Recommendations
if st.button("Get Content-Based Recommendations"):
    recommendations = content_based_recommendations(user_filters)
    st.write("Here are some product recommendations based on your preferences:")
    for index, row in recommendations.iterrows():
        st.write(f"**{row['product_name']}** by {row['brand_name']} - ${row['price_usd']}")

# Collaborative Filtering Recommendations
if st.button("Get Collaborative Recommendations") and user_id:
    recommendations = collaborative_recommendations(user_id)
    st.write("Products you might like based on similar users:")
    for index, row in recommendations.iterrows():
        st.write(f"**{row['product_name']}** by {row['brand_name']} - ${row['price_usd']}")
