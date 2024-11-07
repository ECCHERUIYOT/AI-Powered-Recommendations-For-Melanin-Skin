import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Set up Streamlit page configuration
st.set_page_config(page_title="AI-Powered Recommendations for Melanin-Rich Skin", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
        body { background-color: #F8F9FA; color: #333; }
        .title { font-size: 36px; color: #9053c8; text-align: center; font-weight: bold; }
        .subtitle { font-size: 24px; color: #8d7f8b; text-align: center; }
        .header { background-color: #efe657; padding: 10px; border-radius: 5px; }
        .recommendations { border: 1px solid #f683ff; border-radius: 5px; padding: 10px; background-color: #fdf4fb; }
        .stButton>button { background-color: #60b5dd; color: white; font-size: 16px; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; transition: background-color 0.3s; }
        .stButton>button:hover { background-color: #4DA2B7; }
        .sidebar .sidebar-content { background-color: #2C3E50; color: white; }
    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('data/preprocessed_data.csv')

# Load data
data = load_data()
melanated_data = data[data['skin_tone_category'] == 'melanated']

# Initialize models for content-based filtering
tfidf = TfidfVectorizer(stop_words='english')
ingredients_matrix = tfidf.fit_transform(melanated_data['ingredients'].fillna(''))
content_based_model = cosine_similarity(ingredients_matrix, ingredients_matrix)

# Function to extract top N terms for each ingredient based on TF-IDF score
def get_top_ingredients(tfidf_vector, feature_names, top_n=5):
    sorted_indices = tfidf_vector.argsort()[::-1][:top_n]
    top_terms = [feature_names[i] for i in sorted_indices]
    return ', '.join(top_terms)

# Collaborative model setup
user_product_matrix = melanated_data.pivot_table(index='author_id', columns='product_id', values='rating').fillna(0)
svd_model = TruncatedSVD(n_components=20)
latent_matrix = svd_model.fit_transform(user_product_matrix)

def content_based_recommendations(product_name, top_n=5):
    try:
        product_index = melanated_data[melanated_data['product_name'] == product_name].index[0]
    except IndexError:
        st.write("Product not found in the dataset.")
        return None

    sim_scores = list(enumerate(content_based_model[product_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    unique_recommendations = []
    seen_products = set()

    for score in sim_scores:
        product_idx = score[0]
        product_name_candidate = melanated_data.iloc[product_idx]['product_name']
        if product_name_candidate != product_name and product_name_candidate not in seen_products:
            unique_recommendations.append(product_idx)
            seen_products.add(product_name_candidate)
        if len(unique_recommendations) >= top_n:
            break

    recommendations = melanated_data.iloc[unique_recommendations][['product_name', 'brand_name', 'price_ksh']]
    feature_names = tfidf.get_feature_names_out()
    top_ingredients = [get_top_ingredients(ingredients_matrix[idx].toarray().flatten(), feature_names) for idx in unique_recommendations]
    recommendations['top_ingredients'] = top_ingredients

    return recommendations

def collaborative_recommendations(user_id, top_n=5):
    user_index = melanated_data.index[melanated_data['author_id'] == user_id].tolist()[0]
    scores = latent_matrix[user_index].dot(latent_matrix.T)
    top_recommendations = scores.argsort()[-top_n:][::-1]
    recommended_products = melanated_data.iloc[top_recommendations][['product_name', 'brand_name', 'price_ksh']]
    return recommended_products

def hybrid_recommendations(product_name, user_id=None, content_weight=0.5, collaborative_weight=0.5, top_n=2):
    content_recs = content_based_recommendations(product_name, top_n * 2)
    if content_recs is None:
        return None
    if user_id is not None:
        collaborative_recs = collaborative_recommendations(user_id, top_n * 2)
    else:
        collaborative_recs = pd.DataFrame(columns=content_recs.columns)

    hybrid_recs = pd.concat([content_recs, collaborative_recs]).drop_duplicates().reset_index(drop=True)
    hybrid_recs['score'] = (content_weight * np.arange(len(hybrid_recs)) + collaborative_weight * np.arange(len(hybrid_recs))[::-1])
    hybrid_recs = hybrid_recs.sort_values('score', ascending=False).head(top_n)
    return hybrid_recs[['product_name', 'brand_name', 'price_ksh', 'top_ingredients']]

# Sidebar for Navigation
st.sidebar.title("Navigation")
st.sidebar.markdown('<div class="header">Recommendation Options</div>', unsafe_allow_html=True)

page_selection = st.sidebar.radio("Do you want products based on:", ("Your Features/Interests?", "Your Condition/Concern?", "A Product You Already Like?", "Hybrid Recommendations"))


# Home Page: Customer-Feature Based Recommender
if page_selection == "Your Features/Interests?":
    st.markdown('<div class="title">Personalized Beauty Products Recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Get recommendations tailored to your unique skin profile!</div>', unsafe_allow_html=True)

    # Skin Type Selection  
    skin_type = st.selectbox("Skin Type", options=['dry', 'combination', 'oily', 'normal'])

    # Display Skin Type Characteristics
    if skin_type == 'oily':
        st.markdown("""
            *Oily skin tends to have enlarged pores, a shiny complexion, and a thick, dull-colored appearance. 
            People with oily skin are often prone to pimples and may notice that makeup tends to slide off more easily. 
            If this seems like your skin type, let's proceed!*     """)
    elif skin_type == 'normal':
        st.markdown(""" 
            *Normal skin is characterized by a balanced tone, smooth texture, and no visible pores. 
            People with normal skin generally have no significant issues with pimples or dryness. 
            If this seems like your skin type, let's proceed*      """)
    elif skin_type == 'dry':
        st.markdown("""
           *Dry skin often feels tight, especially after washing, and can appear dull. 
            People with dry skin may notice invisible pores, a bright complexion, and focal scales. 
            If this seems like your skin type, let's proceed!*    """)
    elif skin_type == 'combination':
        st.markdown("""
          *Combination skin typically features a greasy T-zone, which includes the forehead, nose, and chin, 
            while the U-zone (the cheeks and eyelids) may feel dry. 
            If this seems like your skin type, let's proceed!*     """)

    skin_tone = st.selectbox("Skin Tone", options=data['skin_tone'].unique())
    price_tier = st.selectbox("Price Tier", options=['Budget', 'Mid-Range', 'Premium'])

    primary_category = st.selectbox("Primary Category", options=data['primary_category'].unique())
    secondary_categories = data[data['primary_category'] == primary_category]['secondary_category'].unique()
    secondary_category = st.selectbox("Secondary Category", options=secondary_categories)

    tertiary_categories = data[data['secondary_category'] == secondary_category]['tertiary_category'].unique()
    tertiary_category = st.selectbox("Tertiary Category", options=tertiary_categories)

    # Filtered data based on user selection
    if primary_category in ['Skincare', 'Makeup']:  # Link skin type and skin tone only for these categories
        filtered_data = data[(data['skin_tone'] == skin_tone) & 
                             (data['skin_type'] == skin_type) & 
                             (data['price_tier'] == price_tier) & 
                             (data['secondary_category'] == secondary_category) & 
                             (data['primary_category'] == primary_category) & 
                             (data['tertiary_category'] == tertiary_category)]
    else:  # For other categories, skip the skin type and skin tone filter
        filtered_data = data[(data['price_tier'] == price_tier) & 
                             (data['secondary_category'] == secondary_category) & 
                             (data['primary_category'] == primary_category) & 
                             (data['tertiary_category'] == tertiary_category)]
    
    if not filtered_data.empty:
        unique_recommendations = filtered_data[['product_name', 'highlights', 'price_ksh']].drop_duplicates()
        st.markdown('<div class="recommendations">Recommended Products for Your Features:</div>', unsafe_allow_html=True)
        st.write(unique_recommendations[['product_name', 'highlights', 'price_ksh']])
    else:
        st.write("No products found that match your criteria. Try adjusting your filters.")

# Page for Ingredient Similarity-based Recommender
elif page_selection == "A Product You Already Like?":
    st.markdown("## Find products similar to the product you specify.")
    product_name = st.text_input("Enter Product Name: (eg. Deep Exfoliating Cleanser) ",  'Vitamin A Serum with 0.5% Retinol')

    if st.button("Get Recommendations"):
        recommendations = content_based_recommendations(product_name)
        if recommendations is not None:
            st.write("#### Top Products Based on Similar Ingredients:")
            st.write(recommendations)
        else:
            st.error("Product not found. Please check the name and try again.")

# Page for Hybrid Recommendation
elif page_selection == "Hybrid Recommendations":
    st.markdown('''
    ### This hybrid recommendation system helps you discover products based on: 
    1. **What you like:** Similar products based on the ingredients and features of products you’ve liked. 
    2. **What others like:** Recommendations based on the tastes of users who have similar preferences to you.
    ''')

    # Input for user ID and product name
    user_id = st.text_input("Enter your User ID: e.g., '' ", '1030820541')
    product_name = st.text_input("Enter a Product Name (e.g., 'Beauty Oil with Gold Leaf'):",  'GENIUS Collagen Calming Relief')

    if st.button("Get Hybrid Recommendations"):
        recommendations = hybrid_recommendations(user_id, product_name)
        if recommendations is not None:
            st.write("#### Recommended Products for You:")
            st.write(recommendations)
        else:
            st.error("Unable to generate recommendations. Please check your inputs.")
