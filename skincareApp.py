import streamlit as st
import pandas as pd
import pickle
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Set up Streamlit page configuration
st.set_page_config(page_title="AI-Powered Skincare Recommendations for Melanin-Rich Skin", layout="wide")

# Load pre-trained models and data
@st.cache_data
def load_data():
    return pd.read_csv('data/preprocessed_data.csv')

@st.cache_resource
def load_models():
    with open('models/sentiment_analysis_model.pkl', 'rb') as file:
        sia = pickle.load(file)
    with open('models/knn_model.pkl', 'rb') as file:
        knn = pickle.load(file)
    with open('models/svd_model.pkl', 'rb') as file:
        svd_model = pickle.load(file)
    with open('models/xgboost_model.pkl', 'rb') as file:
        xgb_model = pickle.load(file)
    return sia, knn, svd_model, xgb_model

# Load data and models
data = load_data()
sia, knn, svd_model, xgb_model = load_models()

# Vectorize ingredients and create a cosine similarity matrix for content-based filtering
tfidf = TfidfVectorizer(stop_words='english')
ingredients_matrix = tfidf.fit_transform(data['ingredients'].fillna(''))
cosine_sim = cosine_similarity(ingredients_matrix, ingredients_matrix)

# Apply SVD for Collaborative Filtering
user_product_matrix = data.pivot_table(index='author_id', columns='product_id', values='rating').fillna(0)
svd = TruncatedSVD(n_components=20)
latent_matrix = svd.fit_transform(user_product_matrix)

# Define function to get top ingredients based on TF-IDF score
def get_top_ingredients(tfidf_vector, feature_names, top_n=5):
    sorted_indices = tfidf_vector.argsort()[::-1][:top_n]
    top_terms = [feature_names[i] for i in sorted_indices]
    return ', '.join(top_terms)

# Collaborative Filtering Recommendation function
def collaborative_recommendations(user_id, top_n=5):
    try:
        user_index = data.index[data['author_id'] == user_id].tolist()[0]
    except IndexError:
        st.error("User not found in the dataset.")
        return None

    scores = latent_matrix[user_index].dot(latent_matrix.T)
    top_recommendations = scores.argsort()[-top_n:][::-1]
    recommended_products = data.iloc[top_recommendations][['product_name', 'brand_name', 'rating', 'price_ksh']]
    return recommended_products

# Content-Based Recommendation function with sorting by rating
def content_based_recommendations(product_name, top_n=5):
    try:
        product_index = data[data['product_name'] == product_name].index[0]
    except IndexError:
        st.error("Product not found in the dataset.")
        return None

    sim_scores = list(enumerate(cosine_sim[product_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    unique_recommendations = []
    seen_products = set()
    for score in sim_scores:
        product_idx = score[0]
        product_name_candidate = data.iloc[product_idx]['product_name']
        if product_name_candidate != product_name and product_name_candidate not in seen_products:
            unique_recommendations.append(product_idx)
            seen_products.add(product_name_candidate)
        if len(unique_recommendations) >= top_n:
            break

    recommendations = data.iloc[unique_recommendations][['product_name', 'brand_name', 'rating', 'price_ksh']]
    feature_names = tfidf.get_feature_names_out()
    top_ingredients = [get_top_ingredients(ingredients_matrix[i].toarray().flatten(), feature_names) for i in unique_recommendations]
    recommendations['top_ingredients'] = top_ingredients

    # Sort recommendations by rating in descending order
    recommendations = recommendations.sort_values(by='rating', ascending=False)

    return recommendations

# Hybrid Recommendation function
def hybrid_recommendations(user_id, product_name, top_n=5, alpha=0.5):
    collaborative_recs = collaborative_recommendations(user_id, top_n)
    content_recs = content_based_recommendations(product_name, top_n)

    if collaborative_recs is None and content_recs is None:
        st.error("No recommendations found.")
        return None

    # Combining the scores using a weighted approach
    combined_recs = collaborative_recs.copy()
    combined_recs['score'] = alpha * collaborative_recs['rating'] + (1 - alpha) * content_recs['rating'].values[:len(combined_recs)]
    
    # Sort by the combined score
    combined_recs = combined_recs.sort_values(by='score', ascending=False)

    return combined_recs[['product_name', 'brand_name', 'price_ksh', 'rating', 'score']]

# Sidebar for Navigation
st.sidebar.title("Pages")
page_selection = st.sidebar.radio(
    "Do you want recommendations based on:",
    ("Your Features?", "A Product You Already Like?", "Hybrid Recommendations")
)

# Home Page: Customer-Feature Based Recommender
if page_selection == "Your Features?":
    st.title("Personalized Skincare Recommendations")
    st.markdown("### Get recommendations tailored to your unique skin profile!")

    # Filters for user input
    skin_type = st.selectbox("Select Skin Type", options=['combination', 'oily', 'dry', 'normal'])
    skin_tone_category = st.selectbox("Select Skin Tone Category", options=['melanated', 'non-melanated'])
    price_tier = st.selectbox("Select Price Tier", options=['Budget', 'Mid-Range', 'Premium'])
    secondary_category = st.selectbox("Select Secondary Category", options=data['secondary_category'].unique())

    # Filter data based on user input
    filtered_data = data[
        (data['skin_type'] == skin_type) & 
        (data['skin_tone_category'] == skin_tone_category) & 
        (data['price_tier'] == price_tier) &
        (data['secondary_category'] == secondary_category)
    ]

    if not filtered_data.empty:
        unique_recommendations = filtered_data[['product_name', 'brand_name', 'rating', 'price_ksh']].drop_duplicates()
        unique_recommendations = unique_recommendations.sort_values(by='rating', ascending=False).head(10)
        st.write("#### Recommended Products for Your Features:")
        st.write(unique_recommendations)
    else:
        st.write("No products found that match your criteria. Try adjusting your filters.")

# Page for Ingredient Similarity-based Recommender
elif page_selection == "A Product You Already Like?":
    st.markdown("## Find products with similar ingredients to the product you specify.")
    product_name = st.text_input("Enter Your favorite  Product Name (e.g., 'Deep Exfoliating Cleanser'):", 'Aloe Vera Gel')

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
    ### This hybrid recommendation system helps you discover new products based on: 
    1. **What you like:** Similar products based on the ingredients and features of products you’ve liked. 
    2. **What others like:** Recommendations based on the tastes of users who have similar preferences to you.
    ''')

    # Input for user ID and product name
    user_id = st.text_input("Enter your User ID: e.g., '2117812169' ", '7085042638')
    product_name = st.text_input("Enter a Product Name (e.g., 'Next Level Moisturizer'):", 'Aloe Vera Gel')

    if st.button("Get Hybrid Recommendations"):
        recommendations = hybrid_recommendations(user_id, product_name)
        if recommendations is not None:
            st.write("#### Recommended Products for You:")
            st.write(recommendations)
        else:
            st.error("Unable to generate recommendations. Please check your inputs.")
