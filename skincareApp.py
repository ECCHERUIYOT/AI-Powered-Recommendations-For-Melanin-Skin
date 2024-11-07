pip install streamlit
import streamlit as st
import pandas as pd
import pickle
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Set up Streamlit page configuration
st.set_page_config(page_title="AI-Powered Skincare Recommendations for Melanin-Rich Skin", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
        body { background-color: #F8F9FA; color: #333; }
        .title { font-size: 36px; color: #9053c8; text-align: center; font-weight: bold; }
        .subtitle { font-size: 24px; color: #7F8C8D; text-align: center; }
        .header { background-color: #efe657; padding: 10px; border-radius: 5px; }
        .recommendations { border: 1px solid #f683ff; border-radius: 5px; padding: 10px; background-color: #fdf4fb; }
        .stButton>button { background-color: #60b5dd; color: white; font-size: 16px; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; transition: background-color 0.3s; }
        .stButton>button:hover { background-color: #4DA2B7; }
        .sidebar .sidebar-content { background-color: #2C3E50; color: white; }
    </style>
""", unsafe_allow_html=True)


# Load pre-trained models and data
@st.cache_data
def load_data():
    return pd.read_csv('data/preprocessed_data.csv')

@st.cache_resource
def load_models():
    with open('models/content_based_model.pkl', 'rb') as file:
        content_based_model = pickle.load(file)
    with open('models/svd_model.pkl', 'rb') as file:
        svd_model = pickle.load(file)
    return content_based_model, svd_model

# Load data and models
data = load_data()
content_based_model, svd_model = load_models()

# Vectorize ingredients and create sparse matrix for content-based filtering
tfidf = TfidfVectorizer(stop_words='english')
ingredients_matrix = tfidf.fit_transform(data['ingredients'].fillna(''))
# Initialize NearestNeighbors model for content-based filtering
content_based_model = NearestNeighbors(n_neighbors=6, metric='cosine', n_jobs=-1)
content_based_model.fit(ingredients_matrix)

# Collaborative Filtering Setup
top_users = data['author_id'].value_counts().head(2000).index
top_products = data['product_id'].value_counts().head(10000).index
filtered_data = data[data['author_id'].isin(top_users) & data['product_id'].isin(top_products)]
user_product_matrix = filtered_data.pivot_table(index='author_id', columns='product_id', values='rating').fillna(0)
user_product_sparse = csr_matrix(user_product_matrix.values)
svd = TruncatedSVD(n_components=20)
latent_matrix = svd.fit_transform(user_product_sparse)
user_index_map = {user_id: idx for idx, user_id in enumerate(user_product_matrix.index)}

# Define function to get top ingredients based on TF-IDF score
def get_top_ingredients(tfidf_vector, feature_names, top_n=5):
    sorted_indices = tfidf_vector.argsort()[::-1][:top_n]
    top_terms = [feature_names[i] for i in sorted_indices]
    return ', '.join(top_terms)

# Collaborative Filtering Recommendation function
def collaborative_recommendations(user_id, top_n=5):
    if user_id not in user_index_map:
        st.error("User ID not found in the dataset.")
        return None

    user_index = user_index_map[user_id]
    scores = latent_matrix[user_index].dot(latent_matrix.T)
    top_recommendations = scores.argsort()[-top_n-1:][::-1][1:]
    recommended_indices = user_product_matrix.columns[top_recommendations]
    recommended_products = data[data['product_id'].isin(recommended_indices)][['product_name', 'brand_name', 'rating', 'price_ksh']].drop_duplicates()
    sorted_recommendations = recommended_products.sort_values(by='rating', ascending=False).head(top_n)
    
    return sorted_recommendations

# Content-Based Recommendation function with sorting by rating
def content_based_recommendations(product_name, top_n=5):
    try:
        product_index = data[data['product_name'] == product_name].index[0]
    except IndexError:
        st.error("Product not found in the dataset.")
        return None

    distances, indices = content_based_model.kneighbors(ingredients_matrix[product_index], n_neighbors=top_n+1)

    # Skip the first result as it will be the product itself
    recommendations = data.iloc[indices[0][1:]][['product_name', 'brand_name', 'rating', 'price_ksh']]

    # Get the top ingredients for each recommended product
    feature_names = tfidf.get_feature_names_out()
    top_ingredients = []
    for index in indices[0][1:]:
        row_vector = ingredients_matrix[index].toarray().flatten()
        top_ingredients.append(get_top_ingredients(row_vector, feature_names))

    recommendations['top_ingredients'] = top_ingredients

    # Sort recommendations by rating in descending order
    sorted_recommendations = recommendations.sort_values(by='rating', ascending=False)

    return sorted_recommendations

# Hybrid Recommendation function
def hybrid_recommendations(user_id, product_name, top_n=5, alpha=0.5):
    content_recs = content_based_recommendations(product_name, top_n)
    collaborative_recs = collaborative_recommendations(user_id, top_n)

    if content_recs is None or collaborative_recs is None:
        return None

    combined_recs = content_recs.copy()
    combined_recs['collab_rating'] = collaborative_recs['rating'].values[:len(combined_recs)]
    combined_recs['score'] = alpha * combined_recs['rating'] + (1 - alpha) * combined_recs['collab_rating']
    combined_recs = combined_recs.sort_values(by='score', ascending=False)

    return combined_recs[['product_name', 'brand_name', 'price_ksh', 'rating', 'score']]

# Sidebar for Navigation
st.sidebar.title("Navigation")
st.sidebar.markdown('<div class="header">Recommendation Options</div>', unsafe_allow_html=True)

page_selection = st.sidebar.radio( "Do you want products based on:",
    ("Your Features?", "A Product You Already Like?", "Hybrid Recommendations"))

# Home Page: Customer-Feature Based Recommender
# Updated Sidebar for user selection
if page_selection == "Your Features?":
    st.markdown('<div class="title">Personalized Skincare Recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Get recommendations tailored to your unique skin profile!</div>', unsafe_allow_html=True)

    # Filters for user input
    skin_type = st.selectbox("Select Skin Type", options=['dry', 'combination', 'oily', 'normal'])
    skin_tone_category = st.selectbox("Select Skin Tone Category", options=['melanated', 'non-melanated'])

    # Filter skin_tone values based on selected skin_tone_category
    light_skin_tones = ['light', 'fair', 'lightMedium', 'fairLight', 'porcelain', 'olive']
    dark_skin_tones = ['deep', 'mediumTan', 'medium', 'tan', 'rich', 'dark']

    if skin_tone_category == 'non-melanated':
        skin_tone = st.selectbox("Select Skin Tone", options=light_skin_tones)
    else:  # melanated
        skin_tone = st.selectbox("Select Skin Tone", options=dark_skin_tones)
    
    price_tier = st.selectbox("Select Price Tier", options=['Budget', 'Mid-Range', 'Premium'])
    primary_category = st.selectbox("Select Primary Category", options=data['primary_category'].unique())
    secondary_category = st.selectbox("Select Secondary Category", options=data['secondary_category'].unique())
    tertiary_category = st.selectbox("Select Tertiary Category", options=data['tertiary_category'].unique())

    # Filter data based on user input
    filtered_data = data[
        (data['skin_type'] == skin_type) & 
        (data['skin_tone_category'] == skin_tone_category) & 
        (data['skin_tone'] == skin_tone) & 
        (data['price_tier'] == price_tier) &
        (data['primary_category'] == primary_category) & 
        (data['secondary_category'] == secondary_category) & 
        (data['tertiary_category'] == tertiary_category)
    ]

    if not filtered_data.empty:
        unique_recommendations = filtered_data[['product_name', 'brand_name', 'rating', 'price_ksh']].drop_duplicates()
        unique_recommendations = unique_recommendations.sort_values(by='rating', ascending=False).head(10)
        st.markdown('<div class="recommendations">Recommended Products for Your Features:</div>', unsafe_allow_html=True)
        st.write(unique_recommendations)
    else:
        st.write("No products found that match your criteria. Try adjusting your filters.")

# Page for Ingredient Similarity-based Recommender
elif page_selection == "A Product You Already Like?":
    st.markdown("## Find products similar to the product you specify.")
    product_name = st.text_input("Enter Product Name (e.g., 'Aloe Vera Gel'):",  'Deep Exfoliating Cleanser')

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
    user_id = st.text_input("Enter your User ID: e.g., '2117812169' ", '7085042638')
    product_name = st.text_input("Enter a Product Name (e.g., 'Next Level Moisturizer'):", 'Aloe Vera Gel')

    if st.button("Get Hybrid Recommendations"):
        recommendations = hybrid_recommendations(user_id, product_name)
        if recommendations is not None:
            st.write("#### Recommended Products for You:")
            st.write(recommendations)
        else:
            st.error("Unable to generate recommendations. Please check your inputs.")
