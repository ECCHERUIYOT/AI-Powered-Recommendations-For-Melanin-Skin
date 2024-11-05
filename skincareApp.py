import streamlit as st
import pandas as pd
import pickle
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Set up Streamlit page configuration (must be the first Streamlit command)
st.set_page_config(page_title="AI-Powered Skincare Recommendations for Melanin-Rich Skin", layout="wide")

# Load pre-trained models and data (using caching for efficiency)
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

# Sidebar for Navigation
st.sidebar.title("Pages")
page_selection = st.sidebar.radio(
    "Do you want recommendations based on:",
    ("Your Features?", "A Product You Already Like?", "Collaborative Filtering")
)

# Data preprocessing for melanated products
melanated_data = data[data['skin_tone_category'] == 'melanated']

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

    # Display unique, top-rated recommendations sorted by rating
    if not filtered_data.empty:
        unique_recommendations = filtered_data[['product_name', 'brand_name', 'rating', 'price_ksh']].drop_duplicates()
        unique_recommendations = unique_recommendations.sort_values(by='rating', ascending=False).head(10)
        st.write("### Recommended Products for You:")
        st.write(unique_recommendations)
    else:
        st.write("No products found that match your criteria. Try adjusting your filters.")

# Page for Ingredient Similarity-based Recommender
elif page_selection == "A Product You Already Like?":
    st.title("Ingredient Similarity-Based Recommendations")
    st.write("This recommender finds products with similar ingredients to the product you specify. It's ideal if you're looking for alternatives with similar ingredients.")

    # Preprocessing for ingredient-based recommendations
    vectorizer = TfidfVectorizer(stop_words='english')
    ingredients_matrix = vectorizer.fit_transform(melanated_data['ingredients'].fillna(""))
    cosine_sim = cosine_similarity(ingredients_matrix, ingredients_matrix)

    # Function to extract top N terms for each ingredient based on TF-IDF score
    def get_top_ingredients(tfidf_vector, feature_names, top_n=5):
        sorted_indices = tfidf_vector.argsort()[::-1][:top_n]
        top_terms = [feature_names[i] for i in sorted_indices]
        return ', '.join(top_terms)

    # Function to recommend products based on content similarity
    def content_based_recommendations(product_name, top_n=5):
        try:
            product_index = melanated_data[melanated_data['product_name'] == product_name].index[0]
        except IndexError:
            return None  # Return None to handle in Streamlit

        # Calculate similarity scores
        sim_scores = list(enumerate(cosine_sim[product_index]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Initialize a list to keep track of unique product recommendations
        unique_recommendations = []
        seen_products = set()  # Set to track seen product names

        for score in sim_scores:
            product_idx = score[0]
            product_name_candidate = melanated_data.iloc[product_idx]['product_name']

            # Add to recommendations if not already seen and not the input product
            if product_name_candidate != product_name and product_name_candidate not in seen_products:
                unique_recommendations.append(product_idx)
                seen_products.add(product_name_candidate)
            
            # Stop if we've collected enough recommendations
            if len(unique_recommendations) >= top_n:
                break

        # Get recommended products
        recommendations = melanated_data.iloc[unique_recommendations][['product_name', 'brand_name', 'rating', 'price_ksh']]

        # Add a column for top ingredients
        feature_names = vectorizer.get_feature_names_out()
        top_ingredients = []
        for index in unique_recommendations:
            row_vector = ingredients_matrix[index].toarray().flatten()
            top_ingredients.append(get_top_ingredients(row_vector, feature_names))
        recommendations['top_ingredients'] = top_ingredients

        # Sort recommendations by rating in descending order
        sorted_recommendations = recommendations.sort_values(by='rating', ascending=False)

        return sorted_recommendations

    # Input for product name
    product_name = st.text_input("Enter a Product Name (e.g., 'Deep Exfoliating Cleanser', 'Aloe Vera Gel'):")

    # Generate and display recommendations
    if st.button("Get Recommendations"):
        recommendations = content_based_recommendations(product_name)
        if recommendations is not None:
            st.write("### Products with Similar Ingredients:")
            st.write(recommendations)
        else:
            st.error("Product Name not found. Please check the name and try again.")

# Page for Collaborative Filtering-based Recommender
elif page_selection == "Collaborative Filtering":
    st.title("Collaborative Filtering-Based Recommendations")
    st.write("This recommender uses collaborative filtering to suggest products based on user-product interaction data. Enter your user ID to receive recommendations based on your past interactions.")

    # Create user-product interaction matrix for collaborative filtering
    user_product_matrix = data.pivot_table(index='author_id', columns='product_id', values='rating_x').fillna(0)

    # Apply SVD to create latent matrix
    svd = TruncatedSVD(n_components=20)
    latent_matrix = svd.fit_transform(user_product_matrix)

    # Input for user ID
    user_id = st.text_input("Enter your User ID for Collaborative Filtering Recommendations:")

    # Generate and display recommendations
    if st.button("Get Collaborative Recommendations"):
        try:
            user_index = data.index[data['author_id'] == user_id].tolist()[0]
            scores = latent_matrix[user_index].dot(latent_matrix.T)
            top_recommendations = scores.argsort()[-5:][::-1]
            recommendations = data.iloc[top_recommendations][['product_name', 'brand_name', 'rating', 'price_ksh']].drop_duplicates()
            unique_recommendations = recommendations.sort_values(by='rating', ascending=False)
            st.write("### Recommended Products for You:")
            st.write(unique_recommendations)
        except IndexError:
            st.error("User ID not found. Please check the ID and try again.")
