import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load models
sentiment_model_path = 'models/sentiment_analysis_model.pkl'
knn_model_path = 'models/knn_model.pkl'
svd_model_path = 'models/svd_model.pkl'
xgb_model_path = 'models/xgboost_model.pkl'

# Load pre-trained models
with open(sentiment_model_path, 'rb') as file:
    sia = pickle.load(file)
with open(knn_model_path, 'rb') as file:
    knn = pickle.load(file)
with open(svd_model_path, 'rb') as file:
    svd_model = pickle.load(file)
with open(xgb_model_path, 'rb') as file:
    xgb_model = pickle.load(file)

# Load preprocessed data
data = pd.read_csv('data/preprocessed_data.csv')

# Filter melanated skin tone category
melanated_data = data[data['skin_tone_category'] == 'melanated']

# Preprocessing for ingredient-based recommendations
vectorizer = TfidfVectorizer(stop_words='english')
ingredient_matrix = vectorizer.fit_transform(melanated_data['ingredients'].fillna(""))

# Create User-Product Interaction Matrix for collaborative filtering
user_product_matrix = data.pivot_table(index='author_id', columns='product_id', values='rating_x').fillna(0)

# Apply SVD to create latent matrix
svd = TruncatedSVD(n_components=20)
latent_matrix = svd.fit_transform(user_product_matrix)

# Streamlit App Interface
st.title("AI-Powered Skincare Recommendations for Melanin-Rich Skin")
st.sidebar.header("Filter Preferences")

# Sidebar filters
skin_type = st.sidebar.selectbox("Select Skin Type", options=['combination', 'oily', 'dry', 'normal'])
price_tier = st.sidebar.selectbox("Select Price Tier", options=['Budget', 'Mid-Range', 'Premium'])
product_input = st.sidebar.text_input("Enter Product Name (optional)")

# Filtering based on user input
filtered_data = melanated_data[(melanated_data['skin_type'] == skin_type) & (melanated_data['price_tier'] == price_tier)]

# Select recommendation method
rec_method = st.selectbox("Select Recommendation Method", options=['Sentiment-based', 'Ingredient-based (KNN)', 'Collaborative Filtering (SVD)'])

# Check if the sentiment score column exists, calculate if not
if 'sentiment_score' not in data.columns:
    analyzer = SentimentIntensityAnalyzer()
    
    def calculate_sentiment_score(text):
        sentiment = analyzer.polarity_scores(text)
        return sentiment['compound']

    data['sentiment_score'] = data['review_text'].apply(calculate_sentiment_score)

# Filter melanated skin tone category
melanated_data = data[data['skin_tone_category'] == 'melanated']
if rec_method == 'Sentiment-based':
    # Input user ID
    user_id = st.text_input("Enter User ID for Sentiment-based Recommendations", '7085042638')
    
    # Generate and display recommendations
    if st.button("Generate Recommendations"):
        user_reviews = melanated_data[melanated_data['author_id'] == user_id]
        if not user_reviews.empty:  # Check if user_reviews is not empty
            positive_reviews = user_reviews[user_reviews['sentiment_score'] > 0.05]
            recommendations = positive_reviews['product_name'].head(5)
            st.write("Top recommendations based on positive sentiment:")
            st.write(recommendations)
        else:
            st.write("No reviews found for the specified user ID.")

elif rec_method == 'Ingredient-based (KNN)':
    # Input product ID
    product_id = st.text_input("Enter Product ID for Ingredient-based Recommendations", 'P442546')
    
    if st.button("Get Ingredient-based Recommendations"):
        product_index = melanated_data[melanated_data['product_id'] == product_id].index[0]
        distances, indices = knn.kneighbors(ingredient_matrix[product_index], n_neighbors=5)
        recommended_products = melanated_data.iloc[indices[0]]
        st.write("Products with similar ingredients:")
        st.write(recommended_products[['product_id', 'product_name', 'brand_name', 'price_tier']])

elif rec_method == 'Collaborative Filtering (SVD)':
    # Input user ID
    user_id = st.text_input("Enter User ID for Collaborative Filtering", '1030820541')
    
    if st.button("Get Collaborative Recommendations"):
        user_index = data.index[data['author_id'] == user_id].tolist()[0]
        scores = latent_matrix[user_index].dot(latent_matrix.T)
        top_recommendations = scores.argsort()[-5:][::-1]
        recommendations = data['product_name'].iloc[top_recommendations]
        st.write("Collaborative Filtering Recommendations:")
        st.write(recommendations)