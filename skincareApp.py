import streamlit as st
import pandas as pd
import numpy as np
import ast
import pickle
import base64
from sklearn.neighbors import NearestNeighbors

# Set up Streamlit page configuration
st.set_page_config(page_title="AI-Powered Beauty Products Recommendations", layout="wide")

# Function to load image and convert it to base64
def get_img_as_base64(file_path):
    with open(file_path, "rb") as file:
        return base64.b64encode(file.read()).decode()

# Load the image as base64
img_base64 = get_img_as_base64("images/bg.jpeg")

# Define the CSS for sidebar background with reduced opacity
sidebar_bg_img = f"""
<style>
[data-testid="stSidebar"] > div:first-child {{  position: relative; }}
[data-testid="stSidebar"] > div:first-child::before {{ content: ""; position: absolute;  top: 0;
    left: 0;  right: 0;  bottom: 0;  background-image: url("data:image/jpeg;base64,{img_base64}");
    background-size: cover;   background-repeat: no-repeat;
    opacity: 0.2;  z-index: -1;  /* Places image behind the content */ }}
[data-testid="stSidebar"] * {{  font-size: 1.017em;  /* Adjust this value to control the font size */  }}
</style>
"""
# Inject CSS into the Streamlit app
st.markdown(sidebar_bg_img, unsafe_allow_html=True)  # Sidebar styling

# Custom CSS for styling
st.markdown("""
    <style>
        body { background-color: #F8F9FA; color: #333; }
        .title { font-size: 36px; color: #9053c8; text-align: center; font-weight: bold; }
        .subtitle { font-size: 24px; color: #d33f9f; text-align: center; }
        .header { background-color: #efe657; padding: 10px; border-radius: 5px; }
        .recommendations { border: 1px solid #f683ff; border-radius: 5px; padding: 10px; background-color: #fdf4fb; }
        .stButton>button { background-color: #000000; color: white; font-size: 16px; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; transition: background-color 0.3s; }
        .stButton>button:hover { background-color: #4d9c44; color: white; }
        .sidebar .sidebar-content { background-color: #2C3E50; color: white; }
    </style>
""", unsafe_allow_html=True)

# Load and cache data
@st.cache_data(show_spinner=False)
def load_data():
    data = pd.read_csv('Data/preprocessed_data.csv')
    data['highlights_clean'] = data['highlights'].apply(clean_highlights)
    return data

# Function to convert strings representing lists into actual lists
def clean_highlights(highlights):
    try:
        highlights_list = ast.literal_eval(highlights)
        if isinstance(highlights_list, list):
            highlights_list = list(set([term.strip().lower().replace("&", "and") for term in highlights_list]))
        return highlights_list
    except:
        return []

# Cache model components to optimize performance
@st.cache_data(show_spinner=False)
def setup_models(data):
    melanated_data = data[data['skin_tone_category'] == 'melanated']
    
    # Load saved models (Content-based and Collaborative)
    with open('Models/content_based_model.pkl', 'rb') as file:
        content_based_model = pickle.load(file)
    
    with open('Models/tfidf.pkl', 'rb') as file:
        tfidf = pickle.load(file)

    with open('Models/ingredients_matrix.pkl', 'rb') as file:
        ingredients_matrix = pickle.load(file)
    
    return melanated_data, content_based_model, ingredients_matrix, tfidf

# Get top ingredients based on TF-IDF score
def get_top_ingredients(tfidf_vector, feature_names, top_n=5):
    sorted_indices = tfidf_vector.argsort()[::-1][:top_n]
    top_terms = [feature_names[i] for i in sorted_indices]
    return ', '.join(top_terms)

def content_based_recommendations(product_name, melanated_data, content_based_model, ingredients_matrix, tfidf):
    # Get the index of the input product_name
    product_index = melanated_data[melanated_data['product_name'] == product_name].index[0]
    
    # Find the nearest neighbors for the product
    distances, indices = content_based_model.kneighbors(ingredients_matrix[product_index], n_neighbors=11)
    
    # Create a list of tuples with product indices and their corresponding similarity scores (distances)
    sim_scores = list(zip(indices[0], distances[0]))  # [0] to get the first row of results

    # Sort the list of tuples by the similarity score (distance)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=False)  # Sorting by ascending distance

    # Extract the product names for the top recommendations (exclude the product itself)
    recommended_products = []
    for idx, _ in sim_scores[1:]:  # Excluding the first one as it is the input product itself
        recommended_products.append(melanated_data.iloc[idx]['product_name'])
    
    return recommended_products
    
# Load and set up data and models
data = load_data()
melanated_data, content_based_model, ingredients_matrix, tfidf = setup_models(data)
unique_highlights = melanated_data['highlights_clean'].explode().unique()

# Example of fitting the NearestNeighbors model
content_based_model = NearestNeighbors(n_neighbors=11, metric='cosine')
content_based_model.fit(ingredients_matrix)

# Sidebar for Navigation
st.sidebar.title("Welcome ...")
st.sidebar.markdown('<div class="header">Recommendation Options</div>', unsafe_allow_html=True)
page_selection = st.sidebar.radio("Do you want products based on:", ("Your Needs?", "Product Highlights?", "A Product You Already Like?"))

# Home Page: Customer-Feature Based Recommender
if page_selection == "Your Needs?":
    st.markdown('<div class="title">MELANIN-CENTERED SKINCARE RECOMMENDER SYSTEM</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Adjust the filters to match your personal needs.</div>', unsafe_allow_html=True)

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

    skin_tone = st.selectbox("Skin Tone", options=melanated_data['skin_tone'].unique())
    price_tier = st.selectbox("Price Tier", options=['Budget', 'Mid-Range', 'Premium'])

    primary_category = st.selectbox("Primary Category", options=melanated_data['primary_category'].unique())
    secondary_categories = melanated_data[melanated_data['primary_category'] == primary_category]['secondary_category'].unique()
    secondary_category = st.selectbox("Secondary Category", options=secondary_categories)

    tertiary_categories = melanated_data[melanated_data['secondary_category'] == secondary_category]['tertiary_category'].unique()
    tertiary_category = st.selectbox("Tertiary Category", options=tertiary_categories)

    # Filtered data based on user selection
    if primary_category in ['Skincare', 'Makeup']:  # Link skin type and skin tone only for these categories
        filtered_data = melanated_data[(melanated_data['skin_tone'] == skin_tone) & 
                             (melanated_data['skin_type'] == skin_type) & 
                             (melanated_data['price_tier'] == price_tier) & 
                             (melanated_data['secondary_category'] == secondary_category) & 
                             (melanated_data['primary_category'] == primary_category) & 
                             (melanated_data['tertiary_category'] == tertiary_category)]
    else:  # For other categories, skip the skin type and skin tone filter
        filtered_data = melanated_data[(melanated_data['price_tier'] == price_tier) & 
                             (melanated_data['secondary_category'] == secondary_category) & 
                             (melanated_data['primary_category'] == primary_category) & 
                             (melanated_data['tertiary_category'] == tertiary_category)]
    
    if not filtered_data.empty:
        unique_recommendations = filtered_data[['product_name', 'highlights', 'price_ksh']].drop_duplicates()
        st.markdown('<div class="recommendations">Recommended Products:</div>', unsafe_allow_html=True)
        st.write(unique_recommendations[['product_name', 'highlights', 'price_ksh']])
    else:
        st.write("No products found that match your criteria. Try adjusting your filters.")

elif page_selection == "Product Highlights?":
    st.markdown('<div class="title">PRODUCT HIGHLIGHTS-BASED RECOMMENDATIONS</div>',unsafe_allow_html=True)
    st.markdown(""" Select key highlights such as "hydrating," "vegan," or "good for dark circles," and let us recommend products that fit your needs.    """)
    
    selected_highlights = st.multiselect(
    "Product Highlights",
    options=unique_highlights,
    default=['hydrating']  # Default value can be adjusted based on preference
)

    # Get product recommendations based on selected highlights
    def recommend_products_by_highlights(selected_highlights, top_n=10):
        selected_highlights = [highlight.lower() for highlight in selected_highlights]
        filtered_data = melanated_data[melanated_data['highlights_clean'].apply(lambda x: any(highlight in x for highlight in selected_highlights))]
        unique_recommendations = filtered_data.drop_duplicates(subset=['product_name']).head(top_n)
        
        return unique_recommendations[['product_name', 'brand_name', 'price_ksh', 'highlights_clean']]

    recommended_products = recommend_products_by_highlights(selected_highlights, top_n=5)

    if not recommended_products.empty:
        st.markdown('<div class="recommendations">Products that match your selection:</div>', unsafe_allow_html=True)
        st.write(recommended_products)
    else:
        st.write("Sorry, no products match your selected highlights.")

# Page for Ingredient Similarity-based Recommender
elif page_selection == "A Product You Already Like?":
    st.markdown('<div class="title">PRODUCT SIMILARITY-BASED RECOMMENDATIONS</div>', unsafe_allow_html=True)
    st.markdown("""
    If you already have a product you love, simply enter its name below. We'll find similar products with shared ingredients to help you expand your beauty routine.
    """)
    product_name = st.text_input("Product Name: (eg. Vitamin A Serum with 0.5% Retinol) ", 'Deep Exfoliating Cleanser')

    if st.button("Get Recommendations"):
        recommendations = content_based_recommendations(product_name, melanated_data, content_based_model, ingredients_matrix, tfidf)
        if recommendations is not None:
            st.markdown(f'<div class="recommendations">Products similar to {product_name}:</div>', unsafe_allow_html=True)
            st.write(recommendations)
        else:
            st.error("Product not found. Please check the name and try again.")
