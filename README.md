# MELANIN-CENTERED SKINCARE RECOMMENDATION SYSTEM
![product](https://github.com/user-attachments/assets/a99f82c1-111e-4b1d-a447-518fac92dedd)


# Project Overview
This project addresses a significant gap in the skincare industry: **the lack of tailored product recommendations for women of colour**, who often have unique skincare needs such as hyperpigmentation, dryness, and sensitivity. Current recommendation systems tend to generalize recommendations without considering the specific concerns of malanated women, leading to lower customer satisfaction and less effective skincare routines for this demographic.
To bridge this gap, we developed a recommendation system powered by advanced AI techniques, specifically targeting melanin-rich skin. Using a combination of machine learning models, such as content-based filtering (utilizing cosine similarity for ingredient-based similarity), collaborative filtering (using SVD), and sentiment analysis (LSTM model and Sentiment Intensity Analyzer), the system generates personalized product recommendations. Key elements include sentiment classification of reviews, ingredient similarity analysis, and a hybrid recommendation approach to capture and enhance user-product interactions.
The model can provide recommendations based on product similarity, collaborative filtering for user-product interaction, and specific user needs. Moreover, the system offers a user-friendly Streamlit interface, allowing users to input a product name and receive similar product suggestions based on their skincare profile.
By catering specifically to women of color’s skincare needs, **this project aims to increase accessibility to effective skincare solutions and improve customer satisfaction within this underrepresented demographic.**

---

## Problem Statement
Women of color represent a significant demographic in the beauty and skincare industry, yet they face limited access to skincare products tailored to their specific needs, such as hyper- pigmentation, dryness, and sensitivity. Most available recommendation systems overlook the unique skin concerns of women of color, offering general suggestions rather than targeted solutions. This gap impacts consumer satisfaction, as melanated women often struggle to find effective products for their melanin-rich skin. 
This project aims to develop a recommendation system using advanced AI techniques to cater specifically to women’s skincare needs. By integrating machine learning, content- based filtering, collaborative filtering, and sentiment analysis, the system will offer personalized skincare recommendations. Leveraging skin_tone (Author's skin tone (e.g. fair, tan, etc.) as a classification feature, we aim to distinguish and target products that align with melanin-rich skin concerns.


### Objectives
1. Develop a melanin-centered skincare recommendation system using deep learning, tailored for Black women’s unique skin needs.
2. Utilize content-based and collaborative filtering along with sentiment analysis to enhance recommendation accuracy. 
3. Deploy an accessible Streamlit interface for personalized, user-friendly skincare suggestions. 

### Stakeholders
1. *Users*: Black women seeking tailored skincare solutions. 
2. *Skincare Brands*: Companies interested in product insights for melanin-rich skin. 
3. *Healthcare Professionals*: Dermatologists who may use the system as a recommendation tool. 

---

## Data Understanding:
The dataset was collected via a Python scraper and contains:
- Product Information: Over 8,000 beauty products from the Sephora online store, including product and brand names, prices, ingredients, ratings, and various features. 
- User Reviews: Approximately 1 million reviews across over 2,000 products in the skincare category. These reviews include user appearances, skin types, and review ratings.

The key features include:
- Product Features: `product_id`, `product_name`, `brand_name`, `ingredients`, `rating`, `price_ksh`, `new`, `out_of_stock`, `highlights`. 
- Review Features: `author_id`, `rating`, `review_text`, `skin_type`, `skin_tone`, and
`helpfulness`.

---

### Exploratory Data Analysis

#### 1. Count of Products by Skin Type
![skintype](https://github.com/user-attachments/assets/84e65f71-ebc1-4007-b4af-dd5b4f295572)

* The distribution shows that products labelled for combination skin are the most common, followed by those for dry, normal, and then oily skin. This insight can guide product selection based on prevalent skin types and consumer demand within the Black women demographic.

#### 2. Top 20 Recommended Skincare Products
![Top 20 Recommended Products for Melanin Skin](https://github.com/ECCHERUIYOT/AI-Powered-Skincare-Recommendations-For-Melanin-Skin/blob/Echeruiyot/images/image2.png)
* For many skin tones, there's a strong emphasis on gentle yet effective skincare products, with the top three recommendations focusing on hydration and sun protection.
* The Dew Dream Cleansing Balm leads the pack with about 55 recommendations, suggesting that gentle makeup removal is a priority for this skin type. The second most recommended product is the Signature Moisturizer, followed by the NA°39 Facial Sunscreen Mist with SPF 39, which indicates that maintaining skin hydration and sun protection are crucial concerns for melanated skin. The prevalence of brightening, revitalizing, and gentle products in the top 10 (such as Absolue Soft Cream and Evercalm Gentle Cleansing Gel) suggests that users with melanated skin often seek products that address hyperpigmentation and sensitivity while maintaining skin barrier health.

* The list also includes several treatment-focused products like glycolic serums and exfoliators in the middle to lower rankings, indicating that while these are important, they're secondary to basic skin protection and hydration needs.

#### 3. Out of stock Products and Recommendations based on skin tone
![Out of stock Products and Recommendations based on skin tone](https://github.com/ECCHERUIYOT/AI-Powered-Skincare-Recommendations-For-Melanin-Skin/blob/Echeruiyot/images/image3.png)
* The Recommended pie chart illustrates the percentage of products recommended versus those that are not. A significant portion (85%) of products are not marked as recommended, indicating a possible quality or suitability gap. This could help identify where product performance might fall short or suggest a need for more tailored product options.

* The Products pie chart shows the balance between new and existing products. The larger percentage of existing products (91%) suggests that the platform maintains a consistent range of products, with newer items being introduced selectively. This distribution can provide insights into the inventory management and refresh rates of the catalogue over time.

#### 4. Skin Tone Distribution by Price Tier
![skintone](https://github.com/user-attachments/assets/ba640a77-dcf7-4500-a006-b2611bbc9d9a)
* There is a noticeable price difference across skin tones. Products for melanated skin tend to have lower average prices, than non-melanated skin tones, highlighting more affordable options specifically formulated or suited for melanated skin. 

#### 5. Heatmap of Product Count by Skin Tone Category and Skin Type
![Heatmap of Product Count by Skin Tone Category and Skin Type](https://github.com/ECCHERUIYOT/AI-Powered-Skincare-Recommendations-For-Melanin-Skin/blob/Echeruiyot/images/image5.png)
* The heatmap analysis of skin_type and skin_tone_category highlights important insights that align closely with our objectives. Our data reveals a concentration of products available for combination and dry skin types, particularly within lighter skin tones. However, there is a notable scarcity of options for deeper skin tones.This gap underscores the limited market focus on skincare for melanin-rich skin concerns, such as hyperpigmentation, dryness, and sensitivity, which are often more pronounced in deeper skin tones.

* These findings directly support our business problem: many existing recommendation systems fail to provide targeted solutions for specific skin tones. The evident lack of specialized options for drier skin in deeper tones emphasizes an opportunity to develop and recommend products that address this unique need. By prioritizing these underserved areas, our recommendation system can significantly enhance satisfaction and efficacy for women of color seeking products that work for their melanin-rich skin.

---

## Modeling Approach
The recommendation system integrates multiple models:

1. **Customer Feature-Based Recommender**: Recommends products based solely on customer characteristics.
2. **Highlight-Based Recommender**: Recommends products based on selected highlights, catering to specific product features.
3. **Content-Based Filtering Model**: Recommends similar products based on ingredients and features using cosine similarity.
4. **Collaborative Filtering with SVD**: Leverages user-product interactions to recommend products based on user preferences.
5. **Hybrid Model**: Combines content-based and collaborative filtering to provide robust, varied recommendations.

### Evaluation Metrics
- **Precision@k**: Measures the relevance of the top K-recommended products.
- **Recall@k**: Evaluates the system's effectiveness in covering user preferences in the top K recommendations.
- **F1-Score**: Ensures a balance between precision and recall.
- **Hit Rate@k**: Determines if users find a relevant product in the top 5.
- **Normalized Discounted Cumulative Gain (NDCG)**:Considers the position of relevant items, rewarding relevant items appearing earlier in the recommendations list.

---
### Model Performance
1. **High Precision and Recall at *K*=1:**
- **Precision@1** of 0.99 and Recall@1 of 0.91 indicate that the model is very accurate when recommending a single top item. Users are very likely to find a relevant item in the top 1 recommendation.
  
2. **Performance Drops as *𝐾* Increases:**
- Precision drops significantly as **𝐾** increases (e.g., **Precision@5** of 0.24 and **Precision@20** of 0.06). This suggests that as more items are recommended, fewer of them are relevant.

3. **Hit Rate@k == 0.99**
- With a Hit Rate of 0.99 across all values of **𝐾**,the model is consistently recommending at least one relevant item within the top **𝐾** recommendations for almost every user.

4.  The **NDCG@10:** 0.99 result is excellent, indicating that the ranking provided by your recommendation model closely matches the ideal order, where the most relevant items appear at the top.
---
## DEPLOYMENT
The Streamlit [App](https://melanin-centered-skincare-recommendation-system.streamlit.app/) enables users to select product categories and highlights for personalized recommendations, covering the full beauty spectrum:
- **Recommendation Pages**: An intuitive interface where users can select options based on preferences (e.g., skincare, makeup, fragrance).
 ![Home Page 1](https://github.com/user-attachments/assets/367f5794-5e65-4718-b342-5f8ef4e0e6d0)
![Home Page 2](https://github.com/user-attachments/assets/78d7db20-b1e5-4d96-8bdb-7031629aafea)
- **Product Highlights-Based Recommendations**: Users can select specific highlights (e.g., "hydrating," "anti-aging," ) and receive products that meet these needs.
![Highlights](https://github.com/user-attachments/assets/d5d19dc2-b7df-420a-9f38-b7c0121c4784)
- **Recommendation Options**: Users receive product recommendations tailored to specific beauty needs or similar to products they already enjoy.
![Content-Based](https://github.com/user-attachments/assets/86e48078-39c1-428e-8a8d-cf3e34ce3e30)

---

## CONCLUSION
- This project successfully developed a skincare recommendation system specifically tailored for individuals with melanin-rich skin, addressing a significant gap in personalized skincare recommendations. 
- By using content-based filtering on ingredient similarities, collaborative filtering through SVD, and sentiment analysis, the system delivers customized product suggestions that align with users’ unique skin concerns and preferences. The Streamlit interface enhances user experience, providing an intuitive way for users to access their personalized recommendations easily.
- Our approach demonstrates the potential of AI-driven solutions to meet niche market needs and make the skincare industry more inclusive. The system’s design aims not only to recommend relevant products but also to boost user satisfaction by prioritizing highly rated products based on sentiment analysis.

## CHALLENGES
Throughout the project, some notable challenges included:
- **Data Limitations**: The dataset was limited in brand, categories and product diversity, which restricted the range of recommendations.
- **Model Optimization**: Balancing the hybrid model’s performance was challenging, as both content-based and collaborative methods required fine-tuning to avoid biases.
- **Sentiment Analysis Accuracy**: Ensuring accurate sentiment analysis was critical for prioritizing products, but limitations in natural language processing led to occasional inaccuracies in sentiment scoring.

## RECOMMENDATIONS
To enhance the functionality and user engagement of this skincare recommendation system, several improvements are recommended:
- **Enhance User Engagement**: Implement feedback mechanisms to allow users to rate and provide input on recommended products. This feedback loop would help refine the recommendations and improve overall user satisfaction.
- **Expand Dataset Diversity**: Increase the dataset size and diversity to include more skincare brands, product varieties, and user profiles. This would ensure a broader range of personalized recommendations and allow the system to cater to a wider audience.
- **Implement Educational Resources**: Include educational content about skincare for melanin-rich skin, helping users better understand product ingredients, skincare routines, and how to address specific skin concerns.
- **Improve User Interface**: Enhance the user interface for greater ease of use, enabling users to quickly and seamlessly find tailored product recommendations.

## NEXT STEPS:
To further enhance the system and expand its impact, the following steps are recommended:
- **Partner with Skincare Experts and Dermatologists**: Collaborate with skincare specialists to review and validate product recommendations, ensuring that suggestions meet the specific needs of melanin-rich skin.
- **Expand Dataset Coverage**: Increase the dataset size and diversity to cover a broader range of brands, products, and user profiles. This will improve recommendation accuracy and offer users more choices.
- **Integrate Educational Resources**: Include guides and educational content about skincare ingredients and routines for melanin-rich skin, empowering users to make informed decisions.
- **Implement Feedback Loops**: Add mechanisms for users to rate and provide feedback on recommendations, allowing the system to learn and improve continuously.
- **Collaborate with Brands**: Partner with major skincare brands like Garnier and other beauty companies to integrate this recommendation system directly into their platforms, making it accessible to a wider audience.
This project serves as a foundation for creating an inclusive and personalized skincare experience, supporting users with tailored product recommendations that address their unique skincare needs.
---

## Getting Started
To run this project locally, follow these steps:

### Prerequisites
- Python 3.8 or higher
- Required Libraries: `pandas`, `numpy`, `scikit-learn`, `nltk`, `tensorflow`, `streamlit`

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/ECCHERUIYOT/Melanin-Centered-Skincare-Recommendation-System.git
   cd Melanin-Centered-Skincare-Recommendation-System
   ```
2. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Usage
To start the Streamlit application:
   ```bash
    streamlit run skincareApp.py
   ```

---

## Repository Navigation
- **`data/`**: Contains raw and processed data supporting the recommendation system.
- **`images/`**: Saved images.
- **`models/`**: Saved machine learning models.
- **`index.ipynb/`**: Jupyter notebook for data preprocessing, EDA and model training.
- **`skincareApp.py`**: Main Streamlit application file.

[Presentation Link](https://www.canva.com/design/DAGVhd5J2HY/PEtTgT8s4gSNxLThvX2mTA/edit)
# THANK YOU!!
---

