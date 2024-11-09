# AI-Powered-Recommendations-For-Melanin-Skin

![2d248f5507752726ddc2198b39071e4e](https://private-user-images.githubusercontent.com/134943380/384533124-d370cd58-a84b-472c-afa3-ea13a3424085.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzExMzkyNjksIm5iZiI6MTczMTEzODk2OSwicGF0aCI6Ii8xMzQ5NDMzODAvMzg0NTMzMTI0LWQzNzBjZDU4LWE4NGItNDcyYy1hZmEzLWVhMTNhMzQyNDA4NS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQxMTA5JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MTEwOVQwNzU2MDlaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0wY2Q4MjFmNjZkZjkxYzYyYjFmYmJhZDFjMTFhOTc1ZjM3ZjJjZDkwY2ZkYTU0MzQwMWVkODZkMTgyYTczZmM4JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.-RStmEJQo7sAfzQgy0PDmDMf6xAVV1DH0hPSyVIyh0Y)

### Team Members
* Brian Githinji
* Grace Gitau
* Maureen Imanene
* Esther Cheruiyot

### Table of Contents

- [Business Overview](#business-overview)
- [Business Understanding](#business-understanding)
- [Data Understanding](#data-understanding)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Modeling and Evaluation](#modeling-and-evaluation)
- [Conclusion](#conclusion)
- [Repository Navigation](#repository-navigation)

## Project Overview

This project focuses on building an AI-powered recommendation system for a variety of beauty products. While traditional recommendation systems tend to specialize in a single category, this project is designed to suggest a broad spectrum of products including skincare, haircare, supplements, perfumes, makeup, and more. This system aims to deliver personalized beauty recommendations that cater to users’ unique needs and preferences using advanced AI techniques.

## Problem Statement

The vast array of beauty products available today can make it difficult for consumers to find products that match their specific needs and preferences. This project aims to bridge that gap by creating a unified recommendation system that offers tailored beauty product suggestions, addressing the need for personalized solutions across multiple beauty categories.

## Objectives
* Develop a recommendation system that covers multiple beauty product categories using AI and ML techniques.
* Incorporate content-based filtering, collaborative filtering, and sentiment analysis for accurate, varied recommendations.
* Deploy the system through a user-friendly Streamlit application for accessible, personalized product discovery

## Stakeholders
* End Users: Individuals seeking personalized beauty product recommendations across diverse categories.
* Beauty Brands: Companies interested in offering curated recommendations to their customers.
* Retailers: Platforms looking to enhance the shopping experience with targeted product suggestions.

## Data Description
1. Product Data:

* Beauty Products: Data covering various beauty products from brands, with details on type, ingredients, ratings, and price.
* Main Features: `product_id`, `product_name`, `category` (skincare, haircare, makeup, etc.), `brand_name`, `ingredients`, `rating`, `price`, `stock_status`.

2. User Reviews:

* Customer Feedback: Reviews with insights into product effectiveness for various skin types, hair textures, and personal preferences.
* Main Features: `author_id`, `rating`, `review_text`, `skin_type`, `hair_type`, `review_preferences`, `helpfulness`

### Exploratory Data Analysis

#### 1. Count of Products by Skin Type
![Count of Products by Skin Type](https://github.com/ECCHERUIYOT/AI-Powered-Skincare-Recommendations-For-Melanin-Skin/blob/Echeruiyot/images/image1.png)

* The distribution shows that products labeled for combination skin are the most common, followed by those for dry, normal, and then oily skin. This insight can guide product selection based on prevalent skin types and consumer demand within the Black women demographic.

#### 2. Top 20 Recommended Products for Melanin Skin
![Top 20 Recommended Products for Melanin Skin](https://github.com/ECCHERUIYOT/AI-Powered-Skincare-Recommendations-For-Melanin-Skin/blob/Echeruiyot/images/image2.png)
* For melanated skin tones, there's a strong emphasis on gentle yet effective skincare products, with the top three recommendations focusing on hydration and sun protection.

* The Dew Dream Cleansing Balm leads the pack with about 55 recommendations, suggesting that gentle makeup removal is a priority for this skin type. The second most recommended product being the Signature Moisturizer, followed by the NA°39 Facial Sunscreen Mist with SPF 39, indicates that maintaining skin hydration and sun protection are crucial concerns for melanated skin. The prevalence of brightening, revitalizing, and gentle products in the top 10 (such as Absolue Soft Cream and Evercalm Gentle Cleansing Gel) suggests that users with melanated skin often seek products that address hyperpigmentation and sensitivity while maintaining skin barrier health.

* The list also includes several treatment-focused products like glycolic serums and exfoliators in the middle to lower rankings, indicating that while these are important, they're secondary to basic skin protection and hydration needs.

#### 3. Out of stock Products and Recommendations based on skin tone
![Out of stock Products and Recommendations based on skin tone](https://github.com/ECCHERUIYOT/AI-Powered-Skincare-Recommendations-For-Melanin-Skin/blob/Echeruiyot/images/image3.png)
* The Recommended pie chart illustrates the percentage of products recommended versus those that are not. A significant portion (85%) of products are not marked as recommended, indicating a possible quality or suitability gap. This could help identify where product performance might fall short or suggest a need for more tailored product options.

* The Products pie chart shows the balance between new and existing products. The larger percentage of existing products (91%) suggests that the platform maintains a consistent range of products, with newer items being introduced selectively. This distribution can provide insights into the inventory management and refresh rates of the catalog over time.

#### 4. Skin Tone Distribution by Price Tier
![Skin Tone Distribution by Price Tier](https://github.com/ECCHERUIYOT/AI-Powered-Skincare-Recommendations-For-Melanin-Skin/blob/Echeruiyot/images/image4.png)
* There is a noticeable price difference across skin tones. Products for melanated skin tend to have lower average prices, than non-melanated skin tones, highlighting more affordable options specifically formulated or suited for melanated skin. 

#### 5. Heatmap of Product Count by Skin Tone Category and Skin Type
![Heatmap of Product Count by Skin Tone Category and Skin Type](https://github.com/ECCHERUIYOT/AI-Powered-Skincare-Recommendations-For-Melanin-Skin/blob/Echeruiyot/images/image5.png)
* The heatmap analysis of skin_type and skin_tone_category highlights important insights that align closely with our objective of providing tailored skincare recommendations for women of color. Our data reveals a concentration of products available for combination and dry skin types, particularly within lighter skin tones. However, there is a notable scarcity of options for deeper skin tones, suggesting that women of color may have fewer product options specifically suited to their needs. This gap underscores the limited market focus on skincare for melanin-rich skin concerns, such as hyperpigmentation, dryness, and sensitivity, which are often more pronounced in deeper skin tones.

* These findings directly support our business problem: many existing recommendation systems fail to provide targeted solutions for melanated women. The evident lack of specialized options for drier skin in deeper tones emphasizes an opportunity to develop and recommend products that address this unique need. By prioritizing these underserved areas, our recommendation system can significantly enhance satisfaction and efficacy for women of color seeking products that work for their melanin-rich skin.


## Modeling Approach

The recommendation system integrates multiple models:

1. Customer Feature-Based Recommender: Recommends products based solely on customer characteristics.
2. Highlight-Based Recommender: Recommends products based on selected highlights, catering to specific product features.
3. Content-Based Filtering Model: Recommends similar products based on ingredients and features using cosine similarity.
4. Collaborative Filtering with SVD: Leverages user-product interactions to recommend products based on user preferences.
5. Hybrid Model: Combines content-based and collaborative filtering to provide robust, varied recommendations.

## Evaluation Metrics

* Precision: Measures the relevance of recommended products.
* Recall: Evaluates the system's effectiveness in covering user preferences.
* F1-Score: Ensures a balance between precision and recall.


## Conclusion
* According to our metric of successs our metrics indicates that the recommendation system is performing well, with a strong balance between the accuracy of recommendations and user satisfaction.
* By using content-based filtering on ingredient similarities, collaborative filtering through SVD, and sentiment analysis, the system delivers customized product suggestions that align with users’ unique skin concerns and preferences. The Streamlit interface enhances user experience, providing an intuitive way for users to access their personalized recommendations easily.
* The recommendation model was deployed using Streamlit interface.

## Repository Navigation
Link to the final notebook and presentation are shared below;
[Notebook](https://github.com/ECCHERUIYOT/AI-Powered-Skincare-Recommendations-For-Melanin-Skin/blob/main/index.ipynb)

