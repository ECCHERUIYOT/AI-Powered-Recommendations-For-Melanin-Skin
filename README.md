# AI-Powered-Recommendations-For-Melanin-Skin

![2d248f5507752726ddc2198b39071e4e](https://github.com/user-attachments/assets/9c7a2247-b758-4aee-8bc7-50fa0bf76f2d)

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

## Business Overview

Our AI-powered recommendation system addresses the lack of tailored skincare solutions for individuals with melanin-rich skin. By leveraging two datasets on skin care products and products reviews by different users , our model delivers personalized skincare recommendations, achieving an accuracy of over 90% in predicting optimal products for various skin needs and on different budgets.

## Business Understanding

This project aims to develop a recommendation system using advanced AI techniques to cater specifically to Black women’s skincare needs. By integrating machine learning, content- based filtering, collaborative filtering, and sentiment analysis, the system will offer personalized skincare recommendations. Leveraging skin_tone (Author's skin tone (e.g. fair, tan, etc.) as a classification feature, we aim to distinguish and target products that align with melanin-rich skin concerns.

## Data Understanding

The dataset was collected via a Python scraper and contains:
- Product Information: Over 8,000 beauty products from the Sephora online store, including product and brand names, prices, ingredients, ratings, and various features. 
- User Reviews: Approximately 1 million reviews across over 2,000 products in the skincare category. These reviews include user appearances, skin types, and review ratings.

The key features include:
- Product Features: `product_id`, `product_name`, `brand_name`, `ingredients`, `rating`, `price_ksh`, `new`, `out_of_stock`, `highlights`. 
- Review Features: `author_id`, `rating`, `review_text`, `skin_type`, `skin_tone`, and
`helpfulness`.


### Exploratory Data Analysis

#### 1. Count of Products by Skin Type
![Count of Products by Skin Type](https://github.com/ECCHERUIYOT/AI-Powered-Skincare-Recommendations-For-Melanin-Skin/blob/Echeruiyot/images/image1.png)

* The distribution shows that products labeled for combination skin are the most common, followed by those for dry, normal, and then oily skin. This insight can guide product selection based on prevalent skin types and consumer demand within the Black women demographic.

#### 2. Top 20 Recommended Products for Melanin Skin
![Top 20 Recommended Products for Melanin Skin](images\image2.png)
* For melanated skin tones, there's a strong emphasis on gentle yet effective skincare products, with the top three recommendations focusing on hydration and sun protection.

* The Dew Dream Cleansing Balm leads the pack with about 55 recommendations, suggesting that gentle makeup removal is a priority for this skin type. The second most recommended product being the Signature Moisturizer, followed by the NA°39 Facial Sunscreen Mist with SPF 39, indicates that maintaining skin hydration and sun protection are crucial concerns for melanated skin. The prevalence of brightening, revitalizing, and gentle products in the top 10 (such as Absolue Soft Cream and Evercalm Gentle Cleansing Gel) suggests that users with melanated skin often seek products that address hyperpigmentation and sensitivity while maintaining skin barrier health.

* The list also includes several treatment-focused products like glycolic serums and exfoliators in the middle to lower rankings, indicating that while these are important, they're secondary to basic skin protection and hydration needs.

#### 3. Out of stock Products and Recommendations based on skin tone
![Out of stock Products and Recommendations based on skin tone](images\image3.png)
* The Recommended pie chart illustrates the percentage of products recommended versus those that are not. A significant portion (85%) of products are not marked as recommended, indicating a possible quality or suitability gap. This could help identify where product performance might fall short or suggest a need for more tailored product options.

* The Products pie chart shows the balance between new and existing products. The larger percentage of existing products (91%) suggests that the platform maintains a consistent range of products, with newer items being introduced selectively. This distribution can provide insights into the inventory management and refresh rates of the catalog over time.

#### 4. Skin Tone Distribution by Price Tier
![Skin Tone Distribution by Price Tier](images\image4.png)
* There is a noticeable price difference across skin tones. Products for melanated skin tend to have lower average prices, than non-melanated skin tones, highlighting more affordable options specifically formulated or suited for melanated skin. 

#### 5. Heatmap of Product Count by Skin Tone Category and Skin Type
![Heatmap of Product Count by Skin Tone Category and Skin Type](images\image5.png)
* The heatmap analysis of skin_type and skin_tone_category highlights important insights that align closely with our objective of providing tailored skincare recommendations for women of color. Our data reveals a concentration of products available for combination and dry skin types, particularly within lighter skin tones. However, there is a notable scarcity of options for deeper skin tones, suggesting that women of color may have fewer product options specifically suited to their needs. This gap underscores the limited market focus on skincare for melanin-rich skin concerns, such as hyperpigmentation, dryness, and sensitivity, which are often more pronounced in deeper skin tones.

* These findings directly support our business problem: many existing recommendation systems fail to provide targeted solutions for melanated women. The evident lack of specialized options for drier skin in deeper tones emphasizes an opportunity to develop and recommend products that address this unique need. By prioritizing these underserved areas, our recommendation system can significantly enhance satisfaction and efficacy for women of color seeking products that work for their melanin-rich skin.


## Modeling and Evaluation
We used recommendation systems models as follows;
* Content Based Filtering Method
* Collobborative Filtering with SVD
* Hybrid Recommender System
We also supplemented the recommendation system models with machine learning models and a deep learning models in sentiment prediction as below;
* Random Forest Model
* XGBOOST Model
* LSTM deep learning model.

* We evaluated our recommender systems using RMSE and MSE which gave 4.48 and 4.35 consecutively still gave high values after performing hyperparameter tuning,4.59 and 4.41 for RMSE and MAE consecutively.
The Hybrid system was evaluated using the metrics precision and recall which gave a precision of 0.24 and a recall of 0.98 for the first five recommended items meaning that 24% of the items recommended in the top 5 were relevant to users. This suggests that while some relevant items are being recommended, there is still room for improvement in ensuring that more of the top recommendations are relevant.
Recall of 0.98, indicates that 98% of all relevant items (up to the top 5) are being successfully recommended. This is a high recall value, suggesting that the model is good at covering the items users are interested in.

* We also used conversion metrics;
i.) `Hit rate` which gave a Hit Rate of 0.99 across all values of k items meaning,the model is consistently recommending at least one relevant item within the top 𝐾 recommendations for almost every user.
ii.) `Normalized Discounted Cumulative Gain` which gave an average value of 0.99 as well indicating that the ranking provided by your recommendation model closely matches the ideal order, where the most relevant items appear at the top.

* For sentiment prediction the best performing model was XGBOOST.
The Random Forest model performs well,with 94% accuracy and especially in predicting class 1 (positive class), with high precision and recall values. However, the lower recall for class 0 (negative class) suggests that some negative cases are being misclassified.

The XGBoost model exhibits superior performance compared to the Random Forest, achieving higher accuracy of 96% and balanced metrics across both classes, indicating a robust prediction capability.


## Conclusion
* According to our metric of successs our metrics indicates that the recommendation system is performing well, with a strong balance between the accuracy of recommendations and user satisfaction.
* By using content-based filtering on ingredient similarities, collaborative filtering through SVD, and sentiment analysis, the system delivers customized product suggestions that align with users’ unique skin concerns and preferences. The Streamlit interface enhances user experience, providing an intuitive way for users to access their personalized recommendations easily.
* The recommendation model was deployed using Streamlit interface.

## Repository Navigation
Link to the final notebook and presentation are shared below;
[Notebook](https://github.com/ECCHERUIYOT/AI-Powered-Skincare-Recommendations-For-Melanin-Skin/blob/main/index.ipynb)

