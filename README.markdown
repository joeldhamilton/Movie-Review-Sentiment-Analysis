# Custom Sentiment Analysis Web App

## Overview
This web application predicts the sentiment (positive or negative) of movie reviews using a custom-trained BERT model with Logistic Regression from my TripleTen Data Science Bootcamp project. Built with Streamlit and Hugging Face Transformers, it showcases an end-to-end machine learning pipeline, from training to deployment.

### Key Achievements
- **Model**: Trained a BERT-based model with Logistic Regression, achieving an F1 score of 0.83 on the IMDB dataset.
- **Interface**: Developed an intuitive Streamlit app for real-time sentiment prediction.
- **Deployment**: Hosted on Streamlit Community Cloud for public access.
- **Tech Stack**: Python, Streamlit, Transformers, Pandas, NLTK, Joblib, Git.

## How It Works
1. Users input a movie review in a text box.
2. The app preprocesses the text and generates BERT embeddings.
3. A custom-trained Logistic Regression model predicts sentiment and confidence.

## Setup and Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/joeldhamilton/sentiment-analysis-app
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app locally:
   ```bash
   streamlit run app.py
   ```

## Deployment
The app is deployed on Streamlit Community Cloud: [View Live Demo](https://your-app-url.streamlit.app).

## Challenges Overcome
- Integrated custom-trained BERT embeddings with Logistic Regression for efficient deployment.
- Optimized preprocessing and embedding generation with Streamlit caching to reduce latency.
- Ensured compatibility with Streamlit Cloud’s resource constraints.

## Future Improvements
- Fine-tune the BERT model further to improve F1 score.
- Add visualizations (e.g., word importance scores using SHAP).
- Support batch processing for multiple reviews.

## Contact
- **Author**: Joel Hamilton
- **Links**: [GitHub](https://github.com/joeldhamilton) | [LinkedIn](https://www.linkedin.com/in/joel-hamilton) | [Portfolio](https://joeldhamilton.github.io)