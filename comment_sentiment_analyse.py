# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from transformers import pipeline
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
import string
from collections import Counter
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm  # add progress bar

def load_stopwords():
    """Load English stopwords"""
    try:
        return set(stopwords.words('english'))
    except:
        return set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
                   "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves',
                   'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself',
                   'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                   'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those',
                   'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
                   'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and',
                   'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by',
                   'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during',
                   'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
                   'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once'])

def tokenize_and_pos_tag(text, stopwords):
    """Simplified tokenization and POS tagging"""
    # Simple tokenization (split by spaces)
    words = text.split()
    # Filter out stopwords and short words
    words = [word for word in words if word not in stopwords and len(word) > 2]
    return [(word, 'NN') for word in words]  # Simplified processing, treat all words as nouns

def clean_text(text):
    """Clean text data"""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove digits
    text = re.sub(r'\d+', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# 1. Data preparation and preprocessing
def load_data(filepath):
    """Load data and preprocess"""
    print("Loading data...")
    df = pd.read_csv(filepath)
    
    # Remove duplicates
    print("Removing duplicates...")
    df = df.drop_duplicates(subset=['ReviewTitle', 'ReviewBody'])
    
    # Merge title and body
    df['full_text'] = df['ReviewTitle'] + ". " + df['ReviewBody']
    
    # Clean data
    print("Cleaning data...")
    df['cleaned_text'] = df['full_text'].apply(clean_text)
    
    # Tokenization and POS tagging
    print("Tokenizing and POS tagging...")
    stopwords = load_stopwords()
    df['tokens'] = df['cleaned_text'].apply(lambda x: tokenize_and_pos_tag(x, stopwords))
    
    # Extract nouns
    df['nouns'] = df['tokens'].apply(extract_nouns)
    
    # Convert star rating to three-class sentiment labels
    df['true_sentiment'] = df['ReviewStar'].apply(
        lambda x: 'POSITIVE' if x >= 4
        else 'NEGATIVE' if x <= 2
        else 'NEUTRAL'
    )
    
    return df

def extract_nouns(tokens):
    """Extract nouns (NN, NNS, NNP, NNPS)"""
    return [word for word, tag in tokens if tag.startswith('NN')]

# 2. Sentiment analyzer
class SentimentAnalyzer:
    def __init__(self):
        # Use a sentiment analysis model that supports three-class classification
        print("Initializing sentiment analyzer...")
        self.nlp = pipeline(
            task="sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment",
            tokenizer="cardiffnlp/twitter-roberta-base-sentiment"
        )
        # Label mapping
        self.label_mapping = {
            'LABEL_0': 'NEGATIVE',
            'LABEL_1': 'NEUTRAL',
            'LABEL_2': 'POSITIVE'
        }
        print("Sentiment analyzer initialized")

    def predict_sentiment(self, text):
        try:
            result = self.nlp(text[:512])[0]  # Truncate long text
            return {
                'label': self.label_mapping[result['label']],
                'score': result['score']
            }
        except Exception as e:
            print(f"Error processing text: {e}")
            return {'label': 'NEUTRAL', 'score': 0.33}

# 3. Topic modeling
def apply_topic_modeling(df, n_topics=5):
    """Apply LDA topic modeling"""
    print("Applying LDA topic modeling...")
    print("Preparing document-term matrix...")
    
    # Prepare document-term matrix, increase ngram range
    vectorizer = CountVectorizer(max_features=1000,
                               ngram_range=(1, 2),  # Add bigrams
                               max_df=0.95,         # Filter out high-frequency words
                               min_df=2)            # Filter out low-frequency words
    doc_term_matrix = vectorizer.fit_transform(df['cleaned_text'])
    
    print("Training LDA model...")
    # Train LDA model
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=20,        # Increase iteration times
        learning_method='online',  # Use online learning
        batch_size=128      # Set batch size
    )
    lda.fit(doc_term_matrix)
    
    # Get topic words
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-20:-1]]  # Increase the number of keywords for each topic
        topics.append(top_words)
    
    return topics

# 4. Word cloud generation
def generate_wordcloud(df, sentiment):
    """Generate word cloud"""
    print(f"Generating word cloud for {sentiment} reviews...")
    text = ' '.join(df[df['true_sentiment'] == sentiment]['cleaned_text'])
    wordcloud = WordCloud(
        width=1200,         # Increase size
        height=800,
        background_color='white',
        max_words=200,      # Increase word count
        collocations=True,  # Allow word groups
        min_font_size=10,
        max_font_size=150
    ).generate(text)
    
    plt.figure(figsize=(15, 10))  # Increase image size
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.title(f'{sentiment} Reviews WordCloud')
    plt.savefig(f'{sentiment.lower()}_wordcloud.png', dpi=300, bbox_inches='tight')
    plt.close()

# 5. Evaluate model
def evaluate_model(df, predictions):
    """Evaluate model performance"""
    print("Evaluating model performance...")
    # Add prediction results
    eval_df = df.copy()
    eval_df['predicted_sentiment'] = [p['label'] for p in predictions]
    eval_df['prediction_score'] = [p['score'] for p in predictions]

    # Calculate
    accuracy = accuracy_score(eval_df['true_sentiment'], eval_df['predicted_sentiment'])
    report = classification_report(eval_df['true_sentiment'], eval_df['predicted_sentiment'])
    conf_matrix = confusion_matrix(
        eval_df['true_sentiment'],
        eval_df['predicted_sentiment'],
        labels=['NEGATIVE', 'NEUTRAL', 'POSITIVE']
    )

    # Visualize confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['NEGATIVE', 'NEUTRAL', 'POSITIVE'],
                yticklabels=['NEGATIVE', 'NEUTRAL', 'POSITIVE'])
    plt.title('Sentiment analysis confusion matrix')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.savefig('confusion_matrix.png')
    plt.close()

    # Calculate accuracy for each category
    category_accuracy = {}
    for category in ['NEGATIVE', 'NEUTRAL', 'POSITIVE']:
        mask = eval_df['true_sentiment'] == category
        if mask.sum() > 0:
            cat_acc = accuracy_score(
                eval_df[mask]['true_sentiment'],
                eval_df[mask]['predicted_sentiment']
            )
            category_accuracy[category] = cat_acc

    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': conf_matrix,
        'category_accuracy': category_accuracy
    }

# 6. Generate improvement suggestions
def generate_recommendations(df):
    """Generate improvement suggestions based on analysis results"""
    print("Generating improvement suggestions...")
    recommendations = {
        'product_quality': [],
        'customer_service': [],
        'price': []
    }
    
    # Analyze keywords in negative reviews
    negative_reviews = df[df['true_sentiment'] == 'NEGATIVE']
    
    # Extract common nouns
    all_nouns = [noun for nouns in negative_reviews['nouns'] for noun in nouns]
    noun_freq = Counter(all_nouns)
    
    # Generate suggestions based on keywords
    quality_words = ['quality', 'durability', 'material', 'build']
    service_words = ['service', 'support', 'customer', 'warranty']
    price_words = ['price', 'cost', 'value', 'expensive']
    
    for word, freq in noun_freq.most_common(10):
        if any(quality_word in word.lower() for quality_word in quality_words):
            recommendations['product_quality'].append(
                f"Improve quality control related to {word} (mentioned {freq} times)")
        elif any(service_word in word.lower() for service_word in service_words):
            recommendations['customer_service'].append(
                f"Enhance service experience related to {word} (mentioned {freq} times)")
        elif any(price_word in word.lower() for price_word in price_words):
            recommendations['price'].append(
                f"Optimize pricing strategy related to {word} (mentioned {freq} times)")
    
    return recommendations

# Main program
def main():
    input_file = "AllProductReviews.csv"
    
    # 1. Load and preprocess data
    try:
        df = load_data(input_file)
        # Only analyze the first 5000 reviews
        df = df.head(5000)
        print(f"Successfully loaded {len(df)} reviews")
        print(f"Review distribution:\n{df['true_sentiment'].value_counts()}")
        print(f"Review distribution ratio:\n{df['true_sentiment'].value_counts(normalize=True).map('{:.2%}'.format)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Sentiment analysis
    try:
        analyzer = SentimentAnalyzer()
        print("\nPerforming sentiment analysis...")
        predictions = []
        
        # Use tqdm to display progress bar
        for i, text in enumerate(tqdm(df['full_text'], desc="Sentiment analysis progress")):
            pred = analyzer.predict_sentiment(text)
            predictions.append(pred)
            
            # Save intermediate results every 500 reviews
            if (i + 1) % 500 == 0:
                temp_df = df.iloc[:i+1].copy()
                temp_df['predicted_sentiment'] = [p['label'] for p in predictions]
                temp_df['prediction_score'] = [p['score'] for p in predictions]
                temp_df.to_csv(f'sentiment_analysis_results_temp_{i+1}.csv', index=False)
                
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return

    # 3. Topic model analysis
    try:
        # Analyze topics for positive and negative reviews separately
        for sentiment in ['POSITIVE', 'NEGATIVE']:
            sentiment_df = df[df['true_sentiment'] == sentiment]
            topics = apply_topic_modeling(sentiment_df)
            print(f"\n{sentiment} review topic analysis results:")
            for i, topic_words in enumerate(topics):
                print(f"Topic {i+1}: {', '.join(topic_words)}")
    except Exception as e:
        print(f"Error in topic modeling: {e}")

    # 4. Generate word cloud
    try:
        for sentiment in ['POSITIVE', 'NEGATIVE']:
            generate_wordcloud(df, sentiment)
    except Exception as e:
        print(f"Error generating word cloud: {e}")

    # 5. Evaluate model performance
    try:
        evaluation = evaluate_model(df, predictions)
        print("\n=== Model evaluation results ===")
        print(f"Overall accuracy: {evaluation['accuracy']:.2%}")
        print("\nCategory accuracies:")
        for category, acc in evaluation['category_accuracy'].items():
            print(f"{category}: {acc:.2%}")
        print("\nDetailed classification report:")
        print(evaluation['classification_report'])
    except Exception as e:
        print(f"Error evaluating model: {e}")

    # 6. Generate improvement suggestions
    try:
        recommendations = generate_recommendations(df)
        print("\n=== Improvement suggestions ===")
        for category, suggestions in recommendations.items():
            if suggestions:
                print(f"\n{category} related suggestions:")
                for suggestion in suggestions:
                    print(f"- {suggestion}")
    except Exception as e:
        print(f"Error generating improvement suggestions: {e}")

    # 7. Save final results
    try:
        # Save complete analysis results
        df['predicted_sentiment'] = [p['label'] for p in predictions]
        df['prediction_score'] = [p['score'] for p in predictions]
        df.to_csv('sentiment_analysis_results_full.csv', index=False)
        
        # Save evaluation metrics
        with open('evaluation_metrics.txt', 'w') as f:
            f.write("=== Model evaluation results ===\n")
            f.write(f"Overall accuracy: {evaluation['accuracy']:.2%}\n\n")
            f.write("Category accuracies:\n")
            for category, acc in evaluation['category_accuracy'].items():
                f.write(f"{category}: {acc:.2%}\n")
            f.write("\nDetailed classification report:\n")
            f.write(evaluation['classification_report'])
        
        print("\nAnalysis results and visualizations saved to files:")
        print("- sentiment_analysis_results_full.csv: Contains all prediction results")
        print("- evaluation_metrics.txt: Contains detailed evaluation metrics")
        print("- confusion_matrix.png: Confusion matrix visualization")
        print("- positive_wordcloud.png: Positive review word cloud")
        print("- negative_wordcloud.png: Negative review word cloud")
    except Exception as e:
        print(f"Error saving results: {e}")

if __name__ == '__main__':
    main()