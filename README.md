# Comment on the sentiment analysis system(pilot study)

This is a pilot study product review sentiment analysis and evaluation system based on pre-trained models. Combining sentiment analysis and topic acquisition tools, it is possible to obtain users' emotional tendencies based on comment data and put forward suggestions for product improvement

## Functional characteristics

- Text preprocessing and cleaning
- Sentiment analysis based on RoBERTa
- LDA topic modeling and analysis
- Word cloud visualization
- Model performance evaluation

## Dependency library

- transformers
- torch
- pandas
- numpy
- matplotlib
- seaborn
- wordcloud
- scikit-learn
- nltk

## Usage method

1. Install dependencies：
```bash
pip install transformers torch pandas numpy matplotlib seaborn wordcloud scikit-learn nltk
```

2. Download NLTK data：
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

3. Run the program：
```bash
python comment_sentiment_analyse.py
```

## Output result

- sentiment_analysis_results_full.csv：Complete analysis results
- evaluation_metrics.txt：Evaluation indicators
- confusion_matrix.png：Visualization of confusion matrix
- positive_wordcloud.png：Cloud of positive comments
- negative_wordcloud.png：Cloud of negative comments

## Precautions

The first run requires an Internet connection to download the pre-trained model
It is recommended to use GPU acceleration for processing large amounts of data
Support custom comment data input
