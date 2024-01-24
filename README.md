# web_extraction_words-analysis
def extract_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    title = soup.find('title').get_text()
    article_text = ' '.join([p.get_text() for p in soup.find_all('p')])
    return title, article_text
import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np
import csv
with open('negative_words.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    headers = next(csv_reader, None)
    negative_words_list = [row for row in csv_reader]




import csv
with open('positiveword.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    headers = next(csv_reader, None)
    positive_words_list = [row for row in csv_reader]
#tuple_positive=[tuple(i) for i in  positive_words_list]    
#positive_words_list=set(tuple_positive)   
import csv
with open('stop_words.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    headers = next(csv_reader, None)
    stop_words= [row for row in csv_reader]
    import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import cmudict
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('cmudict') 
from textblob import TextBlob
def analyze_text(text):
    blob = TextBlob(text)
    stop_words = set(stopwords.words('english'))
    stop_words.update(stop_words_list)
    words= nltk.word_tokenize(text)
    filtered_words= [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    positive_score = sum(1 for word in filtered_words if word.lower() in positive_words_list) / len(filtered_words) if len(filtered_words) > 0 else 0
    negative_score = sum(1 for word in filtered_words if word.lower() in negative_words_list) / len(filtered_words) if len(filtered_words) > 0 else 0
    polarity_score = blob.sentiment.polarity
    subjectivity_score = blob.sentiment.subjectivity
    return positive_score, negative_score, polarity_score, subjectivity_score
    input_df = pd.read_excel('Input (1).xlsx')

columns = ['URL_ID', 'Title', 'Article_Text', 'Positive_Score', 'Negative_Score', 'Polarity_Score', 'Subjectivity_Score']
output_df = pd.DataFrame(columns=columns)
import pandas as pd
output_data = []
for index, row in input_df.iterrows():
    url_id = row['URL_ID']
    url = row['URL']
    
    title, article_text = extract_text(url)
    
    positive_score, negative_score, polarity_score, subjectivity_score = analyze_text(article_text)
    
    output_data.append({
        'URL_ID': url_id,
        'Title': title,
        'Article_Text': article_text,
        'Positive_Score': positive_score,
        'Negative_Score': negative_score,
        'Polarity_Score': polarity_score,
        'Subjectivity_Score': subjectivity_score
    })

output_df = pd.DataFrame('Output Data Structure (1).xlsx')

output_df.to_excel('Output Data Structure (1).xlsx', index=False)
