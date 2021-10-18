#Import Packages
import pandas as pd
import nltk
from nltk.corpus import stopwords
nltk.download('vader_lexicon')
nltk.download('wordnet')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import typing

#Read in data from group members
mike_data = pd.read_csv('fake_company_pull.csv')
brett_data = pd.read_csv('CompanyPull.csv')
steve_data = pd.read_csv('comps.csv')

#Get rid of word "Purpose"
mike_data['Purpose'] = mike_data['Purpose'].str.replace('Purpose: ', '')
steve_data['Purpose'] = steve_data['Purpose'].str.replace('Purpose: ', '')

#Gwt rid of unnecessary column
brett_data = brett_data.iloc[: , 1:]

#Put dataframes together
allCompanies = pd.DataFrame(columns=['Title', 'Purpose'])
allCompanies = allCompanies.append(mike_data)
allCompanies = allCompanies.append(brett_data)
allCompanies = allCompanies.append(steve_data)

#Define stopwords
stopcorpus: typing.List = stopwords.words('english')

#Define function to remove stop words
def remove_words(em:str,list_of_words_to_remove: typing.List):
    return [item for item in em if item not in list_of_words_to_remove]

#Define function to make list into string
def collapse_list_to_string(string_list):
    return ' '.join(string_list)

#Run functions
allCompanies['Purpose'] = allCompanies['Purpose'].astype(str).apply(lambda x: remove_words(x.split(),stopcorpus))
allCompanies['Purpose'] = allCompanies['Purpose'].apply(collapse_list_to_string)

#Set up tokenizer and lemmatizer
tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

#Define function to use lemmatizer and tokenizer
def root_word(text):
    return [lemmatizer.lemmatize(w) for w in tokenizer.tokenize(text)]

#Apply root word function
allCompanies['Purpose'] = allCompanies['Purpose'].astype(str).apply(root_word)
allCompanies['Purpose'] = allCompanies['Purpose'].apply(collapse_list_to_string)

#Sentiment Analysis
sentiment = []
analyzer = SentimentIntensityAnalyzer()
for purpose in allCompanies.Purpose:
    vs = analyzer.polarity_scores(purpose)
    sentiment.append(vs["compound"])
allCompanies['Sentiment'] = sentiment

#Sort the companies by sentiment score
allCompanies.sort_values(by=['Sentiment'], inplace=True)
worst_sentiment = allCompanies.head(5)
best_sentiment = allCompanies.tail(5)
print(best_sentiment)
print(worst_sentiment)

allCompanies.to_csv('allCompanies.csv')