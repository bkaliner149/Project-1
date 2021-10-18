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
def lose_stop_words(removal:str,stop_words: typing.List):
    return [x for x in removal if x not in stop_words]

#Define function to make companies list into string
def make_string(companies):
    return ' '.join(companies)

#Run functions
allCompanies['Purpose'] = allCompanies['Purpose'].astype(str).apply(lambda x: lose_stop_words(x.split(),stopcorpus))
allCompanies['Purpose'] = allCompanies['Purpose'].apply(make_string)

#Set up tokenizer and lemmatizer
token = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

#Define function to use lemmatizer and tokenizer
def rootWord(text):
    return [lemmatizer.lemmatize(w) for w in token.tokenize(text)]

#Apply root word function
allCompanies['Purpose'] = allCompanies['Purpose'].astype(str).apply(rootWord)
allCompanies['Purpose'] = allCompanies['Purpose'].apply(make_string)

#Sentiment Analysis
sentiment = []
analyze = SentimentIntensityAnalyzer()
for purpose in allCompanies.Purpose:
    vs = analyze.polarity_scores(purpose)
    sentiment.append(vs["compound"])
allCompanies['Sentiment'] = sentiment

#Sort the companies by sentiment score
allCompanies.sort_values(by=['Sentiment'], inplace=True)
Lowest = allCompanies.head(5)
Highest = allCompanies.tail(5)
print(Lowest)
print(Highest)

allCompanies.to_csv('allCompanies.csv')