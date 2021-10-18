import pandas as pd
import nltk
from nltk.corpus import stopwords
nltk.download('vader_lexicon')
nltk.download('wordnet')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import typing

mike_data = pd.read_csv('fake_company_pull.csv')
brett_data = pd.read_csv('CompanyPull.csv')
steve_data = pd.read_csv('comps.csv')

mike_data['Purpose'] = mike_data['Purpose'].str.replace('Purpose: ', '')
steve_data['Purpose'] = steve_data['Purpose'].str.replace('Purpose: ', '')
brett_data = brett_data.iloc[: , 1:]

print(mike_data)
print(steve_data)

allCompanies = pd.DataFrame(columns=['Title', 'Purpose'])
allCompanies = allCompanies.append(mike_data)
allCompanies = allCompanies.append(brett_data)
allCompanies = allCompanies.append(steve_data)
allCompanies.rename( columns={'Unnamed: 0':'axis'}, inplace=True )
allCompanies.set_index('axis', inplace=True)


allCompanies[['junk','Purpose']] = allCompanies.Purpose.str.split(": ", expand=True)
allCompanies.drop(['junk'], axis=1, inplace=True)

stopcorpus: typing.List = stopwords.words('english')
def remove_words(em:str,list_of_words_to_remove: typing.List):
    return [item for item in em if item not in list_of_words_to_remove]
def collapse_list_to_string(string_list):
    return ' '.join(string_list)

def remove_apostrophes(text):
    text = text.replace("'", "")
    text = text.replace('"', "")
    text = text.replace('`', "")
    return text

allCompanies['Purpose'] = allCompanies['Purpose'].astype(str).apply(lambda x: remove_words(x.split(),stopcorpus))
allCompanies['Purpose'] = allCompanies['Purpose'].apply(collapse_list_to_string)
allCompanies['Purpose'] = allCompanies['Purpose'].apply(remove_apostrophes)

tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()
def root_word(text):
    return [lemmatizer.lemmatize(w) for w in tokenizer.tokenize(text)]

allCompanies['Purpose'] = allCompanies['Purpose'].astype(str).apply(root_word)
allCompanies['Purpose'] = allCompanies['Purpose'].apply(collapse_list_to_string)


sentiment = []
analyzer = SentimentIntensityAnalyzer()
for purpose in allCompanies.Purpose:
    vs = analyzer.polarity_scores(purpose)
    sentiment.append(vs["compound"])
allCompanies['Sentiment'] = sentiment


allCompanies.sort_values(by=['Sentiment'], inplace=True)
worst_sentiment = allCompanies.head(5)
best_sentiment = allCompanies.tail(5)
print(best_sentiment)
print(worst_sentiment)



print("done")