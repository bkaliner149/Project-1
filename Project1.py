import pandas as pd
import requests
import nltk
nltk.download('vader_lexicon')
nltk.download('wordnet')
from bs4 import BeautifulSoup


randomCompanies = pd.DataFrame(columns=['Title', 'Purpose'])
companyList = []
linesList = []

for x in range(50):
    url = "http://3.85.131.173:8000/random_company"
    page = requests.get(url)
    data = page.text
    soup = BeautifulSoup(data, 'html.parser')
    for title in soup.find_all('title'):
        companyList.append(title.get_text())


    for li in soup.find_all('li'):
        linesList.append(li.get_text())

matches = [s for s in linesList if "Purpose" in s]
randomCompanies['Purpose'] = matches
randomCompanies['Title'] = companyList

randomCompanies.to_csv('CompanyPull.csv')

print("done")