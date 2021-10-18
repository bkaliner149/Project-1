#Import packages
import pandas as pd
import requests
import nltk
nltk.download('vader_lexicon')
nltk.download('wordnet')
from bs4 import BeautifulSoup

#Create empty dataframe for companies
randomCompanies = pd.DataFrame(columns=['Title', 'Purpose'])
companyList = []
linesList = []

#Scrape website for name and purpose of company
for x in range(50):
    url = "http://3.85.131.173:8000/random_company"
    page = requests.get(url)
    data = page.text
    soup = BeautifulSoup(data, 'html.parser')
    for title in soup.find_all('title'):
        companyList.append(title.get_text())


    for li in soup.find_all('li'):
        linesList.append(li.get_text())

#Fill dataframe with scrape
matches = [s for s in linesList if "Purpose" in s]
randomCompanies['Purpose'] = matches
randomCompanies['Title'] = companyList

#Get rid of word "purpose"
randomCompanies['Purpose'] = randomCompanies['Purpose'].str.replace('Purpose: ', '')

#Write company list to CSV
randomCompanies.to_csv('CompanyPull.csv')