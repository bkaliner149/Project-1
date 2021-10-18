#Import packages
import pandas as pd
import requests
import nltk
nltk.download('vader_lexicon')
nltk.download('wordnet')
from bs4 import BeautifulSoup

#Create empty dataframe for companies
randomCompanies = pd.DataFrame(columns=['Title', 'Purpose'])
Company = []
Information = []

#Scrape website for name and purpose of company
for company in range(50):
    url = "http://3.85.131.173:8000/random_company"
    requestURL = requests.get(url)
    text = requestURL.text
    Parse = BeautifulSoup(text, 'html.parser')
    for title in Parse.find_all('title'):
        Company.append(title.get_text())


    for line in Parse.find_all('li'):
        Information.append(line.get_text())

#Fill dataframe with scrape
purpose = [scrape for scrape in Information if "Purpose" in scrape]
randomCompanies['Purpose'] = purpose
randomCompanies['Title'] = Company

#Get rid of word "purpose"
randomCompanies['Purpose'] = randomCompanies['Purpose'].str.replace('Purpose: ', '')

#Write company list to CSV
randomCompanies.to_csv('CompanyPull.csv')