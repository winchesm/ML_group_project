import requests
import time
from bs4 import BeautifulSoup

max_pages = 30
topics = ['Technology', 'Entertainment', 'Design', 'Business', 'Science', 'Global+issues']
links = []
for topic in topics:
    URL = 'https://www.ted.com/talks?sort=newest&topics%5B%5D=' + topic
    time.sleep(10)
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, 'html.parser')
    max_pages = int(soup.find_all(class_='pagination__item pagination__link')[-1].get_text())
    for page in range(max_pages):
        page += 1
        URL = 'https://www.ted.com/talks?sort=newest&page=' + str(page) + '&topics%5B%5D=' + topic
        page = requests.get(URL)
        time.sleep(1.5)
        soup = BeautifulSoup(page.content, 'html.parser')
        talk_divs = soup.find_all(class_='talk-link')
        for div in talk_divs:
            links.append(div.a.get('href'))

links_nodups = set(links)
print(len(links) != len(links_nodups))

with open('links.txt', 'w') as f:
    for link in links_nodups:
        f.write(link + '\n')

