import requests
import time
from bs4 import BeautifulSoup

max_pages = 30
topics = ['Technology', 'Entertainment', 'Design', 'Business', 'Science', 'Global+issues']
links = []
for topic in topics:
    URL = 'https://www.ted.com/talks?sort=newest&topics%5B%5D=' + topic
    # sleep to avoid  HTTp status code 429 Too Many Requests
    time.sleep(10)
    # request topic URL
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, 'html.parser')
    # Fetch list of all items labelled with below class,
    # the last element is the number of the final page
    max_pages = int(soup.find_all(class_='pagination__item pagination__link')[-1].get_text())
    # Loop through all pages
    for page in range(max_pages):
        page += 1
        URL = 'https://www.ted.com/talks?sort=newest&page=' + str(page) + '&topics%5B%5D=' + topic
        page = requests.get(URL)
        time.sleep(1.5)
        soup = BeautifulSoup(page.content, 'html.parser')
        # Find all talk link URLs and append to list
        talk_divs = soup.find_all(class_='talk-link')
        for div in talk_divs:
            links.append(div.a.get('href'))

#Delete duplicate links
links_nodups = set(links)
print(len(links) != len(links_nodups))

# Save all links to file
with open('links.txt', 'w') as f:
    for link in links_nodups:
        f.write(link + '\n')

