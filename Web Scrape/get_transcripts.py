import requests
import time
from bs4 import BeautifulSoup

# Make list of all links saved in get_links.py
with open('links.txt') as f:
    links = f.read().splitlines()

#skip = True
with open('data.csv', 'a', encoding='utf-8') as f2:
    f2.write('Link,Tags,Transcript\n')
    for link in links:
        # This is used to start off on a specific link after internet disconnected
        # if link == '/talks/dave_brain_what_a_planet_needs_to_sustain_life':
        #     skip = False
        # if skip:
        #     continue
        print(link)
        # Get specific talk url
        URL = 'https://www.ted.com' + link 
        page = requests.get(URL)
        # Sleep to avoid error 429
        time.sleep(1)
        while(page.status_code == 429):
            print('Sleeping tags')
            time.sleep(10)
            page = requests.get(URL)
        print('Tags ok')
        soup = BeautifulSoup(page.content, 'html.parser')
        # Save all topic tags
        tags = []
        for tag in soup.find_all('meta', property='og:video:tag'):
            tags.append(tag.get('content'))
            # Switch over to transcripts tag
        URL = 'https://www.ted.com/' + link + '/transcript'
        page = requests.get(URL)
        time.sleep(1)
        while(page.status_code == 429):
            print('Sleeping transcript')
            time.sleep(10)
            page = requests.get(URL)
        print('Transcript ok')
        soup = BeautifulSoup(page.content, 'html.parser')
        # All transcripts are wrapped with an element with
        # class Grid__cell flx-s:1 p-r:4 
        paragraphs = soup.find_all(class_='Grid__cell flx-s:1 p-r:4')
        transcript = ''
        # Sup up the parts of transcript and add a "" for every "
        # due to csv field formatting
        for p in paragraphs:
            transcript += ' '.join(p.get_text().split()).replace('"', '""') + ' '
        f2.write(link + ',"' + str(tags) + '","' + transcript + '"\n')