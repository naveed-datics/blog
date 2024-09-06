import requests
from bs4 import BeautifulSoup

# def main():
search_keyword = "ana de armas religion"
target_url = f"https://www.google.com/search?q={search_keyword}"

response = requests.get(target_url)
html_data = BeautifulSoup(response.text, 'html.parser')
links = html_data.find('div',id='main').find_all('h3')
top_urls = []
for link in links[:5]:
    try:
        final_url = link.find_parent('a')['href'].split('q=')[1].split('&sa=U&ved')[0]
        top_urls.append(final_url)

    except requests.exceptions.RequestException as e:
        print(f'Error fetching')

    except Exception as e:
        print(f'Error processing')

print(top_urls)
