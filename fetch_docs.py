import requests
from bs4 import BeautifulSoup

res = requests.get('https://docs.limitless.exchange/api-reference/orderbook')
soup = BeautifulSoup(res.text, 'html.parser')

print("=== ENDPOINTS ===")
for code in soup.find_all('code'):
    text = code.get_text()
    if 'GET' in text or 'POST' in text or '/api' in text or 'limitless' in text:
        print(text)

print("\n=== TEXT ===")
for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'li']):
    text = p.get_text()
    if 'order' in text.lower() or 'book' in text.lower() or 'api' in text.lower():
        print(text)
