import os
import requests
from bs4 import BeautifulSoup


base_url = 'https://www.vogue.com/fashion-shows/resort-2024/erdem/slideshow/collection#{}'
start_page = 1
end_page = 10
avoid_alt_tags = ['vogue runway', 'profile']  # Add the alt tags to avoid here


def download_images(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    image_tags = soup.find_all('img')

    for img in image_tags:
        image_url = img.get('src') # Gets src from img tag
        alt_text = img.get('alt', '').lower()  # Convert alt text to lowercase for easier comparison
        if image_url and not any(tag in alt_text for tag in avoid_alt_tags): # If an image is found and not one of the ignored alt tags
            try:
                response = requests.get(image_url) # 
                # print(response)
                if response.status_code == 200:
                    image_name = os.path.basename(image_url)
                    print(image_name)
                    with open(os.path.join('images', image_name), 'wb') as f:
                        f.write(response.content)
            except Exception as e:
                print(f"Failed to download {image_url}: {e}")

if not os.path.exists('images'):
    os.makedirs('images')

for page_number in range(start_page, end_page + 1):
    url = base_url.format(page_number)
    download_images(url)
    print(url)
