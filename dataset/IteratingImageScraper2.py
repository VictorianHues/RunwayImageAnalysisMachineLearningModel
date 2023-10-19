import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support import expected_conditions as EC


base_url = 'https://www.vogue.com/fashion-shows/resort-2024/erdem/slideshow/collection#{}'
end_page = 10
avoid_alt_tags = ['vogue runway', 'profile']  # Add the alt tags to avoid here


service = Service(executable_path=r'C:\Users\Victo\Downloads\chromedriver_win32\chromedriver.exe')
options = webdriver.ChromeOptions()
driver = webdriver.Chrome(service=service, options=options)


driver.get(base_url)

current_page = 1

time.sleep(30)
#close_subscription = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, r'//*[@id="app-root"]/div/div/div[5]/div[1]')))
#close_subscription.click()
#time.sleep(2)  # Give the page some time to load (you can adjust this value as needed)

while current_page <= end_page:
    # Click the "next" button to move to the next page
    try:
        next_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, r'//*[@id="main-content"]/div/div[1]/div[1]/div/div/div[1]/div[2]/div[3]')))
        next_button.click()
        time.sleep(2)  # Give the page some time to load (you can adjust this value as needed)
    except Exception as e:
        print(f"Failed to click 'next' button: {e}")
        break

    current_page += 1

# Close the WebDriver after scraping is done
driver.quit()
