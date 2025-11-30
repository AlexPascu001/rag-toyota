"""
Toyota Manual Scraper
Downloads owner's manuals from the Toyota Europe customer portal.
"""

import time
import os
import urllib.parse
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager

from config import (
    MANUALS_DIR,
    TOYOTA_PORTAL_URL,
    DEFAULT_LANGUAGE,
    DEFAULT_BRAND,
    SCRAPER_TIMEOUTS,
)


def scrape_toyota_manual(model="RAV4", model_type="RAV4", generation_text="2018 - Today"):
    """
    Scrape and download a Toyota owner's manual PDF.
    
    Args:
        model: Vehicle model name (e.g., 'RAV4')
        model_type: Specific model type (e.g., 'RAV4 HEV')
        generation_text: Generation text to match (e.g., '2018 - Today')
    """
    target_dir = str(MANUALS_DIR)
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Created directory: {target_dir}")

    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new") # Comment this line to see the browser
    options.add_argument("--start-maximized")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    
    prefs = {
        "download.default_directory": target_dir,
        "download.prompt_for_download": False,
        "plugins.always_open_pdf_externally": True,
        "profile.default_content_settings.popups": 0,
    }
    options.add_experimental_option("prefs", prefs)

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    timeout = SCRAPER_TIMEOUTS["element_wait"]

    try:
        print("Starting scraper...")
        driver.get(f"{TOYOTA_PORTAL_URL}?language={DEFAULT_LANGUAGE}&brand={DEFAULT_BRAND}")
        
        print(f"Selecting Model: {model}...")
        model_btn = WebDriverWait(driver, timeout).until(
            EC.element_to_be_clickable((By.XPATH, f"//button[normalize-space(.)='{model}']"))
        )
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", model_btn)
        time.sleep(0.5)
        model_btn.click()

        print(f"Selecting Type: {model_type}...")
        for attempt in range(3):
            try:
                time.sleep(1.5)
                type_btn = WebDriverWait(driver, timeout).until(
                    EC.element_to_be_clickable((By.XPATH, f"//button[normalize-space(.)='{model_type}']"))
                )
                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", type_btn)
                time.sleep(0.5)
                type_btn.click()
                break
            except StaleElementReferenceException:
                print(f"Retry {attempt + 1}: Type button stale...")

        print(f"Selecting Generation: {generation_text}...")
        WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.TAG_NAME, "h5")))
        gen_elements = driver.find_elements(By.TAG_NAME, "h5")
        target_gen = None
        for el in gen_elements:
            if generation_text in el.text.strip().replace('\xa0', ' '):
                target_gen = el
                break
        
        if target_gen:
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", target_gen)
            time.sleep(0.5)
            driver.execute_script("arguments[0].click();", target_gen)
        else:
            raise Exception("Generation not found")

        print("Waiting for Manuals List...")
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.XPATH, "//b[contains(text(), 'Owners Manual')]"))
        )
        
        print("Scanning for PDF buttons...")
        titles = driver.find_elements(By.XPATH, "//b[contains(text(), 'Owners Manual')]")
        
        target_row = None
        
        for title in titles:
            try:
                row = title.find_element(By.XPATH, "./../..")
                pdf_btn = row.find_element(By.XPATH, ".//button[contains(., 'PDF')]")
                print("Match Found! This manual has a PDF button.")
                target_row = row
                break
            except NoSuchElementException:
                continue
        
        if not target_row:
            raise Exception("No 'Owners Manual' with a PDF button was found.")

        # Extract Meta from the found row
        meta_element = target_row.find_element(By.CLASS_NAME, "medium")
        meta_text = meta_element.get_attribute("textContent")
        
        clean_meta = meta_text.replace('\xa0', ' ').strip()
        part_no = "UnknownPart"
        date_str = "UnknownDate"
        
        if "Part No:" in clean_meta:
            part_no = clean_meta.split("Part No:")[1].strip().split(" ")[0]
        if "Date:" in clean_meta:
            date_str = clean_meta.split("Date:")[1].split("|")[0].strip()
            
        filename_final = f"{model}_{model_type}_{part_no}_{date_str}.pdf".replace(" ", "_")
        print(f"Target Filename: {filename_final}")

        encoded_model_type = urllib.parse.quote(model_type)
        viewer_url = (
            f"{TOYOTA_PORTAL_URL}/pdf/{part_no}"
            f"?modelType={encoded_model_type}&lineOffDate={date_str}"
            f"&language={DEFAULT_LANGUAGE}&brand={DEFAULT_BRAND}"
        )
        
        print(f"Loading PDF Viewer: {viewer_url}")
        driver.get(viewer_url)
        
        print("Waiting for PDF Rendering...")
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div[role='progressbar']"))
            )
            WebDriverWait(driver, SCRAPER_TIMEOUTS["spinner_wait"]).until(
                EC.invisibility_of_element_located((By.CSS_SELECTOR, "div[role='progressbar']"))
            )
            print("Spinner finished.")
        except TimeoutException:
            print("Spinner check skipped.")

        print("Switching to Viewer Iframe...")
        try:
            iframe = WebDriverWait(driver, SCRAPER_TIMEOUTS["iframe_wait"]).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "iframe[title='webviewer']"))
            )
            driver.switch_to.frame(iframe)
            print("Switched context to iframe.")
        except TimeoutException:
            print("[CRITICAL] Iframe 'webviewer' not found.")
            raise Exception("Iframe missing")

        print("Opening Menu...")
        menu_btn = WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "button[data-element='menuButton']"))
        )
        driver.execute_script("arguments[0].click();", menu_btn)
        
        time.sleep(1) 
        try:
            WebDriverWait(driver, 5).until(lambda d: "active" in menu_btn.get_attribute("class"))
            print("Menu confirmed open.")
        except:
            pass

        print("Attempting Download...")
        files_before = set(os.listdir(target_dir))
        
        for attempt in range(3):
            try:
                time.sleep(1.5)
                download_btn = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "button[data-element='downloadButton']"))
                )
                driver.execute_script("arguments[0].click();", download_btn)
                print(f"Click attempt {attempt + 1} successful.")
                break 
            except StaleElementReferenceException:
                print(f"Attempt {attempt + 1}: Stale element. Retrying...")

        print("Waiting for file download...")
        downloaded_file = None
        for i in range(SCRAPER_TIMEOUTS["download_wait"]):
            current_files = set(os.listdir(target_dir))
            new_files = current_files - files_before
            valid = [
                f for f in new_files 
                if f.endswith('.pdf') and not f.endswith('.crdownload') and not f.endswith('.tmp')
            ]
            if valid:
                downloaded_file = valid[0]
                break
            time.sleep(1)
            if i % 10 == 0:
                print(f"Waiting... {i}s")

        if downloaded_file:
            print(f"[SUCCESS] Downloaded: {downloaded_file}")
            try:
                time.sleep(2)
                os.rename(
                    os.path.join(target_dir, downloaded_file),
                    os.path.join(target_dir, filename_final)
                )
                print(f"Renamed to: {filename_final}")
            except Exception as e:
                print(f"Rename failed: {e}")
        else:
            print("[ERROR] File download timed out.")

    except Exception as e:
        print(f"[CRITICAL ERROR]: {e}")
        driver.save_screenshot("debug_final_error.png")

    finally:
        print("Closing browser...")
        driver.quit()

if __name__ == "__main__":
    scrape_toyota_manual(
        model="RAV4",
        model_type="RAV4",
        generation_text="2018 - Today"
    )