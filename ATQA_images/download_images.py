import os
import zipfile
import urllib.request

IMAGES_URL = "http://aitriplea.net/atqa_images.zip"
ZIP_FILE = "atqa_images.zip"
IMAGES_DIR = "images"

def download_file(url, output_path):
    urllib.request.urlretrieve(url, output_path)

def extract_images(zip_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)

if __name__ == "__main__":
    if not os.path.exists(IMAGES_DIR):
        download_file(IMAGES_URL, ZIP_FILE)
        extract_images(ZIP_FILE, IMAGES_DIR)
        os.remove(ZIP_FILE)
        print("Images downloaded successfully.")
    else:
        print("Images already exist.")