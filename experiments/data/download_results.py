import os
import requests
import zipfile

import gdown
from dotenv import load_dotenv

load_dotenv()

PROJECT_PATH = os.getenv("PROJECT_PATH")

def download_and_extract(url, filename):
    # Path to the directory where the file will be extracted
    extract_dir = os.path.join(PROJECT_PATH)

    # Create the download directory if it doesn't exist
    os.makedirs(extract_dir, exist_ok=True)

    file_path = os.path.join(extract_dir, filename)
    gdown.download(url, file_path, quiet=False)

    # Extract the file
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    # Clean up the downloaded zip file
    os.remove(file_path)

# URL of the file to download
results_url = "https://drive.google.com/uc?id=1FprsHPYWDrE2AJKscRIVET_fvh0LoJX1"

download_and_extract(results_url, "results.zip")

print("Results downloaded and extracted successfully!")