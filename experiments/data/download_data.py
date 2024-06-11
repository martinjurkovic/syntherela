import os
import requests
import zipfile

import gdown
from dotenv import load_dotenv

load_dotenv()

PROJECT_PATH = os.getenv("PROJECT_PATH")

def download_and_extract(url, filename):
    # Path to the directory where the file will be extracted
    extract_dir = os.path.join(PROJECT_PATH, "data")

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
orig_url = "https://drive.google.com/uc?id=11-vhJejKLCA-PAvj9o3MMItvy6RBL8dm"
synth_url = "https://drive.google.com/uc?id=1b6qebgzniF3Zro52WpoIKS9W6C-8O4d-"

download_and_extract(orig_url, "original.zip")
download_and_extract(synth_url, "synthetic.zip")

print("Files downloaded and extracted successfully!")