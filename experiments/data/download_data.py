import os
import zipfile

import gdown

PROJECT_PATH = __file__.split('experiments')[0]


def download_and_extract(url, filename, subdirectory):
    # Path to the directory where the file will be extracted
    extract_dir = os.path.join(PROJECT_PATH, 'data', subdirectory)

    # Create the download directory if it doesn't exist
    os.makedirs(extract_dir, exist_ok=True)

    # Download to a temporary location first
    temp_dir = os.path.join(PROJECT_PATH, 'data')
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, filename)
    gdown.download(url, file_path, quiet=False)

    # Extract the file to the specific subdirectory
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    # Clean up the downloaded zip file
    os.remove(file_path)


# URL of the file to download
orig_url = 'https://drive.google.com/uc?id=1FIBnmdQSVUK4xi5uFpzb_vFseK_KLQUG'
synth_url = 'https://drive.google.com/uc?id=1VRoU57Z-J2QV9J4QTNWo-XTWdD8hqkAl'

download_and_extract(orig_url, 'original.zip', 'original')
download_and_extract(synth_url, 'synthetic.zip', 'synthetic')

print('Files downloaded and extracted successfully!')
