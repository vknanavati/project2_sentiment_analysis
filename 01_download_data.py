# kagglehub lets us download datasets from Kaggle directly in Python
# Think of it like a delivery service — you give it a dataset name and it fetches it for you
import kagglehub

# shutil is Python's built-in file copying tool
# The name stands for "shell utilities" — it lets you move and copy files
# the same way you would with cp in the terminal
import shutil

# os lets us interact with the operating system — create folders, list files, build file paths
# Anything you'd normally do in Finder or the terminal, os can do in Python
import os


def download_dataset():
    # This function's only job is to download the Sentiment140 dataset
    # and move it into our project's data/ folder

    # kagglehub.dataset_download() fetches the dataset from Kaggle and saves it
    # to a temporary cache folder somewhere on your Mac
    # "kazanova/sentiment140" is the dataset's unique ID on Kaggle (username/dataset-name)
    # The variable path now holds the location of where kagglehub put the files
    # Analogy: this is like ordering a package online — it gets delivered to a holding facility first
    print("Downloading Sentiment140 dataset from Kaggle...")
    path = kagglehub.dataset_download("kazanova/sentiment140")

    # Confirm where kagglehub put the temporary files so we know what we're working with
    print(f"Dataset cached at: {path}")


    # os.makedirs() creates the data/ folder inside our project directory
    # exist_ok=True means "if the folder already exists, don't crash — just move on"
    # Analogy: like telling someone to set up a filing cabinet, but only if one isn't already there
    os.makedirs("data", exist_ok=True)


    # os.listdir(path) returns a list of every filename in the temporary kagglehub folder
    # We loop through each one so we can copy them all over
    for file in os.listdir(path):

        # os.path.join(path, file) builds the full file path by combining the folder
        # location and the filename — like writing the full address on an envelope
        # shutil.copy() then copies that file into our project's data/ folder
        # Analogy: driving to the holding facility and bringing the package back to your own filing cabinet
        shutil.copy(os.path.join(path, file), "data/")

    # Confirm the script finished and show what landed in data/
    print("Files saved to data/:")
    print(os.listdir("data"))


# This block only runs if you execute this file directly (python 01_download_data.py)
# If another script imports this file, this block is skipped
if __name__ == "__main__":
    download_dataset()