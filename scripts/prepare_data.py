import os
import subprocess
import splitfolders

def download_dataset():
    """Downloads the dataset using the Kaggle API."""
    print("Downloading dataset from Kaggle...")
    # Note: ensure kaggle API token is located at ~/.kaggle/kaggle.json
    try:
        subprocess.run(["python", "-m", "kaggle", "datasets", "download", "-d", "emmarex/plantdisease", "--unzip"], check=True)
        print("Dataset downloaded and unzipped successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset. Ensure your kaggle.json is set up properly. {e}")
        raise

def split_dataset():
    """Splits the dataset into train, val, and test sets."""
    print("Splitting dataset into train (80%), val (10%), test (10%)...")
    # The emmarex/plantdisease dataset creates a "PlantVillage" folder containing classes
    input_folder = "PlantVillage" 
    
    if not os.path.exists(input_folder):
        # Sometime it unzips into another subfolder structure. Let's look for it
        if os.path.exists("plantvillage dataset/color"):
            input_folder = "plantvillage dataset/color"
        else:
            print("Could not find the extracted PlantVillage dataset folder.")
            return

    output_folder = "data"
    
    # splitfolders handles the ratio
    splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(0.8, 0.1, 0.1), group_prefix=None, move=False)
    print(f"Data successfully split and saved to '{output_folder}'.")

if __name__ == "__main__":
    download_dataset()
    split_dataset()
