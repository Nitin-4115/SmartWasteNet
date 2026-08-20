import os
import shutil
import kagglehub

def main():
    print("⏳ Downloading TrashNet dataset from Kaggle...")
    
    # Download latest version to Kaggle's cache
    cache_path = kagglehub.dataset_download("vishwasmishra1234/trash-net")
    print(f"✅ Downloaded to cache: {cache_path}")

    # Set destination to the 'dataset' folder in your project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dest_dir = os.path.join(project_root, "dataset")
    
    os.makedirs(dest_dir, exist_ok=True)

    print(f"🚚 Moving files to {dest_dir}...")
    
    # Copy all downloaded contents into your local dataset folder
    for item in os.listdir(cache_path):
        s = os.path.join(cache_path, item)
        d = os.path.join(dest_dir, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)
            
    print("🎉 Dataset successfully placed in the 'dataset/' folder! Ready for conversion.")

if __name__ == "__main__":
    main()