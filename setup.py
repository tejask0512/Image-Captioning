import os

# Define the directory structure
structure = {
    "models": ["__init__.py", "encoder.py", "decoder.py", "caption_model.py"],
    "utils": ["__init__.py", "data_loader.py", "preprocessing.py", "metrics.py"],
    "configs": ["config.py"],
    "": ["main.py", "train.py", "evaluate.py", "predict.py", "requirements.txt"]
}

def create_structure():
    for folder, files in structure.items():
        if folder and not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created folder: {folder}")
        for file in files:
            path = os.path.join(folder, file)
            if not os.path.exists(path):
                open(path, 'w').close()
                print(f"Created file: {path}")
            else:
                print(f"File already exists: {path}")

if __name__ == "__main__":
    create_structure()
