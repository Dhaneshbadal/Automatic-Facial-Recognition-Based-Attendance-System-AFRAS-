# cleanup_dataset.py
import os
import shutil
from pathlib import Path

dataset_path = Path("recognition/dataset")
folders_to_delete = [
    "Chris_Andrews", "Don_Hewitt", "Elena_de_Chavez", "Ernie_Stewart",
    "Hernan_Diaz", "Jaouad_Gharib", "Jeff_Bezos", "Jeff_Feldman",
    "Joe_Vandever", "John_Burkett", "John_Thune", "Jon_Stewart",
    "models", "Tatiana_Gratcheva", "training_data", "Vecdi_Gonul", "Yoelbi_Quesada"
]

for folder in folders_to_delete:
    folder_path = dataset_path / folder
    if folder_path.exists():
        shutil.rmtree(folder_path)
        print(f"Deleted: {folder}")

print("Cleanup complete!")