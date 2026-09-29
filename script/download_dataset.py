from datasets import load_dataset
from huggingface_hub import hf_hub_download
import os
import shutil

output_dir = "/Users/phil/Desktop/coding_temp_file/HW1/data/original"
os.makedirs(output_dir, exist_ok=True)

# 1. Load and save train/test
dataset = load_dataset("bkonkle/snips-joint-intent")

dataset["train"].to_csv(
    os.path.join(output_dir, "train.csv"),
    index=False
)

dataset["test"].to_csv(
    os.path.join(output_dir, "test.csv"),
    index=False
)

# 2. Download label files directly from the Hugging Face dataset repo
for filename in ["intent_labels.txt", "slot_labels.txt"]:
    downloaded_path = hf_hub_download(
        repo_id="bkonkle/snips-joint-intent",
        filename=filename,
        repo_type="dataset"
    )

    shutil.copy(
        downloaded_path,
        os.path.join(output_dir, filename)
    )

print("Saved all files to:")
print(output_dir)