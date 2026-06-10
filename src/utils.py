from datetime import datetime
import os

def get_results_folder(tag, base_dir="results"):
    """Creates a unique subfolder name based on the current time."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_path = os.path.join(base_dir, f"{tag}_{timestamp}")

    os.makedirs(folder_path, exist_ok=True)

    return folder_path