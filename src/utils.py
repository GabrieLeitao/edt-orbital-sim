from datetime import datetime
import os

def get_results_folder(tag="run"):
    """Creates a unique subfolder name based on the current time."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_path = os.path.join("results", f"{tag}_{timestamp}")
    
    os.makedirs(folder_path, exist_ok=True)
    
    return folder_path