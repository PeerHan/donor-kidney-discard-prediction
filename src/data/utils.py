import os
import time
import logging

def safe_delete_file(file_path: str, max_retries: int = 3) -> bool:
    """Safely delete a file with retries and error handling.
    Returns True if file was successfully deleted, False otherwise."""
    for attempt in range(max_retries):
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
        except (IOError, PermissionError) as e:
            wait_time = 2 ** attempt  # Exponential backoff
            logging.warning("Attempt %d/%d to delete %s failed: %s",
                          attempt + 1, max_retries, file_path, e)
            if attempt < max_retries - 1:
                time.sleep(wait_time)
            else:
                logging.error("Failed to delete %s after %d attempts",
                            file_path, max_retries)
    return False

def clear_csv_files(dataset_path: str) -> None:
    """Clear all CSV files in the given directory."""
    print(f"Clearing existing CSV files in {dataset_path}...")
    failed_files = []
    for file in os.listdir(dataset_path):
        if file.endswith('.csv'):
            file_path = os.path.join(dataset_path, file)
            if not safe_delete_file(file_path):
                failed_files.append(file)

    if failed_files:
        print(f"Failed to clear some CSV files: {', '.join(failed_files)}")
    else:
        print("Successfully cleared all CSV files")
