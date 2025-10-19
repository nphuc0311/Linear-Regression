import pandas as pd
from pathlib import Path


def download_data(url: str) -> None:
    """Download the advertising dataset from URL."""
    # Create data/raw directory if it doesn't exist
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Download data
    try:
        df = pd.read_csv(url)
        output_path = raw_dir / "advertising.csv"
        df.to_csv(output_path, index=False)
        print(f"Data downloaded successfully to {output_path}")
    except Exception as e:
        print(f"Error downloading data: {e}")
        raise

if __name__ == "__main__":
    url = "https://media.geeksforgeeks.org/wp-content/uploads/20240522145649/advertising.csv"
    download_data(url)