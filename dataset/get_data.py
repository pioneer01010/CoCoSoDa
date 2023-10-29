from pathlib import Path
import kaggle

path = Path("dataset")
path.parent.mkdir(parents=True, exist_ok=True)
kaggle.api.authenticate()
kaggle.api.dataset_download_files("omduggineni/codesearchnet", path=path, unzip=True)