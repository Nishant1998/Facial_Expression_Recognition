import os
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset

class RAFDBDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        """
        :param csv_file: Path to CSV file containing image names and labels (relative or absolute).
        :param root_dir: Root directory where images are stored (relative or absolute).
        """
        PROJECT_ROOT = Path(__file__).resolve().parents[3]

        csv_path = PROJECT_ROOT / csv_file if not Path(csv_file).is_absolute() else Path(csv_file)
        images_root = PROJECT_ROOT / root_dir if not Path(root_dir).is_absolute() else Path(root_dir)

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        if not images_root.exists():
            raise FileNotFoundError(f"Image root dir not found: {images_root}")

        self.labels = pd.read_csv(csv_path)
        self.root_dir = images_root

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, str(self.labels.iloc[idx, 1]), self.labels.iloc[idx, 0])
        label = int(self.labels.iloc[idx, 1]) - 1
        return img_path, label
