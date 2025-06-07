from app.const import DATASET_NAME, SPLIT
from datasets import load_dataset
import pandas as pd


class Dataset:
    def __init__(self, dataset_name=DATASET_NAME, split=SPLIT):
        self.dataset_name = dataset_name
        self.split = split
        self.dataset = self.load_dataset()

    def load_dataset(self):
        try:
            dataset = load_dataset(self.dataset_name, split=self.split)
            return dataset.to_pandas()
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
