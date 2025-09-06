# infinite/train/datasets/rl.py
import copy
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset

class RLDataset(Dataset):
    """Wraps a HuggingFace dataset for the RL trainer."""
    def __init__(self, dataset: HFDataset, responses_per_prompt: int):
        self.dataset = dataset
        self.responses_per_prompt = responses_per_prompt

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    def collate_fn(self, batch):
        return [copy.deepcopy(ex) for ex in batch for _ in range(self.responses_per_prompt)]