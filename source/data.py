import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).parent.parent / "data"


class VQADataset:
    def __init__(self, split: str, data_dir: Path = None, max_samples: int = None):
        """
        Initialize the VQA dataset.

        Args:
            split: Dataset split ('train', 'valid')
            data_dir: Directory containing the dataset (default: DATA_DIR)
        """
        self.data_dir = data_dir or DATA_DIR

        # Load all QA pairs for the split
        self.qa_pairs = []

        with (DATA_DIR / f"{split}.jsonl").open() as f:
            for line in f:
                self.qa_pairs.append(json.loads(line))

        if max_samples is not None:
            self.qa_pairs = self.qa_pairs[:max_samples]

        print(f"Loaded {len(self.qa_pairs)} QA pairs for {split} split")

    def __len__(self):
        return len(self.qa_pairs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Get a QA pair by index.

        Args:
            idx: Index of the QA pair

        Returns:
            Dictionary containing the QA pair and image path
        """
        qa_pair = self.qa_pairs[idx]

        # Construct the full path to the image
        image_path = qa_pair["image"]

        return {
            "image_path": image_path,
            "system": qa_pair["system"],
            "question": qa_pair["user"],
            "answer": qa_pair["response"],
        }

if __name__ == "__main__":
    # Test the dataset
    dataset = VQADataset("train")
    print(f"Dataset size: {len(dataset)}")

    # Print a sample
    sample = dataset[0]
    print("\nSample:")
    print(f"Image: {sample['image_path']}")
    print(f"Question: {sample['question']}")
    print(f"Answer: {sample['answer']}")
