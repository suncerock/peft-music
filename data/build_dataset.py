import json
from typing import Dict

from .dataset import ALL_DATASETS


def build_datasets(exp: str, config: Dict):
    manifest_path = config.pop("manifest_path")
    base_audio_path = config.pop("base_audio_path")

    with open(manifest_path) as f:
        data_list = [json.loads(line) for line in f]
    train_data_list = [data for data in data_list if data["split"] == "train"]
    valid_data_list = [data for data in data_list if data["split"] == "val"]
    test_data_list = [data for data in data_list if data["split"] == "test"]
    
    train_dataset = ALL_DATASETS[exp](train_data_list, base_audio_path, split="train", **config)
    valid_dataset = ALL_DATASETS[exp](valid_data_list, base_audio_path, split="val", **config)
    test_dataset = ALL_DATASETS[exp](test_data_list, base_audio_path, split="test", **config)

    return train_dataset, valid_dataset, test_dataset