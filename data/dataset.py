import os

import soundfile as sf

import numpy as np
import torch.utils.data as Data


class BaseSongLevelDataset(Data.Dataset):
    def __init__(
        self,
        data_list,
        base_audio_path,
        split,

        length=-1
    ) -> None:
        super().__init__()

        self.data = data_list
        self.base_audio_path = base_audio_path

        self.length = length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]

        output_data = dict()
        audio_path = os.path.join(self.base_audio_path, data["audio_path"])
        audio, start_frame, length_frame = self.load_audio(audio_path, original_length=data["length"])

        output_data["x"] = audio
        output_data["length"] = length_frame
        
        output_data["x_start_time"] = start_frame / data["sr"]
        output_data["x_end_time"] = (start_frame + length_frame) / data["sr"]

        output_data["y"] = self.convert_label(
            data["label"], start_time=output_data["x_start_time"], end_time=output_data["x_end_time"], sr=data["sr"])

        return output_data

    def load_audio(self, path, original_length):
        if self.length < 0:
            start_frame = 0
            audio, _ = sf.read(path, dtype=np.float32)
        else:
            start_frame = np.random.randint(low=0, high=original_length - self.length)
            audio, _ = sf.read(path, frames=self.length, start=start_frame, dtype=np.float32)
        return audio, start_frame, len(audio)

    def convert_label(self, label):
        raise NotImplementedError


class AutoTaggingDataset(BaseSongLevelDataset):
    def __init__(self, data_list, base_audio_path, split, length=-1) -> None:
        super().__init__(data_list, base_audio_path, split, length=length)

        if split == "test":
            self.length = -1

    def convert_label(self, label, **kwargs):
        y = np.array(label)
        return (y == 1).astype(np.int64)


ALL_DATASETS = dict(
    mtat=AutoTaggingDataset,

    mtg_top50=AutoTaggingDataset,
    mtg_genre=AutoTaggingDataset,
    mtg_instrument=AutoTaggingDataset,
    mtg_moodtheme=AutoTaggingDataset,
)
