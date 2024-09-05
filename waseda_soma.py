from torch.utils.data import ConcatDataset, Dataset
import pytorch_lightning as L
import torch
import os
import soundfile as sf
import numpy as np

class VapDataset(Dataset):
    def __init__(self, dataset_dir, split):
        self.dataset_dir = dataset_dir
        self.split = split
        self.paths = []
        self._load_dataset()

    def _load_dataset(self):
        utternces_txt_filename = os.path.join(self.dataset_dir, self.split, "utterances.txt")
        with open(utternces_txt_filename, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.rstrip()
                utt_id, wav_path, vad_path = line.split(" ")
                wav_relative_path = os.path.join(*(wav_path.split("/")[2:]))
                vad_relative_path = os.path.join(*(vad_path.split("/")[2:]))
                wav_path = os.path.join(self.dataset_dir, wav_relative_path)
                vad_path = os.path.join(self.dataset_dir, vad_relative_path)
                self.paths.append((utt_id, wav_path, vad_path))
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        utt_id, wav_path, vad_path = self.paths[idx]
        wav, fs = sf.read(wav_path)
        wav = wav.T
        wav = torch.from_numpy(wav).float()

        vad = np.load(vad_path)
        vad = torch.from_numpy(vad).float()

        return {"waveform": wav, "vad": vad}


class VapDataModuleWasedaSoma(L.LightningDataModule):
    def __init__(self, 
                 dataset_dir,
                 batch_size=10, num_workers=1, pin_memory=True):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.train_dataset = VapDataset(dataset_dir, "train")
        self.val_dataset = VapDataset(dataset_dir, "dev")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            # self.test_dataset,
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
