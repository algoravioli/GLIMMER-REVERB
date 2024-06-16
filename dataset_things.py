import glob
import torchaudio
import torch
import os


class AudioEffectDataset(torch.nn.Module):
    def __init__(self, root_dir: str) -> None:
        super().__init__()
        # find all audio files in the root directory, input = "data/input", output = "data/output"
        self.input_files = glob.glob(os.path.join(root_dir, "input", "*.wav"))
        self.output_files = glob.glob(os.path.join(root_dir, "output", "*.wav"))

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx: int):
        input_filepath = self.input_files[idx]

        # read segment of audio from file
        x, sr = torchaudio.load(
            input_filepath,
            backend="soundfile",
        )
        # reduce x to 1000 samples but remain 2 channels
        x = x[:, :1000].clone()
        x = x.unsqueeze(0).clone()  # add batch dim

        y, sr = torchaudio.load(
            self.output_files[idx],
            backend="soundfile",
        )
        # recude y to 1000 samples but remain 2 channels
        y = y[:, :1000].clone()
        y = y.unsqueeze(0).clone()

        # remove batch dim
        x = x.squeeze(0).clone()
        y = y.squeeze(0).clone()

        return x, y
