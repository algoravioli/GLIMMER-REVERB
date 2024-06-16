import numpy as np
import torch
import torch.nn as nn
import soundfile as sf
from tqdm import tqdm


class NNCombFilterClone(nn.Module):
    def __init__(self, delay, feedback):
        super(NNCombFilterClone, self).__init__()
        self.delay = delay
        self.feedback = feedback
        self.buffer = torch.zeros(int(delay), dtype=torch.float32)
        self.buffer_index = 0

    def forward(self, x):
        self.output_sample = self.buffer[int(self.buffer_index)]
        self.feedbacked_sample = self.output_sample * self.feedback
        new_buffer = self.buffer.clone()  # Avoid in-place operation
        new_buffer[int(self.buffer_index)] = x + self.feedbacked_sample
        self.buffer = new_buffer.clone()
        self.dummy_delay = self.delay + 1e-6
        self.buffer_index = int((self.buffer_index + 1) % self.dummy_delay)
        return self.output_sample


class NNAllpassFilterClone(nn.Module):
    def __init__(self, delay, feedback):
        super(NNAllpassFilterClone, self).__init__()
        self.delay = delay
        self.feedback = feedback
        self.buffer = torch.zeros(int(delay), dtype=torch.float32)
        self.buffer_index = 0

    def forward(self, x):
        self.buffer_output = self.buffer[int(self.buffer_index)]
        self.output_sample = -x + self.buffer_output
        new_buffer = self.buffer.clone()  # Avoid in-place operation
        new_buffer[int(self.buffer_index)] = x + self.buffer_output * self.feedback
        self.buffer = new_buffer.clone()
        self.dummy_delay = self.delay + 1e-6
        self.buffer_index = int((self.buffer_index + 1) % self.dummy_delay)
        return self.output_sample


class NNFreeverbModuleClone(nn.Module):
    def __init__(self, sample_rate=44100):
        super(NNFreeverbModuleClone, self).__init__()
        self.sample_rate = sample_rate
        self.combL = [
            NNCombFilterClone(2205, 0.805),
            NNCombFilterClone(2469, 0.827),
            NNCombFilterClone(2690, 0.783),
            NNCombFilterClone(2998, 0.764),
            NNCombFilterClone(3175, 0.742),
            NNCombFilterClone(3439, 0.733),
            NNCombFilterClone(3627, 0.715),
            NNCombFilterClone(4001, 0.697),
        ]
        self.combR = [
            NNCombFilterClone(2277, 0.805),
            NNCombFilterClone(2709, 0.827),
            NNCombFilterClone(2924, 0.783),
            NNCombFilterClone(3175, 0.764),
            NNCombFilterClone(3351, 0.742),
            NNCombFilterClone(3487, 0.733),
            NNCombFilterClone(3660, 0.715),
            NNCombFilterClone(4117, 0.697),
        ]
        self.allpassL = [
            NNAllpassFilterClone(556, 0.5),
            NNAllpassFilterClone(441, 0.5),
            NNAllpassFilterClone(341, 0.5),
            NNAllpassFilterClone(225, 0.5),
        ]
        self.allpassR = [
            NNAllpassFilterClone(579, 0.5),
            NNAllpassFilterClone(464, 0.5),
            NNAllpassFilterClone(396, 0.5),
            NNAllpassFilterClone(289, 0.5),
        ]

    def update_params(self, params):
        # Update comb filter delays and feedbacks
        for i in range(8):
            self.combL[i].delay = int(params[5 + i] * 5000)
            self.combL[i].feedback = params[13 + i]
            self.combR[i].delay = int(params[21 + i] * 5000)
            self.combR[i].feedback = params[29 + i]

        # Update allpass filter delays and feedbacks
        for i in range(4):
            self.allpassL[i].delay = int(params[37 + i])
            self.allpassL[i].feedback = params[41 + i]
            self.allpassR[i].delay = int(params[45 + i])
            self.allpassR[i].feedback = params[49 + i]

        self.roomSize = params[0]
        self.damp = params[1]
        self.wet = params[2]
        self.dry = params[3]
        self.width = params[4]
        self.wet1 = self.wet * (self.width + 1) * 0.5
        self.wet2 = self.wet * (1 - self.width) * 0.5

    def forward(self, input_audio):
        input_audio = input_audio.squeeze(0)
        output_audio = torch.zeros_like(input_audio)
        input_length = input_audio.size(1)

        for i in tqdm(range(input_length)):
            inL, inR = input_audio[0, i], input_audio[1, i]

            outL = sum([comb(inL) for comb in self.combL])
            outR = sum([comb(inR) for comb in self.combR])

            outL = sum([allpass(outL.clone()) for allpass in self.allpassL])
            outR = sum([allpass(outR.clone()) for allpass in self.allpassR])

            output_audio[0, i] = (
                outL * self.wet1 + outR.clone() * self.wet2 + inL.clone() * self.dry
            )
            output_audio[1, i] = (
                outR * self.wet1 + outL.clone() * self.wet2 + inR.clone() * self.dry
            )

        return output_audio


def test_model():
    model = NNFreeverbModuleClone()
    input_parameters = torch.tensor(
        [
            0.5,
            0.5,
            1.0,
            0.0,
            1.0,
            2205,
            2469,
            2690,
            2998,
            3175,
            3439,
            3627,
            4001,
            0.805,
            0.827,
            0.783,
            0.764,
            0.742,
            0.733,
            0.715,
            0.697,
            2277,
            2709,
            2924,
            3175,
            3351,
            3487,
            3660,
            4117,
            0.805,
            0.827,
            0.783,
            0.764,
            0.742,
            0.733,
            0.715,
            0.697,
            556,
            441,
            341,
            225,
            0.5,
            0.5,
            0.5,
            0.5,
            579,
            464,
            396,
            289,
            0.5,
            0.5,
            0.5,
            0.5,
        ]
    )
    output_audio = model(torch.zeros((2, 1000)), input_parameters)
    output_audio = output_audio.detach().numpy().T
    output_audio = output_audio / np.max(np.abs(output_audio))
    sf.write("output.wav", output_audio, 44100)
