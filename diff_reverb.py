# %%
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
from tqdm import tqdm


class NNCombFilter(nn.Module):
    def __init__(self, delay, feedback):
        super(NNCombFilter, self).__init__()
        self.delay = delay
        self.feedback = feedback
        self.buffer = torch.zeros(int(delay.item()), dtype=torch.float32)
        self.buffer_index = nn.Parameter(torch.tensor(0.0))
        self.output_sample = 0
        self.remainder_before_int = 0

    def forward(self, x):

        self.output_sample = self.buffer[self.buffer_index.int()]
        # y[n] = x[n] + feedback * y[n - delay]
        self.new_buffer = self.buffer.clone()
        self.feedbacked_sample = torch.multiply(self.output_sample, self.feedback)
        self.new_buffer[self.buffer_index.int()] = torch.add(
            x.clone(), self.feedbacked_sample
        )
        self.dummy_delay = torch.add(self.delay, 1e-6)
        self.remainder_before_int = torch.remainder(
            self.buffer_index + 1, self.dummy_delay
        )
        self.buffer = self.new_buffer
        self.buffer_index = nn.Parameter(self.remainder_before_int)
        return self.output_sample


class NNAllpassFilter(nn.Module):
    def __init__(self, delay, feedback):
        super(NNAllpassFilter, self).__init__()
        self.delay = delay
        self.feedback = feedback
        self.buffer = torch.zeros(int(delay.item()), dtype=torch.float32)
        self.buffer_index = 0
        self.output_sample = 0
        self.buffer_output = 0

    def forward(self, x):
        self.buffer_output = self.buffer[self.buffer_index]
        # y[n] = -x[n] + buffer_output
        self.output_sample = torch.add(torch.multiply(x, -1), self.buffer_output)
        self.new_buffer = self.buffer.clone()
        self.new_buffer[self.buffer_index] = torch.add(
            x, torch.multiply(self.buffer_output, self.feedback)
        )
        self.buffer = self.new_buffer
        self.dummy_delay = torch.add(self.delay, 1e-6)
        self.buffer_index = torch.remainder(
            self.buffer_index + 1, self.dummy_delay
        ).int()
        return self.output_sample


class NNFreeverbModule(nn.Module):
    def __init__(self, sample_rate=44100):
        super(NNFreeverbModule, self).__init__()
        self.sample_rate = sample_rate
        # combL has params [2205, 2469, 2690, 2998, 3175, 3439, 3627, 4001], [0.805, 0.827, 0.783, 0.764, 0.742, 0.733, 0.715, 0.697]
        self.combL = nn.ModuleList(
            [
                NNCombFilter(
                    nn.Parameter(torch.tensor(2205.0), requires_grad=True),
                    nn.Parameter(torch.tensor(0.805), requires_grad=True),
                ),
                NNCombFilter(
                    nn.Parameter(torch.tensor(2469.0), requires_grad=True),
                    nn.Parameter(torch.tensor(0.827), requires_grad=True),
                ),
                NNCombFilter(
                    nn.Parameter(torch.tensor(2690.0), requires_grad=True),
                    nn.Parameter(torch.tensor(0.783), requires_grad=True),
                ),
                NNCombFilter(
                    nn.Parameter(torch.tensor(2998.0), requires_grad=True),
                    nn.Parameter(torch.tensor(0.764), requires_grad=True),
                ),
                NNCombFilter(
                    nn.Parameter(torch.tensor(3175.0), requires_grad=True),
                    nn.Parameter(torch.tensor(0.742), requires_grad=True),
                ),
                NNCombFilter(
                    nn.Parameter(torch.tensor(3439.0), requires_grad=True),
                    nn.Parameter(torch.tensor(0.733), requires_grad=True),
                ),
                NNCombFilter(
                    nn.Parameter(torch.tensor(3627.0), requires_grad=True),
                    nn.Parameter(torch.tensor(0.715), requires_grad=True),
                ),
                NNCombFilter(
                    nn.Parameter(torch.tensor(4001.0), requires_grad=True),
                    nn.Parameter(torch.tensor(0.697), requires_grad=True),
                ),
            ]
        )
        # combR has params [2277, 2709, 2924, 3175, 3351, 3487, 3660, 4117], [0.805, 0.827, 0.783, 0.764, 0.742, 0.733, 0.715, 0.697]
        self.combR = nn.ModuleList(
            [
                NNCombFilter(
                    nn.Parameter(torch.tensor(2277.0), requires_grad=True),
                    nn.Parameter(torch.tensor(0.805), requires_grad=True),
                ),
                NNCombFilter(
                    nn.Parameter(torch.tensor(2709.0), requires_grad=True),
                    nn.Parameter(torch.tensor(0.827), requires_grad=True),
                ),
                NNCombFilter(
                    nn.Parameter(torch.tensor(2924.0), requires_grad=True),
                    nn.Parameter(torch.tensor(0.783), requires_grad=True),
                ),
                NNCombFilter(
                    nn.Parameter(torch.tensor(3175.0), requires_grad=True),
                    nn.Parameter(torch.tensor(0.764), requires_grad=True),
                ),
                NNCombFilter(
                    nn.Parameter(torch.tensor(3351.0), requires_grad=True),
                    nn.Parameter(torch.tensor(0.742), requires_grad=True),
                ),
                NNCombFilter(
                    nn.Parameter(torch.tensor(3487.0), requires_grad=True),
                    nn.Parameter(torch.tensor(0.733), requires_grad=True),
                ),
                NNCombFilter(
                    nn.Parameter(torch.tensor(3660.0), requires_grad=True),
                    nn.Parameter(torch.tensor(0.715), requires_grad=True),
                ),
                NNCombFilter(
                    nn.Parameter(torch.tensor(4117.0), requires_grad=True),
                    nn.Parameter(torch.tensor(0.697), requires_grad=True),
                ),
            ]
        )
        # allpassL has params [556, 441, 341, 225], [0.5, 0.5, 0.5, 0.5]
        self.allpassL = nn.ModuleList(
            [
                NNAllpassFilter(
                    nn.Parameter(torch.tensor(556.0), requires_grad=True),
                    nn.Parameter(torch.tensor(0.5), requires_grad=True),
                ),
                NNAllpassFilter(
                    nn.Parameter(torch.tensor(441.0), requires_grad=True),
                    nn.Parameter(torch.tensor(0.5), requires_grad=True),
                ),
                NNAllpassFilter(
                    nn.Parameter(torch.tensor(341.0), requires_grad=True),
                    nn.Parameter(torch.tensor(0.5), requires_grad=True),
                ),
                NNAllpassFilter(
                    nn.Parameter(torch.tensor(225.0), requires_grad=True),
                    nn.Parameter(torch.tensor(0.5), requires_grad=True),
                ),
            ]
        )
        # allpassR has params [579, 464, 396, 289], [0.5, 0.5, 0.5, 0.5]
        self.allpassR = nn.ModuleList(
            [
                NNAllpassFilter(
                    nn.Parameter(torch.tensor(579.0), requires_grad=True),
                    nn.Parameter(torch.tensor(0.5), requires_grad=True),
                ),
                NNAllpassFilter(
                    nn.Parameter(torch.tensor(464.0), requires_grad=True),
                    nn.Parameter(torch.tensor(0.5), requires_grad=True),
                ),
                NNAllpassFilter(
                    nn.Parameter(torch.tensor(396.0), requires_grad=True),
                    nn.Parameter(torch.tensor(0.5), requires_grad=True),
                ),
                NNAllpassFilter(
                    nn.Parameter(torch.tensor(289.0), requires_grad=True),
                    nn.Parameter(torch.tensor(0.5), requires_grad=True),
                ),
            ]
        )

        self.roomSize = nn.Parameter(torch.tensor(0.5, requires_grad=True))
        self.damp = nn.Parameter(torch.tensor(0.5, requires_grad=True))
        self.wet = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.dry = nn.Parameter(torch.tensor(0.0, requires_grad=True))
        self.width = nn.Parameter(torch.tensor(0.0, requires_grad=True))
        self.wet1 = torch.multiply(
            self.wet, torch.multiply(torch.add(self.width, 1), 0.5)
        )
        self.wet2 = torch.multiply(
            self.wet, torch.multiply(torch.add(1, torch.multiply(self.width, -1)), 0.5)
        )
        # self.update_params()

    def update_params(self, params=None):
        # print(params.shape)
        # print(params[0])

        if params is not None:
            self.roomSize = nn.Parameter(params[0].clone())
            self.damp = nn.Parameter(params[1].clone())
            self.wet = nn.Parameter(params[2].clone())
            self.dry = nn.Parameter(params[3].clone())
            self.width = nn.Parameter(params[4].clone())
            self.combL[0].delay = nn.Parameter(params[5].clone() * 5000)
            self.combL[1].delay = nn.Parameter(params[6].clone() * 5000)
            self.combL[2].delay = nn.Parameter(params[7].clone() * 5000)
            self.combL[3].delay = nn.Parameter(params[8].clone() * 5000)
            self.combL[4].delay = nn.Parameter(params[9].clone() * 5000)
            self.combL[5].delay = nn.Parameter(params[10].clone() * 5000)
            self.combL[6].delay = nn.Parameter(params[11].clone() * 5000)
            self.combL[7].delay = nn.Parameter(params[12].clone() * 5000)
            self.combL[0].feedback = nn.Parameter(params[13].clone())
            self.combL[1].feedback = nn.Parameter(params[14].clone())
            self.combL[2].feedback = nn.Parameter(params[15].clone())
            self.combL[3].feedback = nn.Parameter(params[16].clone())
            self.combL[4].feedback = nn.Parameter(params[17].clone())
            self.combL[5].feedback = nn.Parameter(params[18].clone())
            self.combL[6].feedback = nn.Parameter(params[19].clone())
            self.combL[7].feedback = nn.Parameter(params[20].clone())
            self.combR[0].delay = nn.Parameter(params[21].clone())
            self.combR[1].delay = nn.Parameter(params[22].clone())
            self.combR[2].delay = nn.Parameter(params[23].clone())
            self.combR[3].delay = nn.Parameter(params[24].clone())
            self.combR[4].delay = nn.Parameter(params[25].clone())
            self.combR[5].delay = nn.Parameter(params[26].clone())
            self.combR[6].delay = nn.Parameter(params[27].clone())
            self.combR[7].delay = nn.Parameter(params[28].clone())
            self.combR[0].feedback = nn.Parameter(params[29].clone())
            self.combR[1].feedback = nn.Parameter(params[30].clone())
            self.combR[2].feedback = nn.Parameter(params[31].clone())
            self.combR[3].feedback = nn.Parameter(params[32].clone())
            self.combR[4].feedback = nn.Parameter(params[33].clone())
            self.combR[5].feedback = nn.Parameter(params[34].clone())
            self.combR[6].feedback = nn.Parameter(params[35].clone())
            self.combR[7].feedback = nn.Parameter(params[36].clone())
            self.allpassL[0].delay = nn.Parameter(params[37].clone())
            self.allpassL[1].delay = nn.Parameter(params[38].clone())
            self.allpassL[2].delay = nn.Parameter(params[39].clone())
            self.allpassL[3].delay = nn.Parameter(params[40].clone())
            self.allpassL[0].feedback = nn.Parameter(params[41].clone())
            self.allpassL[1].feedback = nn.Parameter(params[42].clone())
            self.allpassL[2].feedback = nn.Parameter(params[43].clone())
            self.allpassL[3].feedback = nn.Parameter(params[44].clone())
            self.allpassR[0].delay = nn.Parameter(params[45].clone())
            self.allpassR[1].delay = nn.Parameter(params[46].clone())
            self.allpassR[2].delay = nn.Parameter(params[47].clone())
            self.allpassR[3].delay = nn.Parameter(params[48].clone())
            self.allpassR[0].feedback = nn.Parameter(params[49].clone())
            self.allpassR[1].feedback = nn.Parameter(params[50].clone())
            self.allpassR[2].feedback = nn.Parameter(params[51].clone())
            self.allpassR[3].feedback = nn.Parameter(params[52].clone())
            self.wet1 = torch.multiply(
                self.wet.clone(), torch.multiply(torch.add(self.width.clone(), 1), 0.5)
            )
            self.wet2 = torch.multiply(
                self.wet.clone(),
                torch.multiply(
                    torch.add(1, torch.multiply(self.width.clone(), -1)), 0.5
                ),
            )

        else:
            print("No parameters to update")

    def forward(self, input_audio):
        input_audio = input_audio.squeeze(0)
        output_audio = torch.zeros_like(input_audio, requires_grad=True)
        input_length = input_audio.size(1)
        # print(input_audio.shape)
        for i in tqdm(range(input_length)):
            inL, inR = input_audio[0, i], input_audio[1, i]

            outL = 0
            outR = 0

            # for comb in self.combL:
            #     outL_2 = comb(inL) + outL

            # for comb in self.combR:
            #     outR_2 = comb(inR) + outR

            # for allpass in self.allpassL:
            #     outL_3 = allpass(outL_2)

            # for allpass in self.allpassR:
            #     outR_3 = allpass(outR_2)

            outL_2 = self.combL[0](inL) + outL
            outL_3 = self.combL[1](inL) + outL_2
            out_L4 = self.combL[2](inL) + outL_3
            outL_5 = self.combL[3](inL) + out_L4
            outL_6 = self.combL[4](inL) + outL_5
            outL_7 = self.combL[5](inL) + outL_6
            outL_8 = self.combL[6](inL) + outL_7
            outL_9 = self.combL[7](inL) + outL_8

            outR_2 = self.combR[0](inR) + outR
            outR_3 = self.combR[1](inR) + outR_2
            out_R4 = self.combR[2](inR) + outR_3
            outR_5 = self.combR[3](inR) + out_R4
            outR_6 = self.combR[4](inR) + outR_5
            outR_7 = self.combR[5](inR) + outR_6
            outR_8 = self.combR[6](inR) + outR_7
            outR_9 = self.combR[7](inR) + outR_8

            outL_10 = self.allpassL[0](outL_9)
            outL_11 = self.allpassL[1](outL_10)
            outL_12 = self.allpassL[2](outL_11)
            outL_13 = self.allpassL[3](outL_12)

            outR_10 = self.allpassR[0](outR_9)
            outR_11 = self.allpassR[1](outR_10)
            outR_12 = self.allpassR[2](outR_11)
            outR_13 = self.allpassR[3](outR_12)

            new_output_audio = output_audio.clone()
            new_output_audio[0, i] = torch.add(
                torch.add(outL_13 * self.wet1.clone(), outR_13 * self.wet2.clone()),
                inL * self.dry.clone(),
            )
            new_output_audio[1, i] = torch.add(
                torch.add(outR_13 * self.wet1.clone(), outL_13 * self.wet2.clone()),
                inR * self.dry.clone(),
            )
            output_audio = new_output_audio

        return output_audio


def test_model():
    model = NNFreeverbModule()
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
            0.2,
            1.4142,
            0,
        ]
    )
    # model.update_params(input_parameters)
    output_audio = model(input_parameters)
    output_audio = output_audio.detach().numpy().T
    output_audio = output_audio / np.max(np.abs(output_audio))

    sf.write("output.wav", output_audio, 44100)


# %%
