# %%
import soundfile as sf
import numpy as np
import os
import time


def roll_zero(x, k):
    result = x[k:]
    result = np.append(x[k:], np.zeros(k))
    return result


def zero_pad(x, k):
    return np.append(x, np.zeros(k))


def precompute_frequency_responses(h, L, k, num_blocks):
    H = np.zeros((num_blocks, L + k)).astype("complex128")
    for j in range(num_blocks):
        H[j] += np.fft.fft(zero_pad(h[j * k : (j + 1) * k], L))
    return H


def fft_conv(x, h):
    L, P = len(x), len(h)
    h_zp = zero_pad(h, L - 1)
    x_zp = zero_pad(x, P - 1)
    X = np.fft.fft(x_zp)
    start_time = time.time()
    output = np.fft.ifft(X * np.fft.fft(h_zp)).real
    output = output + x_zp
    end_time = time.time()
    return output, end_time - start_time


def convolution_reverb(input, impulse_response):
    if input.ndim == 1:
        input = input[:, np.newaxis]
    if impulse_response.ndim == 1:
        impulse_response = impulse_response[:, np.newaxis]

    outputs = []
    for channel in range(input.shape[1]):
        output, _ = fft_conv(input[:, channel], impulse_response[:, 0])
        outputs.append(
            output[: len(input)]
        )  # Truncate the output to match the length of the input

    return np.stack(outputs, axis=1)


def create_stereo_dirac_delta_signal(
    duration, fs, name, float_start=0.0001, folder_path="data/input"
):
    # float start represents the start of the diract delta as a ratio of the total duration
    # float start should be between 0 and 1
    # duration is in seconds
    # fs is the sampling rate
    # returns a stereo signal of a dirac delta. i.e [0, 0, 0, 1, 0, 0 .....] for both channels

    # create the dirac delta signal
    signal = np.zeros((2, int(duration * fs)))
    signal[0, int(float_start * fs)] = 1
    signal[1, int(float_start * fs)] = 1

    # export the signal to a wav file with name
    sf.write(f"{folder_path}/input_{name}.wav", signal.T, fs)


def create_output_signal(
    input_signal, impulse_response, name, folder_path="data/output"
):
    # input_signal is the input signal
    # impulse_response is the impulse response
    # name is the name of the output file
    # type is the type of convolution to use

    # Load the input signal
    input_signal, fs1 = sf.read(input_signal)

    # Load the impulse response
    impulse_response, fs2 = sf.read(impulse_response)

    # Convolve the input signal with the impulse response
    output_signal = convolution_reverb(input_signal, impulse_response)
    output_signal = output_signal / np.max(np.abs(output_signal))

    # Export the output signal to a wav file with name
    sf.write(f"{folder_path}/output_{name}.wav", output_signal, fs1)


# files in directory of IRs
impulse_responses = os.listdir("rirs")
remove = [".DS_Store", "._.DS_Store"]
for file in remove:
    if file in impulse_responses:
        impulse_responses.remove(file)

# find longest duration file in impulse_responses
max_duration = 0
for file in impulse_responses:
    impulse_response, fs = sf.read(f"rirs/{file}")
    if len(impulse_response) > max_duration:
        max_duration = len(impulse_response)
print(str(max_duration) + " is the longest duration file in impulse_responses")
# for each file, remove .wav
for i in range(len(impulse_responses)):
    impulse_responses[i] = impulse_responses[i].replace(".wav", "")

# longest duration in seconds
max_duration = max_duration / fs
print(
    str(max_duration)
    + " is the longest duration file in impulse_responses in seconds, with FS = "
    + str(fs)
)

# # create signal for each file
# run only if "data/input" is empty

if len(os.listdir("data/input")) == 0:
    for i in range(len(impulse_responses)):
        create_stereo_dirac_delta_signal(max_duration, fs, impulse_responses[i])

# create output signal for each file
for i in range(len(impulse_responses)):
    create_output_signal(
        f"data/input/input_{impulse_responses[i]}.wav",
        f"rirs/{impulse_responses[i]}.wav",
        impulse_responses[i],
    )
# %%
