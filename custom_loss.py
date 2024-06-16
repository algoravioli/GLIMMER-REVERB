# CUSTOM LOSS FUNCTION
import torch
import auraloss


class CustomLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.stft_loss = auraloss.freq.STFTLoss()
        self.esr_loss = auraloss.time.ESRLoss()
        self.dc_loss = auraloss.time.DCLoss()
        # self.sum_diff = auraloss.freq.SumAndDifferenceSTFTLoss(
        #     fft_sizes=[1024, 2048, 8192],
        #     hop_sizes=[256, 512, 2048],
        #     win_lengths=[1024, 2048, 8192],
        #     perceptual_weighting=True,
        #     sample_rate=44100,
        #     scale="mel",
        #     n_bins=128,
        # )

    def forward(self, y_hat, y):
        mse_loss = torch.nn.functional.mse_loss(y_hat, y)
        stft_loss = self.stft_loss(y_hat, y)
        L1_loss = torch.nn.functional.l1_loss(y_hat, y)
        esr_loss = self.esr_loss(y_hat, y)
        dc_loss = self.dc_loss(y_hat, y)
        # sum_diff_loss = self.sum_diff(y_hat, y)

        print(
            f"MSE Loss: {mse_loss} "
            + f"STFT Loss: {stft_loss} "
            + f"L1 Loss: {L1_loss} "
            + f"ESR Loss: {esr_loss} "
            + f"DC Loss: {dc_loss} "
            # + f"SumDiff Loss: {sum_diff_loss}"
        )
        return mse_loss + stft_loss + L1_loss + esr_loss + dc_loss  # + sum_diff_loss
