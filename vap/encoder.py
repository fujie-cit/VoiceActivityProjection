import torch
import torch.nn as nn
import einops
from vap.encoder_components import load_CPC, get_cnn_layer


class EncoderCPC(nn.Module):
    """
    Encoder: waveform -> h
    pretrained: default='cpc'

    A simpler version of the Encoder
    check paper (branch) version to see other encoders...
    """

    def __init__(self, load_pretrained=True, freeze=True):
        super().__init__()
        self.sample_rate = 16000
        self.encoder = load_CPC(load_pretrained)
        self.output_dim = self.encoder.gEncoder.conv4.out_channels
        self.dim = self.output_dim

        self.downsample_ratio = 160
        self.downsample = get_cnn_layer(
            dim=self.output_dim,
            kernel=[5],
            stride=[2],
            dilation=[1],
            activation="GELU",
        )
        self.downsample_ratio = 320

        if freeze:
            self.freeze()

    def get_default_conf(self):
        return {""}

    def freeze(self):
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        print(f"Froze {self.__class__.__name__}!")

    def unfreeze(self):
        for p in self.encoder.parameters():
            p.requires_grad_(True)
        print(f"Trainable {self.__class__.__name__}!")

    def forward(self, waveform, context=None):
        if waveform.ndim < 3:
            waveform = waveform.unsqueeze(1)  # channel dim

        # Backwards using only the encoder encounters:
        # ---------------------------------------------------
        # RuntimeError: one of the variables needed for gradient computation
        # has been modified by an inplace operation:
        # [torch.FloatTensor [4, 256, 1000]], which is output 0 of ReluBackward0, is at version 1;
        # expected version 0 instead. Hint: enable anomaly detection to find
        # the operation that failed to compute its gradient, with
        # torch.autograd.set_detect_anomaly(True).
        # HOWEVER, if we feed through encoder.gAR we do not encounter that problem...
        z = self.encoder.gEncoder(waveform)

        # streamingモードかどうかのチェック
        is_streaming = context is not None

        # streamingモードの場合は，真ん中の2つのフレームを抽出
        if is_streaming:
            length = z.shape[2]
            z = z[:, :, length//2-1:length//2+1]
                
        z = einops.rearrange(z, "b c n -> b n c")

        if is_streaming:
            z, context["ar_hidden"] = self.encoder.gAR(z, context.get("ar_hidden", None))
        else:
            z, _ = self.encoder.gAR(z)

        if is_streaming: 
            if "ar_output" not in context:
                b, n ,c = z.shape
                context["ar_output"] = torch.zeros(b, 0, c, device=z.device)

            new_ar_output = torch.cat((context["ar_output"], z), dim=1)
            new_ar_output = new_ar_output[:, -5:, :] # TODO: 5 is hardcoded
            context["ar_output"] = new_ar_output

            z = new_ar_output
            z = self.downsample(z)[:, -1:, :]
        else:
            z = self.downsample(z)

        return z, context
