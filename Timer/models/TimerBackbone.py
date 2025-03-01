import torch
from torch import nn

from ..layers.Embed import PatchEmbedding
from ..layers.SelfAttention_Family import AttentionLayer, FullAttention
from ..layers.Transformer_EncDec import Encoder, EncoderLayer


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, configs):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.task_name = configs.task_name
        self.patch_len = configs.patch_len
        self.stride = configs.patch_len
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.layers = configs.e_layers
        self.n_heads = configs.n_heads
        self.dropout = configs.dropout
        padding = 0

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            self.d_model, self.patch_len, self.stride, padding, self.dropout)

        # Decoder
        self.decoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        # 这里改成true就是causal attention
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=True), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Prediction Head
        self.proj = nn.Linear(self.d_model, configs.patch_len, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        B, L, M = x_enc.shape

        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        dec_in, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        dec_out, attns = self.decoder(dec_in)
        # z: [bs x nvars x patch_num x d_model]
        dec_out = self.proj(dec_out)
        dec_out = dec_out.reshape(B, -1)
        dec_out = dec_out.unsqueeze(-1)
        # S = 1440 = 96 * 15

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * stdev + means
        return dec_out
