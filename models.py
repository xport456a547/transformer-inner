import torch
import torch.nn as nn
from torch.nn import *
import torch.nn.functional as F
import math
import warnings
warnings.filterwarnings("ignore")

def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def compress_time(x, mask=None, keepdim=False):
    if mask is not None:
        x = x*mask.unsqueeze(-1)
        mask = mask.sum(-1).unsqueeze(-1) + 10e-8
        x = x.sum(1).squeeze(1) / mask
    else:
        x = x.mean(1)

    if keepdim:
        x = x.unsqueeze(1)
    return x

def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    #index = x.topk(k, dim=dim, sorted=False)[1]
    return x.gather(dim, index)

class CustomLayerNorm(nn.Module):

    def __init__(self, cfg, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(cfg.hidden))
        self.beta = nn.Parameter(torch.zeros(cfg.hidden))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class PointWiseFeedForward(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.hidden, cfg.hidden_ff)
        self.fc2 = nn.Linear(cfg.hidden_ff, cfg.hidden)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

class Attention(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        assert cfg.hidden % cfg.n_heads == 0, "hidden must be divisible by n_heads"

        self.n_heads = cfg.n_heads
        self.efficient_attn = cfg.efficient_attn
        self.scaled_attn = cfg.scaled_attn

        self.proj_q = nn.Linear(cfg.hidden, cfg.hidden, bias=cfg.bias)
        self.proj_k = nn.Linear(cfg.hidden, cfg.hidden, bias=cfg.bias)
        self.proj_v = nn.Linear(cfg.hidden, cfg.hidden, bias=cfg.bias)

        self.dropout_q = nn.Dropout(cfg.dropout)
        self.dropout_k = nn.Dropout(cfg.dropout)
        self.dropout_v = nn.Dropout(cfg.dropout)

    def forward(self, q, k, v, mask_attn=None, mask_out=None):
        q = self.dropout_q(self.proj_q(q))
        k = self.dropout_k(self.proj_k(k))
        v = self.dropout_v(self.proj_v(v))

        # (n, t, d) -> (n, t, heads, d//heads) -> (n, heads, t, d//heads)
        n, t, d = q.size()
        q = q.reshape(n, -1, self.n_heads, d//self.n_heads).transpose(1, 2)

        n, t, d = k.size()
        k = k.reshape(n, -1, self.n_heads, d//self.n_heads).transpose(1, 2)

        n, t, d = v.size()
        v = v.reshape(n, -1, self.n_heads, d//self.n_heads).transpose(1, 2)

        if not self.efficient_attn:
            if not self.scaled_attn:
                output = q @ k.transpose(-1, -2) / math.sqrt(d//self.n_heads)
                if mask_attn is not None:
                    output = output - 10000.0 * \
                        (1.0 - mask_attn[:, None, None, :].float())
                output = torch.softmax(output, dim=-1)
                output = (output @ v)
            else:
                output = (q @ k.transpose(-1, -2)) / t
                if mask_attn is not None:
                    output = output * mask_attn[:, None, None, :].float()
                output = (output @ v)
        else:
            if not self.scaled_attn:
                k = torch.softmax(k, dim=-2)
                if mask_attn is not None:
                    k = k * mask_attn[:, None, :, None].float()
                output = k.transpose(-1, -2) @ v
                output = torch.softmax(q, dim=-1) @ output
            else:
                k = k / t
                if mask_attn is not None:
                    k = k * mask_attn[:, None, :, None].float()
                output = k.transpose(-1, -2) @ v
                output = q @ output

        n, h, t, d = output.size()

        output = output.transpose(1, 2)
        output = output.reshape(n, t, d*h)

        if mask_out is not None:
            return output * mask_out.unsqueeze(-1)

        return output


class ConvProj(nn.Module):

    def __init__(self, input_dim, hidden, n_blocks=4, block_size=4):
        super().__init__()
        assert n_blocks >= 2, "n_blocks must be >= 2"

        self.conv_1 = nn.Conv1d(input_dim, hidden,
                                kernel_size=block_size, stride=block_size//2)
        self.conv_2 = nn.MaxPool1d(kernel_size=n_blocks, stride=1)

    def forward(self, x):

        size = len(x.size())
        if size > 3:
            n, h, t, d = x.size()
            x = x.view(n*h, t, d)
        else:
            n, t, d = x.size()

        x = x.transpose(-1, -2)
        x = self.conv_1(x)
        x = self.conv_2(x)

        if size > 3:
            return x.transpose(-1, -2).view(n, h, -1, d)
        return x.transpose(-1, -2)

class Projection(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        assert cfg.projection in ["lstm", "gru", "mean",
                              "max", "cnn", "dense", "block", "topmax"]

        self.projection = cfg.projection
        self.n_blocks = cfg.n_blocks

        if cfg.projection == "lstm":
            self.proj = nn.LSTM(cfg.hidden, cfg.hidden//2,
                                bidirectional=True, batch_first=True)
        elif cfg.projection == "gru":
            self.proj = nn.GRU(cfg.hidden, cfg.hidden//2,
                               bidirectional=True, batch_first=True)
        elif cfg.projection == "mean":
            self.proj = nn.AvgPool1d(cfg.block_size, cfg.block_size)
        elif cfg.projection == "max":
            self.proj = nn.MaxPool1d(cfg.block_size, cfg.block_size)
        elif cfg.projection == "cnn":
            self.proj = ConvProj(cfg.hidden, cfg.hidden, cfg.n_blocks, cfg.block_size)
        elif cfg.projection == "dense":
            self.proj = nn.Linear(cfg.n_blocks*cfg.block_size, cfg.n_blocks, bias=cfg.bias)
        elif cfg.projection == "block":
            self.proj = Attention(cfg)

        self.linear = nn.Linear(cfg.hidden, cfg.hidden)

    def forward(self, x, mask=None):
        n, t, d = x.size()
        x = self.linear(x)

        if mask is not None:
            x = x*mask.unsqueeze(-1)

        if self.projection in ["mean", "max", "dense"]:
            x = self.proj(x.transpose(-1, -2)).transpose(-1, -2)

        if self.projection == "topmax":
            x = kmax_pooling(x, dim=-2, k=self.n_blocks)

        if self.projection == "cnn":
            x = self.proj(x)

        if self.projection in ["gru", "lstm"]:

            x = self.pack_sequence(x, mask)
            if self.projection == "lstm":
                _, (x, _) = self.proj(x)
            else:
                _, x = self.proj(x)

            x = x.transpose(0, 1).reshape(x.shape[0], -1)
            x = x.reshape(n, self.n_blocks, -1)

        if self.projection == "block":
            x = x.reshape(n*self.n_blocks, t//self.n_blocks, d)
            new_mask = mask.reshape(n*self.n_blocks, t//self.n_blocks)
            x = self.proj(compress_time(
                x, new_mask, keepdim=True), x, x, mask, new_mask)
            x = x.reshape(n, self.n_blocks, d)

        if mask is not None:
            mask = mask.reshape(n, self.n_blocks, t//self.n_blocks).sum(-1)
            mask = mask.clamp(0., 1.)
            return x*mask.unsqueeze(-1), mask

        return x, mask

    def pack_sequence(self, x, mask=None):
        n, t, d = x.size()

        if mask is not None:
            mask = mask.reshape(n*self.n_blocks, t//self.n_blocks).mean(-1)
            mask = mask.clamp(1., t)
        else:
            mask = [t//self.n_blocks for _ in range(n*self.n_blocks)]

        x = x.reshape(n*self.n_blocks, t//self.n_blocks, d)
        x = torch.nn.utils.rnn.pack_padded_sequence(
            x, mask, batch_first=True, enforce_sorted=False)

        return x

class LinearAttention(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        if cfg.inner_attn:
            self.proj = Projection(cfg)
            self.proj_attn = Attention(cfg)
        else:
            self.proj_k = Projection(cfg)
            self.proj_v = Projection(cfg)

        self.output_attn = Attention(cfg)

    def forward(self, x, mask=None):
        if self.cfg.inner_attn:
            x_proj, proj_mask = self.proj(x, mask)
            x_proj = self.proj_attn(
                x_proj, x_proj, x_proj, proj_mask, proj_mask)
            x = self.output_attn(x, x_proj, x_proj, proj_mask, mask)
        else:
            x_proj_k, proj_mask = self.proj_k(x, mask)
            x_proj_v, proj_mask = self.proj_v(x, mask)
            x = self.output_attn(x, x_proj_k, x_proj_v, proj_mask, mask)

        return x

class TransformerLayer(nn.Module):

    def __init__(self, cfg, attn=None, pwff=None):
        super().__init__()

        if attn is not None:
            self.attn = attn
        else:
            self.attn = LinearAttention(cfg)

        if pwff is not None:
            self.pwff = pwff
        else:
            self.pwff = PointWiseFeedForward(cfg)

        self.proj = nn.Linear(cfg.hidden, cfg.hidden)

        self.norm_1 = CustomLayerNorm(cfg)
        self.norm_2 = CustomLayerNorm(cfg)

    def forward(self, x, mask):
        x = self.attn(x, mask)
        x = self.norm_1(x + self.proj(x))
        x = self.norm_2(x + self.pwff(x))

        return x

class Embeddings(nn.Module):
    "The embedding module from word, position and token_type embeddings."

    def __init__(self, cfg):
        super().__init__()

        self.embedding = nn.Embedding(cfg.vocab_size, cfg.embedding)
        self.proj = nn.Linear(cfg.embedding, cfg.hidden)
        self.norm = CustomLayerNorm(cfg)

    def forward(self, x):

        x = self.embedding(x)
        x = self.proj(x)
        return self.norm(x)

class Transformers(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.attn = LinearAttention(cfg)

        if cfg.share_attn and not cfg.share_all:
            attn = LinearAttention(cfg)
        else:
            attn = None

        if cfg.share_pwff and not cfg.share_all:
            pwff = PointWiseFeedForward(cfg)
        else:
            pwff = None

        if cfg.share_all:
            transformer = Transformer(cfg)
            self.transformers = nn.ModuleList(
                [transformer for _ in range(cfg.n_layers)])
        else:
            self.transformers = nn.ModuleList(
                [TransformerLayer(cfg, attn, pwff) for _ in range(cfg.n_layers)])

    def forward(self, x, mask):
        for transformer in self.transformers:
            x = transformer(x, mask)
        return x

class BertInner(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.embedding = Embeddings(cfg)
        self.transformers = Transformers(cfg)

        # decoder is shared with embedding layer
        # project hidden layer to embedding layer
        embed_weight2 = self.embedding.proj.weight
        n_hidden, n_embedding = embed_weight2.size()
        self.decoder1 = nn.Linear(n_hidden, n_embedding, bias=False)
        self.decoder1.weight.data = embed_weight2.data.t()
        self.decoder1_bias = nn.Parameter(torch.zeros(n_embedding))

        # project embedding layer to vocabulary layer
        embed_weight1 = self.embedding.embedding.weight
        n_vocab, n_embedding = embed_weight1.size()
        self.decoder2 = nn.Linear(n_embedding, n_vocab, bias=False)
        self.decoder2.weight = embed_weight1

        self.decoder2_bias = nn.Parameter(torch.zeros(n_vocab))

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, mask, labels, labels_mask):
        x = self.embedding(x)
        x = self.transformers(x, mask)

        n, t = labels.size()
        labels_mask = labels_mask.reshape(n*t).bool() == True
        x = x.reshape(n*t, -1)[labels_mask]
        labels = labels.reshape(n*t)[labels_mask]

        x = self.decoder1(x) + self.decoder1_bias
        x = self.decoder2(x) + self.decoder2_bias

        return self.lm_loss(x, labels)

    def lm_loss(self, outputs, labels):
        return self.criterion(outputs, labels)