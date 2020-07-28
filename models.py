import torch
import torch.nn as nn
from torch.nn import *
import torch.nn.functional as F
import math
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

def BertLayerNorm(shape, eps=1e-5, elementwise_affine=True):
    if torch.cuda.is_available():
        try:
            from apex.normalization import FusedLayerNorm

            return FusedLayerNorm(shape, eps=eps, elementwise_affine=elementwise_affine)
        except ImportError:
            pass
    return nn.LayerNorm(shape, eps=eps, elementwise_affine=elementwise_affine)

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
        assert cfg.hidden_attn % cfg.n_heads == 0, "hidden_attn must be divisible by n_heads"

        self.n_heads = cfg.n_heads
        self.efficient_attn = cfg.efficient_attn
        self.outer_attn = cfg.outer_attn

        self.proj_q = nn.Linear(cfg.hidden, cfg.hidden_attn, bias=cfg.bias)
        self.proj_k = nn.Linear(cfg.hidden, cfg.hidden_attn, bias=cfg.bias)
        self.proj_v = nn.Linear(cfg.hidden, cfg.hidden, bias=cfg.bias)

        self.dropout_q = nn.Dropout(cfg.dropout)
        self.dropout_k = nn.Dropout(cfg.dropout)
        self.dropout_v = nn.Dropout(cfg.dropout)
        self.drop_attn = nn.Dropout(cfg.dropout_attn)

    def forward(self, q, k, v, mask_q=None, mask_attn=None, mask_out=None, vanilla=False):

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

        if self.efficient_attn and not vanilla:

            if self.outer_attn == "scaled":
                if mask_attn is not None:
                    k = k * mask_attn[:, None, :, None].float() 
                if mask_q is not None:
                    q = q * mask_q[:, None, :, None].float()

                k = self.drop_attn(k)
                output = k.transpose(-1, -2) @ v
                output = self.drop_attn(q) @ output

                k = self.drop_attn(k)
                output = k.transpose(-1, -2) @ v
                output = self.drop_attn(q) @ output / mask_out.sum(-1, keepdim=True)[:, :, None, None]

            else:

                if mask_attn is not None:
                    k = k - 10000 * (1.0 - mask_attn[:, None, :, None].float())
                if mask_q is not None:
                    q = q - 10000 * (1.0 - mask_q[:, None, :, None].float())

                k = self.drop_attn(torch.softmax(k , dim=-2)) 
                output = k.transpose(-1, -2) @ v / math.sqrt(d//self.n_heads)
                output = self.drop_attn(torch.softmax(q, dim=-1)) @ output 

        else:

            if self.outer_attn == "scaled" and not vanilla:
                if mask_attn is not None:
                    k = k * mask_attn[:, None, :, None].float()
                if mask_q is not None:
                    q = q * mask_q[:, None, :, None].float()

                output = self.drop_attn(q @ k.transpose(-1, -2))
                output = output @ v / mask_out.sum(-1, keepdim=True)[:, :, None, None]

            else:
                
                if mask_q is not None:
                    q = q * mask_q[:, None, :, None].float()
                output = q @ k.transpose(-1, -2) / math.sqrt(d//self.n_heads)
                if mask_attn is not None:
                    output = output - 10000.0 * \
                        (1.0 - mask_attn[:, None, None, :].float())
                output = self.drop_attn(torch.softmax(output, dim=-1))
                output = (output @ v)

        n, h, t, d = output.size()

        output = output.transpose(1, 2)
        output = output.reshape(n, t, d*h)

        """
        if mask_out is not None:
            return output * mask_out.unsqueeze(-1)
        """
        return output

        
class ConvProj(nn.Module):

    def __init__(self, input_dim, hidden, n_blocks=4, block_size=4):
        super().__init__()
        assert n_blocks >= 2, "n_blocks must be >= 2"

        self.conv_1 = nn.Conv1d(input_dim, hidden,
                                kernel_size=block_size, stride=block_size//2)
        #self.conv_2 = nn.MaxPool1d(kernel_size=n_blocks, stride=1)
        self.conv_2 = nn.AvgPool1d(kernel_size=n_blocks, stride=1)

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
        self.block_size = cfg.block_size

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
            x = x.reshape(n, t//self.block_size, -1)

        if self.projection == "block":

            x = x.reshape(-1, self.block_size, d)
            new_mask = mask.reshape(-1, self.block_size)
            x = self.proj(compress_time(
                x, new_mask, keepdim=True), x, x, new_mask, mask, new_mask)
            x = x.reshape(n, -1, d)

        if mask is not None:

            mask = mask.reshape(n, -1, self.block_size).sum(-1)
            mask = mask.clamp(0., 1.)
            #return x*mask.unsqueeze(-1), mask
            return x, mask
        
        return x, mask

    def pack_sequence(self, x, mask=None):
        n, t, d = x.size()

        if mask is not None:
            mask = mask.reshape(-1, self.block_size).sum(-1)
            mask = mask.clamp(1., t)
        else:
            mask = [self.block_size for _ in range(n*t//self.block_size)]

        x = x.reshape(-1, self.block_size, d)
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

        if self.cfg.attn_skip_connection:
            self.norm = BertLayerNorm(cfg.hidden)
            self.hidden = nn.Linear(cfg.hidden, cfg.hidden)
            
        else:
            self.proj_k = Projection(cfg)
            self.proj_v = Projection(cfg)

        self.output_attn = Attention(cfg)

    def forward(self, x, mask=None):

        if self.cfg.inner_attn:
            x_proj, proj_mask = self.proj(x, mask)
            x_proj_out = self.proj_attn(
                x_proj, x_proj, x_proj, proj_mask, proj_mask, proj_mask, vanilla=True)

            if self.cfg.attn_skip_connection:
                x_proj_out = self.norm(x_proj + self.hidden(x_proj_out))
            h = self.output_attn(x, x_proj_out, x_proj_out, mask, proj_mask, mask)

        else:
            x_proj_k, proj_mask = self.proj_k(x, mask)
            x_proj_v, proj_mask = self.proj_v(x, mask)
            h = self.output_attn(x, x_proj_k, x_proj_v, mask, proj_mask, mask)
            
        return h

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

        self.norm_1 = BertLayerNorm(cfg.hidden)
        self.norm_2 = BertLayerNorm(cfg.hidden)

    def forward(self, x, mask):
        h = self.attn(x, mask)
        h = self.norm_1(x + self.proj(h))
        h = self.norm_2(h + self.pwff(h))

        return h

class Embeddings(nn.Module):
    "The embedding module from word, position and token_type embeddings."

    def __init__(self, cfg):
        super().__init__()

        self.norm = BertLayerNorm(cfg.hidden)
        self.dropout = nn.Dropout(cfg.dropout)

        self.cfg = cfg

        self.positional_embedding = cfg.positional_embedding
        self.block_positional_embedding = cfg.block_positional_embedding
        self.inner_block_positional_embedding = cfg.inner_block_positional_embedding
        self.embedding_factorization = cfg.embedding_factorization

        if cfg.embedding_factorization > 0:
            self.embedding = nn.Embedding(cfg.vocab_size, cfg.embedding_factorization, padding_idx=cfg.padding_idx)
            self.factorization = nn.Linear(cfg.embedding_factorization, cfg.hidden)
        else:
            self.embedding = nn.Embedding(cfg.vocab_size, cfg.hidden, padding_idx=cfg.padding_idx)

        if cfg.positional_embedding:
            self.pos_embedding = nn.Embedding(cfg.max_len + 1, cfg.hidden, padding_idx=cfg.padding_idx)

        if cfg.block_positional_embedding:
            self.block_pos_embedding = nn.Embedding(cfg.n_blocks + 1, cfg.hidden, padding_idx=cfg.padding_idx)

        if cfg.inner_block_positional_embedding:
            self.inner_block_pos_embedding = nn.Embedding(cfg.block_size + 1, cfg.hidden, padding_idx=cfg.padding_idx)

    def forward(self, x, mask=None):

        n, t = x.size()
        
        x = self.embedding(x)

        if self.embedding_factorization > 0:
            x = self.factorization(x) 

        if self.positional_embedding:
            pos = torch.arange(t, device=x.device) + 1
            pos = pos.unsqueeze(0).expand_as(x[:,:,0]) 
            pos = self.pos_embedding(pos)
            x = x + pos

        if self.block_positional_embedding:
            pos = torch.cat([torch.ones(self.cfg.block_size) * i for i in range(self.cfg.n_blocks)]).to(x.device).long() + 1
            pos = pos.unsqueeze(0).expand_as(x[:,:,0]) 
            pos = self.block_pos_embedding(pos)
            x = x + pos

        if self.inner_block_positional_embedding:
            pos = torch.cat([torch.arange(self.cfg.block_size) for i in range(self.cfg.n_blocks)]).to(x.device).long() + 1
            pos = pos.unsqueeze(0).expand_as(x[:,:,0]) 
            pos = self.inner_block_pos_embedding(pos)
            x = x + pos

        return self.dropout(self.norm(x))

class BertInnerModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.embedding = Embeddings(cfg)

        if cfg.share_all:
            transformer = TransformerLayer(cfg)
            self.transformers = nn.ModuleList(
                [transformer for _ in range(cfg.n_layers)])
        else:
            attn, pwff = None, None

            if cfg.share_attn:
                attn = LinearAttention(cfg)
            if cfg.share_pwff:
                pwff = PointWiseFeedForward(cfg)

            self.transformers = nn.ModuleList(
                [TransformerLayer(cfg, attn, pwff) for _ in range(cfg.n_layers)])

        self.pooler = nn.Linear(cfg.hidden, cfg.hidden)
        self.pooler_activation = nn.Tanh()

    def forward(self, x, mask=None):
        x = self.embedding(x, mask)

        for transformer in self.transformers:
            x = transformer(x, mask)

        x = self.pooler(x)
        return self.pooler_activation(x)

class BertInnerLMHead(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()

        self.embedding_factorization = cfg.embedding_factorization

        self.dense = nn.Linear(cfg.hidden, cfg.hidden)
        self.layer_norm = BertLayerNorm(cfg.hidden)

        if cfg.embedding_factorization > 0:
            self.decoder_factorization = nn.Linear(cfg.hidden, cfg.embedding_factorization)
            self.decoder = nn.Linear(cfg.embedding_factorization, cfg.vocab_size)
        else:
            self.decoder = nn.Linear(cfg.hidden, cfg.vocab_size)

        self.bias = nn.Parameter(torch.zeros(cfg.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, x):
        x = self.dense(x)
        x = F.gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        if self.embedding_factorization > 0:
            x = self.decoder_factorization(x)

        x = self.decoder(x)

        return x

class BertInnerClassificationHead(nn.Module):

    def __init__(self, cfg, n_labels):
        super().__init__()
        self.dense = nn.Linear(cfg.hidden, cfg.hidden)
        self.dropout = nn.Dropout(cfg.dropout)
        self.out_proj = nn.Linear(cfg.hidden, n_labels)

    def forward(self, x):
        x = x[:, 0, :] 
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class BertInnerForMaskedLM(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.encoder = BertInnerModel(cfg)
        self.lm = BertInnerLMHead(cfg)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, mask, labels, labels_mask):
        x = self.encoder(x, mask) 
        x, labels = self.select_masked_words(x, labels, labels_mask)
        x = self.lm(x)
        return self.lm_loss(x, labels), x, labels

    def lm_loss(self, outputs, labels):
        return self.criterion(outputs, labels)

    def select_masked_words(self, x, labels, labels_mask):
        n, t = labels.size()
        labels_mask = labels_mask.reshape(n*t).bool() == True
        x = x.reshape(n*t, -1)[labels_mask]
        labels = labels.reshape(n*t)[labels_mask]
        return x, labels

class BertInnerForSequenceClassification(nn.Module):

    def __init__(self, cfg, n_labels, criterion):
        super().__init__()
        
        self.encoder = BertInnerModel(cfg)
        self.classifier = BertInnerClassificationHead(cfg, n_labels)
        self.criterion = criterion

    def forward(self, x, mask, labels):
        x = self.encoder(x, mask)
        x = self.classifier(x)
        return self.compute_loss(x, labels), x, labels

    def compute_loss(self, outputs, labels):
        return self.criterion(outputs, labels)

