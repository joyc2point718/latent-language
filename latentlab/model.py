from dataclasses import dataclass, asdict
import math
import torch
from torch import nn

PAD, BOS, EOS, UNK, MASK = range(5)

@dataclass
class Config:
    vocab_size: int
    width: int = 256
    heads: int = 4
    layers: int = 4
    slots: int = 8
    max_len: int = 64
    dropout: float = 0.1

class Autoencoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config
        self.embedding = nn.Embedding(c.vocab_size, c.width, padding_idx=PAD)
        self.position = nn.Embedding(c.max_len, c.width)
        enc = nn.TransformerEncoderLayer(c.width, c.heads, 4*c.width,
                                        c.dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, c.layers, nn.LayerNorm(c.width),
                                            enable_nested_tensor=False)
        self.queries = nn.Parameter(torch.randn(c.slots, c.width) / math.sqrt(c.width))
        self.pool = nn.MultiheadAttention(c.width, c.heads, c.dropout, batch_first=True)
        self.latent_norm = nn.LayerNorm(c.width)
        dec = nn.TransformerDecoderLayer(c.width, c.heads, 4*c.width,
                                        c.dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec, c.layers, nn.LayerNorm(c.width))
        self.output = nn.Linear(c.width, c.vocab_size, bias=False)
        self.output.weight = self.embedding.weight

    def embed(self, ids):
        positions = torch.arange(ids.shape[1], device=ids.device)
        return self.embedding(ids) * math.sqrt(self.config.width) + self.position(positions)

    def encode(self, ids):
        padding = ids.eq(PAD)
        h = self.encoder(self.embed(ids), src_key_padding_mask=padding)
        q = self.queries.unsqueeze(0).expand(ids.shape[0], -1, -1)
        z, _ = self.pool(q, h, h, key_padding_mask=padding, need_weights=False)
        return self.latent_norm(q + z)

    def decode(self, prefix, z):
        n = prefix.shape[1]
        causal = torch.ones(n, n, device=prefix.device, dtype=torch.bool).triu(1)
        h = self.decoder(self.embed(prefix), z, tgt_mask=causal,
                         tgt_key_padding_mask=prefix.eq(PAD))
        return self.output(h)

    def forward(self, source, prefix):
        return self.decode(prefix, self.encode(source))

    @torch.no_grad()
    def generate(self, z):
        ids = torch.full((z.shape[0], 1), BOS, dtype=torch.long, device=z.device)
        finished = torch.zeros(z.shape[0], dtype=torch.bool, device=z.device)
        for _ in range(self.config.max_len - 1):
            logits = self.decode(ids, z)[:, -1].clone()
            logits[:, [PAD, BOS, MASK, UNK]] = -torch.inf
            token = logits.argmax(-1)
            token = torch.where(finished, torch.full_like(token, PAD), token)
            ids = torch.cat([ids, token[:, None]], dim=1)
            finished |= token.eq(EOS)
            if finished.all():
                break
        return ids

    def encoder_state(self):
        prefixes = ('embedding.', 'position.', 'encoder.', 'queries', 'pool.', 'latent_norm.')
        return {k: v for k, v in self.state_dict().items() if k.startswith(prefixes)}

    def decoder_state(self):
        return {k: v for k, v in self.state_dict().items()
                if k.startswith(('embedding.', 'position.', 'decoder.', 'output.'))}
