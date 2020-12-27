import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        x = x + self.pe.narrow(0, 0, x.size(0))
        return self.dropout(x)

class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, input_vocab_size, target_vocab_size, d_model=512, nhead=8, dim_feedforward=2048, num_encoder_layers=6, num_decoder_layers=6, dropout=0.1, activation="relu"):
        super(TransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoderLayer, TransformerDecoderLayer
        except:
            raise ImportError(
                'TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'

        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = TransformerEncoder(
            encoder_layers, num_encoder_layers, d_model, input_vocab_size, dropout, encoder_norm)

        decoder_layers = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layers, num_decoder_layers, d_model, target_vocab_size, dropout, decoder_norm)

        self.linear = nn.Linear(d_model, target_vocab_size)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    # def init_weights(self):
    #     initrange = 0.1
    #     nn.init.uniform_(self.encoder.weight, -initrange, initrange)
    #     nn.init.zeros_(self.decoder.weight)
    #     nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    # Order of arguments is important for onnx export, in python/pth_to_onnx.py
    def forward(self, src, tgt, tgt_mask=None, src_key_padding_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None, src_mask=None, memory_mask=None):
        # if has_mask:
        #     device = src.device
        #     if self.src_mask is None or self.src_mask.size(0) != len(src):
        #         mask = self._generate_square_subsequent_mask(
        #             len(src)).to(device)
        #         self.src_mask = mask
        # else:
        #     self.src_mask = None
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask= src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask,
                               memory_key_padding_mask=memory_key_padding_mask)
        output = self.linear(output)
        return F.log_softmax(output, dim=-1)


class TransformerEncoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, d_model, vocab_size, dropout, norm=None):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        self.norm = norm

    def forward(self, src, mask = None, src_key_padding_mask = None):
        output = self.embedding(src) * math.sqrt(self.d_model)
        output = self.pos_encoder(output)

        for mod in self.layers:
            output = mod(output, src_mask=mask,
                         src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerDecoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, d_model, vocab_size, dropout, norm=None):
        super(TransformerDecoder, self).__init__()
        self.d_model = d_model
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = self.embedding(tgt) * math.sqrt(self.d_model)
        output = self.pos_encoder(output)

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


def _get_clones(module, N):
  return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
