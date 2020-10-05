# ported from https://github.com/pytorch/pytorch/blob/626e410e1dedcdb9d5a410a8827cc7a8a9fbcce1/torch/nn/modules/transformer.py

module Secryst
  class Transformer < Torch::NN::Module
    # A transformer model. User is able to modify the attributes as needed. The architecture
    # is based on the paper "Attention Is All You Need". Ashish Vaswani, Noam Shazeer,
    # Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and
    # Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information
    # Processing Systems, pages 6000-6010. Users can build the BERT(https://arxiv.org/abs/1810.04805)
    # model with corresponding parameters.
    # Args:
    #     d_model: the number of expected features in the encoder/decoder inputs (default=512).
    #     nhead: the number of heads in the multiheadattention models (default=8).
    #     num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
    #     num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
    #     dim_feedforward: the dimension of the feedforward network model (default=2048).
    #     dropout: the dropout value (default=0.1).
    #     activation: the activation function of encoder/decoder intermediate layer, relu or gelu (default=relu).
    #     custom_encoder: custom encoder (default=nil).
    #     custom_decoder: custom decoder (default=nil).
    #     input_vocab_size: size of vocabulary for input sequence (number of different possible tokens).
    #     target_vocab_size: size of vocabulary for target sequence (number of different possible tokens).
    # Examples::
    #     >>> transformer_model = Transformer.new(nhead: 16, num_encoder_layers: 12)
    #     >>> src = Torch.rand((10, 32, 512))
    #     >>> tgt = Torch.rand((20, 32, 512))
    #     >>> out = transformer_model.call(src, tgt)
    def initialize(d_model: 512, nhead: 8, num_encoder_layers: 6, num_decoder_layers: 6,
      dim_feedforward: 2048, dropout: 0.1, activation: 'relu', custom_encoder: nil, custom_decoder: nil, input_vocab_size:, target_vocab_size:)

      super()

      if custom_encoder
        @encoder = custom_encoder
      else
        encoder_layers = num_encoder_layers.times.map { TransformerEncoderLayer.new(d_model, nhead, dim_feedforward: dim_feedforward, dropout: dropout, activation: activation) }
        encoder_norm = Torch::NN::LayerNorm.new(d_model)
        @encoder = TransformerEncoder.new(encoder_layers, encoder_norm, d_model, input_vocab_size, dropout)
      end

      if custom_decoder
        @decoder = custom_decoder
      else
        decoder_layers = num_decoder_layers.times.map { TransformerDecoderLayer.new(d_model, nhead, dim_feedforward: dim_feedforward, dropout: dropout, activation: activation) }
        decoder_norm = Torch::NN::LayerNorm.new(d_model)
        @decoder = TransformerDecoder.new(decoder_layers, decoder_norm, d_model, target_vocab_size, dropout)
      end

      @linear = Torch::NN::Linear.new(d_model, target_vocab_size)
      @softmax = Torch::NN::LogSoftmax.new(dim: -1)
      _reset_parameters()

      @d_model = d_model
      @nhead = nhead

    end

    # Take in and process masked source/target sequences.
    # Args:
    #     src: the sequence to the encoder (required).
    #     tgt: the sequence to the decoder (required).
    #     src_mask: the additive mask for the src sequence (optional).
    #     tgt_mask: the additive mask for the tgt sequence (optional).
    #     memory_mask: the additive mask for the encoder output (optional).
    #     src_key_padding_mask: the ByteTensor mask for src keys per batch (optional).
    #     tgt_key_padding_mask: the ByteTensor mask for tgt keys per batch (optional).
    #     memory_key_padding_mask: the ByteTensor mask for memory keys per batch (optional).
    # Shape:
    #     - src: :math:`(S, N, E)`.
    #     - tgt: :math:`(T, N, E)`.
    #     - src_mask: :math:`(S, S)`.
    #     - tgt_mask: :math:`(T, T)`.
    #     - memory_mask: :math:`(T, S)`.
    #     - src_key_padding_mask: :math:`(N, S)`.
    #     - tgt_key_padding_mask: :math:`(N, T)`.
    #     - memory_key_padding_mask: :math:`(N, S)`.
    #     Note: [src/tgt/memory]_mask ensures that position i is allowed to attend the unmasked
    #     positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
    #     while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``true``
    #     are not allowed to attend while ``false`` values will be unchanged. If a FloatTensor
    #     is provided, it will be added to the attention weight.
    #     [src/tgt/memory]_key_padding_mask provides specified elements in the key to be ignored by
    #     the attention. If a ByteTensor is provided, the non-zero positions will be ignored while the zero
    #     positions will be unchanged. If a BoolTensor is provided, the positions with the
    #     value of ``true`` will be ignored while the position with the value of ``false`` will be unchanged.
    #     - output: :math:`(T, N, E)`.
    #     Note: Due to the multi-head attention architecture in the transformer model,
    #     the output sequence length of a transformer is same as the input sequence
    #     (i.e. target) length of the decode.
    #     where S is the source sequence length, T is the target sequence length, N is the
    #     batch size, E is the feature number
    # Examples:
    #     >>> output = transformer_model.call(src, tgt, src_mask: src_mask, tgt_mask: tgt_mask)
    def forward(src, tgt, src_mask: nil, tgt_mask: nil,
              memory_mask: nil, src_key_padding_mask: nil,
              tgt_key_padding_mask: nil, memory_key_padding_mask: nil)
      if src.size(1) != tgt.size(1)
        raise RuntimeError, "the batch number of src and tgt must be equal"
      end

      memory = @encoder.call(src, mask: src_mask, src_key_padding_mask: src_key_padding_mask)
      output = @decoder.call(tgt, memory, tgt_mask: tgt_mask, memory_mask: memory_mask,
                            tgt_key_padding_mask: tgt_key_padding_mask,
                            memory_key_padding_mask: memory_key_padding_mask)
      output = @linear.call(output)
      output = @softmax.call(output)

      return output
    end

    def _reset_parameters
      parameters.each do |p|
        Torch::NN::Init.xavier_uniform!(p) if p.dim > 1
      end
    end
  end


  class TransformerEncoder < Torch::NN::Module
    # TransformerEncoder is a stack of N encoder layers
    # Args:
    #     encoder_layers: an array of instances of the TransformerEncoderLayer class (required).
    #     norm: the layer normalization component (optional).
    #     d_model: the number of expected features in the encoder/decoder inputs.
    #     vocab_size: size of vocabulary (number of different possible tokens).
    # Examples::
    #     >>> encoder_layers = 6.times.map {|i| TransformerEncoderLayer.new(512, 8) }
    #     >>> transformer_encoder = nn.TransformerEncoder(encoder_layers, nil, 512, 72, 0.1)
    #     >>> src = Torch.rand(10, 32, 512)
    #     >>> out = transformer_encoder.call(src)
    def initialize(encoder_layers, norm=nil, d_model, vocab_size, dropout)
      super()
      @d_model = d_model
      encoder_layers.each.with_index do |l, i|
        instance_variable_set("@layer#{i}", l)
      end
      @layers = encoder_layers.length.times.map {|i| instance_variable_get("@layer#{i}") }
      @num_layers = encoder_layers.length
      @embedding = Torch::NN::Embedding.new(vocab_size, d_model)
      @pos_encoder = PositionalEncoding.new(d_model, dropout: dropout)
      @norm = norm
    end

    # Pass the input through the encoder layers in turn.
    # Args:
    #     src: the sequence to the encoder (required).
    #     mask: the mask for the src sequence (optional).
    #     src_key_padding_mask: the mask for the src keys per batch (optional).
    # Shape:
    #     see the docs in Transformer class.
    def forward(src, mask: nil, src_key_padding_mask: nil)
      output = @embedding.call(src) * Math.sqrt(@d_model)
      output = @pos_encoder.call(output)

      @layers.each { |mod|
        output = mod.call(output, src_mask: mask, src_key_padding_mask: src_key_padding_mask)
      }

      if @norm
        output = @norm.call(output)
      end

      return output
    end
  end


  class TransformerEncoderLayer < Torch::NN::Module
    # TransformerEncoderLayer is made up of self-attn and feedforward network.
    # This standard encoder layer is based on the paper "Attention Is All You Need".
    # Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    # Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    # Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    # in a different way during application.
    # Args:
    #     d_model: the number of expected features in the input (required).
    #     nhead: the number of heads in the multiheadattention models (required).
    #     dim_feedforward: the dimension of the feedforward network model (default=2048).
    #     dropout: the dropout value (default=0.1).
    #     activation: the activation function of intermediate layer, relu or gelu (default=relu).
    # Examples::
    #     >>> encoder_layer = TransformerEncoderLayer.new(512, 8)
    #     >>> src = Torch.rand(10, 32, 512)
    #     >>> out = encoder_layer.call(src)
    def initialize(d_model, nhead, dim_feedforward:2048, dropout:0.1, activation:"relu")
      super()
      @self_attn = MultiheadAttention.new(d_model, nhead, dropout: dropout)
      # Implementation of Feedforward model
      @linear1 = Torch::NN::Linear.new(d_model, dim_feedforward)
      @dropout = Torch::NN::Dropout.new(p: dropout)
      @linear2 = Torch::NN::Linear.new(dim_feedforward, d_model)

      @norm1 = Torch::NN::LayerNorm.new(d_model)
      @norm2 = Torch::NN::LayerNorm.new(d_model)
      @dropout1 = Torch::NN::Dropout.new(p: dropout)
      @dropout2 = Torch::NN::Dropout.new(p: dropout)

      @activation = _get_activation_fn(activation)
    end

    # Pass the input through the encoder layer.
    # Args:
    #     src: the sequence to the encoder layer (required).
    #     src_mask: the mask for the src sequence (optional).
    #     src_key_padding_mask: the mask for the src keys per batch (optional).
    # Shape:
    #     see the docs in Transformer class.
    def forward(src, src_mask: nil, src_key_padding_mask: nil)
      src2 = @self_attn.call(src, src, src, attn_mask: src_mask,
                            key_padding_mask: src_key_padding_mask)[0]
      src = src + @dropout1.call(src2)
      src = @norm1.call(src)
      src2 = @linear2.call(@dropout.call(@activation.call(@linear1.call(src))))
      src = src + @dropout2.call(src2)
      src = @norm2.call(src)
      return src
    end
  end


  class TransformerDecoder < Torch::NN::Module
    # TransformerDecoder is a stack of N decoder layers
    # Args:
    #     decoder_layers: an array of instances of the TransformerDecoderLayer class (required).
    #     norm: the layer normalization component (optional).
    #     d_model: the number of expected features in the encoder/decoder inputs.
    #     vocab_size: size of vocabulary (number of different possible tokens).
    # Examples::
    #     >>> decoder_layers = 6.times.map {|i| TransformerDecoderLayer.new(512, 8) }
    #     >>> transformer_decoder = TransformerDecoder.new(encoder_layers, nil, 512, 72, 0.1)
    #     >>> memory = Torch.rand(10, 32, 512)
    #     >>> tgt = Torch.rand(20, 32, 512)
    #     >>> out = transformer_decoder.call(tgt, memory)
    def initialize(decoder_layers, norm=nil, d_model, vocab_size, dropout)
      super()
      @d_model = d_model
      decoder_layers.each.with_index do |l, i|
        instance_variable_set("@layer#{i}", l)
      end
      @layers = decoder_layers.length.times.map {|i| instance_variable_get("@layer#{i}") }
      @num_layers = decoder_layers.length
      @embedding = Torch::NN::Embedding.new(vocab_size, d_model)
      @pos_encoder = PositionalEncoding.new(d_model, dropout: dropout)
      @norm = norm
    end

    # Pass the inputs (and mask) through the decoder layer in turn.
    # Args:
    #     tgt: the sequence to the decoder (required).
    #     memory: the sequence from the last layer of the encoder (required).
    #     tgt_mask: the mask for the tgt sequence (optional).
    #     memory_mask: the mask for the memory sequence (optional).
    #     tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
    #     memory_key_padding_mask: the mask for the memory keys per batch (optional).
    # Shape:
    #     see the docs in Transformer class.
    def forward(tgt, memory, tgt_mask: nil,
              memory_mask: nil, tgt_key_padding_mask: nil,
              memory_key_padding_mask: nil)

      output = @embedding.call(tgt) * Math.sqrt(@d_model)
      output = @pos_encoder.call(output)

      @layers.each { |mod|
        output = mod.call(output, memory, tgt_mask: tgt_mask,
                       memory_mask: memory_mask,
                       tgt_key_padding_mask: tgt_key_padding_mask,
                       memory_key_padding_mask: memory_key_padding_mask)
      }

      if @norm
        output = @norm.call(output)
      end

      return output
    end
  end

  class TransformerDecoderLayer < Torch::NN::Module
    # TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    # This standard decoder layer is based on the paper "Attention Is All You Need".
    # Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    # Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    # Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    # in a different way during application.
    # Args:
    #     d_model: the number of expected features in the input (required).
    #     nhead: the number of heads in the multiheadattention models (required).
    #     dim_feedforward: the dimension of the feedforward network model (default=2048).
    #     dropout: the dropout value (default=0.1).
    #     activation: the activation function of intermediate layer, relu or gelu (default=relu).
    # Examples::
    #     >>> decoder_layer = TransformerDecoderLayer(512, 8)
    #     >>> memory = Torch.rand(10, 32, 512)
    #     >>> tgt = Torch.rand(20, 32, 512)
    #     >>> out = decoder_layer.call(tgt, memory)

    def initialize(d_model, nhead, dim_feedforward: 2048, dropout: 0.1, activation: "relu")
      super()
      @self_attn = MultiheadAttention.new(d_model, nhead, dropout: dropout)
      @multihead_attn = MultiheadAttention.new(d_model, nhead, dropout: dropout)
      # Implementation of Feedforward model
      @linear1 = Torch::NN::Linear.new(d_model, dim_feedforward)
      @dropout = Torch::NN::Dropout.new(p: dropout)
      @linear2 = Torch::NN::Linear.new(dim_feedforward, d_model)

      @norm1 = Torch::NN::LayerNorm.new(d_model)
      @norm2 = Torch::NN::LayerNorm.new(d_model)
      @norm3 = Torch::NN::LayerNorm.new(d_model)
      @dropout1 = Torch::NN::Dropout.new(p: dropout)
      @dropout2 = Torch::NN::Dropout.new(p: dropout)
      @dropout3 = Torch::NN::Dropout.new(p: dropout)

      @activation = _get_activation_fn(activation)
    end

    # Pass the inputs (and mask) through the decoder layer.
    # Args:
    #     tgt: the sequence to the decoder layer (required).
    #     memory: the sequence from the last layer of the encoder (required).
    #     tgt_mask: the mask for the tgt sequence (optional).
    #     memory_mask: the mask for the memory sequence (optional).
    #     tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
    #     memory_key_padding_mask: the mask for the memory keys per batch (optional).
    # Shape:
    #     see the docs in Transformer class.
    def forward(tgt, memory, tgt_mask: nil, memory_mask: nil,
              tgt_key_padding_mask: nil, memory_key_padding_mask: nil)

      tgt2 = @self_attn.call(tgt, tgt, tgt, attn_mask: tgt_mask,
                            key_padding_mask: tgt_key_padding_mask)[0]
      tgt = tgt + @dropout1.call(tgt2)
      tgt = @norm1.call(tgt)
      tgt2 = @multihead_attn.call(tgt, memory, memory, attn_mask: memory_mask,
                                 key_padding_mask: memory_key_padding_mask)[0]
      tgt = tgt + @dropout2.call(tgt2)
      tgt = @norm2.call(tgt)
      tgt2 = @linear2.call(@dropout.call(@activation.call(@linear1.call(tgt))))
      tgt = tgt + @dropout3.call(tgt2)
      tgt = @norm3.call(tgt)
      return tgt
    end
  end

  class PositionalEncoding < Torch::NN::Module
    # PositionalEncoding module injects some information about the relative or absolute position of the tokens in the sequence. The positional encodings have the same dimension as the embeddings so that the two can be summed. Here, we use sine and cosine functions of different frequencies.
    def initialize(d_model, dropout: 0.1, max_len: 5000)
      super()
      @dropout = Torch::NN::Dropout.new(p: dropout)

      pe = Torch.zeros(max_len, d_model)
      position = Torch.arange(0, max_len, dtype: :float).unsqueeze(1)
      div_term = Torch.exp(Torch.arange(0, d_model, 2).float() * (-Math.log(10000.0) / d_model))
      sin = Torch.sin(position * div_term).t
      cos = Torch.cos(position * div_term).t
      pe.t!
      pe.each.with_index do |row, i|
        pe[i] = sin[i / 2] if i % 2 == 0
        pe[i] = cos[(i-1)/2] if i % 2 != 0
      end
      pe.t!
      pe = pe.unsqueeze(0).transpose(0, 1)
      register_buffer('pe', pe)
    end

    def forward(x)
      x = x + pe.narrow(0, 0, x.size(0))
      return x
    end
  end
end


def _get_activation_fn(activation)
  if activation == "relu"
    return Torch::NN::F.method(:relu)
  elsif activation == "gelu"
    return Torch::NN::F.method(:gelu)
  end

  raise RuntimeError, "activation should be relu/gelu, not %s" % activation
end
