# ported from https://github.com/pytorch/pytorch/blob/4ae832e1060c72cb89de1d9693629783dbe0c9a6/torch/csrc/api/include/torch/nn/functional/activation.h

require_relative 'multi_head_attention_forward'
module Secryst
  class MultiheadAttention < Torch::NN::Module
    # Allows the model to jointly attend to information
    # from different representation subspaces.
    # See reference: Attention Is All You Need
    # .. math::
    #     \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
    #     \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
    # Args:
    #     embed_dim: total dimension of the model.
    #     num_heads: parallel attention heads.
    #     dropout: a Dropout layer on attn_output_weights. Default: 0.0.
    #     bias: add bias as module parameter. Default: true.
    #     add_bias_kv: add bias to the key and value sequences at dim=0.
    #     add_zero_attn: add a new batch of zeros to the key and
    #                     value sequences at dim=1.
    #     kdim: total number of features in key. Default: nil.
    #     vdim: total number of features in value. Default: nil.
    #     Note: if kdim and vdim are nil, they will be set to embed_dim such that
    #     query, key, and value have the same number of features.
    # Examples::
    #     >>> multihead_attn = MultiheadAttention.new(embed_dim: embed_dim, num_heads: num_heads)
    #     >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    # bias_k: Optional[Torch::Tensor]
    # bias_v: Optional[Torch::Tensor]

    def initialize(embed_dim, num_heads, dropout:0.0, bias: true, add_bias_kv: false, add_zero_attn: false, kdim: nil, vdim: nil)
      super()
      @embed_dim = embed_dim
      @kdim = kdim || embed_dim
      @vdim = vdim || embed_dim
      @_qkv_same_embed_dim = @kdim == @embed_dim && @vdim == @embed_dim

      @num_heads = num_heads
      @dropout = dropout
      @head_dim = embed_dim / num_heads
      raise ArgumentError, "embed_dim must be divisible by num_heads" if @head_dim * num_heads != @embed_dim

      if !@_qkv_same_embed_dim
        @q_proj_weight = Torch::NN::Parameter.new(Torch::Tensor.new(embed_dim, embed_dim))
        @k_proj_weight = Torch::NN::Parameter.new(Torch::Tensor.new(embed_dim, @kdim))
        @v_proj_weight = Torch::NN::Parameter.new(Torch::Tensor.new(embed_dim, @vdim))
        register_parameter('in_proj_weight', nil)
      else
        @in_proj_weight = Torch::NN::Parameter.new(Torch.empty(3 * embed_dim, embed_dim))
        register_parameter('q_proj_weight', nil)
        register_parameter('k_proj_weight', nil)
        register_parameter('v_proj_weight', nil)
      end

      if bias
        @in_proj_bias = Torch::NN::Parameter.new(Torch.empty(3 * embed_dim))
      else
        register_parameter('in_proj_bias', nil)
      end
      @out_proj = Torch::NN::Linear.new(embed_dim, embed_dim)

      if add_bias_kv
        @bias_k = Torch::NN::Parameter.new(Torch.empty(1, 1, embed_dim))
        @bias_v = Torch::NN::Parameter.new(Torch.empty(1, 1, embed_dim))
      else
        @bias_k = @bias_v = nil
      end

      @add_zero_attn = add_zero_attn

      _reset_parameters
    end

    def _reset_parameters
      if @_qkv_same_embed_dim
        Torch::NN::Init.xavier_uniform!(@in_proj_weight)
      else
        Torch::NN::Init.xavier_uniform!(@q_proj_weight)
        Torch::NN::Init.xavier_uniform!(@k_proj_weight)
        Torch::NN::Init.xavier_uniform!(@v_proj_weight)
      end

      if @in_proj_bias
        Torch::NN::Init.constant!(@in_proj_bias, 0.0)
        Torch::NN::Init.constant!(@out_proj.bias, 0.0)
      end

      if @bias_k
        Torch::NN::Init.xavier_normal!(@bias_k)
      end

      if @bias_v
        Torch::NN::Init.xavier_normal!(@bias_v)
      end
    end

    # Args:
    #     query, key, value: map a query and a set of key-value pairs to an output.
    #         See "Attention Is All You Need" for more details.
    #     key_padding_mask: if provided, specified padding elements in the key will
    #         be ignored by the attention. When given a binary mask and a value is true,
    #         the corresponding value on the attention layer will be ignored. When given
    #         a byte mask and a value is non-zero, the corresponding value on the attention
    #         layer will be ignored
    #     need_weights: output attn_output_weights.
    #     attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
    #         the batches while a 3D mask allows to specify a different mask for the entries of each batch.
    # Shape:
    #     - Inputs:
    #     - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
    #       the embedding dimension.
    #     - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
    #       the embedding dimension.
    #     - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
    #       the embedding dimension.
    #     - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
    #       If a ByteTensor is provided, the non-zero positions will be ignored while the position
    #       with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
    #       value of ``true`` will be ignored while the position with the value of ``false`` will be unchanged.
    #     - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
    #       3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
    #       S is the source sequence length. attn_mask ensure that position i is allowed to attend the unmasked
    #       positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
    #       while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``true``
    #       is not allowed to attend while ``false`` values will be unchanged. If a FloatTensor
    #       is provided, it will be added to the attention weight.
    #     - Outputs:
    #     - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
    #       E is the embedding dimension.
    #     - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
    #       L is the target sequence length, S is the source sequence length.
    def forward(query, key, value, key_padding_mask:nil,
                need_weights:true, attn_mask:nil)
      if !@_qkv_same_embed_dim
        return Secryst::MultiHeadAttentionForward.multi_head_attention_forward(
          query, key, value, @embed_dim, @num_heads,
          @in_proj_weight, @in_proj_bias,
          @bias_k, @bias_v, @add_zero_attn,
          @dropout, @out_proj.weight, @out_proj.bias,
          training: @training,
          key_padding_mask: key_padding_mask, need_weights: need_weights,
          attn_mask: attn_mask, use_separate_proj_weight: true,
          q_proj_weight: @q_proj_weight, k_proj_weight: @k_proj_weight,
          v_proj_weight: @v_proj_weight)
      else
        return Secryst::MultiHeadAttentionForward.multi_head_attention_forward(
            query, key, value, @embed_dim, @num_heads,
            @in_proj_weight, @in_proj_bias,
            @bias_k, @bias_v, @add_zero_attn,
            @dropout, @out_proj.weight, @out_proj.bias,
            training: @training,
            key_padding_mask: key_padding_mask, need_weights: need_weights,
            attn_mask: attn_mask)
      end
    end
  end
end
