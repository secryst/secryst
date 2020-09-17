module Torch
  module NN
    class Functional
      class << self
        # Args:
        #   query, key, value: map a query and a set of key-value pairs to an output.
        #       See "Attention Is All You Need" for more details.
        #   embed_dim_to_check: total dimension of the model.
        #   num_heads: parallel attention heads.
        #   in_proj_weight, in_proj_bias: input projection weight and bias.
        #   bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        #   add_zero_attn: add a new batch of zeros to the key and
        #                   value sequences at dim=1.
        #   dropout_p: probability of an element to be zeroed.
        #   out_proj_weight, out_proj_bias: the output projection weight and bias.
        #   training: apply dropout if is ``True``.
        #   key_padding_mask: if provided, specified padding elements in the key will
        #       be ignored by the attention. This is an binary mask. When the value is True,
        #       the corresponding value on the attention layer will be filled with -inf.
        #   need_weights: output attn_output_weights.
        #   attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
        #       the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        #   use_separate_proj_weight: the function accept the proj. weights for query, key,
        #       and value in different forms. If false, in_proj_weight will be used, which is
        #       a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        #   q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        #   static_k, static_v: static key and value used for attention operators.
        # Shape:
        #   Inputs:
        #   - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
        #     the embedding dimension.
        #   - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
        #     the embedding dimension.
        #   - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
        #     the embedding dimension.
        #   - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
        #     If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
        #     will be unchanged. If a BoolTensor is provided, the positions with the
        #     value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        #   - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
        #     3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
        #     S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
        #     positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
        #     while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
        #     are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
        #     is provided, it will be added to the attention weight.
        #   - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
        #     N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        #   - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
        #     N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        #   Outputs:
        #   - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
        #     E is the embedding dimension.
        #   - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
        #     L is the target sequence length, S is the source sequence length.

        # if not torch.jit.is_scripting():
        #   tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v,
        #               out_proj_weight, out_proj_bias)
        #   if any([type(t) is not Tensor for t in tens_ops]) and has_torch_function(tens_ops):
        #       return handle_torch_function(
        #           multi_head_attention_forward, tens_ops, query, key, value,
        #           embed_dim_to_check, num_heads, in_proj_weight, in_proj_bias,
        #           bias_k, bias_v, add_zero_attn, dropout_p, out_proj_weight,
        #           out_proj_bias, training=training, key_padding_mask=key_padding_mask,
        #           need_weights=need_weights, attn_mask=attn_mask,
        #           use_separate_proj_weight=use_separate_proj_weight,
        #           q_proj_weight=q_proj_weight, k_proj_weight=k_proj_weight,
        #           v_proj_weight=v_proj_weight, static_k=static_k, static_v=static_v)
        def multi_head_attention_forward(query,
                                 key,
                                 value,
                                 embed_dim_to_check,
                                 num_heads,
                                 in_proj_weight,
                                 in_proj_bias,
                                 bias_k,
                                 bias_v,
                                 add_zero_attn,
                                 dropout_p,
                                 out_proj_weight,
                                 out_proj_bias,
                                 training: true,
                                 key_padding_mask: nil,
                                 need_weights: true,
                                 attn_mask: nil,
                                 use_separate_proj_weight: false,
                                 q_proj_weight: nil,
                                 k_proj_weight: nil,
                                 v_proj_weight: nil,
                                 static_k: nil,
                                 static_v: nil)
          tgt_len, bsz, embed_dim = query.size()
          raise ArgumentError if embed_dim != embed_dim_to_check
          # allow MHA to have different sizes for the feature dimension
          raise ArgumentError if key.size(0) != value.size(0) or key.size(1) != value.size(1)

          head_dim = embed_dim / num_heads
          raise ArgumentError, "embed_dim must be divisible by num_heads" if head_dim * num_heads != embed_dim
          scaling = head_dim.to_f ** -0.5

          if !use_separate_proj_weight
            if Torch.equal(query, key) && Torch.equal(key, value)
              # self-attention
              q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, -1)

            elsif Torch.equal(key, value)
              # encoder-decoder attention
              # This is inline in_proj function with in_proj_weight and in_proj_bias
              _b = in_proj_bias
              _start = 0
              _end = embed_dim
              _w = in_proj_weight.slice(0, _start, _end) # NOTE: inc-trspl
              if _b
                _b = _b.slice(0, _start, _end)
              end
              q = linear(query, _w, _b)

              if !key
                raise ArgumentError if value
                k = nil
                v = nil
              else
                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = nil
                _w = in_proj_weight.slice(0, _start)
                if _b
                  _b = _b.slice(0, _start)
                end
                k, v = linear(key, _w, _b).chunk(2, -1)
              end

            else
              # This is inline in_proj function with in_proj_weight and in_proj_bias
              _b = in_proj_bias
              _start = 0
              _end = embed_dim
              _w = in_proj_weight.slice(0, _start, _end)
              if _b
                _b = _b.slice(0, _start, _end)
              end
              q = linear(query, _w, _b)

              # This is inline in_proj function with in_proj_weight and in_proj_bias
              _b = in_proj_bias
              _start = embed_dim
              _end = embed_dim * 2
              _w = in_proj_weight.slice(0, _start, _end)
              if _b
                _b = _b.slice(0, _start, _end)
              end
              k = linear(key, _w, _b)

              # This is inline in_proj function with in_proj_weight and in_proj_bias
              _b = in_proj_bias
              _start = embed_dim * 2
              _end = nil
              _w = in_proj_weight.slice(0, _start)
              if _b
                _b = _b.slice(0, _start)
              end
              v = linear(value, _w, _b)
            end
          else
            q_proj_weight_non_opt = q_proj_weight # NOTE: inc-trspl
            len1, len2 = q_proj_weight_non_opt.size()
            raise ArgumentError if len1 != embed_dim || len2 != query.size(-1)

            k_proj_weight_non_opt = k_proj_weight # NOTE: inc-trspl
            len1, len2 = k_proj_weight_non_opt.size()
            raise ArgumentError if len1 != embed_dim || len2 != key.size(-1)

            v_proj_weight_non_opt = v_proj_weight # NOTE: inc-trspl
            len1, len2 = v_proj_weight_non_opt.size()
            raise ArgumentError if len1 != embed_dim || len2 != value.size(-1)

            if in_proj_bias
              q = linear(query, q_proj_weight_non_opt, in_proj_bias.slice(0,0,embed_dim))
              k = linear(key, k_proj_weight_non_opt, in_proj_bias.slice(0, embed_dim, embed_dim * 2))
              v = linear(value, v_proj_weight_non_opt, in_proj_bias.slice(0, embed_dim * 2))
            else
              q = linear(query, q_proj_weight_non_opt, in_proj_bias)
              k = linear(key, k_proj_weight_non_opt, in_proj_bias)
              v = linear(value, v_proj_weight_non_opt, in_proj_bias)
            end
          end
          q = q * scaling

          if attn_mask
            raise ArgumentError, 'Only float, byte, and bool types are supported for attn_mask, not %s' % attn_mask.dtype unless attn_mask.dtype == Torch.float32 || attn_mask.dtype == Torch.float64 || attn_mask.dtype == Torch.float16 || attn_mask.dtype == Torch.uint8 || attn_mask.dtype == Torch.bool
            if attn_mask.dtype == Torch.uint8
              puts "Byte tensor for attn_mask in NN::MultiheadAttention is deprecated. Use bool tensor instead."
              attn_mask = attn_mask.to(Torch.bool)
            end

            if attn_mask.dim() == 2
              attn_mask = attn_mask.unsqueeze(0)
              raise ArgumentError, 'The size of the 2D attn_mask is not correct.' if attn_mask.size() != [1, query.size(0), key.size(0)]
            elsif attn_mask.dim() == 3
              raise ArgumentError, 'The size of the 3D attn_mask is not correct.' if attn_mask.size() != [bsz * num_heads, query.size(0), key.size(0)]
            else
              raise ArgumentError, "attn_mask's dimension %s is not supported" % attn_mask.dim()
            end
            # attn_mask's dim is 3 now.
          end

          # convert ByteTensor key_padding_mask to bool
          if key_padding_mask && key_padding_mask.dtype == Torch.uint8
            puts("Byte tensor for key_padding_mask in NN::MultiheadAttention is deprecated. Use bool tensor instead.")
            key_padding_mask = key_padding_mask.to(Torch.bool) # NOTE: inc-trspl
          end

          if bias_k && bias_v
            if !static_k && !static_v
              k = Torch.cat([k, bias_k.repeat(1, bsz, 1)])
              v = Torch.cat([v, bias_v.repeat(1, bsz, 1)])
              attn_mask = pad(attn_mask, [0, 1]) if attn_mask
              key_padding_mask = pad(key_padding_mask, [0, 1]) if key_padding_mask
            else
              raise ArgumentError, "bias cannot be added to static key." unless !static_k
              raise ArgumentError, "bias cannot be added to static value." unless !static_v
            end
          else
            raise ArgumentError unless !bias_k
            raise ArgumentError unless !bias_v
          end

          q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
          k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1) if k
          v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1) if v

          if static_k
            raise ArgumentError unless static_k.size(0) == bsz * num_heads
            raise ArgumentError unless static_k.size(2) == head_dim
            k = static_k
          end

          if static_v
            raise ArgumentError unless static_v.size(0) == bsz * num_heads
            raise ArgumentError unless static_v.size(2) == head_dim
            v = static_v
          end

          src_len = k.size(1)

          if key_padding_mask
            raise ArgumentError unless key_padding_mask.size(0) == bsz
            raise ArgumentError unless key_padding_mask.size(1) == src_len
          end

          if add_zero_attn
            src_len += 1
            k_sizes = k.size()
            k_sizes[1] = 1
            k = Torch.cat([k, Torch.zeros(k_sizes, dtype: k.dtype, device: k.device)], 1)
            v_sizes = v.size()
            v_sizes[1] = 1
            v = Torch.cat([v, Torch.zeros(v_sizes, dtype: v.dtype, device: v.device)], 1)
            attn_mask = pad(attn_mask, [0, 1]) if attn_mask
            key_padding_mask = pad(key_padding_mask, [0, 1]) if key_padding_mask
          end

          attn_output_weights = Torch.bmm(q, k.transpose(1, 2))
          raise ArgumentError unless attn_output_weights.size() == [bsz * num_heads, tgt_len, src_len]

          if attn_mask
            if attn_mask.dtype == Torch.bool
              attn_output_weights.masked_fill!(attn_mask, -1.0/0.0) # NOTE: inc-trsplc float inf
            else
              attn_output_weights += attn_mask
            end
          end


          if key_padding_mask
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(
              key_padding_mask.unsqueeze(1).unsqueeze(2),
              -1.0/0.0
            ) # NOTE: inc-trspl
            attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)
          end

          attn_output_weights = softmax(
              attn_output_weights, dim: -1)
          attn_output_weights = dropout(attn_output_weights, p: dropout_p, training: training)

          attn_output = Torch.bmm(attn_output_weights, v)
          raise ArgumentError unless attn_output.size() == [bsz * num_heads, tgt_len, head_dim]
          attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
          attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

          if need_weights
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            return attn_output, attn_output_weights.sum(1) / num_heads
          else
            return attn_output, nil
          end
        end
      end
    end
  end
end