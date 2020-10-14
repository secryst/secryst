module Secryst
  class Translator
    def initialize(model_file:)
      model_name, model, metadata, model_state_dict, vocabs = nil
      @device = "cpu"
      # Unzip model in memory
      Zip::File.open(model_file) do |zip_file|
        metadata = zip_file.glob('metadata.yaml').first
        raise 'metadata.yaml is missing in model zip!' if !metadata
        metadata = YAML.load(metadata.get_input_stream.read)

        model = zip_file.glob('model.pth').first
        raise 'model.pth is missing in model zip!' if !model
        model_state_dict = Torch.send :to_ruby, Torch._load(model.get_input_stream.read)

        vocabs = zip_file.glob('vocabs.yaml').first
        raise 'vocabs.yaml is missing in model zip!' if !vocabs
        vocabs = YAML.load(vocabs.get_input_stream.read)
      end

      model_name = metadata.delete("name")
      # load vocabs
      @input_vocab = Vocab.new(vocabs["input"])
      @target_vocab = Vocab.new(vocabs["target"])

      if model_name == 'transformer'
        @model = Secryst::Transformer.new({
          d_model: metadata[:d_model],
          nhead: metadata[:nhead],
          num_encoder_layers: metadata[:num_encoder_layers],
          num_decoder_layers: metadata[:num_decoder_layers],
          dim_feedforward: metadata[:dim_feedforward],
          dropout: metadata[:dropout],
          activation: metadata[:activation],
          input_vocab_size: @input_vocab.length,
          target_vocab_size: @target_vocab.length,
        })
      else
        raise ArgumentError, 'Only transformer model is currently supported'
      end

      @model.load_state_dict(model_state_dict)
      @model.eval
    end

    def translate(phrase, max_seq_length: 100)
      input = ['<sos>'] + phrase.chars + ['<eos>']
      input = Torch.tensor([input.map {|i| @input_vocab.stoi[i]}]).t
      output = Torch.tensor([[@target_vocab.stoi['<sos>']]])
      src_key_padding_mask = input.t.eq(1)

      max_seq_length.times do |i|
        tgt_key_padding_mask = output.t.eq(1)
        tgt_mask = Torch.triu(Torch.ones(i+1,i+1)).eq(0).transpose(0,1)
        opts = {
          tgt_mask: tgt_mask,
          src_key_padding_mask: src_key_padding_mask,
          tgt_key_padding_mask: tgt_key_padding_mask,
          memory_key_padding_mask: src_key_padding_mask,
        }
        prediction = @model.call(input, output, opts).map {|i| i.argmax.item }
        break if @target_vocab.itos[prediction[i]] == '<eos>'
        output = Torch.cat([output, Torch.tensor([[prediction[i]]])])
      end

      "#{output[1..-1].map {|i| @target_vocab.itos[i.item]}.join('')}"
    end
  end
end