module Secryst
  class Translator
    def initialize(model:, vocabs_dir:, hyperparameters:, model_file:)
      @device = "cpu"
      @vocabs_dir = vocabs_dir

      load_vocabs()

      if model == 'transformer'
        @model = Torch::NN::Transformer.new(hyperparameters.merge({
          input_vocab_size: @input_vocab.length,
          target_vocab_size: @target_vocab.length,
        }))
      else
        raise ArgumentError, 'Only transformer model is currently supported'
      end

      @model.load_state_dict(Torch.load(model_file))
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
      puts "#{output[1..-1].map {|i| @target_vocab.itos[i.item]}.join('')}"
    end

    private def load_vocabs
      @input_vocab = Vocab.new(JSON.parse(File.read("#{@vocabs_dir}/input_vocab.json")))
      @target_vocab = Vocab.new(JSON.parse(File.read("#{@vocabs_dir}/target_vocab.json")))
    end
  end
end