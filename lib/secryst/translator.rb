module Secryst
  class Translator
    def initialize(model:, data:, hyperparameters:, model_file:)
      data_dir = File.expand_path("../../data", __dir__)
      @data_input = File.readlines("#{data_dir}/#{data}/input.csv", chomp: true)
      @data_target = File.readlines("#{data_dir}/#{data}/target.csv", chomp: true)

      @device = "cpu"

      generate_vocabs()

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

    private def generate_vocabs
      input_texts = []
      target_texts = []
      input_vocab_counter = Hash.new(0)
      target_vocab_counter = Hash.new(0)

      @data_input.each do |input_text|
        input_text.strip!
        input_texts.push(input_text)
        input_text.each_char do |char|
          input_vocab_counter[char] += 1
        end
      end

      @data_target.each do |target_text|
        target_text.strip!
        target_texts.push(target_text)
        target_text.each_char do |char|
          target_vocab_counter[char] += 1
        end
      end

      @input_vocab = TorchText::Vocab.new(input_vocab_counter)
      @target_vocab = TorchText::Vocab.new(target_vocab_counter)
    end
  end
end