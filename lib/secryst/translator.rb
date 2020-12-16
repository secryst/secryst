module Secryst
  class Translator
    attr_accessor :model
    def initialize(model_file:)
      @device = "cpu"
      @model = Model.from_file(model_file)
    end

    def translate(phrase, max_seq_length: 100)
      input = ['<sos>'] + phrase.chars + ['<eos>']
      input = Torch.tensor([input.map {|i| @model.input_vocab.stoi[i]}]).t
      output = Torch.tensor([[@model.target_vocab.stoi['<sos>']]])
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
        puts input.size.inspect
        puts output.size.inspect
        prediction = @model.argmax(input, output, opts)
        # require 'byebug'
        # byebug
        puts prediction.inspect
        break if @model.target_vocab.itos[prediction[i]] == '<eos>'
        output = Torch.cat([output, Torch.tensor([[prediction[i]]])])
      end

      "#{output[1..-1].map {|i| @model.target_vocab.itos[i.item]}.join('')}"
    end
  end
end