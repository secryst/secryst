module Secryst
  class Translator
    attr_accessor :model, :input_vocab, :target_vocab
    def initialize(model_file:)
      model_name, model, metadata, model_state_dict, vocabs = nil
      @device = "cpu"
      # Unzip model in memory
      Zip::File.open(model_file) do |zip_file|
        metadata = zip_file.glob('metadata.yaml').first
        metadata = YAML.load(metadata.get_input_stream.read) if metadata

        vocabs = zip_file.glob('vocabs.yaml').first
        raise 'vocabs.yaml is missing in model zip!' if !vocabs
        vocabs = YAML.load(vocabs.get_input_stream.read)
        @input_vocab = Vocab.new(vocabs["input"], specials: [])
        @target_vocab = Vocab.new(vocabs["target"], specials: [])

        model = zip_file.glob('*.pth').first
        if !model
          model = zip_file.glob('*.onnx').first
          raise 'Both .pth and .onnx model files is missing in model zip!' if !model
        end
        @model = Model.from_file(model, metadata, @input_vocab.length, @target_vocab.length)
      end
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