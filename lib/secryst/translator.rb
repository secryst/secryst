module Secryst
  class Translator
    attr_accessor :model
    def initialize(model_file:)
      @device = "cpu"
      @model = Model.from_file(model_file)
    end

    def translate(phrase, max_seq_length: 100)
      input = ['<sos>'] + phrase.chars + ['<eos>']
      input = Numo::NArray[input.map {|i| @model.input_vocab.stoi[i]}].transpose
      output = Numo::NArray[[@model.target_vocab.stoi['<sos>']]]
      src_key_padding_mask = input.transpose.eq(1)

      max_seq_length.times do |i|
        tgt_key_padding_mask = output.transpose.eq(1)
        tgt_mask = Numo::DFloat.ones(i+1,i+1).triu.transpose.eq(0)
        if defined?(Torch)
          input = Torch.from_numo(input).to(:long) if input.kind_of?(Numo::NArray)
          output = Torch.from_numo(output).to(:long) if output.kind_of?(Numo::NArray)
          tgt_mask = Torch.from_numo(Numo::UInt8.cast(tgt_mask)) if tgt_mask.kind_of?(Numo::NArray)
          src_key_padding_mask = Torch.from_numo(Numo::UInt8.cast(src_key_padding_mask)) if src_key_padding_mask.kind_of?(Numo::NArray)
          tgt_key_padding_mask = Torch.from_numo(Numo::UInt8.cast(tgt_key_padding_mask)) if tgt_key_padding_mask.kind_of?(Numo::NArray)
        end
        puts "inp", input.inspect
        puts "out", output.inspect
        puts "tgt_mask", tgt_mask.inspect
        puts "src_key_padding_mask", src_key_padding_mask.inspect
        puts "tgt_key_padding_mask", tgt_key_padding_mask.inspect
        opts = {
          tgt_mask: tgt_mask,
          src_key_padding_mask: src_key_padding_mask,
          tgt_key_padding_mask: tgt_key_padding_mask,
          memory_key_padding_mask: src_key_padding_mask,
        }

        # We are cloning the output tensor because of weird bug
        # that it is being mutated https://github.com/secryst/secryst/pull/34#issuecomment-769935874
        dupped_output = if defined?(Torch)
          Torch.tensor(output)
        else
          output.dup
        end
        puts "dup0", dupped_output.inspect
        prediction = @model.argmax(input, dupped_output, **opts)
        break if @model.target_vocab.itos[prediction[i]] == '<eos>'
        puts "prediction", prediction.inspect
        puts "before .numo", output.inspect
        if defined?(Torch)
          output = output.numo
        end
        puts "after .numo", output.inspect
        output = Numo::NArray.concatenate([output, Numo::NArray[[prediction[i]]]])
        puts "after concat", output.inspect
      end

      "#{output[1..-1].to_a.flatten.map {|i| @model.target_vocab.itos[i]}.join('')}"
    end
  end
end
