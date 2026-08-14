require 'zip'
require 'onnxruntime'

module Secryst
  # Byte-level seq2seq (ByT5 family) inference over ONNX encoder/decoder
  # sessions. The tokenizer is UTF-8 bytes themselves, so no vocab files
  # are needed: pad = 0, EOS = 1 (ByT5 convention; the decoder starts
  # with token 0).
  class Byt5Onnx
    PAD_ID = 0
    EOS_ID = 1

    attr_reader :encoder, :decoder

    def initialize(zip_path)
      Zip::File.open(zip_path) do |z|
        enc = z.glob('encoder.onnx').first
        dec = z.glob('decoder.onnx').first
        raise 'encoder.onnx is missing in model zip!' unless enc
        raise 'decoder.onnx is missing in model zip!' unless dec
        @encoder = OnnxRuntime::Model.new(enc.get_input_stream.read)
        @decoder = OnnxRuntime::Model.new(dec.get_input_stream.read)
      end
    end

    def translate(text, max_seq_length: 256)
      input_ids = text.bytes
      return '' if input_ids.empty?

      hidden = @encoder.predict({ input_ids: [input_ids] }).values.first

      generated = []
      decoder_ids = [PAD_ID]
      max_seq_length.times do
        logits = @decoder.predict({
          input_ids: [decoder_ids],
          encoder_hidden_states: hidden,
        }).values.first
        next_id = argmax(logits.last)
        break if next_id == EOS_ID

        generated << next_id
        decoder_ids << next_id
      end

      generated.pack('C*').force_encoding(Encoding::UTF_8)
    end

    private

    def argmax(row)
      best = 0
      row.each_with_index.reduce(-Float::INFINITY) do |best_val, (v, i)|
        if v > best_val
          best = i
          v
        else
          best_val
        end
      end
      best
    end
  end
end
