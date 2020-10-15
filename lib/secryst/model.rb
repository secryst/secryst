module Secryst
  class Model
    attr_accessor :model
    def initialize(model)
      @model = model
    end

    def self.from_file(model_file, metadata, input_vocab_length, target_vocab_length)
      if model_file.name.end_with?('.pth')
        raise 'metadata.yaml is missing in model zip!' if !metadata
        model_state_dict = Torch.send :to_ruby, Torch._load(model_file.get_input_stream.read)
        model_name = metadata.delete("name")
        if model_name == 'transformer'
          model = Secryst::Transformer.new({
            d_model: metadata[:d_model],
            nhead: metadata[:nhead],
            num_encoder_layers: metadata[:num_encoder_layers],
            num_decoder_layers: metadata[:num_decoder_layers],
            dim_feedforward: metadata[:dim_feedforward],
            dropout: metadata[:dropout],
            activation: metadata[:activation],
            input_vocab_size: input_vocab_length,
            target_vocab_size: target_vocab_length,
          })
        else
          raise ArgumentError, 'Only transformer model is currently supported'
        end
        model.load_state_dict(model_state_dict)
        model.eval
        return self.new(model)
      elsif model_file.name.end_with?('.onnx')
        return Onnx.new(model_file.get_input_stream.read)
      else
        raise "Model extension #{model_file.name.split('.').last} not supported"
      end

    end

    def call(*args)
      @model.call(*args)
    end

    class Onnx < Model
      def initialize(model_path_or_bytes)
        @model = OnnxRuntime::Model.new(model_path_or_bytes)
      end

      def call(*args)
        @model.predict(*args)
      end
    end
  end
end