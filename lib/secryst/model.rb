module Secryst
  class Model
    attr_accessor :model, :input_vocab, :target_vocab
    def initialize(model, input_vocab, target_vocab)
      @model = model
      @input_vocab = input_vocab
      @target_vocab = target_vocab
    end

    def self.from_file(model_file)
      model_name, model, metadata, model_state_dict, vocabs, input_vocab, target_vocab = nil
      # Locate the model name
      model_file = Provisioning.locate(model_file)

      # Unzip model in memory
      Zip::File.open(model_file) do |zip_file|
        metadata = zip_file.glob('metadata.yaml').first
        metadata = YAML.load(metadata.get_input_stream.read) if metadata

        vocabs = zip_file.glob('vocabs.yaml').first
        raise 'vocabs.yaml is missing in model zip!' if !vocabs
        vocabs = YAML.load(vocabs.get_input_stream.read)
        puts vocabs
        input_vocab = Vocab.new(vocabs["input"], specials: [])
        target_vocab = Vocab.new(vocabs["target"], specials: [])

        model = zip_file.glob('*.pth').first
        if !model
          model = zip_file.glob('*.onnx').first
          raise 'Both .pth and .onnx model files is missing in model zip!' if !model
        end
      end
      if model.name.end_with?('.pth')
        require 'torch'
        require "secryst/multihead_attention"
        require "secryst/transformer"
        raise 'metadata.yaml is missing in model zip!' if !metadata
        model_state_dict = Torch.send :to_ruby, Torch._load(model.get_input_stream.read)
        # Python-trained model have other key namess, so we transform
        # them in this case
        new_dict = {}
        model_state_dict.each {|k,v|
          key = k.gsub(/layers\.(\d+)\./, "layer\1")
          new_dict[key] = v
        }

        model_name = metadata.delete("name")
        if model_name == 'transformer'
          model = Secryst::Transformer.new(
            d_model: metadata[:d_model],
            nhead: metadata[:nhead],
            num_encoder_layers: metadata[:num_encoder_layers],
            num_decoder_layers: metadata[:num_decoder_layers],
            dim_feedforward: metadata[:dim_feedforward],
            dropout: metadata[:dropout],
            activation: metadata[:activation],
            input_vocab_size: input_vocab.length,
            target_vocab_size: target_vocab.length,
          )
        else
          raise ArgumentError, 'Only transformer model is currently supported'
        end
        model.load_state_dict(new_dict)
        model.eval
        return self.new(model, input_vocab, target_vocab)
      elsif model.name.end_with?('.onnx')
        return Onnx.new(model.get_input_stream.read, input_vocab, target_vocab)
      else
        raise "Model extension #{model.name.split('.').last} not supported"
      end

    end

    def call(*args, **kwargs)
      @model.call(*args, **kwargs)
    end

    def argmax(*args, **kwargs)
      self.call(*args, **kwargs).map {|i| i.argmax.item }
    end

    def method_missing(name, *args, **kwargs, &block)
      @model.public_send(name, *args, **kwargs, &block)
    end

    class Onnx < Model
      def initialize(model_path_or_bytes, input_vocab, target_vocab)
        @model = OnnxRuntime::Model.new(model_path_or_bytes)
        @input_vocab = input_vocab
        @target_vocab = target_vocab
      end

      def call(input, output, opts)
        @model.predict({src: input, tgt: output}.merge(opts))["output"]
      end

      def argmax(*args)
        self.call(*args).map {|i| 
          i.flatten.each_with_index.max[1] 
        }
      end

      def predict(args)
        @model.predict(args)
      end
    end
  end
end
