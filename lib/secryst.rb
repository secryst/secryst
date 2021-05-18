Bundler.require(:development)
require 'yaml'
require 'zip'

# numo
require 'numo/narray'

# transformer model
if defined?(Torch)
  require "secryst/multihead_attention"
  require "secryst/transformer"
end
require "secryst/vocab"

require "secryst/translator"
require "secryst/model"

require "secryst/provisioning"

require 'onnxruntime'

module Secryst
  DEFAULT_HYPERPARAMETERS = {
    d_model: 64,
    nhead: 8,
    num_encoder_layers: 4,
    num_decoder_layers: 4,
    dim_feedforward: 256,
    dropout: 0.05,
    activation: 'relu',
  }
end
