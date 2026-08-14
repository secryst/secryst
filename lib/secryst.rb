require 'yaml'
require 'zip'
require 'numo/narray'

require 'secryst/vocab'
require 'secryst/byt5_onnx'
require 'secryst/translator'
require 'secryst/model'
require 'secryst/provisioning'

module Secryst
  DEFAULT_HYPERPARAMETERS = {
    d_model: 64,
    nhead: 8,
    num_encoder_layers: 4,
    num_decoder_layers: 4,
    dim_feedforward: 256,
    dropout: 0.05,
    activation: 'relu',
  }.freeze
end
