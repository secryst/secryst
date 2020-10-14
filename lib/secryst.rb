require 'yaml'
require 'zip'

# torch
require "torch-rb"

# transformer model
require "secryst/multihead_attention"
require "secryst/vocab"
require "secryst/transformer"

require "secryst/translator"

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