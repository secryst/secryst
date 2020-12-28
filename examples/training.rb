#!/usr/bin/env ruby

require "secryst-trainer"

kh2en = Secryst::Trainer.new(
  model: 'transformer',
  data_input: File.expand_path('../examples/khm-latn-small/input.csv', __dir__),
  data_target: File.expand_path('../examples/khm-latn-small/target.csv', __dir__),
  batch_size: 32,
  lr: 0.1,
  scheduler_step_size: 70,
  gamma: 0.9,
  max_epochs: 500, # nil for unlimited
  checkpoint_every: 50,
  checkpoint_dir: File.expand_path('../examples/khm-latn-small/checkpoints', __dir__),
  hyperparameters: {
    d_model: 64,
    nhead: 8,
    num_encoder_layers: 4,
    num_decoder_layers: 4,
    dim_feedforward: 256,
    dropout: 0.05,
    activation: 'relu',
  }
)

kh2en.train