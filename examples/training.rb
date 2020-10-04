require "secryst"

kh2en = Secryst::Trainer.new({
  model: 'transformer',
  data: 'kh-rom',
  batch_size: 32,
  lr: 0.1,
  scheduler_step_size: 70,
  gamma: 0.9,
  max_epochs: nil,
  checkpoint_every: 50,
  checkpoint_dir: File.expand_path('./checkpoints', __dir__),
  hyperparameters: {
    d_model: 64,
    nhead: 8,
    num_encoder_layers: 4,
    num_decoder_layers: 4,
    dim_feedforward: 256,
    dropout: 0.05,
    activation: 'relu',
  },

})

kh2en.train()