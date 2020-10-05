require "secryst"

kh2en = Secryst::Translator.new({
  model: 'transformer',
  vocabs_dir: 'checkpoints',
  model_file: 'checkpoints/checkpoint-120.pth',
  hyperparameters: {
    d_model: 64,
    nhead: 8,
    num_encoder_layers: 4,
    num_decoder_layers: 4,
    dim_feedforward: 256, # 128
    dropout: 0.05, # 0.1
    activation: 'relu'
  },
})

puts kh2en.translate('បាត់ទៅណា?')
puts kh2en.translate('ប្រាសាទ')
puts kh2en.translate('អោយ')
puts kh2en.translate('អង')
