#!/usr/bin/ruby

require "torch-rb"
require_relative "transformer"
require_relative "vocab"

d_model = 64
nhead = 8
num_encoder_layers = num_decoder_layers = 4
dim_feedforward = 256 # 128
dropout = 0.05 # 0.1
activation = 'relu'


data_kh = File.readlines('data_kh.csv', chomp: true)
data_rom = File.readlines('data_rom.csv', chomp: true)
epochs = 3
batch_size = 32
eval_batch_size = 16

input_texts = []
target_texts = []
input_vocab_counter = Hash.new(0)
target_vocab_counter = Hash.new(0)
# i = k = 0

data_kh.each do |input_text|
  input_text.strip!
  input_texts.push(input_text)
  input_text.each_char do |char|
    input_vocab_counter[char] += 1
  end
end

data_rom.each do |target_text|
  target_text.strip!
  # target_text = "\t" + target_text.strip + "\n"
  target_texts.push(target_text)
  target_text.each_char do |char|
    target_vocab_counter[char] += 1
  end
end

# num_encoder_tokens = input_characters.length
# num_decoder_tokens = target_characters.length


input_vocab = TorchText::Vocab.new(input_vocab_counter)
target_vocab = TorchText::Vocab.new(target_vocab_counter)
train_txt = input_texts.zip(target_texts)
max_input_seq_length = input_texts.max_by(&:length).length
max_target_seq_length = target_texts.max_by(&:length).length


def pad(arr, length)
  (arr + ["<eos>"]).fill("<pad>", (arr.length+1)...length)
end

input_data = input_texts.map {|line| pad(line.chars, max_input_seq_length).map {|c| input_vocab[c]} }
target_data = target_texts.map {|line| pad(line.chars, max_target_seq_length).map {|c| target_vocab[c]} }

device = "cpu"

model = Torch::NN::Transformer.new(d_model: d_model,
  nhead: nhead,
  num_encoder_layers: num_encoder_layers,
  num_decoder_layers: num_decoder_layers,
  dim_feedforward: dim_feedforward,
  dropout: dropout,
  activation: activation,
  input_vocab_size: input_vocab.length,
  target_vocab_size: target_vocab.length,
)

model.load_state_dict(Torch.load('net-last.pth'))
model.eval



loop do
  puts "Enter some Khmer phrase to transliterate:"

  input = ['<sos>'] + gets().chars + ['<eos>']
  input = Torch.tensor([input.map {|i| input_vocab.stoi[i]}]).t
  output = Torch.tensor([[target_vocab.stoi['<sos>']]])

  100.times do |i|
    begin
    src_key_padding_mask = input.t.eq(1)
    tgt_key_padding_mask = output.t.eq(1)
    tgt_mask = Torch.triu(Torch.ones(i+1,i+1)).eq(0).transpose(0,1)
    opts = {
      tgt_mask: tgt_mask,
      src_key_padding_mask: src_key_padding_mask,
      tgt_key_padding_mask: tgt_key_padding_mask,
      memory_key_padding_mask: src_key_padding_mask,
    }
    prediction = model.call(input, output, opts).map {|i| i.argmax.item }
    break if target_vocab.itos[prediction[i]] == '<eos>'
    output = Torch.cat([output, Torch.tensor([[prediction[i]]])])
    rescue StandardError => e
    end
  end
  puts "#{output[1..-1].map {|i| target_vocab.itos[i.item]}.join('')}"
end
