require "numo/linalg/use/openblas"
require "torch-rb"
require "torchtext"
require_relative "transformer"
require_relative "vocab"
require "byebug"

d_model = 64
nhead = 8
num_encoder_layers = 6
num_decoder_layers = 6
dim_feedforward = 2048 # 128
dropout = 0.01 # 0.1
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
  input_vocab_size: 80,
  target_vocab_size: 72,
)

model.load_state_dict(Torch.load('net-1200.pth'))
model.eval


# input = input_data[0]
input = ["ប", "ា", "ន", "<eos>"]
puts "Testing model with input:"
puts input.join('')
puts input.map {|i| input_vocab.stoi[i]}.select {|i| i != '<pad>' && i != '<eos>'}.inspect
output = Torch.full([4,1], 1, dtype: :long)
output[0][0] = 0
# output = Torch.tensor([target_data[0]]).t
# puts output.map {|i| target_vocab.itos[i.item]}.join('').inspect
input = Torch.tensor([input.map {|i| input_vocab.stoi[i]}]).t

45.times do |i|
  src_key_padding_mask = input.t.eq(1)
  tgt_key_padding_mask = output.t.eq(1)
  src_mask = Torch.triu(Torch.ones(input.size(0),input.size(0))).eq(0).transpose(0,1)
  tgt_mask = Torch.triu(Torch.ones(output.size(0),output.size(0))).eq(0).transpose(0,1)
  memory_mask = Torch.triu(Torch.ones(input.size(0),output.size(0))).eq(0).transpose(0,1)
  opts = {
    src_mask: src_mask,
    tgt_mask: tgt_mask,
    memory_mask: memory_mask,
    src_key_padding_mask: src_key_padding_mask,
    tgt_key_padding_mask: tgt_key_padding_mask,
  }
  prediction = model.call(input, output, opts).map {|i| i.argmax.item }
  # output = Torch.tensor([output.map {|i| i.argmax }], dtype: :long).t
  puts i, prediction.map {|i| target_vocab.itos[i]}.join('').inspect
  output[0][i] = prediction[i]
end
