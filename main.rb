#!/usr/bin/env ruby

require "numo/linalg/use/openblas"
require "torch-rb"
require "torchtext"
require_relative "transformer"
require_relative "vocab"
require "byebug"

def clip_grad_norm(parameters, max_norm:, norm_type:2)
  parameters = parameters.select {|p| p.grad }
  max_norm = max_norm.to_f
  if parameters.length == 0
    return Torch.tensor(0.0)
  end
  device = parameters[0].grad.device
  if norm_type == Float::INFINITY
    # ...
  else
    total_norm = Numo::Linalg.norm(Numo::NArray.concatenate(parameters.map {|p| Numo::Linalg.norm(p.grad.detach.numo, norm_type)}), norm_type)
  end
  clip_coef = max_norm / (total_norm + 1e-6)
  # puts clip_coef
  if clip_coef < 1
    parameters.each {|p| p.grad = p.grad.detach * clip_coef}
  end
  # puts total_norm
  return total_norm
end

# transformer = Torch::NN::Transformer.new()

# puts transformer.inspect

# input = Torch.randn(10, 32, 512)
# tgt = Torch.randn(20, 32, 512)

# out = transformer.call(input, tgt)

# puts out.inspect

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


@input_vocab = TorchText::Vocab.new(input_vocab_counter)
@target_vocab = TorchText::Vocab.new(target_vocab_counter)
train_txt = input_texts.zip(target_texts)



# def batchify(data, bsz, vocab)
#   numericalized_data = Torch.tensor(data.map {|line| line.chars.map {|c| vocab[c]} })
#   # Divide the dataset into bsz parts.
#   nbatch = numericalized_data.size(0) / bsz
# end

def pad(arr, length, no_eos:false)
  (no_eos ? arr : arr + ["<eos>"]).fill("<pad>", (arr.length+(no_eos ? 0 : 1))...length)
end

# input_tensor = Torch.tensor(input_texts.map {|line| pad(line.chars, max_input_seq_length).map {|c| input_vocab[c]} })

# target_tensor = Torch.tensor(target_texts.map {|line| pad(line.chars, max_target_seq_length).map {|c| target_vocab[c]} })

batches = []
batch_size = 1
( input_texts.length / batch_size ).times do |i|
  input_data = input_texts[i*batch_size, batch_size]
  decoder_input_data = target_texts[i*batch_size, batch_size]
  target_data = target_texts[i*batch_size, batch_size]
  max_input_seq_length = input_data.max_by(&:length).length + 1
  max_target_seq_length = target_data.max_by(&:length).length + 1
  src_mask = Torch.triu(Torch.ones(max_input_seq_length,max_input_seq_length)).eq(0).transpose(0,1)
  tgt_mask = Torch.triu(Torch.ones(max_target_seq_length,max_target_seq_length)).eq(0).transpose(0,1)
  memory_mask = Torch.triu(Torch.ones(max_input_seq_length,max_target_seq_length)).eq(0).transpose(0,1)
  batches << [
    input_data.map {|line| pad(line.chars, max_input_seq_length).map {|c| @input_vocab[c]} },
    target_data.map {|line| pad(line.chars, max_target_seq_length).map {|c| @target_vocab[c]} },
    decoder_input_data.map {|line| pad(line.chars, max_target_seq_length, no_eos: true).map {|c| @target_vocab[c]} },
    src_mask,
    tgt_mask,
    memory_mask
  ]
end
# input_data = input_texts.map {|line| pad(line.chars, max_input_seq_length).map {|c| input_vocab[c]} }
# decoder_input_data = target_texts.map {|line| pad(line.chars, max_target_seq_length, no_eos: true).map {|c| target_vocab[c]} }
# target_data = target_texts.map {|line| pad(line.chars, max_target_seq_length).map {|c| target_vocab[c]} }
# input_data = input_texts.map {|line| pad(line.chars, max_input_seq_length).map {|c| input_vocab[c]} }
# decoder_input_data = target_texts.map {|line| pad(line.chars, max_target_seq_length, no_eos: true).map {|c| target_vocab[c]} }
# target_data = target_texts.map {|line| pad(line.chars, max_target_seq_length).map {|c| target_vocab[c]} }

# puts input_data[0].map {|c| input_vocab.itos[c] }.inspect
# puts target_data[0].map {|c| target_vocab.itos[c] }.inspect
# input_data = input_texts.map {|line| pad(line.chars, max_input_seq_length) }
# target_data = target_texts.map {|line| pad(line.chars, max_target_seq_length) }

device = "cpu"

model = Torch::NN::Transformer.new(d_model: d_model,
  nhead: nhead,
  num_encoder_layers: num_encoder_layers,
  num_decoder_layers: num_decoder_layers,
  dim_feedforward: dim_feedforward,
  dropout: dropout,
  activation: activation,
  input_vocab_size: @input_vocab.length,
  target_vocab_size: @target_vocab.length,
)

# criterion = Torch::NN::CrossEntropyLoss.new.to(device)
criterion = Torch::NN::CrossEntropyLoss.new(ignore_index: 1).to(device)
lr = 5.0 # learning rate
optimizer = Torch::Optim::SGD.new(model.parameters, lr: lr)
scheduler = Torch::Optim::LRScheduler::StepLR.new(optimizer, step_size: 1, gamma: 0.9)

model.train
total_loss = 0.0
start_time = Time.now
ntokens = @target_vocab.length
# training_data = input_data.zip(target_data, decoder_input_data)
epoch = 0


# src_mask = Torch.triu(Torch.ones(max_input_seq_length,max_input_seq_length)).eq(0).transpose(0,1)
# tgt_mask = Torch.triu(Torch.ones(max_target_seq_length,max_target_seq_length)).eq(0).transpose(0,1)
# memory_mask = Torch.triu(Torch.ones(max_input_seq_length,max_target_seq_length)).eq(0).transpose(0,1)

def stringify(arr, s='i')
  if s == 'i'
    arr.map {|c| @input_vocab.itos[c.item]}.join('')
  elsif s == 't'
    arr.map {|c| @target_vocab.itos[c.item]}.join('')
  elsif s == 'p'
    arr.map {|c| @target_vocab.itos[c.argmax.item]}.join('')
  end
end

batches.each.with_index do |batch, i|
  # unzip
  inputs, targets, decoder_inputs, src_mask, tgt_mask, memory_mask = batch
  inputs = Torch.tensor(inputs).t
  decoder_inputs = Torch.tensor(decoder_inputs).t
  targets = Torch.tensor(targets).t
  src_key_padding_mask = inputs.t.eq(1)
  tgt_key_padding_mask = decoder_inputs.t.eq(1)

  # puts "input = #{inputs.t[0]} #{stringify(inputs.t[0])}"
  # puts "out = #{targets.t[0]} #{stringify(targets.t[0], 't')}"

  optimizer.zero_grad
  opts = {
    src_mask: src_mask,
    tgt_mask: tgt_mask,
    memory_mask: memory_mask,
    src_key_padding_mask: src_key_padding_mask,
    tgt_key_padding_mask: tgt_key_padding_mask,
  }
  output = model.call(inputs, decoder_inputs, opts)
  loss = criterion.call(output.view(-1, ntokens), targets.t.view(-1))
  loss.backward
  clip_grad_norm(model.parameters, max_norm: 0.5)
  optimizer.step
  # puts "before = #{output.view(-1, ntokens)[0..2000].map {|c| c.argmax.item}}"
  # puts "after = #{model.call(inputs, decoder_inputs, opts).view(-1, ntokens)[0..2000].map {|c| c.argmax.item}}"
  # puts "after.itos = #{model.call(inputs, decoder_inputs, opts).view(-1, ntokens)[0..2000].map {|c| @target_vocab.itos[c.argmax.item]}}"
  # puts "predicting with #{Torch.full(decoder_inputs.size, 1, dtype: :long)}"
  byebug
  # pred = model.call(inputs, Torch.full(decoder_inputs.size, 1, dtype: :long), opts).view(-1, ntokens)[0..2000]
  # puts "first pred = #{pred.map {|c| c.argmax.item}} #{stringify(pred, 'p')}"
  puts "i[#{i}] loss: #{loss}"
  total_loss += loss.item()

  log_interval = 200
  if i > 0 && ( i % log_interval == 0 || i == batches.length )
    cur_loss = total_loss / log_interval
    elapsed = Time.now - start_time
    puts "| epoch #{epoch} | #{i}/#{batches.length} batches |"\
          "lr #{scheduler.get_lr()[0]} | ms/batch #{elapsed.to_i / log_interval} | "\
          "loss #{cur_loss} | ppl #{Math.exp(cur_loss)}"
    total_loss = 0
    start_time = Time.now

    puts "saving model"
    Torch.save(model.state_dict, "net-#{i}.pth")
  end
end
Torch.save(model.state_dict, "net-last.pth")

# best_val_loss = 0
# epochs = 3 # The number of epochs
# best_model = nil

# epochs.times do |epoch|
#   epoch_start_time = Time.now()
#   train(epoch)
# end