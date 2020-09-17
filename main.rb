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
    parameters.each {|p| p.grad.detach * clip_coef}
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

d_model = 512
nhead = 8
num_encoder_layers = 6
num_decoder_layers = 6
dim_feedforward = 2048
dropout = 0.1
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


# def batchify(data, bsz, vocab)
#   numericalized_data = Torch.tensor(data.map {|line| line.chars.map {|c| vocab[c]} })
#   # Divide the dataset into bsz parts.
#   nbatch = numericalized_data.size(0) / bsz
# end

def pad(arr, length)
  (arr + ["<eos>"]).fill("<pad>", (arr.length+1)...length)
end

# input_tensor = Torch.tensor(input_texts.map {|line| pad(line.chars, max_input_seq_length).map {|c| input_vocab[c]} })

# target_tensor = Torch.tensor(target_texts.map {|line| pad(line.chars, max_target_seq_length).map {|c| target_vocab[c]} })

input_data = input_texts.map {|line| pad(line.chars, max_input_seq_length).map {|c| input_vocab[c]} }

target_data = target_texts.map {|line| pad(line.chars, max_target_seq_length).map {|c| target_vocab[c]} }
puts input_data[0].map {|c| input_vocab.itos[c] }.inspect
puts target_data[0].map {|c| target_vocab.itos[c] }.inspect
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
  input_vocab_size: input_vocab.length,
  target_vocab_size: target_vocab.length,
)

criterion = Torch::NN::CrossEntropyLoss.new.to(device)
lr = 5.0 # learning rate
optimizer = Torch::Optim::SGD.new(model.parameters, lr: lr)
scheduler = Torch::Optim::LRScheduler::StepLR.new(optimizer, step_size: 1, gamma: 0.9)

model.train()
total_loss = 0.0
start_time = Time.now
ntokens = target_vocab.length
training_data = input_data.zip(target_data)
i = 0
training_data.each_slice(batch_size) do |batch|
  # unzip
  inputs, targets = batch.transpose
  inputs = Torch.tensor(inputs).t
  targets = Torch.tensor(targets).t
  optimizer.zero_grad()
  output = model.call(inputs, targets)
  loss = criterion.call(output.view(-1, ntokens), targets.t.contiguous.view(-1))
  # output.view(-1, ntokens).map(&:to_a).each {|l| puts l.inspect}
  # output.view(-1, ntokens).map(&:to_a).each {|l| puts l.inspect}
  # puts targets.t.contiguous.view(-1).to_a.inspect
  # exit
  # puts output.view(-1, ntokens).map(&:to_a)[-1].inspect
  puts loss, i
  loss.backward()
  clip_grad_norm(model.parameters, max_norm: 0.5)
  optimizer.step
  total_loss += loss.item()
  log_interval = 200
  if i % log_interval == 0 && i > 0
    cur_loss = total_loss / log_interval
    elapsed = Time.now - start_time
    puts "| epoch #{epoch} | #{i}/#{training_data.length} batches |"\
          "lr #{scheduler.get_lr()[0]} | ms/batch #{elapsed * 1000 / log_interval} | "\
          "loss #{cur_loss} | ppl #{Math.exp(cur_loss)}"
    total_loss = 0
    start_time = time.time()

    puts "saving model"
    Torch.save(model.state_dict, "net-#{i}.pth")
  end
  i += 1
end

byebug

# train_data = batchify(target_texts, batch_size, input_vocab)
# val_data = batchify(val_txt, eval_batch_size)
# test_data = batchify(test_txt, eval_batch_size)



