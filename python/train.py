from zipfile import ZipFile, ZIP_DEFLATED
from io import BytesIO
import torch
import time
import numpy
import math
import yaml
import sys
from vocab import Vocab
import random
import re
import os
import transformer
import pickle
from pathlib import Path
import json
import argparse

default_hyperparameters = {
  'd_model': 64,
  'nhead': 8,
  'num_encoder_layers': 4,
  'num_decoder_layers': 4,
  'dim_feedforward': 256,
  'dropout': 0.05,
  'activation': 'relu',
}

parser = argparse.ArgumentParser(
    description="Train the model on dataset",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument("-m", "--model", help="Model architecture: currently only transformer", default="transformer")
parser.add_argument("-i", "--data-input", type=Path,
                    help="Specify file with input sequences", required=True)

parser.add_argument("-t", "--data-target", type=Path,
                    help="Specify file with target sequences", required=True)
parser.add_argument("-x", "--hyperparameters", type=json.loads, help="Specify model hyperparameters", default='{}')
parser.add_argument("-e", "--max-epochs", type=int, help="Maximum epochs count", default=500)
parser.add_argument("-l", "--log-interval", type=int, help="Training logging interval", default=10)
parser.add_argument("-c", "--checkpoint-every", type=int, help="Save checkpoint every N epochs", default=1)
parser.add_argument("--checkpoint-dir", type=Path, help="Directory where to save checkpoints", default=Path("./checkpoints"))
parser.add_argument("--scheduler-step-size", help="Scheduler step size", type=int, default=70)
parser.add_argument("--gamma", help="Scheduler gamma", type=float, default=0.9)
parser.add_argument("-b", "--batch-size", help="Batch size", type=int, default=32)
parser.add_argument("--lr", help="Learning rate", type=float, default=0.1)
parser.add_argument("-d", "--device", help="Device: CPU or GPU", default="cpu")

## TODO: Test all arguments

args = parser.parse_args()


f = open(args.data_input)
data_input = f.readlines()
f.close()

f = open(args.data_target)
data_target = f.readlines()
f.close()

device = args.device
lr = args.lr
scheduler_step_size = args.scheduler_step_size
gamma = args.gamma
batch_size = args.batch_size
model_name = args.model
max_epochs = args.max_epochs
log_interval = args.log_interval
checkpoint_every = args.checkpoint_every
checkpoint_dir = args.checkpoint_dir

checkpoint_dir.mkdir(parents=True, exist_ok=True)

def find_checkpoint_index(n):
  matches = re.findall("checkpoint-([0-9]+)", n)
  if len(matches) > 0:
    return int(matches[0])
  else:
    return 0

last_checkpoint = None
sorted_checkpoints = sorted(
    filter(lambda f: re.match("checkpoint-([0-9]+)\.zip", f), os.listdir(checkpoint_dir)),
   key=find_checkpoint_index)
if len(sorted_checkpoints) > 0:
  last_checkpoint = sorted_checkpoints[-1]

if last_checkpoint:
  print(f'Starting from checkpoint {last_checkpoint}')
  initial_epoch = find_checkpoint_index(last_checkpoint) + 1
else:
  initial_epoch = 0

input_texts = []
target_texts = []
input_vocab_counter = {}
target_vocab_counter = {}

for input_text in data_input:
  input_text = input_text.strip()
  input_texts.append(input_text)
  for char in input_text:
    if char in input_vocab_counter:
      input_vocab_counter[char] += 1
    else:
      input_vocab_counter[char] = 1

for target_text in data_target:
  target_text = target_text.strip()
  target_texts.append(target_text)
  for char in target_text:
    if char in target_vocab_counter:
      target_vocab_counter[char] += 1
    else:
      target_vocab_counter[char] = 1


input_vocab = Vocab(list(input_vocab_counter.keys()))
target_vocab = Vocab(list(target_vocab_counter.keys()))

# Generate train, eval, and test batches
seed = 1
zipped_texts = list(zip(input_texts, target_texts))
random.Random(seed).shuffle(zipped_texts)



# # train - 90%, eval - 7%, test - 3%
train_texts = zipped_texts[0:int(len(zipped_texts)*0.9)]
eval_texts = zipped_texts[int(len(zipped_texts)*0.9) + 1:int(len(zipped_texts)*0.97)]
test_texts = zipped_texts[int(len(zipped_texts)*0.97)+1:-1]

# prepare batches
def pad(arr, length, no_eos=False, no_sos=False):
  if not no_eos:
    arr = arr + ["<eos>"]
  if not no_sos:
    arr = ["<sos>"] + arr

  return numpy.pad(arr, (0, length - len(arr)), constant_values="<pad>")

def prepare_line(seq_length, vocab, no_sos=False, no_eos=False):
  return lambda line: list(map(
      lambda c: vocab[c],
      pad(list(line), seq_length, no_sos=no_sos, no_eos=no_eos)
    ))

def batchify(data, bsz):
  batches = []
  # do loop at least one time
  for i in range(max(1,int(len(data) / bsz))):
    input_data, decoder_input_data = list(zip(*data[i*bsz:i*bsz + bsz]))
    target_data = decoder_input_data
    max_input_seq_length = len(max(input_data, key=len)) + 2
    max_target_seq_length = len(max(target_data, key=len)) + 1
    src_mask = torch.triu(torch.ones(
        max_input_seq_length, max_input_seq_length)).eq(0).transpose(0, 1)
    tgt_mask = torch.triu(torch.ones(
        max_target_seq_length, max_target_seq_length)).eq(0).transpose(0, 1)
    memory_mask = torch.triu(torch.ones(
        max_input_seq_length, max_target_seq_length)).eq(0).transpose(0, 1)

    batches.append([
      list(map(
        prepare_line(max_input_seq_length, input_vocab),
        list(input_data)
      )),
      list(map(
        prepare_line(max_target_seq_length, target_vocab, no_sos=True),
        list(target_data)
      )),
      list(map(
        prepare_line(max_target_seq_length, target_vocab, no_eos=True),
        list(target_data)
      )),
      src_mask,
      tgt_mask,
      memory_mask
    ])


  return batches


train_data = batchify(train_texts, batch_size)
eval_data = batchify(eval_texts, batch_size)
test_data = batchify(test_texts, batch_size)

hyperparameters = {
  **default_hyperparameters,
  **args.hyperparameters, 
  "input_vocab_size": len(input_vocab),
  "target_vocab_size": len(target_vocab),
}

with open(Path(checkpoint_dir, 'vocabs.yaml'), 'w') as outfile:
    yaml.dump({
      "input": input_vocab.itos,
      "target": target_vocab.itos
    }, outfile)

with open(Path(checkpoint_dir, 'metadata.yaml'), 'w') as outfile:
    yaml.dump({
      "name": "transformer",
      **{f':{k}': v for k, v in hyperparameters.items()} # transform keys for ruby
    }, outfile)


assert model_name == "transformer", "Only transformer model is currently supported"

# load checkpoint data to resume
if last_checkpoint:
  input_zip_file = ZipFile(Path(checkpoint_dir, last_checkpoint))
  assert "metadata.yaml" in input_zip_file.namelist(), "metadata.yaml is missing in model zip!"
  assert "vocabs.yaml" in input_zip_file.namelist(), "vocabs.yaml is missing in model zip!"

  metadata = yaml.safe_load(input_zip_file.read('metadata.yaml'))
  vocabs = yaml.safe_load(input_zip_file.read('vocabs.yaml'))
  input_vocab = Vocab(vocabs['input'], specials=[])
  target_vocab = Vocab(vocabs['target'], specials=[])

  if "model.pth" in input_zip_file.namelist():
    model_dict = torch.load(BytesIO(input_zip_file.read('model.pth')))
    new_dict = {}
    # transforming the keys of model dict so that they
    # fit well to torch Transformer
    for key, value in model_dict.items():
      transformed_key = re.sub(r"layer(\d+)\.", r"layers.\1.", key)
      new_dict[transformed_key] = value
  else:
    print("model.pth is missing in model zip!")
    exit()

  model = transformer.TransformerModel(
    metadata[':input_vocab_size'],
    metadata[':target_vocab_size'],
    d_model=metadata[':d_model'],
    nhead=metadata[':nhead'],
    dim_feedforward=metadata[':dim_feedforward'],
    num_encoder_layers=metadata[':num_encoder_layers'],
    num_decoder_layers=metadata[':num_decoder_layers'],
    dropout=metadata[':dropout'],
    activation=metadata[':activation']
  )

  model.load_state_dict(new_dict)
else:
  # do not pass parameters that are not present, for defaults to pick up
  _h = {k: hyperparameters.get(k) for k, v in hyperparameters.items() if k not in ['input_vocab_size', 'target_vocab_size'] and v is not None}
  model = transformer.TransformerModel(
    hyperparameters['input_vocab_size'],
    hyperparameters['target_vocab_size'],
    **_h
   )

def save_model(epoch):
  start_saving = time.time()
  state_dict = {k: v.data if isinstance(
      v, torch.nn.Parameter) else v for k, v in model.state_dict().items()}
  torch.save(state_dict, Path(checkpoint_dir, "model.pth"),
             _use_new_zipfile_serialization=True)

  # Zip generation
  input_filenames = ['model.pth', 'metadata.yaml', 'vocabs.yaml']
  zipfile_name = Path(checkpoint_dir, f"checkpoint-{epoch}.zip")
  if os.path.exists(zipfile_name):
    os.remove(zipfile_name)

  output_zip_file = ZipFile(zipfile_name, 'w', compression=ZIP_DEFLATED)

  for filename in input_filenames:
    output_zip_file.write(Path(checkpoint_dir, filename), filename)

  output_zip_file.close()

  print(f">> Saved checkpoint '{checkpoint_dir}/checkpoint-{epoch}.zip' in {round(time.time() - start_saving, 3)}s")

## Training
best_model = None
best_val_loss = float("inf")

criterion = torch.nn.CrossEntropyLoss(ignore_index=target_vocab.stoi["<pad>"])
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=gamma)

total_loss = 0.0
start_time = time.time()
ntokens = len(target_vocab)

model = model.to(device)

for epoch in range(initial_epoch, max_epochs + 1):
  epoch_start_time = time.time()
  model.train()
  for i, batch in enumerate(train_data):
    inputs, targets, decoder_inputs, src_mask, tgt_mask, memory_mask = batch
    inputs = list(inputs)
    targets = list(targets)
    decoder_inputs = list(decoder_inputs)
    inputs = torch.tensor(inputs).t()
    decoder_inputs = torch.tensor(decoder_inputs).t()
    targets = torch.tensor(targets).t()
    src_key_padding_mask = inputs.t().eq(1)
    tgt_key_padding_mask = decoder_inputs.t().eq(1)
    optimizer.zero_grad()
    output = model(inputs.to(device), decoder_inputs.to(device), 
      # src_mask=src_mask,
      tgt_mask=tgt_mask.to(device),
      # memory_mask=memory_mask,
      src_key_padding_mask=src_key_padding_mask.to(device),
      tgt_key_padding_mask=tgt_key_padding_mask.to(device),
      memory_key_padding_mask=src_key_padding_mask.to(device),
    )
    loss = criterion(output.transpose(0,1).reshape(-1, ntokens), targets.t().view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()

    total_loss += loss.item()
    if (i + 1) % log_interval == 0:
      cur_loss = total_loss / log_interval
      elapsed = time.time() - start_time
      print('| epoch {:3d} | {:5d}/{:5d} batches | '
            'lr {:02.2f} | ms/batch {:5f} | '
            'loss {:5.4f} | ppl {:8.2f}'.format(
                epoch, i + 1, len(
                    train_data), scheduler.get_last_lr()[0],
                elapsed * 1000 / log_interval,
                cur_loss, math.exp(cur_loss)))
      total_loss = 0
      start_time = time.time()
    
  if epoch > 0 and epoch % checkpoint_every == 0:
    save_model(epoch)

  # Evaluate
  model.eval()
  total_loss = 0.0
  with torch.no_grad():
    for i, batch in enumerate(eval_data):
      inputs, targets, decoder_inputs, src_mask, tgt_mask, memory_mask = batch
      inputs = list(inputs)
      targets = list(targets)
      decoder_inputs = list(decoder_inputs)
      inputs = torch.tensor(inputs).t()
      decoder_inputs = torch.tensor(decoder_inputs).t()
      targets = torch.tensor(targets).t()
      src_key_padding_mask = inputs.t().eq(1)
      tgt_key_padding_mask = decoder_inputs.t().eq(1)
      output = model(inputs.to(device), decoder_inputs.to(device),
                     # src_mask=src_mask,
                     tgt_mask=tgt_mask.to(device),
                     # memory_mask=memory_mask,
                     src_key_padding_mask=src_key_padding_mask.to(device),
                     tgt_key_padding_mask=tgt_key_padding_mask.to(device),
                     memory_key_padding_mask=src_key_padding_mask.to(device),
                     )
      output_flat = output.transpose(0,1).reshape(-1, ntokens)

      total_loss += criterion(output_flat, targets.t().view(-1)).item()
    total_loss = total_loss / len(eval_data)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     total_loss, math.exp(total_loss)))
    print('-' * 89)

    if total_loss < best_val_loss:
      # Question: here we really save best model or only save reference?
      # taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
      best_model = model
      best_val_loss = total_loss

  scheduler.step()

os.remove(Path(checkpoint_dir, "model.pth"))
os.remove(Path(checkpoint_dir, "metadata.yaml"))
os.remove(Path(checkpoint_dir, "vocabs.yaml"))
