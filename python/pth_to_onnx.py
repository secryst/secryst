from zipfile import ZipFile
from io import BytesIO
import torch
import yaml
import sys
import re
import os
import transformer

if len(sys.argv[1]) < 2:
  sys.exit('Input model path not specified')

if len(sys.argv) < 3:
  sys.exit('Output path not specified')

input_zip_path = sys.argv[1]
input_zip_file = ZipFile(input_zip_path)
metadata = yaml.safe_load(input_zip_file.read('metadata.yaml'))


transformer_model = transformer.TransformerModel(
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


model_dict = torch.load(BytesIO(input_zip_file.read('model.pth')))
new_dict = {}

# transforming the keys of model dict so that they
# fit well to torch Transformer
for key, value in model_dict.items():
  transformed_key = re.sub(r"layer(\d+)\.", r"layers.\1.", key)
  new_dict[transformed_key] = value


transformer_model.load_state_dict(new_dict)
transformer_model.eval()
batch_size = 16
source_length = 8 # TODO: Let's see if this size will make a problem for sequences longer than that
target_length = 8

src = torch.randint(
    0, metadata[':input_vocab_size'], (source_length, batch_size))
tgt = torch.randint(
    0, metadata[':target_vocab_size'], (target_length, batch_size))
tgt_mask = torch.randint(
    0, 2, (target_length, target_length)).bool()
src_key_padding_mask = torch.randint(
    0, 2, (batch_size, source_length)).bool()
tgt_key_padding_mask = torch.randint(
    0, 2, (batch_size, target_length)).bool()
memory_key_padding_mask = torch.randint(
    0, 2, (batch_size, source_length)).bool()

out = transformer_model(src, tgt, tgt_mask=tgt_mask,
                        src_key_padding_mask=src_key_padding_mask,
                        tgt_key_padding_mask=tgt_key_padding_mask,
                        memory_key_padding_mask=memory_key_padding_mask)

# Export the model
torch.onnx.export(transformer_model,               # model being run
                  # model input (or a tuple for multiple inputs)
                  (src, tgt, tgt_mask,
                   src_key_padding_mask,
                   tgt_key_padding_mask,
                   memory_key_padding_mask),
                  # where to save the model (can be a file or file-like object)
                  "transformer.onnx",
                  verbose=False,
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=False,  # whether to execute constant folding for optimization
                  input_names=['src', 'tgt', 'tgt_mask',
                               'src_key_padding_mask',
                               'tgt_key_padding_mask',
                               'memory_key_padding_mask'],   # the model's input names
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'src': {0: 'source_length', 1: 'batch_size'},    # variable length axes
                                'tgt': {0: 'target_length', 1: 'batch_size'},
                                'output': {0: 'target_length', 1: 'batch_size'},
                                'tgt_mask': {0: 'target_length', 1: 'target_length'},
                                'src_key_padding_mask': {0: 'batch_size', 1: 'source_length'},
                                'tgt_key_padding_mask': {0: 'batch_size', 1: 'target_length'},
                                'memory_key_padding_mask': {0: 'batch_size', 1: 'source_length'},
                                })


output_zip_path = sys.argv[2]
output_zip_file = ZipFile(output_zip_path, 'w')

output_zip_file.writestr('metadata.yaml', input_zip_file.read('metadata.yaml'))
output_zip_file.writestr('vocabs.yaml', input_zip_file.read('vocabs.yaml'))
output_zip_file.write('transformer.onnx')

os.remove('transformer.onnx')

output_zip_file.close()
input_zip_file.close()