from zipfile import ZipFile
from io import BytesIO
import torch
import yaml
import sys
import re
import transformer

zip_path = sys.argv[1]
zip_file = ZipFile(zip_path)
metadata = yaml.safe_load(zip_file.read('metadata.yaml'))


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


#=# transformer_model = transformer(
#   d_model=metadata[':d_model'], 
#   nhead=metadata[':nhead'], 
#   num_encoder_layers=metadata[':num_encoder_layers'],
#   num_decoder_layers=metadata[':num_decoder_layers'],
#   dim_feedforward=metadata[':dim_feedforward'],
#   dropout=metadata[':dropout'],
#   activation=metadata[':activation']
# )

model_dict = torch.load(BytesIO(zip_file.read('model.pth')))
# print(model_dict.keys())
# exit
new_dict = {}

# transforming the keys of model dict so that they
# fit well to torch Transformer
for key, value in model_dict.items():
  transformed_key = re.sub(r"layer(\d+)\.", r"layers.\1.", key)
  new_dict[transformed_key] = value


# print(torch.load(BytesIO(model_dict)).__class__)

transformer_model.load_state_dict(new_dict)

batch_size = 16
source_length = 20
target_length = 16

src = torch.randint(
    0, metadata[':input_vocab_size'], (source_length, batch_size))
tgt = torch.randint(
    0, metadata[':target_vocab_size'], (target_length, batch_size))
out = transformer_model(src,tgt)
print(metadata)
print(src.size())
print(out.size())
# quit()
# Export the model
torch.onnx.export(transformer_model,               # model being run
                  # model input (or a tuple for multiple inputs)
                  (src, tgt),
                  # where to save the model (can be a file or file-like object)
                  "transformer.onnx",
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['src', 'tgt'],   # the model's input names
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'src': {0: 'source_length', 1: 'batch_size'},    # variable length axes
                                'tgt': {0: 'target_length', 1: 'batch_size'},
                                'output': {0: 'target_length', 1: 'batch_size'}})


# значит там неправильно названы лейера. ок, щас. пофиксил. теперь у нас лишний линеар. пач?
# 5 1 64
# 5 1 64
