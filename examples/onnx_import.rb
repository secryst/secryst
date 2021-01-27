#!/usr/bin/env ruby
require "secryst"
require "json"

# puts "Downloading ONNX Bert Base Uncased model..."
# system('wget ...')
puts "Converting Bert Base Uncased model to ONNX..."
system("python3 -m transformers.convert_graph_to_onnx #{`python3 -c "import transformers as _; print(_.__path__[0])"`.strip}/convert_graph_to_onnx.py --framework pt --pipeline fill-mask --model bert-base-uncased ./bert.onnx")

puts "Downloading Bert tokenizer..."
system('wget https://cdn.huggingface.co/bert-base-uncased-tokenizer.json')

# Make vocabs.yaml from tokenizer
vocab = JSON.parse(File.read('./bert-base-uncased-tokenizer.json'))["model"]["vocab"]
new_vocab = {
  "input" => vocab.keys,
  "target" => vocab.keys
}
File.write('./vocabs.yaml', new_vocab.to_yaml)

# Zip it
input_filenames = ['bert.onnx', 'vocabs.yaml']
zipfile_name = "./model.zip"
FileUtils.rm(zipfile_name) if File.exists?(zipfile_name)
Zip::File.open(zipfile_name, Zip::File::CREATE) do |zipfile|
  input_filenames.each do |filename|
    zipfile.add(filename, File.join('.', filename))
  end
end


bert = Secryst::Translator.new({
  model_file: './model.zip',
})

# Utilize ONNX model for mask filling scenario
input_sentence = ['[CLS]', 'this', 'looks', 'like', 'a', 'wonderful', '[MASK]', '.', '[SEP]']
attn_mask = input_sentence.map { 1 }
input_sentence.map! {|t| bert.model.input_vocab.stoi[t]}
token_type_ids = input_sentence.map { 0 }

res = bert.model.model.predict({input_ids: [input_sentence], attention_mask: [attn_mask], token_type_ids: [token_type_ids]})

# Decode the output
output = res["output_0"][0].map {|part| part.each_with_index.max[1]}
res_tokens = output.map {|t| bert.model.target_vocab.itos[t]}
puts res_tokens.inspect

raise 'ONNX model output is wrong' if res_tokens != ['.', 'this', 'looks', 'like', 'a', 'wonderful', 'place', '.', '.']