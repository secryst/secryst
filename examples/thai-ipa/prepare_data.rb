require 'fileutils'

lines = File.readlines('./data/wiktionary-11-2-2020.tsv')
FileUtils.rm_rf('./data/input.csv')
FileUtils.rm_rf('./data/target.csv')

lines.each do |line|
  input, target = line.strip.split("\t")
  File.write('./data/input.csv', input + "\n", mode: 'a')
  File.write('./data/target.csv', target + "\n", mode: 'a')
end
