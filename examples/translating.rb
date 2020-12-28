#!/usr/bin/env ruby

require "secryst"

kh2en = Secryst::Translator.new({
  model_file: File.expand_path('../examples/khm-latn-small/checkpoints/checkpoint-500.zip', __dir__)
})

puts kh2en.translate('បាត់ទៅណា?')
puts kh2en.translate('ប្រាសាទ')
puts kh2en.translate('អោយ')
puts kh2en.translate('អង')
