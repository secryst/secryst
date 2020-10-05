# coding: utf-8
lib = File.expand_path('../lib', __FILE__)
$LOAD_PATH.unshift(lib) unless $LOAD_PATH.include?(lib)
require 'secryst/version'

Gem::Specification.new do |spec|
  spec.name          = "secryst"
  spec.version       = Secryst::VERSION
  spec.summary       = "Seq2seq transformer for transliteration in Ruby."
  spec.description   = %q{Seq2seq transformer for transliteration in Ruby.}
  spec.homepage      = "https://github.com/secryst/secryst"
  spec.license       = "MIT"

  spec.authors       = ['project_contibutors']

  spec.files         = Dir.glob("{lib,exe,spec,maps}/**/*", File::FNM_DOTMATCH)
  spec.files         += ['README.adoc']
  spec.require_path  = "lib"
  spec.bindir        = "bin"
  spec.executables   = spec.files.grep(%r{^bin/}) { |f| File.basename(f) }

  spec.required_ruby_version = ">= 2.7"

  spec.add_dependency "torch-rb", ">= 0.4.0"
end
