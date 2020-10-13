# coding: utf-8
lib = File.expand_path('../lib', __FILE__)
$LOAD_PATH.unshift(lib) unless $LOAD_PATH.include?(lib)
require 'secryst/version'

Gem::Specification.new do |spec|
  spec.name          = "secryst-trainer"
  spec.version       = Secryst::VERSION
  spec.summary       = "A seq2seq transformer suited for transliteration. Written in Ruby. Includes packages for training models"
  spec.homepage      = "https://github.com/secryst/secryst"
  spec.license       = "BSD-2-Clause"

  spec.authors       = ['project_contibutors']

  spec.files         = Dir.glob("{lib,exe,spec,maps}/**/*", File::FNM_DOTMATCH)
  spec.files         += ['README.adoc']
  spec.require_path  = "lib"
  spec.bindir        = "bin"
  spec.executables   = spec.files.grep(%r{^bin/}) { |f| File.basename(f) }
  spec.executables   << "secryst-trainer"

  spec.required_ruby_version = ">= 2.6"

  spec.add_dependency "torch-rb", '~> 0.4'
  spec.add_dependency "numo", '~> 0.1'
  spec.add_dependency "numo-linalg", '~> 0.1'
  spec.add_dependency "thor", "~> 1.0"
  spec.add_dependency "secryst", Secryst::VERSION

  spec.add_development_dependency "rake"
  spec.add_development_dependency "rspec"
end
