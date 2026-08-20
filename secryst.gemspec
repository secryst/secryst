# coding: utf-8
lib = File.expand_path('../lib', __FILE__)
$LOAD_PATH.unshift(lib) unless $LOAD_PATH.include?(lib)
require 'secryst/version'

Gem::Specification.new do |spec|
  spec.name          = "secryst"
  spec.version       = Secryst::VERSION
  spec.summary       = "Ruby crystal: local ONNX vocalization/G2P implementing the interscript-ml contract."
  spec.description   = %q{Secryst (scrying + crystal) reveals the hidden reading of a script — diacritization, vocalization, grapheme-to-phoneme — via local, sha256-verified IMF v1 ONNX models. Implements the interscript-ml contract (models.yaml index, byte tokenizer, golden parity). Sibling crystals: pip install secryst, npm i secryst.}
  spec.homepage      = "https://www.secryst.org"
  spec.metadata      = {
    "homepage_uri" => "https://www.secryst.org",
    "source_code_uri" => "https://github.com/secryst/secryst",
    "changelog_uri" => "https://github.com/secryst/secryst/blob/master/README.adoc",
  }
  spec.license       = "BSD-2-Clause"

  spec.authors       = ['Interscript / Secryst contributors']

  spec.files         = Dir.glob("{lib,exe,spec,maps}/**/*", File::FNM_DOTMATCH)
  spec.files         += ['README.adoc']
  spec.require_path  = "lib"
  spec.bindir        = "bin"
  spec.executables   << "secryst"

  spec.required_ruby_version = ">= 2.7"

  spec.add_dependency "thor", "~> 1.0"
  spec.add_dependency "numo-narray", "~> 0.9"
  spec.add_dependency "onnxruntime", "~> 0.6"
  spec.add_dependency "rubyzip", "~> 2.3"

  spec.add_development_dependency "rake"
  spec.add_development_dependency "rspec"
end
