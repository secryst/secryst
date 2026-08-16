require 'spec_helper'
require 'secryst'
require 'fileutils'
require 'tmpdir'
require 'json'

RSpec.describe Secryst::IMF do
  it 'uses the canonical ByT5 table (byte + 3, trailing EOS)' do
    expect(described_class.encode('rok')).to eq([117, 114, 110, 1])
    expect(described_class.decode([117, 114, 110])).to eq('rok')
    expect(described_class.decode([117, 1, 114])).to eq('r')
  end

  describe 'with the tiny IMF fixture zip' do
    let(:zip_path) { File.expand_path('../fixtures/tiny-imf.zip', __dir__) }

    it 'parses the manifest and verifies graph checksums' do
      meta = described_class.manifest(zip_path)
      expect(meta['format']).to eq('imf-v1')
      expect(meta['tokenizer']).to eq('bytes')
      graphs = described_class.verify_and_read(zip_path)
      expect(graphs.keys.sort).to eq(['decoder.onnx', 'encoder.onnx'])
    end

    it 'loads sessions from verified bytes' do
      model = Secryst::Byt5Onnx.new(zip_path)
      expect(model.id).to eq('tiny-1.0')
    end

    it 'decodes a real zip against the shared golden set', :e2e do
      zip = ENV['SECRYST_E2E_ZIP']
      skip 'set SECRYST_E2E_ZIP to a real IMF zip' unless zip
      golden = ENV['SECRYST_GOLDEN'] ||
        '/Users/mulgogi/src/interscript/ml-models/golden/khm-latn-100.jsonl'
      translator = Secryst::Translator.new(model_file: zip)
      ok = 0
      total = 0
      File.readlines(golden).each do |line|
        row = JSON.parse(line)
        total += 1
        ok += 1 if translator.translate(row['input'], max_seq_length: 128) == row['output']
      end
      expect(ok).to eq(total)
    end
  end

  it 'rejects a tampered zip loudly' do
    Dir.mktmpdir do |tmp|
      source = File.expand_path('../fixtures/tiny-imf.zip', __dir__)
      tampered = File.join(tmp, 'tampered.zip')
      require 'zip'
      Zip::File.open(source) do |src|
        Zip::File.open(tampered, create: true) do |dst|
          src.entries.each do |e|
            if e.name == 'encoder.onnx'
              dst.get_output_stream(e.name) { |io| io.write('corrupted-bytes') }
            else
              dst.get_output_stream(e.name) { |io| io.write(e.get_input_stream.read) }
            end
          end
        end
      end
      expect { described_class.verify_and_read(tampered) }
        .to raise_error(Secryst::IMF::FormatError, /sha256 mismatch/)
    end
  end

  describe '.resolve' do
    it 'installs a verified copy into the cache from a local index' do
      Dir.mktmpdir do |tmp|
        channel = File.join(tmp, 'channel')
        FileUtils.mkdir_p(channel)
        zip_path = File.expand_path('../fixtures/tiny-imf.zip', __dir__)
        FileUtils.cp(zip_path, File.join(channel, 'tiny.zip'))
        require 'digest'
        index = File.join(tmp, 'models.yaml')
        File.write(index, <<~YAML)
          version: 1
          models:
            tiny-1.0:
              filename: tiny.zip
              url: file://#{channel}/tiny.zip
              sha256: #{Digest::SHA256.file(zip_path).hexdigest}
        YAML
        cache = File.join(tmp, 'cache')
        result = described_class.resolve('tiny-1.0', index_url: index) if false
        # env-based cache (the public API reads ENV at call time)
        ENV['INTERSCRIPT_ML_CACHE'] = cache
        begin
          installed = described_class.resolve('tiny-1.0', index_url: index)
          expect(installed).to eq(File.join(cache, 'models', 'tiny-1.0', 'tiny.zip'))
          expect(File.file?(installed)).to be(true)
          FileUtils.rm_f(File.join(channel, 'tiny.zip'))
          expect(described_class.resolve('tiny-1.0', index_url: index)).to eq(installed)
        ensure
          ENV.delete('INTERSCRIPT_ML_CACHE')
        end
      end
    end

    it 'raises for unknown ids' do
      Dir.mktmpdir do |tmp|
        index = File.join(tmp, 'models.yaml')
        File.write(index, "version: 1\nmodels: {}\n")
        expect { described_class.resolve('nope-1.0', index_url: index) }
          .to raise_error(Secryst::IMF::RegistryError, /unknown model id/)
      end
    end
  end
end
