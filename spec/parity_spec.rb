require "spec_helper"
require "json"
require "secryst/byt5_onnx"

# Cross-crystal parity (interscript-ml v1, requirement C5): the Ruby
# crystal must reproduce the Python reference goldens byte-for-byte on
# the deterministic decode-loop fixture. CI checks out secryst/secryst-py
# (parity/ = zip + golden.jsonl); locally point SECRYST_PARITY_DIR at it.
parity_dir = ENV["SECRYST_PARITY_DIR"] || File.expand_path("../../secryst-py/parity", __dir__)
zip_path = File.join(parity_dir, "tiny-1.0.zip")
goldens_path = File.join(parity_dir, "golden.jsonl")

describe "cross-crystal parity" do
  let(:model) { Secryst::Byt5Onnx.new(zip_path) }

  it "reproduces the reference goldens exactly" do
    skip "parity kit not found (#{parity_dir})" unless File.exist?(goldens_path)

    goldens = File.readlines(goldens_path).map { |l| JSON.parse(l) }
    expect(goldens).not_to be_empty

    failures = goldens.reject do |row|
      model.translate(row["input"], max_seq_length: 8) == row["output"]
    end

    aggregate_failures "golden diff" do
      failures.each do |row|
        expect(model.translate(row["input"], max_seq_length: 8))
          .to eq(row["output"]), "input #{row['input'].inspect}"
      end
    end
  end
end
