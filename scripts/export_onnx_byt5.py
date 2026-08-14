"""Export a ByT5 (byte-level seq2seq) checkpoint to a secryst model.zip.

Produces encoder.onnx + decoder.onnx + metadata.yaml. The Ruby runtime
(lib/secryst/byt5_onnx.rb) tokenizes as UTF-8 bytes (pad=0, EOS=1) and
greedy-decodes — no vocab files needed.

Usage:
    python scripts/export_onnx_byt5.py <hf_checkpoint_dir> <out.zip>
"""

import sys
import zipfile
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForSeq2SeqLM


class DecoderWithHead(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.decoder = model.get_decoder()
        self.lm_head = model.lm_head

    def forward(self, input_ids, encoder_hidden_states):
        hidden = self.decoder(
            input_ids=input_ids, encoder_hidden_states=encoder_hidden_states
        )[0]
        return self.lm_head(hidden)


def main(ckpt: str, out_zip: str) -> None:
    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt).eval()
    ids = torch.tensor([[104, 101]])  # "he"
    with torch.no_grad():
        hidden = model.get_encoder()(input_ids=ids)[0]

    torch.onnx.export(
        model.get_encoder(),
        (ids,),
        "/tmp/encoder.onnx",
        input_names=["input_ids"],
        output_names=["last_hidden_state"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "last_hidden_state": {0: "batch", 1: "seq"}},
    )
    torch.onnx.export(
        DecoderWithHead(model),
        (torch.tensor([[0]]), hidden),
        "/tmp/decoder.onnx",
        input_names=["input_ids", "encoder_hidden_states"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "encoder_hidden_states": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
        },
    )
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("metadata.yaml", "name: byt5\n")
        z.write("/tmp/encoder.onnx", "encoder.onnx")
        z.write("/tmp/decoder.onnx", "decoder.onnx")
    print(f"wrote {out_zip}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
