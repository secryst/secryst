import modal

vol = modal.Volume.from_name("secryst-checkpoints", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.5.1", "transformers==4.46.3", "onnx").add_local_dir("scripts", "/opt/secryst", copy=True)
app = modal.App("secryst-export", image=image)

@app.function(volumes={"/ckpts": vol}, timeout=30 * 60)
def export():
    import sys, zipfile
    sys.path.insert(0, "/opt/secryst")
    from export_onnx_byt5 import main as export_main
    vol.reload()
    import os
    src = "/ckpts/khmer_byt5/run-001/best"
    print("ckpt exists:", os.path.isdir(src))
    export_main(src, "/ckpts/khmer_byt5/khm-latn-byt5.zip")
    with zipfile.ZipFile("/ckpts/khmer_byt5/khm-latn-byt5.zip") as z:
        print("zip contents:", z.namelist())
    vol.commit()

if __name__ == "__main__":
    with app.run():
        export.remote()
