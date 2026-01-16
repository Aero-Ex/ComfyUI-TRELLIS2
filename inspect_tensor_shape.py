import gguf
import torch
import warnings

reader = gguf.GGUFReader(r"D:\@home\aero\comfy\ComfyUI\models\trellis2\ckpts\ss_flow_img_dit_1_3B_64_bf16_Q8_0.gguf")
t = [x for x in reader.tensors if "to_qkv.weight" in x.name][0]
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    tt = torch.from_numpy(t.data)
print(f"GGUF shape: {t.shape}")
print(f"Tensor type: {t.tensor_type}")
print(f"Torch tensor shape: {tt.shape}")
print(f"Torch tensor numel: {tt.numel()}")
