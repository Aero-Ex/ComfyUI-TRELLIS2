import gguf
import sys
import os

path = r"D:\@home\aero\comfy\ComfyUI\models\trellis2\ckpts\ss_flow_img_dit_1_3B_64_bf16_Q8_0.gguf"
if not os.path.exists(path):
    print(f"File not found: {path}")
    sys.exit(1)

reader = gguf.GGUFReader(path)
print(f"GGUF File: {os.path.basename(path)}")
print(f"Metadata fields: {len(reader.fields)}")
for field_name in reader.fields:
    if "orig_shape" in field_name or "architecture" in field_name:
        field = reader.get_field(field_name)
        print(f"  {field_name}: {field.parts[field.data[-1]] if field.types[0] == gguf.GGUFValueType.STRING else field.data}")

import torch
import numpy as np

print(f"\nGuidance Tensors:")
for tensor in reader.tensors:
    if "guidance_in" in tensor.name:
        print(f"  {tensor.name}: shape={tensor.shape}, type={tensor.tensor_type.name}")
