import os
import torch
import numpy as np

def convert_pt_to_txt(pt_dir="./debug_logs", txt_dir="./debug_txt_logs", fmt="%.6f", max_rows=None, max_cols=None):
    os.makedirs(txt_dir, exist_ok=True)

    for fname in sorted(os.listdir(pt_dir)):
        if not fname.endswith(".pt"):
            continue

        pt_path = os.path.join(pt_dir, fname)
        txt_name = fname.replace(".pt", ".txt")
        txt_path = os.path.join(txt_dir, txt_name)
        if os.path.exists(txt_path) == True:
            continue
        tensor = torch.load(pt_path, map_location="cpu")

        if tensor.dtype == torch.bfloat16:
            tensor = tensor.to(torch.float32)

        array = tensor.numpy()
        # print(array.shape)
        if len(array.shape) == 2:
            if max_rows is not None:
                array = array[:max_rows]
            if max_cols is not None:
                array = array[:, :max_cols]
        if len(array.shape) == 3:
            if max_rows is not None:
                array = array[:max_rows]
            if max_cols is not None:
                array = array[:, :4, :max_cols]
            array = array.reshape(array.shape[0], -1)
        
        np.savetxt(txt_path, array, fmt=fmt)
        print(f"Converted {fname} -> {txt_name}")

convert_pt_to_txt(
    pt_dir="./debug_logs",
    txt_dir="./debug_txt_logs",
    fmt="%.6f",
    max_rows=None,
    max_cols=None
)