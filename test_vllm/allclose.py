import torch
import argparse

def load_tensor(file_path: str, tensor_name: str=None) -> torch.Tensor:
    """从 pt 文件中加载指定名字的 tensor。"""
    
    if tensor_name is None:
        state_dict = torch.load(file_path, map_location="cpu")
        return state_dict#[tensor_name]
    state_dict = torch.load(file_path, map_location="cpu", weights_only=False)
    if tensor_name not in state_dict:
        raise KeyError(f"Tensor '{tensor_name}' not found in {file_path}")
    return state_dict[tensor_name]

def compare_tensors(file1: str, file2: str, atol=1e-5, rtol=1e-4):
    """加载两个文件的同名 tensor，拼接并与目标 tensor 比较。"""
    tensor1 = load_tensor(file1)
    tensor2 = load_tensor(file2)
    print(f"tensor1 shape{tensor1.shape}, tensor2 shape{tensor2.shape}")
    full_tensor = torch.cat([tensor1, tensor2], dim=1)

    print(f"Combined tensor shape: {full_tensor.shape}")
    return full_tensor

def compute_rms(tensor: torch.Tensor) -> float:
    return (tensor.float() ** 2).mean().sqrt().item()

def compute_rms_diff(t1: torch.Tensor, t2: torch.Tensor) -> float:
    return ((t1.float() - t2.float()) ** 2).mean().sqrt().item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file1", type=str, required=True, help="第一个分片 pt 文件路径")
    parser.add_argument("--file2", type=str, required=True, help="第二个分片 pt 文件路径")
    parser.add_argument("--tensor_name", type=str, required=True, help="要比较的 tensor 名称")
    parser.add_argument("--reference", type=str, required=True, help="参考完整 tensor 的 pt 文件路径")
    parser.add_argument("--atol", type=float, default=5e-3)
    parser.add_argument("--rtol", type=float, default=5e-3)
    args = parser.parse_args()

    # 加载 tensor_parallel 的拼接结果
    tp_tensor = compare_tensors(args.file1, args.file2, args.atol, args.rtol)

    # 加载 reference tensor
    ref_tensor = load_tensor(args.reference, args.tensor_name)
    ref_tensor = ref_tensor.squeeze(1)
    # ref_tensor = ref_tensor.reshape(8320, -1)
    print(f"Reference tensor shape: {ref_tensor.shape}")
    # tp_tensor = tp_tensor[:, :3584]
    if tp_tensor.shape != ref_tensor.shape:
        print("Shape mismatch:")
        print(f"TensorParallel tensor: {tp_tensor.shape}")
        print(f"Reference tensor    : {ref_tensor.shape}")
        return

    if torch.allclose(tp_tensor, ref_tensor, atol=args.atol, rtol=args.rtol):
        print("✅ The tensors match (torch.allclose passed)")
    else:
        print("❌ The tensors do NOT match (torch.allclose failed)")
        rms_tp = compute_rms(tp_tensor)
        rms_ref = compute_rms(ref_tensor)
        rms_diff = compute_rms_diff(tp_tensor, ref_tensor)
        print(f"RMS(TP)    = {rms_tp:.6f}")
        print(f"RMS(REF)   = {rms_ref:.6f}")
        print(f"RMS_DIFF   = {rms_diff:.6f}")
        print(f"MAX_DIFF   = {(tp_tensor - ref_tensor).abs().max().item():.6f}")
        print(f"MEAN_DIFF  = {(tp_tensor - ref_tensor).abs().mean().item():.6f}")

if __name__ == "__main__":
    main()

"""
python allclose.py \
    --file1 ./ref_pt/layer000_after_attn_norm_device0.pt \
    --file2 ./ref_pt/layer000_after_attn_norm_device0.pt \
    --tensor_name layers.1.after_attn_norm \
    --reference mid_res.pt
    ✅ The tensors match (torch.allclose passed)
"""

"""
python allclose.py \
    --file1 ./ref_pt/layer000_out_attn_device0.pt \
    --file2 ./ref_pt/layer000_out_attn_device1.pt \
    --tensor_name layers.1.attn.swa_output \
    --reference mid_attn_res.pt
    ❌ The tensors do NOT match (torch.allclose failed)
    RMS(TP)    = 0.340962
    RMS(REF)   = 0.340852
    RMS_DIFF   = 0.001930
    MAX_DIFF   = 0.285156
    MEAN_DIFF  = 0.000782
"""

"""
python allclose.py \
    --file1 ./ref_pt/layer000_out_linear_device0.pt \
    --file2 ./ref_pt/layer000_out_linear_device1.pt \
    --tensor_name layers.1.attn.linattn_output \
    --reference mid_attn_res.pt
    ❌ The tensors do NOT match (torch.allclose failed)
    RMS(TP)    = 4.537794
    RMS(REF)   = 7.581432
    RMS_DIFF   = 8.617321
    MAX_DIFF   = 532.000000
    MEAN_DIFF  = 1.718750
"""

"""
python allclose.py \
    --file1 ./ref_pt/layer000_linear_attn_q_device0.pt \
    --file2 ./ref_pt/layer000_linear_attn_q_device1.pt \
    --tensor_name layers.1.attn.linattn_q \
    --reference mid_attn_res.pt
    ❌ The tensors do NOT match (torch.allclose failed)
    RMS(TP)    = 3.417003
    RMS(REF)   = 3.416966
    RMS_DIFF   = 0.005255
    MAX_DIFF   = 0.500000
    MEAN_DIFF  = 0.001068
"""

"""
python allclose.py \
    --file1 ./ref_pt/layer000_linear_attn_k_device0.pt \
    --file2 ./ref_pt/layer000_linear_attn_k_device1.pt \
    --tensor_name layers.1.attn.linattn_k \
    --reference mid_attn_res.pt
    ❌ The tensors do NOT match (torch.allclose failed)
    RMS(TP)    = 26.720959
    RMS(REF)   = 26.721111
    RMS_DIFF   = 0.011701
    MAX_DIFF   = 1.000000
    MEAN_DIFF  = 0.001678
"""


"""
python allclose.py    \
      --file1 ./ref_pt/layer000_out_linear_norm_device0.pt  \
      --file2 ./ref_pt/layer000_out_linear_norm_device1.pt    \
      --tensor_name layers.1.attn.linattn_norm_output    \
      --reference mid_attn_res.pt
❌ The tensors do NOT match (torch.allclose failed)
RMS(TP)    = 0.982180
RMS(REF)   = 0.988026
RMS_DIFF   = 1.354023
MAX_DIFF   = 55.520996
MEAN_DIFF  = 0.351592
"""


"""
python allclose.py    \
      --file1 ./ref_pt/layer000_linear_attn_g_after_device0.pt  \
      --file2 ./ref_pt/layer000_linear_attn_g_after_device1.pt    \
      --tensor_name layers.1.attn.linattn_g    \
      --reference mid_attn_res.pt
❌ The tensors do NOT match (torch.allclose failed)
RMS(TP)    = 1.255611
RMS(REF)   = 1.255617
RMS_DIFF   = 0.807224
MAX_DIFF   = 2.995732
MEAN_DIFF  = 0.353373
"""


"""
python allclose.py    \
      --file1 ./ref_pt/layer000_linear_attn_k_after_device0.pt  \
      --file2 ./ref_pt/layer000_linear_attn_k_after_device1.pt    \
      --tensor_name layers.1.attn.linattn_k_after    \
      --reference mid_attn_res.pt
❌ The tensors do NOT match (torch.allclose failed)
RMS(TP)    = 0.511904
RMS(REF)   = 0.511904
RMS_DIFF   = 0.282313
MAX_DIFF   = 0.949219
MEAN_DIFF  = 0.142578
"""

"""
python allclose.py    \
      --file1 ./ref_pt/layer000_linear_attn_q_after_device0.pt  \
      --file2 ./ref_pt/layer000_linear_attn_q_after_device1.pt    \
      --tensor_name layers.1.attn.linattn_q_after    \
      --reference mid_attn_res.pt
❌ The tensors do NOT match (torch.allclose failed)
RMS(TP)    = 2.572599
RMS(REF)   = 2.572539
RMS_DIFF   = 0.003978
MAX_DIFF   = 0.500000
MEAN_DIFF  = 0.000507
"""

"""
python allclose.py    \
      --file1 ./ref_pt/layer000_flash_attn_k_device0.pt  \
      --file2 ./ref_pt/layer000_flash_attn_k_device1.pt    \
      --tensor_name layers.1.attn.swa_k    \
      --reference mid_attn_res.pt
❌ The tensors do NOT match (torch.allclose failed)
RMS(TP)    = 26.731997
RMS(REF)   = 26.732229
RMS_DIFF   = 0.013265
MAX_DIFF   = 1.000000
MEAN_DIFF  = 0.001457
"""
