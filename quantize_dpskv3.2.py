#
# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import os
import json
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm

import torch
import triton
import triton.language as tl

from safetensors.torch import load_file, save_file
import quark
from quark.torch.utils.pack import Pack_fp4
from quark.torch.quantization.tensor_quantize import FakeQuantizeBase
from quark.torch.quantization import FP8E4M3PerChannelSpec
from quark.torch.quantization.config.config import OCP_MXFP4Spec
from quark.torch.quantization.utils import calculate_qmin_qmax
from quark.torch.quantization.config.type import Dtype

'''
Note the "exclude" here, which specifies in regular expression format which layers are not quantized. 
If you modify the code below to add layers that need to be quantized, please also modify this section accordingly.
'''
mxfp4_quant_config = {"quantization_config": {
    "algo_config": None,
    "exclude": [
      "lm_head",
      "re:.*mlp.gate$",
      "re:model.layers.61.*",
      "re:.*self_attn.*",
    ],
    "export": {
      "kv_cache_group": [],
      "min_kv_scale": 0.0,
      "pack_method": "reorder",
      "weight_format": "real_quantized",
      "weight_merge_groups": None
    },
    "global_quant_config": {
      "bias": None,
      "input_tensors": {
        "ch_axis": -1,
        "dtype": "fp4",
        "group_size": 32,
        "is_dynamic": True,
        "is_scale_quant": False,
        "mx_element_dtype": None,
        "observer_cls": "PerBlockMXObserver",
        "qscheme": "per_group",
        "round_method": "half_even",
        "scale_calculation_mode": "even",
        "scale_format": "e8m0",
        "scale_type": "float",
        "symmetric": None
      },
      "output_tensors": None,
      "target_device": None,
      "weight": {
        "ch_axis": -1,
        "dtype": "fp4",
        "group_size": 32,
        "is_dynamic": False,
        "is_scale_quant": False,
        "mx_element_dtype": None,
        "observer_cls": "PerBlockMXObserver",
        "qscheme": "per_group",
        "round_method": "half_even",
        "scale_calculation_mode": "even",
        "scale_format": "e8m0",
        "scale_type": "float",
        "symmetric": None
      }
    },
    "kv_cache_post_rope": False,
    "kv_cache_quant_config": {},
    "layer_quant_config": {},
    "layer_type_quant_config": {},
    "quant_method": "quark",
    "quant_mode": "eager_mode",
    "softmax_quant_spec": None,
    "version": "0.11rc1+5ff5fbf6b2"
  }}

class Quantizer:
    '''
    Implementing MXFP4 quantization by calling the Quark API
    '''
    def __init__(self, quant_scheme, device="cuda"):
        self.quant_scheme = quant_scheme
        self.device = device
        is_dynamic = False
        self.reorder = True
        self.pack_method = None
        if quant_scheme == "ptpc_fp8":
            self.qspec = FP8E4M3PerChannelSpec(
                is_dynamic=is_dynamic,
                ch_axis=0
            ).to_quantization_spec()
        elif quant_scheme == "mxfp4":
            self.qspec = OCP_MXFP4Spec(ch_axis=-1, is_dynamic=is_dynamic).to_quantization_spec()
        else:
            raise ValueError(f"Unsupported quant_scheme: {quant_scheme}")
        self.dtype = self.qspec.dtype.value
        self.ch_axis = self.qspec.ch_axis
        self.group_size = self.qspec.group_size
        self.round_method = getattr(self.qspec.round_method, "value", None)
        self.qscheme_str_name = getattr(self.qspec.qscheme, "value", None)
        self.quant_min, self.quant_max = calculate_qmin_qmax(self.qspec.dtype)
        self.pack_method = Pack_fp4(getattr(self.qspec.qscheme, "value", None), self.dtype) if self.qspec.dtype is Dtype.fp4 else None


    def real_quantize(self, bf_weight: torch.Tensor):
        # cal scale and zero_point
        self._weight_quantizer = FakeQuantizeBase.get_fake_quantize(
            self.qspec,
            device=self.device
        )
        param = self._weight_quantizer(bf_weight)
        assert isinstance(param, torch.Tensor)
        scale = self._weight_quantizer.scale
        zero_point = self._weight_quantizer.zero_point
        # realquant
        w_res = quark.torch.kernel.scaled_real_quantize(
            self.dtype,
            param,
            scale,
            zero_point,
            self.ch_axis,
            self.group_size,
            self.quant_min,
            self.quant_max,
            self.round_method,
            self.qscheme_str_name,
        )
        # pack weight and scale for mxfp4 according to the quark format
        if self.pack_method is not None:
            w_res = self.pack_method.pack(w_res, self.reorder)
            if getattr(self.qspec, "scale_format", None) == "e8m0":
                scale = (torch.log2(scale).round().to(torch.int16).clamp(-127, 127) + 127).to(torch.uint8)

        return w_res, scale, zero_point

    
@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    """
    Dequantizes weights using the provided scaling factors and stores the result.

    Args:
        x_ptr (tl.pointer): Pointer to the quantized weights.
        s_ptr (tl.pointer): Pointer to the scaling factors.
        y_ptr (tl.pointer): Pointer to the output buffer for dequantized weights.
        M (int): Number of rows in the weight matrix.
        N (int): Number of columns in the weight matrix.
        BLOCK_SIZE (tl.constexpr): Size of the block for tiling.

    Returns:
        None
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)

def weight_dequant(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """
    This function is provided by deepseek-ai.
    Dequantizes the given weight tensor using the provided scale tensor.

    Args:
        x (torch.Tensor): The quantized weight tensor of shape (M, N).
        s (torch.Tensor): The scale tensor of shape (M//block_size, N//block_size).
        block_size (int, optional): The block size to use for dequantization. Defaults to 128.

    Returns:
        torch.Tensor: The dequantized weight tensor of the same shape as `x`.

    Raises:
        AssertionError: If `x` or `s` are not contiguous or if their dimensions are not 2.
    """
    assert x.is_contiguous() and s.is_contiguous(), 'Input tensors must be contiguous'
    assert x.dim() == 2 and s.dim() == 2, 'Input tensors must have 2 dimensions'
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.get_default_dtype())
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE']), triton.cdiv(N, meta['BLOCK_SIZE']))
    weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y

def copy_file(inp: str, out: str):
    '''
    Copy the *.py and *.json files from the original model dir to quantized model dir
    '''
    import shutil
    from pathlib import Path
    src = Path(inp)
    dst = Path(out)
    def ignore_safetensors(dir, files):
        return [f for f in files if f.endswith(".safetensors") or f == "model.safetensors.index.json"]
    shutil.copytree(
        src,
        dst,
        ignore=ignore_safetensors,
        dirs_exist_ok=True
    )

def main(input_path, output_path, quant_scheme):
    """

    This script accepts a raw deepseekv3.2 model (data type: fp8), 
    first dequantizes it to the bf16 data type, 
    and then quantizes it into either mxfp4 or ptpc_fp8 format.

    """
    quantizer = Quantizer(quant_scheme=quant_scheme)

    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(output_path, exist_ok=True)
    copy_file(input_path, output_path)
    model_index_file = os.path.join(input_path, "model.safetensors.index.json")
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]
    
    # Cache for loaded safetensor files
    loaded_files = {}
    fp8_weight_names = []


    # Helper function to get tensor from the correct file
    def get_tensor(tensor_name):
        """
        Retrieves a tensor from the cached safetensor files or loads it from disk if not cached.

        Args:
            tensor_name (str): The name of the tensor to retrieve.

        Returns:
            torch.Tensor: The retrieved tensor.

        Raises:
            KeyError: If the tensor does not exist in the safetensor file.
        """
        file_name = weight_map[tensor_name]
        if file_name not in loaded_files:
            file_path = os.path.join(input_path, file_name)
            loaded_files[file_name] = load_file(file_path, device="cuda")
        return loaded_files[file_name][tensor_name]

    safetensor_files = list(glob(os.path.join(input_path, "*.safetensors")))
    safetensor_files.sort()
    for safetensor_file in tqdm(safetensor_files):
        file_name = os.path.basename(safetensor_file)
        current_state_dict = load_file(safetensor_file, device="cuda")
        loaded_files[file_name] = current_state_dict
        
        new_state_dict = {}
        for weight_name, weight in current_state_dict.items():
            # We skip "weight_scale_inv" and then save "weight_scale" below.
            if weight_name.endswith("_scale_inv"): 
                continue
            elif weight.element_size() == 1:  # FP8 weight
                scale_inv_name = f"{weight_name}_scale_inv"
                scale_name = f"{weight_name}_scale"
                try:
                    # Get scale_inv from the correct file
                    scale_inv = get_tensor(scale_inv_name)
                    fp8_weight_names.append(weight_name)

                    bf_weight = weight_dequant(weight, scale_inv)
                    # dequant parts
                    # skip self_attn and mtp layers quantization
                    if "self_attn" in weight_name or "model.layers.61" in weight_name:
                        # clear model.safetensors.index.json
                        if scale_inv_name in weight_map: 
                            which_safe_file = weight_map[scale_inv_name]
                            weight_map.pop(scale_inv_name)
                        new_state_dict[weight_name] = bf_weight

                    # requant parts
                    else:
                        # real Q
                        w_res, scale, _ = quantizer.real_quantize(bf_weight)
                        # clear model.safetensors.index.json
                        if scale_inv_name in weight_map:
                            which_safe_file = weight_map[weight_name]
                            weight_map.pop(scale_inv_name)
                            weight_map[scale_name] = which_safe_file
                        # add new k-v
                        new_state_dict[scale_name] = scale
                        new_state_dict[weight_name] = w_res

                except KeyError:
                    print(f"Warning: Missing scale_inv tensor for {weight_name}, skipping conversion")
                    new_state_dict[weight_name] = weight
            else:
                new_state_dict[weight_name] = weight
                
        new_safetensor_file = os.path.join(output_path, file_name)
        save_file(new_state_dict, new_safetensor_file)
        
        # Memory management: keep only the 2 most recently used files
        if len(loaded_files) > 2:
            oldest_file = next(iter(loaded_files))
            del loaded_files[oldest_file]
            torch.cuda.empty_cache()
    
    # Update model index json
    new_model_index_file = os.path.join(output_path, "model.safetensors.index.json")
    with open(new_model_index_file, "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2)
    # Update config json
    old_config_path = os.path.join(input_path, "config.json")
    with open(old_config_path, "r") as f:
        config = json.load(f)
    if quant_scheme == "mxfp4":
        config.update(mxfp4_quant_config)
    new_config_path = os.path.join(output_path, "config.json")
    with open(new_config_path, "w") as f:
        json.dump(config, f, indent=2) 


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-fp8-hf-path", type=str, default="/shareddata/deepseek-ai/DeepSeek-V3.2-Exp")
    parser.add_argument("--output-quantized-hf-path", type=str, default="/shareddata/deepseek-ai/DeepSeek-V3.2-Exp-mxfp4")
    parser.add_argument("--quant-scheme", type=str, choices=["ptpc_fp8", "mxfp4"], default="mxfp4")
    args = parser.parse_args()
    main(args.input_fp8_hf_path, args.output_quantized_hf_path, args.quant_scheme)

'''
This file is used to quantize the original deepseek-v3.2 model into mxfp4 format (default: attn and mtp parts remain unquantized). 
Triton and Quark must be installed. 
Parameters include the original model path, output model path, and quantization method (default: mxfp4).
Use it like this
python3 this_script.py --input-fp8-hf-path "original_model" --output-quantized-hf-path "quantized model path"
'''
