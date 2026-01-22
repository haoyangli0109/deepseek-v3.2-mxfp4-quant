# deepseek-v3.2-mxfp4-quant
This file is used to quantize the original deepseek-v3.2 model into mxfp4 format (default: attn and mtp parts remain unquantized). 
Triton and Quark must be installed. 
Parameters include the original model path, output model path, and quantization method (default: mxfp4).
Use it like this
```
python3 this_script.py --input-fp8-hf-path "original_model" --output-quantized-hf-path "quantized model path"
