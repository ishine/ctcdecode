from onnxruntime_tools import optimizer
from transformers.convert_graph_to_onnx import convert
from pathlib import Path
from transformers import GPT2Tokenizer
import onnx
import onnxruntime
from onnxruntime.quantization import quantize_dynamic, QuantType
import numpy as np
# Handles all the above steps for you
convert(framework="pt", model="gpt2", output=Path("onnx/gpt2.onnx"), opset=11)
# # Mixed precision conversion for bert-base-cased model converted from Pytorch
optimized_model = optimizer.optimize_model("onnx1/gpt2.onnx", model_type='gpt2', num_heads=12, hidden_size=768)
optimized_model.save_model_to_file("gpt2.onnx")
onnx_model_path = "onnx/gpt2.onnx"
quantized_model_path = "quant_gpt2.onnx"
onnx_opt_model = onnx.load(onnx_model_path)
quantize_dynamic(onnx_model_path,
                quantized_model_path,
                weight_type=QuantType.QInt8)
