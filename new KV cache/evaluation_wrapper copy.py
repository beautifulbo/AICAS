"""
AICAS 2026 - Participant Core Modification File

Participants should modify the VLMModel class to implement optimizations.

Note:
- Benchmark directly calls self.model.generate() for performance testing.
- Your optimizations should modify self.model or its operators in __init__ via Monkey Patch.
- The generate() method is optional and mainly for debugging.
"""
import inspect
import math
import types
from typing import Dict, Optional
try:
    from PIL import Image
except ImportError:
    # For testing without PIL
    class Image:
        pass
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


def _expand_kv_heads(hidden_states: torch.Tensor, groups: int) -> torch.Tensor:
    batch, kv_heads, seq_len, head_dim = hidden_states.shape
    if groups == 1:
        return hidden_states
    expanded = hidden_states[:, :, None, :, :].expand(batch, kv_heads, groups, seq_len, head_dim)
    return expanded.reshape(batch, kv_heads * groups, seq_len, head_dim)


def _build_random_orthogonal_matrix(
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    gaussian = torch.randn(dim, dim, generator=generator, device=device, dtype=torch.float32)
    q_matrix, _ = torch.linalg.qr(gaussian, mode="reduced")
    return q_matrix.to(dtype=dtype)


def _build_scalar_codebook(levels: int, device: torch.device) -> torch.Tensor:
    if levels <= 1:
        return torch.zeros(1, device=device, dtype=torch.float32)
    probs = (torch.arange(levels, device=device, dtype=torch.float32) + 0.5) / levels
    probs = probs.clamp(1e-6, 1.0 - 1e-6)
    return (math.sqrt(2.0) * torch.erfinv(2.0 * probs - 1.0)).to(torch.float32)


def _pack_lowbit(values: torch.Tensor, bits: int) -> tuple[torch.Tensor, int]:
    values = values.to(torch.uint8)
    values_per_byte = 8 // bits
    original_size = values.shape[-1]
    pad = (-original_size) % values_per_byte
    if pad:
        values = torch.nn.functional.pad(values, (0, pad))

    packed = torch.zeros(*values.shape[:-1], values.shape[-1] // values_per_byte, dtype=torch.uint8, device=values.device)
    bit_mask = (1 << bits) - 1
    for i in range(values_per_byte):
        packed |= ((values[..., i::values_per_byte] & bit_mask) << (i * bits))
    return packed.contiguous(), original_size


def _unpack_lowbit(packed: torch.Tensor, bits: int, original_size: int) -> torch.Tensor:
    values_per_byte = 8 // bits
    bit_mask = (1 << bits) - 1
    unpacked = []
    for i in range(values_per_byte):
        unpacked.append((packed >> (i * bits)) & bit_mask)
    return torch.stack(unpacked, dim=-1).reshape(*packed.shape[:-1], -1)[..., :original_size].contiguous()


def _pack_sign_bits(signs: torch.Tensor) -> tuple[torch.Tensor, int]:
    bits = (signs > 0).to(torch.uint8)
    return _pack_lowbit(bits, bits=1)


def _unpack_sign_bits(packed: torch.Tensor, original_size: int) -> torch.Tensor:
    unpacked = _unpack_lowbit(packed, bits=1, original_size=original_size)
    return torch.where(unpacked > 0, 1.0, -1.0)


class _TurboQuantMSE:
    def __init__(self, dim: int, bits: int, device: torch.device, seed: int):
        self.dim = dim
        self.bits = bits
        self.levels = 1 << bits
        self.rotation = _build_random_orthogonal_matrix(dim, device, torch.float32, seed)
        self.codebook = _build_scalar_codebook(self.levels, device)

    def quantize(self, values: torch.Tensor) -> dict:
        norms = values.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
        normalized_values = values.float() / norms
        rotated = torch.matmul(normalized_values, self.rotation.transpose(0, 1))
        distance = (rotated.unsqueeze(-1) - self.codebook).abs()
        indices = distance.argmin(dim=-1).to(torch.uint8)
        packed_indices, original_dim = _pack_lowbit(indices, bits=self.bits)
        return {
            "packed_indices": packed_indices,
            "original_dim": original_dim,
            "norms": norms.to(torch.float16).contiguous(),
        }

    def dequantize(self, quantized: dict, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        packed_indices = quantized["packed_indices"].to(device=device, non_blocking=True)
        indices = _unpack_lowbit(
            packed_indices,
            bits=self.bits,
            original_size=quantized["original_dim"],
        ).long()
        norms = quantized["norms"].to(device=device, dtype=torch.float32, non_blocking=True)
        rotated = self.codebook.to(device=device)[indices]
        restored = torch.matmul(rotated, self.rotation.to(device=device)) * norms
        return restored.to(dtype=dtype)


class _TurboQuantProd:
    def __init__(self, dim: int, bits: int, device: torch.device, seed: int, qjl_dim: Optional[int] = None):
        self.dim = dim
        self.bits = bits
        self.mse_quantizer = _TurboQuantMSE(dim, max(1, bits - 1), device, seed)
        self.qjl_dim = qjl_dim or dim
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + 1)
        self.projection = torch.randn(self.qjl_dim, dim, generator=generator, dtype=torch.float32, device=device)
        self.residual_scale = math.sqrt(math.pi / 2.0) / self.qjl_dim

    def quantize(self, values: torch.Tensor) -> dict:
        mse_quantized = self.mse_quantizer.quantize(values)
        base = self.mse_quantizer.dequantize(mse_quantized, values.device, torch.float32)
        residual = values.float() - base
        projected = torch.matmul(residual, self.projection.transpose(0, 1))
        qjl = torch.where(projected >= 0, 1.0, -1.0)
        packed_qjl, qjl_dim = _pack_sign_bits(qjl)
        gamma = residual.norm(dim=-1, keepdim=True).to(torch.float16)
        return {
            "mse": mse_quantized,
            "packed_qjl": packed_qjl,
            "qjl_dim": qjl_dim,
            "gamma": gamma.contiguous(),
        }

    def dequantize(self, quantized: dict, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        base = self.mse_quantizer.dequantize(quantized["mse"], device, torch.float32)
        packed_qjl = quantized["packed_qjl"].to(device=device, non_blocking=True)
        qjl = _unpack_sign_bits(packed_qjl, quantized["qjl_dim"]).to(device=device, dtype=torch.float32)
        gamma = quantized["gamma"].to(device=device, dtype=torch.float32, non_blocking=True)
        residual = torch.matmul(qjl, self.projection.to(device=device))
        residual = residual * gamma * self.residual_scale
        return (base + residual).to(dtype=dtype)

    def inner_product(self, query: torch.Tensor, quantized: dict) -> torch.Tensor:
        base = self.mse_quantizer.dequantize(quantized["mse"], query.device, torch.float32)
        term1 = (query.float() * base).sum(dim=-1)
        packed_qjl = quantized["packed_qjl"].to(device=query.device, non_blocking=True)
        qjl = _unpack_sign_bits(packed_qjl, quantized["qjl_dim"]).to(device=query.device, dtype=torch.float32)
        gamma = quantized["gamma"].to(device=query.device, dtype=torch.float32, non_blocking=True).squeeze(-1)
        projected_query = torch.matmul(query.float(), self.projection.to(device=query.device).transpose(0, 1))
        term2 = gamma * self.residual_scale * (projected_query * qjl).sum(dim=-1)
        return term1 + term2


class _TurboQuantCodec:
    def __init__(self, key_bits: int, value_bits: int, layer_seed: int):
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.layer_seed = layer_seed
        self._key_quantizers: Dict[tuple[int, str], _TurboQuantProd] = {}
        self._value_quantizers: Dict[tuple[int, str], _TurboQuantMSE] = {}

    def _get_key_quantizer(self, dim: int, device: torch.device) -> _TurboQuantProd:
        key = (dim, str(device))
        quantizer = self._key_quantizers.get(key)
        if quantizer is None:
            quantizer = _TurboQuantProd(dim, self.key_bits, device, self.layer_seed + dim * 17)
            self._key_quantizers[key] = quantizer
        return quantizer

    def _get_value_quantizer(self, dim: int, device: torch.device) -> _TurboQuantMSE:
        key = (dim, str(device))
        quantizer = self._value_quantizers.get(key)
        if quantizer is None:
            quantizer = _TurboQuantMSE(dim, self.value_bits, device, self.layer_seed + 100 + dim * 19)
            self._value_quantizers[key] = quantizer
        return quantizer

    def quantize_key_page(self, values: torch.Tensor) -> dict:
        quantizer = self._get_key_quantizer(values.shape[-1], values.device)
        flat = values.detach().contiguous().view(-1, values.shape[-1])
        quantized = quantizer.quantize(flat)
        return {
            "shape": tuple(values.shape),
            "payload": quantized,
        }

    def quantize_value_page(self, values: torch.Tensor) -> dict:
        quantizer = self._get_value_quantizer(values.shape[-1], values.device)
        flat = values.detach().contiguous().view(-1, values.shape[-1])
        quantized = quantizer.quantize(flat)
        return {
            "shape": tuple(values.shape),
            "payload": quantized,
        }

    def dequantize_value_page(self, page: dict, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        dim = page["shape"][-1]
        quantizer = self._get_value_quantizer(dim, device)
        restored = quantizer.dequantize(page["payload"], device, dtype)
        return restored.view(*page["shape"]).contiguous()

    def key_inner_product(self, query: torch.Tensor, page: dict) -> torch.Tensor:
        dim = page["shape"][-1]
        quantizer = self._get_key_quantizer(dim, query.device)
        flat_query = query.contiguous().view(-1, dim)
        scores = quantizer.inner_product(flat_query, page["payload"])
        return scores


class _KVPageAllocator:
    def __init__(self):
        self.page_table = []
        self.physical_pages = {}
        self.free_physical_pages = []
        self.next_physical_page = 0
        self.next_logical_page = 0

    def reset(self):
        self.page_table.clear()
        self.physical_pages.clear()
        self.free_physical_pages.clear()
        self.next_physical_page = 0
        self.next_logical_page = 0

    def allocate(self, page_payload: dict, valid_tokens: int):
        if self.free_physical_pages:
            physical_page = self.free_physical_pages.pop()
        else:
            physical_page = self.next_physical_page
            self.next_physical_page += 1

        logical_page = self.next_logical_page
        self.next_logical_page += 1
        self.physical_pages[physical_page] = page_payload
        self.page_table.append(
            {
                "logical_page": logical_page,
                "physical_page": physical_page,
                "valid_tokens": valid_tokens,
            }
        )

    def iter_pages(self):
        for page_entry in self.page_table:
            yield page_entry, self.physical_pages[page_entry["physical_page"]]


class _LayerKVRuntime:
    def __init__(self, page_size: int, key_bits: int, value_bits: int, layer_seed: int):
        self.page_size = page_size
        self.codec = _TurboQuantCodec(key_bits=key_bits, value_bits=value_bits, layer_seed=layer_seed)
        self.allocator = _KVPageAllocator()
        self.buffer_key: Optional[torch.Tensor] = None
        self.buffer_value: Optional[torch.Tensor] = None
        self.decode_tokens = 0
        self.prefill_ready = False

    def reset(self):
        self.allocator.reset()
        self.buffer_key = None
        self.buffer_value = None
        self.decode_tokens = 0
        self.prefill_ready = False

    def _append_dense(self, key_states: torch.Tensor, value_states: torch.Tensor):
        if self.buffer_key is None:
            self.buffer_key = key_states.detach().contiguous()
            self.buffer_value = value_states.detach().contiguous()
        else:
            self.buffer_key = torch.cat([self.buffer_key, key_states.detach()], dim=-2).contiguous()
            self.buffer_value = torch.cat([self.buffer_value, value_states.detach()], dim=-2).contiguous()

        while self.buffer_key is not None and self.buffer_key.shape[-2] >= self.page_size:
            page_key = self.buffer_key[:, :, : self.page_size, :].contiguous()
            page_value = self.buffer_value[:, :, : self.page_size, :].contiguous()
            self.allocator.allocate(
                {
                    "key": self.codec.quantize_key_page(page_key),
                    "value": self.codec.quantize_value_page(page_value),
                },
                valid_tokens=self.page_size,
            )
            self.buffer_key = self.buffer_key[:, :, self.page_size :, :].contiguous()
            self.buffer_value = self.buffer_value[:, :, self.page_size :, :].contiguous()

        self.prefill_ready = True

    def ingest_prefill(self, key_states: torch.Tensor, value_states: torch.Tensor):
        self.reset()
        self._append_dense(key_states, value_states)

    def append_decode(self, key_states: torch.Tensor, value_states: torch.Tensor):
        self._append_dense(key_states, value_states)
        self.decode_tokens += key_states.shape[-2]

    def materialize_values(self, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        value_chunks = []

        for page_entry, page_payload in self.allocator.iter_pages():
            page_value = self.codec.dequantize_value_page(page_payload["value"], device, dtype)
            valid_tokens = page_entry["valid_tokens"]
            value_chunks.append(page_value[:, :, :valid_tokens, :])

        if self.buffer_key is not None and self.buffer_value is not None:
            value_chunks.append(self.buffer_value.to(device=device, dtype=dtype, non_blocking=True))

        if not value_chunks:
            return None
        return torch.cat(value_chunks, dim=-2)

    def attention_scores(self, query_states: torch.Tensor) -> Optional[torch.Tensor]:
        batch, heads, query_len, head_dim = query_states.shape
        flat_query = query_states.contiguous().view(-1, head_dim)
        score_chunks = []

        for page_entry, page_payload in self.allocator.iter_pages():
            scores = self.codec.key_inner_product(flat_query, page_payload["key"])
            scores = scores.view(batch, heads, query_len, -1)
            valid_tokens = page_entry["valid_tokens"]
            score_chunks.append(scores[..., :valid_tokens])

        if self.buffer_key is not None:
            expanded_key = _expand_kv_heads(self.buffer_key, heads // self.buffer_key.shape[1])
            dense_scores = torch.matmul(query_states, expanded_key.transpose(2, 3))
            score_chunks.append(dense_scores)

        if not score_chunks:
            return None
        return torch.cat(score_chunks, dim=-1)


def _get_layer_kv_runtime(module) -> _LayerKVRuntime:
    runtime = getattr(module, "_turbo_kv_runtime", None)
    if runtime is None:
        runtime = _LayerKVRuntime(
            page_size=max(1, getattr(module, "_turbo_kv_page_size", 64)),
            key_bits=max(2, getattr(module, "_turbo_kv_key_bits", 3)),
            value_bits=max(1, getattr(module, "_turbo_kv_value_bits", 3)),
            layer_seed=1009 + int(getattr(module, "layer_idx", 0)),
        )
        module._turbo_kv_runtime = runtime
    return runtime


def _turboquant_attention(
    module,
    runtime: _LayerKVRuntime,
    query_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
):
    scores = runtime.attention_scores(query_states)
    if scores is None:
        return None, None
    values = runtime.materialize_values(query_states.device, query_states.dtype)
    if values is None:
        return None, None
    expanded_value = _expand_kv_heads(values, module.num_key_value_groups)
    scores = scores * module.scaling
    if attention_mask is not None:
        scores = scores + attention_mask
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    output = torch.matmul(probs, expanded_value).transpose(1, 2).contiguous()
    return output, probs


def _build_swiftkv_decoder_layer_forward(original_forward):
    """
    SwiftKV-style approximation for prefill only.

    During prefill, selected decoder layers are skipped to reduce TTFT.
    Decode stage keeps the normal layer path.
    """

    def patched_forward(self, hidden_states, *args, **kwargs):
        # Prefill usually has sequence length > 1, while decode is typically 1 token.
        is_prefill = hidden_states.shape[1] > 1
        if is_prefill and getattr(self, "_swiftkv_skip_in_prefill", False):
            outputs = (hidden_states,)
            if kwargs.get("output_attentions", False):
                outputs += (None,)
            if kwargs.get("use_cache", False):
                outputs += (kwargs.get("past_key_value", None),)
            return outputs
        return original_forward(hidden_states, *args, **kwargs)

    return patched_forward


def _build_turboquant_kv_forward(original_forward):
    """
    Replace the layer KV path with a TurboQuant-based paged cache.
    """

    multimodal_rope_fn = original_forward.__globals__.get("apply_multimodal_rotary_pos_emb")
    rope_fn = original_forward.__globals__.get("apply_rotary_pos_emb") or multimodal_rope_fn
    original_signature = inspect.signature(original_forward)

    def _call_original(hidden_states, position_embeddings, attention_mask, past_key_values, kwargs):
        call_kwargs = dict(kwargs)
        call_kwargs.pop("hidden_states", None)
        call_kwargs.pop("position_embeddings", None)
        call_kwargs.pop("attention_mask", None)
        call_kwargs.pop("past_key_values", None)
        call_kwargs.pop("past_key_value", None)
        if "hidden_states" in original_signature.parameters:
            call_kwargs["hidden_states"] = hidden_states
        if "position_embeddings" in original_signature.parameters:
            call_kwargs["position_embeddings"] = position_embeddings
        if "attention_mask" in original_signature.parameters:
            call_kwargs["attention_mask"] = attention_mask
        if "past_key_values" in original_signature.parameters:
            call_kwargs["past_key_values"] = past_key_values
        elif "past_key_value" in original_signature.parameters:
            call_kwargs["past_key_value"] = past_key_values
        return original_forward(**call_kwargs)

    def patched_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings=None,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        if past_key_values is None and "past_key_value" in kwargs:
            past_key_values = kwargs["past_key_value"]

        if rope_fn is None:
            return _call_original(
                hidden_states, position_embeddings, attention_mask, past_key_values, kwargs
            )

        try:
            cos = None
            sin = None
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)

            query_states = self.q_proj(hidden_states).view(hidden_shape)
            key_states = self.k_proj(hidden_states).view(hidden_shape)
            value_states = self.v_proj(hidden_states).view(hidden_shape)

            if hasattr(self, "q_norm"):
                query_states = self.q_norm(query_states)
            if hasattr(self, "k_norm"):
                key_states = self.k_norm(key_states)

            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)

            if position_embeddings is not None:
                cos, sin = position_embeddings
                if multimodal_rope_fn is not None and rope_fn is multimodal_rope_fn:
                    rope_scaling = getattr(self, "rope_scaling", None)
                    mrope_section = None
                    if rope_scaling is not None:
                        mrope_section = rope_scaling.get("mrope_section")
                    if mrope_section is None and hasattr(self, "config"):
                        config_rope_scaling = getattr(self.config, "rope_scaling", None)
                        if config_rope_scaling is not None:
                            mrope_section = config_rope_scaling.get("mrope_section")
                    if mrope_section is None:
                        return _call_original(
                            hidden_states, position_embeddings, attention_mask, past_key_values, kwargs
                        )
                    query_states, key_states = rope_fn(
                        query_states,
                        key_states,
                        cos,
                        sin,
                        mrope_section,
                    )
                else:
                    query_states, key_states = rope_fn(query_states, key_states, cos, sin)

            runtime = _get_layer_kv_runtime(self)
            decode_seq_len = query_states.shape[-2]

            cache_kwargs = {}
            if cos is not None and sin is not None:
                cache_kwargs["cos"] = cos
                cache_kwargs["sin"] = sin
            if "cache_position" in kwargs:
                cache_kwargs["cache_position"] = kwargs["cache_position"]

            if decode_seq_len > 1:
                if past_key_values is not None:
                    key_states, value_states = past_key_values.update(
                        key_states, value_states, self.layer_idx, cache_kwargs
                    )
                runtime.ingest_prefill(key_states, value_states)
            else:
                if past_key_values is not None and not runtime.prefill_ready:
                    key_states, value_states = past_key_values.update(
                        key_states, value_states, self.layer_idx, cache_kwargs
                    )
                    runtime.ingest_prefill(key_states, value_states)
                else:
                    runtime.append_decode(key_states, value_states)

            attn_output, attn_weights = _turboquant_attention(
                self,
                runtime,
                query_states,
                attention_mask=attention_mask,
            )
            if attn_output is None or attn_weights is None:
                return _call_original(
                    hidden_states, position_embeddings, attention_mask, past_key_values, kwargs
                )

            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            return attn_output, attn_weights
        except Exception:
            return _call_original(
                hidden_states, position_embeddings, attention_mask, past_key_values, kwargs
            )

    return patched_forward


def _reset_turboquant_kv_cache(model):
    text_model = getattr(model, "model", None)
    layers = getattr(text_model, "layers", None)
    if layers is None:
        return

    for layer in layers:
        self_attn = getattr(layer, "self_attn", None)
        if self_attn is None:
            continue
        runtime = getattr(self_attn, "_turbo_kv_runtime", None)
        if runtime is not None:
            runtime.reset()


def _build_resetting_generate(original_generate):
    """
    Clear TurboQuant KV cache before every generation call.
    """

    def patched_generate(self, *args, **kwargs):
        _reset_turboquant_kv_cache(self)
        return original_generate(*args, **kwargs)

    return patched_generate


class VLMModel:
    """
    Participant optimization class - modify this to implement optimizations.
    
    Optimization Architecture:
    - Split optimizations into separate methods for isolation and testing
    - Enable/disable each optimization independently in __init__
    - Each optimization method can be tested individually
    
    Important Notes:
    1. Benchmark directly calls self.model.generate() for performance testing.
    2. Your optimizations should modify self.model or its operators via Monkey Patch.
    3. All optimizations are applied in __init__ by calling optimization methods.
    """
    
    def __init__(self, model_path: str, device: str = "cuda:0"):
        """
        Initialize model and apply optimizations.
        
        Args:
            model_path: Qwen3-VL-2B-Instruct model path
            device: CUDA device, e.g., "cuda:0"
        """
        self._device = device
        self.model_path = model_path
        
        # Load processor
        print(f"[VLMModel] Loading processor from {model_path}...")
        self._processor = AutoProcessor.from_pretrained(model_path)
        
        # Load model
        print(f"[VLMModel] Loading model with FP16...")
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device
        )
        self._model.eval()
        
        # Track applied optimizations
        self._optimizations_applied = []
        
        # ================================================================
        # Participant Optimization Area - Enable/disable optimizations here
        # Uncomment the optimization methods you want to apply
        # ================================================================
        
        # 1. Vision Encoder Acceleration
        # self._optimize_vision_encoder()
        
        # 2. KV Cache Management
        self._optimize_kv_cache()
        
        # 3. Cross-modal Connector Optimization
        # self._optimize_cross_modal_connector()
        
        # 4. Flash Attention Optimization
        # self._enable_flash_attention()
        
        # 5. Quantization
        # self._apply_quantization()
        
        # Optional: Explore model structure before optimization
        # self._explore_model_structure()
        
        # print(self._model)
        # ================================================================
        
        print(f"[VLMModel] Model loaded successfully on {device}")
        if self._optimizations_applied:
            print(f"[VLMModel] Applied optimizations: {', '.join(self._optimizations_applied)}")
    
    # ================================================================
    # Optimization Methods - Implement your optimizations here
    # ================================================================
    
    def _explore_model_structure(self):
        """
        Helper method to explore model structure.
        
        Use this to understand the model architecture before implementing optimizations.
        This helps identify where to apply monkey patches.
        """
        print("=" * 60)
        print("Model Structure Exploration")
        print("=" * 60)
        
        # Explore vision model structure
        if hasattr(self._model, 'vision_model'):
            print(f"Vision Model: {type(self._model.vision_model)}")
            if hasattr(self._model.vision_model, 'encoder'):
                if hasattr(self._model.vision_model.encoder, 'layers'):
                    print(f"  Vision Encoder Layers: {len(self._model.vision_model.encoder.layers)}")
                    # Show first layer structure
                    if len(self._model.vision_model.encoder.layers) > 0:
                        print(f"  First Layer Type: {type(self._model.vision_model.encoder.layers[0])}")
        else:
            print("Vision Model: Not found (model structure may differ)")
        
        # Explore language model structure
        if hasattr(self._model, 'model'):
            print(f"Language Model: {type(self._model.model)}")
            if hasattr(self._model.model, 'layers'):
                print(f"  Language Model Layers: {len(self._model.model.layers)}")
        else:
            print("Language Model: Not found (model structure may differ)")
        
        # Explore cross-modal components
        cross_modal_attrs = ['connector', 'cross_attn', 'cross_attention', 'proj', 'projector']
        found_components = []
        for attr in cross_modal_attrs:
            if hasattr(self._model, attr):
                found_components.append(attr)
        if found_components:
            print(f"Cross-modal Components: {', '.join(found_components)}")
        else:
            print("Cross-modal Components: Explore manually (structure may vary)")
        
        print("=" * 60)
        print("Tip: Use print(self._model) to see full model structure")
        print("=" * 60)
    
    def _optimize_vision_encoder(self):
        """
        Optimize Vision Encoder for high-resolution image inputs.
        
        Optimization Directions:
        1. Patch embedding convolution optimization
        2. Vision Transformer attention mechanism optimization
        3. Layer normalization optimization
        4. Memory-efficient image processing
        
        Implementation Steps:
        1. Inspect model structure: call self._explore_model_structure()
        2. Identify bottlenecks using profiling tools (PyTorch Profiler, nsys, etc.)
        3. Implement optimized operators (Triton/CUDA kernels)
        4. Replace original operators via monkey patch
        
        Target Components:
        - self._model.vision_model (if exists)
        - Vision encoder layers and attention mechanisms
        - Convolution operations in patch embedding
        """
        # TODO: Implement your Vision Encoder optimization here
        # 
        # Example workflow:
        # 1. from your_optimization import optimized_attention, optimized_conv
        # 2. Inspect: print(self._model.vision_model) to find target layers
        # 3. Replace: layer.self_attn.forward = optimized_attention
        # 4. Test: Run benchmark to verify improvement
        
        if 'vision_encoder' not in self._optimizations_applied:
            self._optimizations_applied.append('vision_encoder')
    
    def _optimize_kv_cache(self):
        """
        Replace the default KV path with a TurboQuant paged cache.
        """
        self._model.config.use_cache = True
        if hasattr(self._model.config, 'pad_token_id'):
            if self._model.config.pad_token_id is None:
                self._model.config.pad_token_id = self._model.config.eos_token_id
        if hasattr(self._model, "generation_config"):
            self._model.generation_config.use_cache = True

        if not hasattr(self._model, "_original_generate_for_turbo_kv"):
            self._model._original_generate_for_turbo_kv = self._model.generate
            self._model.generate = types.MethodType(
                _build_resetting_generate(self._model.generate),
                self._model,
            )

        text_model = getattr(self._model, "model", None)
        layers = getattr(text_model, "layers", None)
        patched_layers = 0

        if layers is not None:
            for layer_idx, layer in enumerate(layers):
                self_attn = getattr(layer, "self_attn", None)
                if self_attn is None:
                    continue

                required_attrs = [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "head_dim",
                    "num_key_value_heads",
                    "num_key_value_groups",
                    "scaling",
                    "layer_idx",
                ]
                if not all(hasattr(self_attn, attr) for attr in required_attrs):
                    continue

                self_attn._turbo_kv_page_size = 64
                self_attn._turbo_kv_key_bits = 3
                self_attn._turbo_kv_value_bits = 3
                self_attn._turbo_kv_runtime = _LayerKVRuntime(
                    page_size=self_attn._turbo_kv_page_size,
                    key_bits=self_attn._turbo_kv_key_bits,
                    value_bits=self_attn._turbo_kv_value_bits,
                    layer_seed=1009 + layer_idx,
                )

                if hasattr(self_attn, "_original_forward_for_turbo_kv"):
                    continue

                self_attn._original_forward_for_turbo_kv = self_attn.forward
                self_attn.forward = types.MethodType(
                    _build_turboquant_kv_forward(self_attn.forward),
                    self_attn,
                )
                patched_layers += 1

        if patched_layers > 0:
            print(
                f"[VLMModel] Applied TurboQuant paged KV cache "
                f"to {patched_layers} layers "
                f"(page size = 64, key bits = 3, value bits = 3, "
                f"keys use TurboQuantProd asymmetric attention, "
                f"values use TurboQuantMSE reconstruction, "
                f"page-table allocation, per-layer runtime reset on generate)"
            )
        
        if 'kv_cache' not in self._optimizations_applied:
            self._optimizations_applied.append('kv_cache')
    
    def _optimize_cross_modal_connector(self):
        """
        Optimize Cross-modal Connector computation efficiency.
        
        Optimization Directions:
        1. Cross-attention mechanism optimization
        2. Vision-to-language projection optimization
        3. Multi-modal fusion layer efficiency
        4. Feature alignment and transformation optimization
        
        Implementation Steps:
        1. Identify cross-modal components using self._explore_model_structure()
        2. Profile cross-modal operations to find bottlenecks
        3. Implement optimized cross-attention or projection kernels
        4. Replace original operations via monkey patch
        
        Note: Qwen3-VL's cross-modal structure may vary.
        Use model exploration to identify actual component names and locations.
        """
        # TODO: Implement your Cross-modal Connector optimization here
        # 
        # Example workflow:
        # 1. Explore: self._explore_model_structure() to find connector components
        # 2. from your_optimization import optimized_cross_attention
        # 3. Identify: Inspect model to find cross-attention layers
        # 4. Replace: connector.cross_attention.forward = optimized_cross_attention
        # 5. Test: Verify accuracy and performance improvements
        
        if 'cross_modal' not in self._optimizations_applied:
            self._optimizations_applied.append('cross_modal')
    
    def _enable_flash_attention(self):
        """
        Enable or implement Flash Attention optimization.
        
        Implementation Approaches:
        
        Approach 1: Enable PyTorch's Built-in Flash Attention (Simple)
            - Uses torch.backends.cuda.enable_flash_sdp(True)
            - Easy to enable but limited customization
            - May not work for all attention patterns in Qwen3-VL
        
        Approach 2: Implement Custom Flash Attention (Advanced, Recommended)
            - Write custom Triton/CUDA kernels for attention computation
            - Replace torch.nn.functional.scaled_dot_product_attention
            - Full control over attention computation and memory layout
            - Better performance potential but requires more implementation effort
        
        Recommended: Implement Approach 2 for better performance gains.
        Use profiling to identify which attention operations benefit most from optimization.
        """
        # TODO: Choose and implement your Flash Attention approach
        
        # Approach 1: Simple (enable PyTorch built-in)
        # torch.backends.cuda.enable_flash_sdp(True)
        
        # Approach 2: Advanced (custom implementation - recommended)
        # from your_optimization import custom_flash_attention
        # torch.nn.functional.scaled_dot_product_attention = custom_flash_attention
        # 
        # Or replace at layer level:
        # for layer in self._model.model.layers:
        #     layer.self_attn.forward = custom_attention_with_flash
        
        if 'flash_attention' not in self._optimizations_applied:
            self._optimizations_applied.append('flash_attention')
    
    def _apply_quantization(self):
        """
        Apply quantization to reduce model size and speed up inference.
        
        Optimization Directions:
        1. INT8 quantization (8-bit integer)
        2. FP8 quantization (8-bit floating point)
        3. Mixed precision quantization
        4. Dynamic vs static quantization
        
        Implementation Steps:
        1. Choose quantization strategy based on accuracy/performance trade-off
        2. Use quantization libraries (BitsAndBytes, TensorRT, etc.)
        3. Calibrate quantized model on validation data
        4. Verify accuracy preservation
        
        Note: Quantization may require reloading the model with quantization config.
        Consider applying quantization before other optimizations if model reload is needed.
        """
        # TODO: Implement your quantization here
        # 
        # Example workflow:
        # 1. from transformers import BitsAndBytesConfig
        # 2. quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        # 3. Note: May need to reload model with quantization config
        # 4. Test: Verify accuracy and performance improvements
        
        if 'quantization' not in self._optimizations_applied:
            self._optimizations_applied.append('quantization')
    
    # Required properties for benchmark
    @property
    def processor(self):
        """
        Required by benchmark for input processing.
        
        Benchmark uses this to prepare inputs with unified tokenizer.
        """
        return self._processor
    
    @property
    def model(self):
        """
        Required by benchmark for direct model.generate() calls.
        
        Benchmark directly calls self.model.generate() for performance testing.
        Your optimizations should modify this model object or its operators.
        """
        return self._model
    
    @property
    def device(self):
        """
        Required by benchmark for device information.
        """
        return self._device
    
    def generate(
        self, 
        image: Image.Image, 
        question: str, 
        max_new_tokens: int = 128
    ) -> Dict:
        """
        Generate answer (optional method, mainly for debugging).
        
        Note: Benchmark uses self.model.generate() directly for performance testing.
        This method is provided for convenience and debugging purposes.
        
        Args:
            image: PIL Image object
            question: Question text
            max_new_tokens: Maximum tokens to generate
        
        Returns:
            Dict: {
                "text": str,        # Generated text answer
                "token_count": int  # Generated token count
            }
        """
        # Build Qwen3-VL message format
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]
        }]
        
        # Process inputs
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self._device)
        
        # Generate
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                use_cache=True
            )
        
        # Extract generated tokens (remove input part)
        input_len = inputs.input_ids.shape[1]
        generated_ids = output_ids[0][input_len:]
        
        # Decode
        text = self._processor.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        return {
            "text": text,
            "token_count": len(generated_ids)
        }
