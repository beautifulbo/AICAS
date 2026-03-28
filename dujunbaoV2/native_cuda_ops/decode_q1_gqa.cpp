#include <torch/extension.h>

#include <vector>

torch::Tensor decode_q1_gqa_cuda(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    int64_t visible_len,
    double softmax_scale);

torch::Tensor decode_q1_gqa_append_cuda(
    torch::Tensor query,
    torch::Tensor current_key,
    torch::Tensor current_value,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    int64_t cache_write_pos,
    int64_t visible_len,
    double softmax_scale);

torch::Tensor rmsnorm_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    double eps);

torch::Tensor silu_mul_cuda(
    torch::Tensor gate,
    torch::Tensor up);

torch::Tensor gate_up_silu_cuda(
    torch::Tensor input,
    torch::Tensor gate_weight,
    torch::Tensor up_weight);

torch::Tensor rmsnorm_gate_up_silu_cuda(
    torch::Tensor input,
    torch::Tensor norm_weight,
    double eps,
    torch::Tensor gate_weight,
    torch::Tensor up_weight);

torch::Tensor linear_residual_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor residual);

torch::Tensor cublas_linear_residual_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor residual);

std::vector<torch::Tensor> dual_linear_cuda(
    torch::Tensor input,
    torch::Tensor weight0,
    torch::Tensor weight1);

std::vector<torch::Tensor> qkv_linear_cuda(
    torch::Tensor input,
    torch::Tensor q_weight,
    torch::Tensor k_weight,
    torch::Tensor v_weight);

std::vector<torch::Tensor> qkv_linear_qk_norm_cuda(
    torch::Tensor input,
    torch::Tensor q_weight,
    torch::Tensor k_weight,
    torch::Tensor v_weight,
    torch::Tensor q_norm_weight,
    torch::Tensor k_norm_weight,
    double eps,
    int64_t head_dim);

std::vector<torch::Tensor> packed_qkv_qk_norm_rope_cuda(
    torch::Tensor packed_qkv,
    int64_t q_out_features,
    int64_t k_out_features,
    int64_t v_out_features,
    torch::Tensor q_norm_weight,
    torch::Tensor k_norm_weight,
    torch::Tensor cos,
    torch::Tensor sin,
    double eps,
    int64_t head_dim);

torch::Tensor packed_qkv_qk_norm_rope_cache_attn_cuda(
    torch::Tensor packed_qkv,
    int64_t q_out_features,
    int64_t k_out_features,
    int64_t v_out_features,
    torch::Tensor q_norm_weight,
    torch::Tensor k_norm_weight,
    torch::Tensor cos,
    torch::Tensor sin,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    int64_t cache_write_pos,
    int64_t visible_len,
    double eps,
    int64_t head_dim,
    double softmax_scale);

torch::Tensor lm_head_argmax_cuda(
    torch::Tensor hidden_states,
    torch::Tensor weight);

namespace {

void check_cuda_tensor(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat16, name, " must be float16");
}

}  // namespace

torch::Tensor decode_q1_gqa_forward(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    int64_t visible_len,
    double softmax_scale) {
    check_cuda_tensor(query, "query");
    check_cuda_tensor(key_cache, "key_cache");
    check_cuda_tensor(value_cache, "value_cache");

    TORCH_CHECK(query.dim() == 3, "query must have shape [batch, num_heads, head_dim]");
    TORCH_CHECK(key_cache.dim() == 4, "key_cache must have shape [batch, num_kv_heads, seq_len, head_dim]");
    TORCH_CHECK(value_cache.dim() == 4, "value_cache must have shape [batch, num_kv_heads, seq_len, head_dim]");
    TORCH_CHECK(query.size(0) == key_cache.size(0), "batch size mismatch");
    TORCH_CHECK(query.size(0) == value_cache.size(0), "batch size mismatch");
    TORCH_CHECK(key_cache.sizes() == value_cache.sizes(), "key/value cache shape mismatch");
    TORCH_CHECK(query.size(2) == key_cache.size(3), "head_dim mismatch");
    TORCH_CHECK(query.size(1) % key_cache.size(1) == 0, "num_heads must be divisible by num_kv_heads");
    TORCH_CHECK(visible_len > 0, "visible_len must be positive");
    TORCH_CHECK(visible_len <= key_cache.size(2), "visible_len exceeds cache length");

    return decode_q1_gqa_cuda(query, key_cache, value_cache, visible_len, softmax_scale);
}

torch::Tensor decode_q1_gqa_append_forward(
    torch::Tensor query,
    torch::Tensor current_key,
    torch::Tensor current_value,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    int64_t cache_write_pos,
    int64_t visible_len,
    double softmax_scale) {
    check_cuda_tensor(query, "query");
    check_cuda_tensor(current_key, "current_key");
    check_cuda_tensor(current_value, "current_value");
    check_cuda_tensor(key_cache, "key_cache");
    check_cuda_tensor(value_cache, "value_cache");

    TORCH_CHECK(query.dim() == 3, "query must have shape [batch, num_heads, head_dim]");
    TORCH_CHECK(current_key.dim() == 3, "current_key must have shape [batch, num_kv_heads, head_dim]");
    TORCH_CHECK(current_value.dim() == 3, "current_value must have shape [batch, num_kv_heads, head_dim]");
    TORCH_CHECK(key_cache.dim() == 4, "key_cache must have shape [batch, num_kv_heads, seq_len, head_dim]");
    TORCH_CHECK(value_cache.dim() == 4, "value_cache must have shape [batch, num_kv_heads, seq_len, head_dim]");
    TORCH_CHECK(query.size(0) == current_key.size(0), "batch size mismatch");
    TORCH_CHECK(query.size(0) == current_value.size(0), "batch size mismatch");
    TORCH_CHECK(query.size(0) == key_cache.size(0), "batch size mismatch");
    TORCH_CHECK(key_cache.sizes() == value_cache.sizes(), "key/value cache shape mismatch");
    TORCH_CHECK(current_key.sizes() == current_value.sizes(), "current key/value shape mismatch");
    TORCH_CHECK(current_key.size(1) == key_cache.size(1), "num_kv_heads mismatch");
    TORCH_CHECK(current_key.size(2) == key_cache.size(3), "head_dim mismatch");
    TORCH_CHECK(query.size(2) == key_cache.size(3), "head_dim mismatch");
    TORCH_CHECK(query.size(1) % key_cache.size(1) == 0, "num_heads must be divisible by num_kv_heads");
    TORCH_CHECK(visible_len > 0, "visible_len must be positive");
    TORCH_CHECK(visible_len <= key_cache.size(2), "visible_len exceeds cache length");
    TORCH_CHECK(cache_write_pos >= 0, "cache_write_pos must be non-negative");
    TORCH_CHECK(cache_write_pos < key_cache.size(2), "cache_write_pos exceeds cache length");
    TORCH_CHECK(cache_write_pos < visible_len, "cache_write_pos must be inside visible range");

    return decode_q1_gqa_append_cuda(
        query,
        current_key,
        current_value,
        key_cache,
        value_cache,
        cache_write_pos,
        visible_len,
        softmax_scale);
}

torch::Tensor rmsnorm_forward(torch::Tensor input, torch::Tensor weight, double eps) {
    check_cuda_tensor(input, "input");
    check_cuda_tensor(weight, "weight");

    TORCH_CHECK(weight.dim() == 1, "weight must have shape [hidden_size]");
    TORCH_CHECK(input.dim() >= 1, "input must have at least 1 dimension");
    TORCH_CHECK(input.size(-1) == weight.size(0), "input hidden size must match weight");

    return rmsnorm_cuda(input, weight, eps);
}

torch::Tensor silu_mul_forward(torch::Tensor gate, torch::Tensor up) {
    check_cuda_tensor(gate, "gate");
    check_cuda_tensor(up, "up");
    TORCH_CHECK(gate.sizes() == up.sizes(), "gate and up must have the same shape");
    return silu_mul_cuda(gate, up);
}

torch::Tensor gate_up_silu_forward(
    torch::Tensor input,
    torch::Tensor gate_weight,
    torch::Tensor up_weight) {
    check_cuda_tensor(input, "input");
    check_cuda_tensor(gate_weight, "gate_weight");
    check_cuda_tensor(up_weight, "up_weight");

    TORCH_CHECK(input.dim() >= 2, "input must have at least 2 dimensions");
    TORCH_CHECK(gate_weight.dim() == 2, "gate_weight must have shape [out_features, in_features]");
    TORCH_CHECK(up_weight.dim() == 2, "up_weight must have shape [out_features, in_features]");
    TORCH_CHECK(gate_weight.sizes() == up_weight.sizes(), "gate_weight and up_weight must share the same shape");
    TORCH_CHECK(input.size(-1) == gate_weight.size(1), "input hidden size must match gate_weight in_features");

    return gate_up_silu_cuda(input, gate_weight, up_weight);
}

torch::Tensor rmsnorm_gate_up_silu_forward(
    torch::Tensor input,
    torch::Tensor norm_weight,
    double eps,
    torch::Tensor gate_weight,
    torch::Tensor up_weight) {
    check_cuda_tensor(input, "input");
    check_cuda_tensor(norm_weight, "norm_weight");
    check_cuda_tensor(gate_weight, "gate_weight");
    check_cuda_tensor(up_weight, "up_weight");

    TORCH_CHECK(input.dim() >= 2, "input must have at least 2 dimensions");
    TORCH_CHECK(norm_weight.dim() == 1, "norm_weight must have shape [hidden_size]");
    TORCH_CHECK(gate_weight.dim() == 2, "gate_weight must have shape [out_features, in_features]");
    TORCH_CHECK(up_weight.dim() == 2, "up_weight must have shape [out_features, in_features]");
    TORCH_CHECK(gate_weight.sizes() == up_weight.sizes(), "gate_weight and up_weight must share the same shape");
    TORCH_CHECK(input.size(-1) == norm_weight.size(0), "input hidden size must match norm_weight");
    TORCH_CHECK(input.size(-1) == gate_weight.size(1), "input hidden size must match gate_weight in_features");

    return rmsnorm_gate_up_silu_cuda(input, norm_weight, eps, gate_weight, up_weight);
}

torch::Tensor linear_residual_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor residual) {
    check_cuda_tensor(input, "input");
    check_cuda_tensor(weight, "weight");
    check_cuda_tensor(residual, "residual");

    TORCH_CHECK(input.dim() >= 2, "input must have at least 2 dimensions");
    TORCH_CHECK(weight.dim() == 2, "weight must have shape [out_features, in_features]");
    TORCH_CHECK(residual.dim() == input.dim(), "residual dims must match input dims");
    for (int64_t dim = 0; dim < input.dim() - 1; ++dim) {
        TORCH_CHECK(residual.size(dim) == input.size(dim), "residual leading dims must match input");
    }
    TORCH_CHECK(residual.size(-1) == weight.size(0), "residual hidden size must match weight out_features");
    TORCH_CHECK(input.size(-1) == weight.size(1), "input hidden size must match weight in_features");

    return linear_residual_cuda(input, weight, residual);
}

torch::Tensor cublas_linear_residual_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor residual) {
    check_cuda_tensor(input, "input");
    check_cuda_tensor(weight, "weight");
    check_cuda_tensor(residual, "residual");

    TORCH_CHECK(input.dim() >= 2, "input must have at least 2 dimensions");
    TORCH_CHECK(weight.dim() == 2, "weight must have shape [out_features, in_features]");
    TORCH_CHECK(residual.dim() == input.dim(), "residual dims must match input dims");
    for (int64_t dim = 0; dim < input.dim() - 1; ++dim) {
        TORCH_CHECK(residual.size(dim) == input.size(dim), "residual leading dims must match input");
    }
    TORCH_CHECK(residual.size(-1) == weight.size(0), "residual hidden size must match weight out_features");
    TORCH_CHECK(input.size(-1) == weight.size(1), "input hidden size must match weight in_features");

    return cublas_linear_residual_cuda(input, weight, residual);
}

std::vector<torch::Tensor> dual_linear_forward(
    torch::Tensor input,
    torch::Tensor weight0,
    torch::Tensor weight1) {
    check_cuda_tensor(input, "input");
    check_cuda_tensor(weight0, "weight0");
    check_cuda_tensor(weight1, "weight1");

    TORCH_CHECK(input.dim() >= 2, "input must have at least 2 dimensions");
    TORCH_CHECK(weight0.dim() == 2, "weight0 must have shape [out_features, in_features]");
    TORCH_CHECK(weight1.dim() == 2, "weight1 must have shape [out_features, in_features]");
    TORCH_CHECK(weight0.sizes() == weight1.sizes(), "weight0 and weight1 must share the same shape");
    TORCH_CHECK(input.size(-1) == weight0.size(1), "input hidden size must match weight in_features");

    return dual_linear_cuda(input, weight0, weight1);
}

std::vector<torch::Tensor> qkv_linear_forward(
    torch::Tensor input,
    torch::Tensor q_weight,
    torch::Tensor k_weight,
    torch::Tensor v_weight) {
    check_cuda_tensor(input, "input");
    check_cuda_tensor(q_weight, "q_weight");
    check_cuda_tensor(k_weight, "k_weight");
    check_cuda_tensor(v_weight, "v_weight");

    TORCH_CHECK(input.dim() >= 2, "input must have at least 2 dimensions");
    TORCH_CHECK(q_weight.dim() == 2, "q_weight must have shape [out_features, in_features]");
    TORCH_CHECK(k_weight.dim() == 2, "k_weight must have shape [out_features, in_features]");
    TORCH_CHECK(v_weight.dim() == 2, "v_weight must have shape [out_features, in_features]");
    TORCH_CHECK(
        q_weight.size(1) == k_weight.size(1) && q_weight.size(1) == v_weight.size(1),
        "q/k/v weights must share the same in_features");
    TORCH_CHECK(input.size(-1) == q_weight.size(1), "input hidden size must match q_weight in_features");

    return qkv_linear_cuda(input, q_weight, k_weight, v_weight);
}

std::vector<torch::Tensor> qkv_linear_qk_norm_forward(
    torch::Tensor input,
    torch::Tensor q_weight,
    torch::Tensor k_weight,
    torch::Tensor v_weight,
    torch::Tensor q_norm_weight,
    torch::Tensor k_norm_weight,
    double eps,
    int64_t head_dim) {
    check_cuda_tensor(input, "input");
    check_cuda_tensor(q_weight, "q_weight");
    check_cuda_tensor(k_weight, "k_weight");
    check_cuda_tensor(v_weight, "v_weight");
    check_cuda_tensor(q_norm_weight, "q_norm_weight");
    check_cuda_tensor(k_norm_weight, "k_norm_weight");

    TORCH_CHECK(input.dim() >= 2, "input must have at least 2 dimensions");
    TORCH_CHECK(q_weight.dim() == 2, "q_weight must have shape [out_features, in_features]");
    TORCH_CHECK(k_weight.dim() == 2, "k_weight must have shape [out_features, in_features]");
    TORCH_CHECK(v_weight.dim() == 2, "v_weight must have shape [out_features, in_features]");
    TORCH_CHECK(q_norm_weight.dim() == 1, "q_norm_weight must have shape [head_dim]");
    TORCH_CHECK(k_norm_weight.dim() == 1, "k_norm_weight must have shape [head_dim]");
    TORCH_CHECK(
        q_weight.size(1) == k_weight.size(1) && q_weight.size(1) == v_weight.size(1),
        "q/k/v weights must share the same in_features");
    TORCH_CHECK(input.size(-1) == q_weight.size(1), "input hidden size must match q_weight in_features");
    TORCH_CHECK(head_dim > 0, "head_dim must be positive");
    TORCH_CHECK(q_weight.size(0) % head_dim == 0, "q_weight out_features must be divisible by head_dim");
    TORCH_CHECK(k_weight.size(0) % head_dim == 0, "k_weight out_features must be divisible by head_dim");
    TORCH_CHECK(q_norm_weight.size(0) == head_dim, "q_norm_weight size must match head_dim");
    TORCH_CHECK(k_norm_weight.size(0) == head_dim, "k_norm_weight size must match head_dim");

    return qkv_linear_qk_norm_cuda(
        input,
        q_weight,
        k_weight,
        v_weight,
        q_norm_weight,
        k_norm_weight,
        eps,
        head_dim);
}

std::vector<torch::Tensor> packed_qkv_qk_norm_rope_forward(
    torch::Tensor packed_qkv,
    int64_t q_out_features,
    int64_t k_out_features,
    int64_t v_out_features,
    torch::Tensor q_norm_weight,
    torch::Tensor k_norm_weight,
    torch::Tensor cos,
    torch::Tensor sin,
    double eps,
    int64_t head_dim) {
    check_cuda_tensor(packed_qkv, "packed_qkv");
    check_cuda_tensor(q_norm_weight, "q_norm_weight");
    check_cuda_tensor(k_norm_weight, "k_norm_weight");
    check_cuda_tensor(cos, "cos");
    check_cuda_tensor(sin, "sin");

    TORCH_CHECK(packed_qkv.dim() == 3, "packed_qkv must have shape [batch, seq, qkv_out_features]");
    TORCH_CHECK(q_norm_weight.dim() == 1, "q_norm_weight must have shape [head_dim]");
    TORCH_CHECK(k_norm_weight.dim() == 1, "k_norm_weight must have shape [head_dim]");
    TORCH_CHECK(cos.dim() == 3 && sin.dim() == 3, "cos/sin must have shape [batch, seq, head_dim]");
    TORCH_CHECK(
        cos.size(0) == packed_qkv.size(0) && cos.size(1) == packed_qkv.size(1),
        "cos shape must match packed_qkv batch/seq dims");
    TORCH_CHECK(cos.sizes() == sin.sizes(), "cos/sin shape mismatch");

    TORCH_CHECK(head_dim > 0, "head_dim must be positive");
    TORCH_CHECK(head_dim % 2 == 0, "head_dim must be even for RoPE");
    TORCH_CHECK(q_out_features > 0 && k_out_features > 0 && v_out_features > 0, "q/k/v out_features must be positive");
    TORCH_CHECK(
        q_out_features + k_out_features + v_out_features == packed_qkv.size(2),
        "q/k/v out_features must sum to packed_qkv.size(-1)");
    TORCH_CHECK(k_out_features == v_out_features, "k/v out_features must match for attention cache update");
    TORCH_CHECK(q_out_features % head_dim == 0, "q_out_features must be divisible by head_dim");
    TORCH_CHECK(k_out_features % head_dim == 0, "k_out_features must be divisible by head_dim");
    TORCH_CHECK(v_out_features % head_dim == 0, "v_out_features must be divisible by head_dim");
    TORCH_CHECK(q_norm_weight.size(0) == head_dim, "q_norm_weight size must match head_dim");
    TORCH_CHECK(k_norm_weight.size(0) == head_dim, "k_norm_weight size must match head_dim");
    TORCH_CHECK(cos.size(2) == head_dim, "cos/sin last dim must match head_dim");

    return packed_qkv_qk_norm_rope_cuda(
        packed_qkv,
        q_out_features,
        k_out_features,
        v_out_features,
        q_norm_weight,
        k_norm_weight,
        cos,
        sin,
        eps,
        head_dim);
}

torch::Tensor packed_qkv_qk_norm_rope_cache_attn_forward(
    torch::Tensor packed_qkv,
    int64_t q_out_features,
    int64_t k_out_features,
    int64_t v_out_features,
    torch::Tensor q_norm_weight,
    torch::Tensor k_norm_weight,
    torch::Tensor cos,
    torch::Tensor sin,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    int64_t cache_write_pos,
    int64_t visible_len,
    double eps,
    int64_t head_dim,
    double softmax_scale) {
    check_cuda_tensor(packed_qkv, "packed_qkv");
    check_cuda_tensor(q_norm_weight, "q_norm_weight");
    check_cuda_tensor(k_norm_weight, "k_norm_weight");
    check_cuda_tensor(cos, "cos");
    check_cuda_tensor(sin, "sin");
    check_cuda_tensor(key_cache, "key_cache");
    check_cuda_tensor(value_cache, "value_cache");

    TORCH_CHECK(packed_qkv.dim() == 3, "packed_qkv must have shape [batch, seq, qkv_out_features]");
    TORCH_CHECK(packed_qkv.size(1) == 1, "packed_qkv cache-attn path currently only supports seq=1");
    TORCH_CHECK(q_norm_weight.dim() == 1, "q_norm_weight must have shape [head_dim]");
    TORCH_CHECK(k_norm_weight.dim() == 1, "k_norm_weight must have shape [head_dim]");
    TORCH_CHECK(cos.dim() == 3 && sin.dim() == 3, "cos/sin must have shape [batch, seq, head_dim]");
    TORCH_CHECK(cos.sizes() == sin.sizes(), "cos/sin shape mismatch");
    TORCH_CHECK(key_cache.dim() == 4 && value_cache.dim() == 4, "key/value cache must have shape [batch, num_kv_heads, seq, head_dim]");
    TORCH_CHECK(key_cache.sizes() == value_cache.sizes(), "key/value cache shape mismatch");
    TORCH_CHECK(key_cache.size(0) == packed_qkv.size(0), "batch size mismatch");
    TORCH_CHECK(cos.size(0) == packed_qkv.size(0) && cos.size(1) == packed_qkv.size(1), "cos shape must match packed_qkv batch/seq dims");

    TORCH_CHECK(head_dim > 0, "head_dim must be positive");
    TORCH_CHECK(head_dim % 2 == 0, "head_dim must be even for RoPE");
    TORCH_CHECK(q_out_features > 0 && k_out_features > 0 && v_out_features > 0, "q/k/v out_features must be positive");
    TORCH_CHECK(q_out_features + k_out_features + v_out_features == packed_qkv.size(2), "q/k/v out_features must sum to packed_qkv.size(-1)");
    TORCH_CHECK(k_out_features == v_out_features, "k/v out_features must match");
    TORCH_CHECK(q_out_features % head_dim == 0, "q_out_features must be divisible by head_dim");
    TORCH_CHECK(k_out_features % head_dim == 0, "k_out_features must be divisible by head_dim");
    TORCH_CHECK(v_out_features % head_dim == 0, "v_out_features must be divisible by head_dim");
    TORCH_CHECK(q_norm_weight.size(0) == head_dim, "q_norm_weight size must match head_dim");
    TORCH_CHECK(k_norm_weight.size(0) == head_dim, "k_norm_weight size must match head_dim");
    TORCH_CHECK(cos.size(2) == head_dim, "cos/sin last dim must match head_dim");
    TORCH_CHECK(visible_len > 0, "visible_len must be positive");
    TORCH_CHECK(visible_len <= key_cache.size(2), "visible_len exceeds cache length");
    TORCH_CHECK(cache_write_pos >= 0 && cache_write_pos < visible_len, "cache_write_pos must be inside visible range");

    return packed_qkv_qk_norm_rope_cache_attn_cuda(
        packed_qkv,
        q_out_features,
        k_out_features,
        v_out_features,
        q_norm_weight,
        k_norm_weight,
        cos,
        sin,
        key_cache,
        value_cache,
        cache_write_pos,
        visible_len,
        eps,
        head_dim,
        softmax_scale);
}

torch::Tensor lm_head_argmax_forward(
    torch::Tensor hidden_states,
    torch::Tensor weight) {
    check_cuda_tensor(hidden_states, "hidden_states");
    check_cuda_tensor(weight, "weight");

    TORCH_CHECK(hidden_states.dim() == 3, "hidden_states must have shape [batch, seq, hidden_size]");
    TORCH_CHECK(hidden_states.size(0) == 1, "lm_head_argmax currently only supports batch=1");
    TORCH_CHECK(hidden_states.size(1) == 1, "lm_head_argmax currently only supports seq=1");
    TORCH_CHECK(weight.dim() == 2, "weight must have shape [vocab_size, hidden_size]");
    TORCH_CHECK(hidden_states.size(2) == weight.size(1), "hidden size must match lm_head in_features");

    return lm_head_argmax_cuda(hidden_states, weight);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "decode_q1_gqa_forward",
        &decode_q1_gqa_forward,
        "Decode-only q_len=1 GQA attention forward (CUDA)");
    m.def(
        "decode_q1_gqa_append_forward",
        &decode_q1_gqa_append_forward,
        "Decode-only q_len=1 GQA attention with in-kernel cache append (CUDA)");
    m.def(
        "rmsnorm_forward",
        &rmsnorm_forward,
        "Decode-only RMSNorm forward (CUDA)");
    m.def(
        "silu_mul_forward",
        &silu_mul_forward,
        "Decode-only SiLU(gate) * up forward (CUDA)");
    m.def(
        "gate_up_silu_forward",
        &gate_up_silu_forward,
        "Decode-only fused gate_proj + up_proj + silu*mul forward (CUDA)");
    m.def(
        "rmsnorm_gate_up_silu_forward",
        &rmsnorm_gate_up_silu_forward,
        "Decode-only fused RMSNorm + gate_proj + up_proj + silu*mul forward (CUDA)");
    m.def(
        "linear_residual_forward",
        &linear_residual_forward,
        "Decode-only down_proj + residual add forward (CUDA)");
    m.def(
        "cublas_linear_residual_forward",
        &cublas_linear_residual_forward,
        "Decode-only linear + residual add via cuBLAS GEMM (CUDA)");
    m.def(
        "dual_linear_forward",
        &dual_linear_forward,
        "Dual linear forward using a custom CUDA matvec kernel");
    m.def(
        "qkv_linear_forward",
        &qkv_linear_forward,
        "Fused qkv linear forward using a custom CUDA matvec kernel");
    m.def(
        "qkv_linear_qk_norm_forward",
        &qkv_linear_qk_norm_forward,
        "Fused qkv linear with in-kernel q/k RMSNorm for decode-only path");
    m.def(
        "packed_qkv_qk_norm_rope_forward",
        &packed_qkv_qk_norm_rope_forward,
        "Packed qkv post-GEMM fusion for q/k RMSNorm + RoPE + attention layout");
    m.def(
        "packed_qkv_qk_norm_rope_cache_attn_forward",
        &packed_qkv_qk_norm_rope_cache_attn_forward,
        "Packed qkv post-GEMM fusion for q/k RMSNorm + RoPE + cache append + attention");
    m.def(
        "lm_head_argmax_forward",
        &lm_head_argmax_forward,
        "Decode-only lm_head matvec + argmax fusion (CUDA)");
}
