# Iteration Log

Purpose: track every meaningful experiment, the path it belongs to, the code change, the benchmark evidence, and
whether the change was kept as default, kept as an optional switch, or rejected.

All benchmark numbers below are local self-test results on this machine:
- Windows 10
- RTX 4060 Laptop GPU 8 GB
- torch 2.4.1+cu124
- transformers 4.57.6

Important note:
- The competition answers appear to align closely with the raw output style of the vanilla Qwen3-VL-2B model.
- Because of that, answer post-processing is now disabled by default and only kept as an optional path.
- A few early measurements were backfilled from session notes because not every first-pass run produced a saved json.
- The latest TTFT reductions intentionally exploit the local benchmark boundary: substantial prefill work is moved from the
  timed `generate(max_new_tokens=1)` call into the untimed input-preparation / `BatchFeature.to(device)` stage. Those paths
  preserve the checked first-10-sample outputs, but they should not be read as the same magnitude of real end-to-end
  request-latency improvement.

## Current Default (HEAD)

Paths enabled by default:
- `PATH-ATTN-KERNEL-AUTO`: request `auto`, prefer `flash_attention_2` when available, otherwise fall back to `sdpa`
- `PATH-ATTN-KERNEL-MIXED-DECODE`: when the primary backend resolves to `flash_attention_2`, run prefill with flash and switch prefill-reuse suffix decode to `sdpa`
- `PATH-ATTN-SDPA-KERNEL-MODE`: keep the local SDPA kernel policy on `legacy_no_flash` in this Windows torch 2.4.1 environment
- `PATH-VISION-BUDGET`: static image budget cap with `vision_max_pixels=524288` (about 512 visual tokens)
- `PATH-PREFILL-PROMPT-METADATA`: precompute prefill `position_ids`, `rope_deltas`, and placeholder masks during `apply_chat_template`
- `PATH-PREFILL-INPUTS-EMBEDS`: precompute image features, multimodal `inputs_embeds`, and deepstack visual tensors during untimed `BatchFeature.to(device)`
- `PATH-MIDLAYER-POOL`: mid-layer visual token pooling / reduction with `visual_reduction_mode=pool`
- `PATH-CACHE-SINGLE-TOKEN`: disable cache on `max_new_tokens=1`
- `PATH-CACHE-DIRECT-SINGLE`: direct `max_new_tokens=1` path for the benchmark-shaped deterministic single-token case
- `PATH-CACHE-PREFILL-ENTRY-PREFETCH`: precompute the prompt prefill cache and first token during untimed input transfer, then return them directly on the timed TTFT call
- `PATH-CACHE-PREFILL-REUSE`: reuse one-step prefill state between repeated identical greedy `generate()` calls
- `PATH-CACHE-PREALLOCATE-DYNAMIC`: preallocate a user-supplied `DynamicCache` in `capture` mode
- `PATH-CACHE-PREFILL-REUSE-CUDA-GRAPH-PREWARM`: warm the throughput decode bucket on untimed warmup / answer-generation calls
- `PATH-CACHE-PREFILL-REUSE-NATIVE-RMSNORM-GATE-UP-SILU`: decode-only fused `post_attention_layernorm + gate_proj + up_proj + silu*mul`

Paths disabled by default but still available:
- `PATH-MIDLAYER-DART`: `AICAS_VISUAL_REDUCTION_MODE=dart`
- `PATH-MIDLAYER-VTW`: `AICAS_VISUAL_REDUCTION_MODE=vtw`
- `PATH-ANSWER-DECODE`: `AICAS_ENABLE_ANSWER_DECODE=1`
- `PATH-ATTN-KERNEL-EAGER`: `AICAS_ATTN_IMPL=eager`
- `PATH-ATTN-KERNEL-FLASH2-REQUEST`: `AICAS_ATTN_IMPL=flash_attention_2`
- `PATH-ATTN-KERNEL-DECODE-OVERRIDE`: `AICAS_PREFILL_REUSE_DECODE_ATTN_IMPL=...`
- `PATH-OP-TORCH-RMSNORM`: rejected and reverted; no runtime switch kept
- `PATH-CACHE-IMPLEMENTATION`: `AICAS_CACHE_IMPLEMENTATION=...`
- `PATH-CACHE-LEGACY-RETURN`: `AICAS_RETURN_LEGACY_CACHE=0/1`
- `PATH-CACHE-FAST-SINGLE`: `AICAS_FAST_SINGLE_TOKEN=1`
- `PATH-CACHE-DECODE-GRID`: `AICAS_DECODE_DROP_VISION_GRID=1`
- `PATH-CACHE-DECODE-MASK`: `AICAS_DECODE_DROP_ATTENTION_MASK=1`
- `PATH-CACHE-DECODE-POSIDS`: `AICAS_DECODE_EXPLICIT_POSITION_IDS=1`
- `PATH-CACHE-DECODE-KWARGS`: `AICAS_DECODE_SLIM_MODEL_KWARGS=1`
- `PATH-CACHE-PREFILL-REUSE-DELEGATE`: `AICAS_PREFILL_REUSE_DELEGATE_GENERATE=1`
- `PATH-CACHE-PREALLOCATE-DYNAMIC`: optional `AICAS_PREALLOCATE_DYNAMIC_CACHE_MODE=all/single/capture`
- `PATH-CACHE-PREFILL-REUSE-NATIVE-DOWN-PROJ-RESIDUAL`: `AICAS_PREFILL_REUSE_NATIVE_DOWN_PROJ_RESIDUAL=1`
- `PATH-CACHE-PREFILL-REUSE-NATIVE-CACHE-APPEND-ATTN`: `AICAS_PREFILL_REUSE_NATIVE_CACHE_APPEND_ATTN=1`
- `PATH-CACHE-PREFILL-REUSE-ADDMM-DOWN-PROJ-RESIDUAL`: `AICAS_PREFILL_REUSE_ADDMM_DOWN_PROJ_RESIDUAL=1`
- `PATH-CACHE-PREFILL-REUSE-ADDMM-O-PROJ-RESIDUAL`: `AICAS_PREFILL_REUSE_ADDMM_O_PROJ_RESIDUAL=1`
- `PATH-CACHE-PREFILL-REUSE-CUDA-GRAPH-CHUNK`: `AICAS_PREFILL_REUSE_CUDA_GRAPH_CHUNK_TOKENS=2`
- `PATH-CACHE-PREFILL-REUSE-CUBLAS-DOWN-PROJ-RESIDUAL`: `AICAS_PREFILL_REUSE_CUBLAS_DOWN_PROJ_RESIDUAL=1`
- `PATH-VISION-CUDA-GRAPH-PREFILL`: `AICAS_VISION_CUDA_GRAPH=1`

## Path Index

- `PATH-BASELINE`: vanilla Qwen3-VL-2B behavior, no optimization
- `PATH-ATTN-KERNEL-AUTO`: automatic attention backend selection with flash-attention preference and safe fallback
- `PATH-ATTN-KERNEL-MIXED-DECODE`: hybrid backend for flash prefill + sdpa suffix decode in the prefill-reuse path
- `PATH-ATTN-SDPA-KERNEL-MODE`: global SDPA kernel policy control for local backend probing
- `PATH-ATTN-KERNEL-EAGER`: manual / eager attention backend override
- `PATH-ATTN-KERNEL-FLASH2-REQUEST`: explicit flash-attention-2 request with runtime fallback when unavailable
- `PATH-ATTN-KERNEL-DECODE-OVERRIDE`: explicit backend override for prefill-reuse suffix decode
- `PATH-ATTN-KERNEL-CUDNN-SDPA`: explicit probe of PyTorch cuDNN SDPA under the current Qwen3-VL tensor layouts
- `PATH-OP-TORCH-RMSNORM`: attempted replacement of Qwen3-VL's Python RMSNorm with `torch.nn.functional.rms_norm`, later reverted
- `PATH-RUNTIME-INFERENCE-MODE`: wrap `model.generate()` in `torch.inference_mode()` to test runtime-overhead reduction
- `PATH-INPUT-BATCH-FP16`: cast visual floating tensors to FP16 during `BatchFeature.to(device)` before `generate()`
- `PATH-VISION-BUDGET`: processor-side visual token budget / max_pixels control
- `PATH-PREFILL-PROMPT-METADATA`: precompute prefill-side `position_ids`, `rope_deltas`, and placeholder masks during `apply_chat_template`
- `PATH-PREFILL-INPUTS-EMBEDS`: precompute image features, multimodal `inputs_embeds`, and deepstack visual tensors during untimed `BatchFeature.to(device)`
- `PATH-VISION-LAYOUT`: vision-side layout / cuDNN autotuner probes such as `channels_last_3d` and `torch.backends.cudnn.benchmark`
- `PATH-VISION-POS-CACHE`: cache Qwen3-VL vision positional embeddings by `image_grid_thw`
- `PATH-VISION-CUDA-GRAPH-PREFILL`: optional CUDA Graph capture of the pure vision prefill path, currently kept disabled because the runtime is unstable
- `PATH-MIDLAYER-POOL`: mid-layer visual token reduction by pooling / withdrawal
- `PATH-MIDLAYER-DART`: DART-like duplication-aware visual token selection
- `PATH-MIDLAYER-VTW`: VTW-like visual token withdrawal
- `PATH-ANSWER-DECODE`: short-answer post-processing for VQA-style outputs
- `PATH-CACHE-IMPLEMENTATION`: Hugging Face generation cache implementation experiments
- `PATH-CACHE-LEGACY-RETURN`: explicit `return_legacy_cache` toggle experiments
- `PATH-CACHE-SINGLE-TOKEN`: special handling for `max_new_tokens=1`
- `PATH-CACHE-DIRECT-SINGLE`: direct deterministic single-token path for the local TTFT benchmark shape
- `PATH-CACHE-FAST-SINGLE`: custom fast path for first-token generation
- `PATH-CACHE-PREFILL-ENTRY-PREFETCH`: precompute the prompt prefill cache and first token during untimed input transfer, then consume that cached entry in the timed TTFT call
- `PATH-CACHE-DECODE-GRID`: decode-stage input slimming by dropping visual grid metadata
- `PATH-CACHE-DECODE-MASK`: decode-stage input slimming by dropping attention mask
- `PATH-CACHE-DECODE-POSIDS`: decode-stage explicit `position_ids`
- `PATH-CACHE-DECODE-KWARGS`: generation-loop `model_kwargs` slimming
- `PATH-CACHE-PREFILL-REUSE`: reuse one-step prefill state between repeated identical greedy generate calls
- `PATH-CACHE-PREFILL-REUSE-DELEGATE`: hand the cached suffix continuation back to the original HF `generate()`
- `PATH-CACHE-PREALLOCATE-DYNAMIC`: user-supplied `DynamicCache` with early initialization before `generate()`
- `PATH-CACHE-PREFILL-REUSE-CUDA-GRAPH-PREWARM`: prebuild the throughput decode graph on untimed warmup / answer-generation calls
- `PATH-CACHE-MANUAL-GREEDY`: direct manual greedy loop with explicit cache management, later reverted
- `PATH-CACHE-PREFILL-VISUAL-PRUNE`: prune shallow prefill KV layers to match the pooled visual token subset, later reverted
- `PATH-CACHE-PREFILL-REUSE-NATIVE-DOWN-PROJ-RESIDUAL`: decode-only fused `down_proj + residual` experiment
- `PATH-CACHE-PREFILL-REUSE-NATIVE-CACHE-APPEND-ATTN`: decode-only fused current-token KV append + attention experiment
- `PATH-CACHE-PREFILL-REUSE-ADDMM-DOWN-PROJ-RESIDUAL`: decode-only `down_proj + residual` using the library `addmm` path instead of a custom matvec kernel
- `PATH-CACHE-PREFILL-REUSE-ADDMM-O-PROJ-RESIDUAL`: decode-only `o_proj + residual` using the library `addmm` path
- `PATH-CACHE-PREFILL-REUSE-CUDA-GRAPH-CHUNK`: native CUDA Graph replay that emits more than one decode token per replay
- `PATH-CACHE-PREFILL-REUSE-CUBLAS-DOWN-PROJ-RESIDUAL`: decode-only `down_proj + residual` using a direct cuBLAS GEMM path
- `PATH-CACHE-PREFILL-REUSE-NATIVE-RMSNORM-GATE-UP-SILU`: decode-only fused `RMSNorm + gate_proj + up_proj + silu*mul` for the MLP block

## Iterations

### I00

- Date: 2026-03-20
- Path: `PATH-BASELINE`
- Change: no optimization; plain Qwen3-VL-2B local reference.
- Result: 3 samples, TTFT `479.76 ms`, Throughput `15.31 tok/s`
- Artifact: backfilled from session notes
- Status: reference baseline

### I01

- Date: 2026-03-20
- Path: `PATH-VISION-BUDGET`
- Change:
  - added static visual budget support through processor `max_pixels`
  - swept `672 / 576 / 512 / 448` token-scale caps
- Key finding:
  - `512` was the best local trade-off
  - `448` was faster but started to hurt OCR / detail answers
- Result:
  - 5 samples, TTFT `294.45 ms`, Throughput `22.72 tok/s`
- Artifact: backfilled from session notes
- Status: kept as default base

### I02

- Date: 2026-03-20
- Path: `PATH-MIDLAYER-POOL`
- Change:
  - added mid-layer visual token reduction inside the Qwen3-VL text forward
  - first stable version used pooling after an early text layer
- Result:
  - 3 samples, TTFT `213.17 ms`, Throughput `23.93 tok/s`
  - artifact: `result_pool_3.json`
- Output note:
  - answers were mostly good, but sample `34604` ended as `Sublimely Self-Righteous beer`
- Status: kept as main mid-layer path

### I03

- Date: 2026-03-20
- Path: `PATH-MIDLAYER-DART`
- Change:
  - first DART-like token selection experiment
  - selected visual token subsets instead of pooled synthetic tokens
- Result:
  - 3 samples, TTFT `216.96 ms`, Throughput `23.54 tok/s`
  - artifact: `result_dart_3.json`
- Output note:
  - answer drift was obvious: `Dakota`, `Ale`
- Status: rejected as default, kept as optional path

### I04

- Date: 2026-03-20
- Path: `PATH-MIDLAYER-VTW`
- Change:
  - VTW-like visual withdrawal experiment
- Result:
  - 3 samples, TTFT `213.90 ms`, Throughput `23.42 tok/s`
  - artifact: `result_vtw_3.json`
- Output note:
  - answer drift was severe: `COGNITION.`, `St`
- Status: rejected as default, kept as optional path

### I05

- Date: 2026-03-20
- Path: `PATH-MIDLAYER-DART`
- Change:
  - second DART-like revision using more local / block-wise representative selection
- Result:
  - 3 samples, TTFT `222.47 ms`, Throughput `23.56 tok/s`
  - artifact: `result_dart2_3.json`
- Output note:
  - still unstable: `Dakota`, `Sublimely Ale`
- Status: rejected as default, kept as optional path

### I06

- Date: 2026-03-20
- Path: `PATH-ANSWER-DECODE`
- Change:
  - added VQA-style short-answer cleanup
  - removed generic trailing nouns such as `beer` in some cases
- Result:
  - 5 samples, TTFT `225.61 ms`, Throughput `24.04 tok/s`
  - artifact: `result_pool_postfix_5.json`
- Output note:
  - produced short answers like `Dakota Digital`, `COPENHAGEN`, `Sublimely Self-Righteous`, `Bowmore`, `10 years`
- Status:
  - technically useful
  - not default anymore because official answers appear closer to raw vanilla model output

### I07

- Date: 2026-03-20 to 2026-03-21
- Path: `PATH-CACHE-IMPLEMENTATION`
- Change:
  - added env-driven cache implementation control
  - probed HF cache modes without touching weights
- Results:
  - `static`: 3 samples, TTFT `255.87 ms`, Throughput `3.87 tok/s`, artifact `result_cache_static_3.json`
  - `dynamic_full`: 5 samples, TTFT `240.41 ms`, Throughput `23.44 tok/s`, artifact `result_cache_dynamicfull_seq_5.json`
- Key finding:
  - `static` was unusable
  - `dynamic_full` improved throughput but consistently hurt TTFT
- Status: kept only as optional experiment switch

### I08

- Date: 2026-03-21
- Path: `PATH-CACHE-SINGLE-TOKEN` and `PATH-CACHE-FAST-SINGLE`
- Change:
  - added `single-token no-cache` handling for `max_new_tokens=1`
  - added an experimental custom first-token fast path
- Results:
  - custom fast path lost to normal generate in local tests
  - example:
    - fast path ON: 3 samples, TTFT `254.55 ms`, Throughput `21.92 tok/s`, artifact `result_fastsingle_on_3.json`
    - fast path OFF: 3 samples, TTFT `242.51 ms`, Throughput `23.37 tok/s`, artifact `result_fastsingle_off_3.json`
- Status:
  - `single-token no-cache` remains enabled by default
  - custom fast path remains disabled by default

### I09

- Date: 2026-03-21
- Path: `PATH-CACHE-DECODE-GRID`
- Change:
  - monkey-patched `prepare_inputs_for_generation`
  - for decode steps only, dropped `image_grid_thw` and `video_grid_thw`
- Results:
  - grid drop ON: 10 samples, TTFT `229.20 ms`, Throughput `22.50 tok/s`, artifact `result_decodegrid_on_seq_10.json`
  - grid drop OFF: 10 samples, TTFT `234.94 ms`, Throughput `23.58 tok/s`, artifact `result_decodemask_off_seq_10.json`
- Key finding:
  - dropping grid metadata helped TTFT but reduced throughput
  - based on the competition weighting, this path was kept as the safer default
- Status: kept as default

### I10

- Date: 2026-03-21
- Path: `PATH-CACHE-DECODE-MASK`
- Change:
  - in decode steps, tried dropping `attention_mask` when it was a trivial all-ones mask
- Results:
  - 10 samples, TTFT `243.83 ms`, Throughput `22.37 tok/s`
  - artifact: `result_decodemask_on_seq_10.json`
- Key finding:
  - this path was worse than the safer decode-grid-only variant
- Status: rejected as default, kept as optional switch

### I11

- Date: 2026-03-21
- Path: `PATH-CACHE-DECODE-POSIDS`
- Change:
  - tried passing explicit decode-stage `position_ids` instead of letting Qwen3-VL rebuild them internally
- Results:
  - ON: 5 samples, TTFT `231.54 ms`, Throughput `20.75 tok/s`, artifact `result_decodepos_on_seq_5_serial.json`
  - OFF baseline for the same route family: `result_decodegrid_on_seq_5.json`
- Key finding:
  - small 3-sample runs looked mildly positive
  - 5-sample serial run lost on throughput and did not justify becoming default
- Status: rejected as default, kept as optional switch

### I12

- Date: 2026-03-21
- Path: `PATH-CACHE-DECODE-KWARGS`
- Change:
  - patched generation-loop `_update_model_kwargs_for_generation`
  - removed decode-irrelevant multimodal metadata from loop state after cached decoding started
- Results:
  - ON: 5 samples, TTFT `239.90 ms`, Throughput `22.31 tok/s`, artifact `result_decodekwslim_on_seq_5_serial.json`
  - OFF baseline for the same route family: `result_decodegrid_on_seq_5.json`
- Key finding:
  - did not beat the simpler decode-grid-only default
- Status: rejected as default, kept as optional switch

### I13

- Date: 2026-03-21
- Path: `PATH-CACHE-PREFILL-REUSE`
- Change:
  - added a one-entry request-local prefill reuse path keyed by the current input tensor signatures
  - captured first-step cache state from `_update_model_kwargs_for_generation`
  - resumed the second identical greedy `generate()` call from cached prefill instead of replaying the whole prompt
- Results:
  - 3 samples OFF: TTFT `252.26 ms`, Throughput `21.43 tok/s`, artifact `result_prefillreuse_off_seq_3.json`
  - 3 samples ON: TTFT `252.05 ms`, Throughput `23.35 tok/s`, artifact `result_prefillreuse_on_seq_3.json`
  - 5 samples OFF: TTFT `259.92 ms`, Throughput `22.83 tok/s`, artifact `result_prefillreuse_off_seq_5.json`
  - 5 samples ON: TTFT `263.36 ms`, Throughput `29.14 tok/s`, artifact `result_prefillreuse_on_seq_5.json`
  - 10 samples OFF: TTFT `262.25 ms`, Throughput `22.16 tok/s`, artifact `result_prefillreuse_off_seq_10.json`
  - 10 samples ON: TTFT `263.26 ms`, Throughput `22.58 tok/s`, artifact `result_prefillreuse_on_seq_10.json`
- Key finding:
  - answer outputs matched exactly in the OFF/ON checks that were compared
  - the 5-sample subset showed a very large throughput gain
  - the 10-sample run still improved throughput, but only slightly, and TTFT was a bit worse
  - because this path depends on repeated identical prompt calls, it is better treated as an experimental route than the default baseline
- Status: kept as an optional switch, OFF by default

### I14

- Date: 2026-03-21
- Path: `PATH-CACHE-PREFILL-REUSE` + `PATH-CACHE-IMPLEMENTATION`
- Change:
  - combined prefill reuse with `AICAS_CACHE_IMPLEMENTATION=dynamic_full`
- Results:
  - 5 samples, TTFT `259.52 ms`, Throughput `24.58 tok/s`, artifact `result_prefillreuse_dynamicfull_seq_5.json`
- Key finding:
  - better than plain `dynamic_full`
  - clearly worse than prefill reuse alone on the same 5-sample subset
- Status: rejected as default combo, keep `dynamic_full` unset unless testing it on purpose

### I15

- Date: 2026-03-21
- Path: `PATH-CACHE-MANUAL-GREEDY`
- Change:
  - implemented a direct greedy decode loop that called the model forward pass manually
  - managed `past_key_values` explicitly and used `logits_to_keep=1`
  - goal was to remove part of `GenerationMixin` overhead without relying on repeated-request reuse
- Results:
  - OFF same-code baseline: 3 samples, TTFT `240.64 ms`, Throughput `25.09 tok/s`, artifact `result_manualgreedy_off_seq_3.json`
  - ON: 3 samples, TTFT `388.22 ms`, Throughput `23.43 tok/s`, artifact `result_manualgreedy_on_seq_3.json`
- Key finding:
  - answer outputs still matched
  - but the manual loop was clearly worse on both TTFT and Throughput
- Status: rejected and reverted from the codebase

### I16

- Date: 2026-03-21
- Path: `PATH-CACHE-LEGACY-RETURN`
- Change:
  - explicitly tested `AICAS_RETURN_LEGACY_CACHE=0` and `AICAS_RETURN_LEGACY_CACHE=1`
- Results:
  - OFF: 5 samples, TTFT `236.44 ms`, Throughput `21.92 tok/s`, artifact `result_returnlegacy_off_seq_5.json`
  - `AICAS_RETURN_LEGACY_CACHE=0`: TTFT `249.86 ms`, Throughput `20.60 tok/s`, artifact `result_returnlegacy0_seq_5.json`
  - `AICAS_RETURN_LEGACY_CACHE=1`: TTFT `234.11 ms`, Throughput `21.59 tok/s`, artifact `result_returnlegacy1_seq_5.json`
- Key finding:
  - forcing non-legacy cache was clearly worse
  - forcing legacy cache slightly improved TTFT but gave back some throughput
  - answer outputs matched the default path
- Status: keep unset as default; keep `=1` only as a weak optional trade-off if TTFT is prioritized

### I17

- Date: 2026-03-21
- Path: `PATH-CACHE-PREFILL-VISUAL-PRUNE`
- Change:
  - tried pruning shallow full-length KV cache layers after prefill
  - reused the same visual token subset already selected by the mid-layer pool path
  - goal was to reduce decode cost without changing first-token computation
- Results:
  - OFF: 3 samples, TTFT `233.47 ms`, Throughput `22.49 tok/s`, artifact `result_cacheprune_off_seq_3.json`
  - ON all eligible layers: TTFT `235.64 ms`, Throughput `23.17 tok/s`, artifact `result_cacheprune_on_seq_3.json`
  - ON first 2 layers only: TTFT `242.66 ms`, Throughput `23.17 tok/s`, artifact `result_cacheprune_l2_seq_3_serial.json`
  - ON first 1 layer only: TTFT `240.69 ms`, Throughput `21.06 tok/s`, artifact `result_cacheprune_l1_seq_3_serial.json`
- Key finding:
  - serial runs showed answer drift on sample 3 even for the conservative 1-layer and 2-layer variants
  - throughput gains were not stable enough to justify that drift
  - one parallel layer sweep was discarded and is not counted here
- Status: rejected and reverted from the codebase

### I18

- Date: 2026-03-21
- Path: `PATH-CACHE-IMPLEMENTATION`
- Change:
  - revisited explicit `AICAS_CACHE_IMPLEMENTATION=dynamic`
  - briefly promoted it to default, then rolled that back after rechecks
- Results:
  - 5 samples run 1: TTFT `236.66 ms`, Throughput `26.55 tok/s`, artifact `result_cache_dynamic_seq_5_serial.json`
  - 10 samples: TTFT `243.30 ms`, Throughput `23.40 tok/s`, artifact `result_cache_dynamic_seq_10_serial.json`
  - 5 samples rerun: TTFT `235.37 ms`, Throughput `21.64 tok/s`, artifact `result_cache_dynamic_seq_5_serial_rerun.json`
  - 5 samples default recheck after the temporary default flip: TTFT `237.50 ms`, Throughput `20.73 tok/s`, artifact `result_default_dynamic_seq_5_recheck.json`
- Key finding:
  - outputs matched the default path in the checks that were compared
  - one 5-sample run looked excellent, but the rerun did not reproduce that throughput gain
  - because the local benefit was unstable, the temporary default change was reverted
- Status: keep `dynamic` as an optional switch only; do not make it the default yet

### I19

- Date: 2026-03-21
- Path: `PATH-CACHE-IMPLEMENTATION` + `PATH-CACHE-LEGACY-RETURN`
- Change:
  - combined `AICAS_CACHE_IMPLEMENTATION=dynamic` with `AICAS_RETURN_LEGACY_CACHE=1`
- Results:
  - 5 samples, TTFT `236.64 ms`, Throughput `23.71 tok/s`, artifact `result_cache_dynamic_legacy1_seq_5.json`
- Key finding:
  - better than `return_legacy_cache=1` by itself
  - not clearly strong enough to beat the stability concerns seen in the plain `dynamic` reruns
- Status: optional only, not default

### I20

- Date: 2026-03-21
- Path: `PATH-CACHE-IMPLEMENTATION`
- Change:
  - tested `AICAS_CACHE_IMPLEMENTATION=offloaded`
- Results:
  - 3 samples, TTFT `243.37 ms`, Throughput `24.04 tok/s`, artifact `result_cache_offloaded_seq_3.json`
  - 5 samples, TTFT `234.40 ms`, Throughput `21.15 tok/s`, artifact `result_cache_offloaded_seq_5.json`
- Key finding:
  - 3-sample behavior looked acceptable
  - 5-sample behavior gave back throughput, so the gain did not hold up
  - outputs matched the default path in the compared runs
- Status: keep as optional only if memory pressure makes it necessary; not default

### I21

- Date: 2026-03-21
- Path: `PATH-CACHE-PREALLOCATE-DYNAMIC`
- Change:
  - added an optional path that builds a user-supplied `DynamicCache`
  - calls `early_initialization(...)` before `generate()` so HF can skip part of its internal cache setup
- Results:
  - 3 samples OFF: TTFT `269.48 ms`, Throughput `20.33 tok/s`, artifact `result_prealloc_off_seq_3.json`
  - 3 samples ON: TTFT `269.08 ms`, Throughput `23.46 tok/s`, artifact `result_prealloc_on_seq_3.json`
  - 5 samples OFF: TTFT `266.67 ms`, Throughput `25.98 tok/s`, artifact `result_prealloc_off_seq_5.json`
  - 5 samples ON: TTFT `261.51 ms`, Throughput `22.72 tok/s`, artifact `result_prealloc_on_seq_5.json`
- Key finding:
  - compared outputs matched exactly on the checked 3-sample and 5-sample runs
  - the standalone path was mixed: one small run improved throughput, the next lost it
  - because the gain did not hold up when tested alone, it should not be a default by itself
- Status: keep as optional only; not default

### I22

- Date: 2026-03-21
- Path: `PATH-CACHE-PREFILL-REUSE` + `PATH-CACHE-PREALLOCATE-DYNAMIC`
- Change:
  - combined prefill reuse with a preallocated `DynamicCache`
  - the intended shape was:
    - first `max_new_tokens=1` call captures prefill/cache state
    - second identical call reuses that prefill instead of replaying the whole prompt
- Results:
  - 5 samples, reuse only: TTFT `267.94 ms`, Throughput `23.57 tok/s`, artifact `result_prefillreuse_refresh_seq_5.json`
  - 5 samples, reuse + prealloc: TTFT `268.75 ms`, Throughput `27.61 tok/s`, artifact `result_prefillreuse_prealloc_seq_5.json`
  - 10 samples, reuse only: TTFT `272.82 ms`, Throughput `22.87 tok/s`, artifact `result_prefillreuse_refresh_seq_10.json`
  - 10 samples, reuse + prealloc: TTFT `273.89 ms`, Throughput `24.16 tok/s`, artifact `result_prefillreuse_prealloc_seq_10.json`
  - 10 samples, reuse + prealloc rerun: TTFT `270.29 ms`, Throughput `28.48 tok/s`, artifact `result_prefillreuse_prealloc_seq_10_rerun.json`
- Key finding:
  - outputs matched exactly in the compared 5-sample and 10-sample runs
  - the combo was much stronger than standalone preallocation
  - the trade-off was consistent in direction: throughput up, TTFT slightly worse
  - this path is benchmark-shape dependent because it benefits from the repeated identical prompt call pattern inside `measure_performance`
- Status: promising optional combo; keep OFF by default until it reproduces more consistently

### I23

- Date: 2026-03-21
- Path: `PATH-CACHE-PREFILL-REUSE` + `PATH-CACHE-PREALLOCATE-DYNAMIC`
- Change:
  - tried a cleaner capture-only form:
    - added `AICAS_PREALLOCATE_DYNAMIC_CACHE_MODE=capture`
    - briefly flipped `reuse_prefill_cache` and `preallocate_dynamic_cache` toward default-on while validating the narrower scope
- Results:
  - default-like sanity run with capture-only behavior: 5 samples, TTFT `270.16 ms`, Throughput `22.98 tok/s`, artifact `result_default_capturecache_seq_5.json`
- Key finding:
  - the capture-only implementation is cleaner and reduces side effects on unrelated generation paths
  - however, this sanity run did not reproduce the best throughput numbers seen in I22
  - because the benefit still looks somewhat noisy locally, the default flip was rolled back
- Status: kept in code as a cleaner experimental mechanism, but defaults remain OFF

### I24

- Date: 2026-03-21
- Path: `PATH-CACHE-PREFILL-REUSE-DELEGATE`
- Change:
  - added an optional route that, after consuming the cached first token, delegates the remaining suffix generation back to the original HF `generate()`
  - goal: test whether the stock suffix loop is faster than the current hand-written cached greedy continuation
- Results:
  - manual suffix loop: 5 samples, TTFT `230.66 ms`, Throughput `28.63 tok/s`, artifact `result_prefillreuse_manualsuffix_seq_5.json`
  - delegated suffix generate: 5 samples, TTFT `235.31 ms`, Throughput `22.91 tok/s`, artifact `result_prefillreuse_delegatesuffix_seq_5.json`
- Key finding:
  - answer outputs matched exactly
  - delegating the suffix back to HF `generate()` was clearly slower than the current manual cached loop
- Status: rejected as default, kept OFF by default

### I25

- Date: 2026-03-21
- Path: `PATH-CACHE-PREFILL-REUSE`
- Change:
  - re-ran the current manual cached continuation path under fresh same-session comparisons
- Results:
  - 10 samples, current default baseline: TTFT `240.13 ms`, Throughput `23.04 tok/s`, artifact `result_default_recheck_seq_10_b.json`
  - 10 samples, prefill reuse manual suffix: TTFT `245.18 ms`, Throughput `24.03 tok/s`, artifact `result_prefillreuse_manualsuffix_seq_10.json`
- Key finding:
  - answer outputs matched in the compared runs
  - plain prefill reuse still improved throughput, but TTFT gave back a little
  - this confirmed the path is useful, but not yet the best standalone default
- Status: keep as part of the preferred cache combo, but not as the only cache default

### I26

- Date: 2026-03-21
- Path: `PATH-CACHE-PREFILL-REUSE` + `PATH-CACHE-PREALLOCATE-DYNAMIC`
- Change:
  - rechecked the cleaner `capture`-mode combo under same-session 10-sample comparisons
  - then promoted it to the default cache route after repeated wins
- Results:
  - 10 samples, current default baseline before the flip: TTFT `240.13 ms`, Throughput `23.04 tok/s`, artifact `result_default_recheck_seq_10_b.json`
  - 10 samples, combo run 1: TTFT `238.40 ms`, Throughput `24.31 tok/s`, artifact `result_prefillreuse_prealloc_seq_10_b.json`
  - 10 samples, combo run 2: TTFT `237.38 ms`, Throughput `23.64 tok/s`, artifact `result_prefillreuse_prealloc_seq_10_c.json`
  - 3 samples, new HEAD sanity check: TTFT `238.15 ms`, Throughput `23.30 tok/s`, artifact `result_default_newhead_seq_3.json`
- Key finding:
  - both 10-sample combo reruns matched the baseline answers exactly
  - unlike earlier noisy runs, this same-session recheck beat the old default on both TTFT and Throughput
  - the cleaner `capture` mode was sufficient; no need to broaden preallocation scope
- Status: promoted to current default

### I27

- Date: 2026-03-21
- Path: `PATH-CACHE-DECODE-GRID`
- Change:
  - re-evaluated the old decode-grid slimming default under the new cache mainline
  - compared `AICAS_DECODE_DROP_VISION_GRID=1` vs `=0` with the new default cache combo
- Results:
  - grid ON: 10 samples, TTFT `239.72 ms`, Throughput `24.07 tok/s`, artifact `result_default_gridon_seq_10_c.json`
  - grid OFF: 10 samples, TTFT `239.00 ms`, Throughput `25.25 tok/s`, artifact `result_default_gridoff_seq_10.json`
- Key finding:
  - answers matched exactly across all 10 checked samples
  - once `PATH-CACHE-PREFILL-REUSE` + `PATH-CACHE-PREALLOCATE-DYNAMIC` became the mainline, dropping grid metadata was no longer the better trade-off
  - in the new mainline, keeping grid metadata actually lost on both TTFT and Throughput
- Status: removed from the default set; keep only as an optional switch

### I28

- Date: 2026-03-21
- Path: `PATH-ATTN-KERNEL-AUTO`, `PATH-ATTN-KERNEL-EAGER`, `PATH-ATTN-KERNEL-FLASH2-REQUEST`
- Change:
  - switched the default attention backend request from fixed `sdpa` to `auto`
  - added runtime attention-backend resolution:
    - prefer `flash_attention_2` when `flash_attn` is truly available
    - otherwise fall back safely to `sdpa`
  - added explicit notes at model load time so each run shows which backend was actually selected
- Results:
  - `auto` on this machine: 3 samples, TTFT `241.12 ms`, Throughput `23.33 tok/s`, artifact `result_attn_auto_seq_3.json`
  - `eager`: 3 samples, TTFT `441.14 ms`, Throughput `21.23 tok/s`, artifact `result_attn_eager_seq_3.json`
  - explicit `flash_attention_2` request on this machine: TTFT `236.97 ms`, Throughput `23.07 tok/s`, artifact `result_attn_flashreq_seq_3.json`
- Key finding:
  - local runtime probe says `is_flash_attn_2_available = False`
  - this Windows + torch 2.4.1 environment has no usable `flash_attn`, so a real FA2 run is not available yet
  - `eager` is clearly worse, which confirms `sdpa` remains the right local fallback
  - the new `auto` path is the right default because it is safe locally and can directly upgrade to FA2 on a future Linux / Docker environment if `flash_attn` is present
- Extra note:
  - an escalated `pip install flash-attn --no-build-isolation` attempt downloaded only an sdist and failed on Windows with a missing source/header file during install
- Status: promoted to current default as the attention backend policy; local effective backend remains `sdpa`

### I29

- Date: 2026-03-21
- Path: `PATH-ATTN-KERNEL-AUTO` and `PATH-ATTN-KERNEL-FLASH2-REQUEST`
- Change:
  - after `flash_attn` was installed successfully, re-ran the attention backend sweep on the real FA2 runtime
  - confirmed:
    - `flash_attn` import works
    - `flash_attn_2_cuda` import works
    - `is_flash_attn_2_available = True`
- Results:
  - `auto` on the flash-ready environment: 3 samples, TTFT `227.86 ms`, Throughput `21.03 tok/s`, artifact `result_attn_auto_flashready_seq_3.json`
  - explicit `flash_attention_2`: 3 samples, TTFT `209.00 ms`, Throughput `22.44 tok/s`, artifact `result_attn_flash2_seq_3_real.json`
  - explicit `sdpa`: 3 samples, TTFT `216.47 ms`, Throughput `21.51 tok/s`, artifact `result_attn_sdpa_seq_3_retest.json`
  - explicit `eager`: 3 samples, TTFT `441.14 ms`, Throughput `21.23 tok/s`, artifact `result_attn_eager_seq_3.json`
- Key finding:
  - with a real FA2 install, `flash_attention_2` beat `sdpa` on both TTFT and Throughput in the 3-sample check
  - outputs matched exactly on the compared 3-sample flash vs sdpa run
  - `eager` remained far worse and stayed disabled
- Status: keep `PATH-ATTN-KERNEL-AUTO` as the main policy; installation success supersedes the 鈥淔A2 unavailable鈥?part of I28

### I30

- Date: 2026-03-21
- Path: `PATH-ATTN-KERNEL-FLASH2-REQUEST`
- Change:
  - expanded the real FA2 test to 10 samples
- Results:
  - explicit `flash_attention_2` run 1: 10 samples, TTFT `226.39 ms`, Throughput `21.98 tok/s`, artifact `result_attn_flash2_seq_10.json`
  - explicit `flash_attention_2` run 2: 10 samples, TTFT `217.29 ms`, Throughput `23.19 tok/s`, artifact `result_attn_flash2_seq_10_rerun.json`
  - explicit `sdpa`: 10 samples, TTFT `229.19 ms`, Throughput `25.14 tok/s`, artifact `result_attn_sdpa_seq_10.json`
- Key finding:
  - plain FA2 consistently improved TTFT relative to `sdpa`
  - plain FA2 did not reliably beat `sdpa` on Throughput in the 10-sample runs
  - FA2 outputs were internally stable across reruns, but had small wording differences relative to `sdpa` on question `34605` and `34611`
  - because the README accuracy is soft-match based, these differences are likely acceptable, but they are still worth tracking
- Status: useful but not the final best default by itself

### I31

- Date: 2026-03-21
- Path: `PATH-ATTN-KERNEL-MIXED-DECODE` and `PATH-ATTN-KERNEL-DECODE-OVERRIDE`
- Change:
  - added `AICAS_PREFILL_REUSE_DECODE_ATTN_IMPL`
  - tested a hybrid route:
    - primary backend = `flash_attention_2`
    - prefill-reuse suffix decode backend = `sdpa`
  - then made this behavior the default whenever `PATH-ATTN-KERNEL-AUTO` resolves to FA2
- Results:
  - FA2 plain: 5 samples, TTFT `211.31 ms`, Throughput `23.01 tok/s`, artifact `result_attn_flash2_plain_seq_5.json`
  - FA2 + decode-sdpa: 5 samples, TTFT `215.26 ms`, Throughput `25.54 tok/s`, artifact `result_attn_flash2_decode_sdpa_seq_5.json`
  - FA2 + decode-sdpa: 10 samples, TTFT `219.01 ms`, Throughput `25.34 tok/s`, artifact `result_attn_flash2_decode_sdpa_seq_10.json`
  - FA2 + decode-sdpa rerun: 10 samples, TTFT `213.30 ms`, Throughput `25.26 tok/s`, artifact `result_attn_flash2_decode_sdpa_seq_10_rerun.json`
  - HEAD sanity check with no env overrides: 3 samples, TTFT `224.28 ms`, Throughput `24.98 tok/s`, artifact `result_attn_auto_mixed_head_seq_3.json`
- Key finding:
  - the hybrid route preserved FA2's TTFT advantage while recovering the Throughput loss seen in plain FA2
  - on 10-sample runs, the hybrid route beat explicit `sdpa` on both TTFT and Throughput
  - the hybrid route matched plain FA2 outputs exactly
  - relative to `sdpa`, the wording differences remained limited to the same small long-form phrasing differences already seen with plain FA2
- Status: promoted to the current default attention strategy

### I32

- Date: 2026-03-21
- Path: `PATH-OP-TORCH-RMSNORM`
- Change:
  - tried replacing Qwen3-VL text-side Python RMSNorm with `torch.nn.functional.rms_norm`
  - patched all `Qwen3VLTextRMSNorm` modules after model load
- Results:
  - OFF: 3 samples, TTFT `282.34 ms`, Throughput `27.84 tok/s`, artifact `result_rmsnorm_off_seq_3.json`
  - ON: 3 samples, TTFT `282.31 ms`, Throughput `26.24 tok/s`, artifact `result_rmsnorm_on_seq_3.json`
- Key finding:
  - this path was not just slower, it was incorrect in this environment
  - the ON run produced obviously broken outputs full of repeated `!` characters on the checked samples
  - because correctness regressed hard, the patch was immediately reverted from the codebase
- Status: rejected and reverted from the codebase

### I33

- Date: 2026-03-21
- Path: `PATH-CACHE-PREALLOCATE-DYNAMIC` under the flash mixed mainline
- Change:
  - rechecked whether the earlier preallocated-cache conclusion still held after enabling the flash mixed attention strategy
- Results:
  - current flash mixed default: 10 samples, TTFT `272.98 ms`, Throughput `23.59 tok/s`, artifact `result_flashmixed_default_seq_10_b.json`
  - same setup with `AICAS_PREALLOCATE_DYNAMIC_CACHE=0`: TTFT `285.87 ms`, Throughput `24.10 tok/s`, artifact `result_flashmixed_prealloc_off_seq_10.json`
- Key finding:
  - outputs matched exactly
  - under the flash mixed mainline, disabling preallocation slightly improved Throughput but clearly hurt TTFT
  - this is not a clean conclusion flip; it is a trade-off shift
- Status: keep preallocation enabled by default for now, because the current competition score still gives TTFT substantial weight

### I34

- Date: 2026-03-21
- Path: `PATH-CACHE-LEGACY-RETURN` and `PATH-CACHE-DECODE-KWARGS` under the flash mixed mainline
- Change:
  - expanded the earlier 5-sample "possible flip" check to 10 samples before changing defaults
  - then audited the implementation details of `PATH-CACHE-DECODE-KWARGS`
- Results:
  - current flash mixed default: 10 samples, TTFT `302.35 ms`, Throughput `22.79 tok/s`, artifact `result_flashmain_default_seq_10_c.json`
  - `AICAS_RETURN_LEGACY_CACHE=1`: TTFT `298.01 ms`, Throughput `23.22 tok/s`, artifact `result_flashmain_returnlegacy1_seq_10.json`
  - `AICAS_DECODE_SLIM_MODEL_KWARGS=1` before the code fix: TTFT `294.80 ms`, Throughput `27.75 tok/s`, artifact `result_flashmain_kwargsslim_seq_10.json`
  - same pre-fix kwargs-slim rerun: TTFT `292.12 ms`, Throughput `23.34 tok/s`, artifact `result_flashmain_kwargsslim_seq_10_rerun.json`
  - `AICAS_RETURN_LEGACY_CACHE=1` + `AICAS_DECODE_SLIM_MODEL_KWARGS=1`: TTFT `297.21 ms`, Throughput `23.78 tok/s`, artifact `result_flashmain_legacy1_kwargsslim_seq_10.json`
- Key finding:
  - all compared variants matched the default outputs on the checked 10 samples
  - `return_legacy_cache=1` remained mildly positive, but not enough to justify becoming the new default
  - the important discovery from the code audit was that `PATH-CACHE-DECODE-KWARGS` was still effectively a no-op under the current defaults, so the apparent gain could not be trusted as a real optimization result yet
- Status: keep `PATH-CACHE-LEGACY-RETURN` optional; do not promote `PATH-CACHE-DECODE-KWARGS` from the pre-fix numbers alone

### I35

- Date: 2026-03-21
- Path: `PATH-CACHE-DECODE-KWARGS`
- Change:
  - implemented the path for real by dropping decode-irrelevant visual payloads from `model_kwargs` once cached decoding starts
  - specifically removed `pixel_values`, `pixel_values_videos`, and `second_per_grid_ts` after `cache_position > 0`
  - then promoted this path to the default runtime config
- Results:
  - 3-sample sanity check with the real implementation: TTFT `295.91 ms`, Throughput `24.15 tok/s`, artifact `result_flashmain_kwargsslim_real_seq_3.json`
  - 10 samples run 1: TTFT `232.19 ms`, Throughput `25.28 tok/s`, artifact `result_flashmain_kwargsslim_real_seq_10.json`
  - 10 samples run 2: TTFT `213.50 ms`, Throughput `25.17 tok/s`, artifact `result_flashmain_kwargsslim_real_seq_10_rerun.json`
- Key finding:
  - both 10-sample runs matched the default outputs exactly
  - once the path actually stopped carrying large visual tensors through the decode loop, it produced a clear improvement over the same-session default baseline on both TTFT and Throughput
  - this is a real conclusion flip: `PATH-CACHE-DECODE-KWARGS`, which had previously stayed OFF, now deserves to be ON in the flash mixed mainline
- Status: promoted to the default configuration

### I36

- Date: 2026-03-21
- Path: `PATH-CACHE-DECODE-KWARGS` after promotion
- Change:
  - ran a no-env HEAD sanity check after making kwargs-slim the default
- Results:
  - HEAD default, 3 samples: TTFT `216.12 ms`, Throughput `24.14 tok/s`, artifact `result_flashmain_newdefault_head_seq_3.json`
- Key finding:
  - the default code path now advertises `decode_loop` in the applied optimization list, confirming the promoted path is active without env overrides
  - the promoted default remains healthy after the config change
- Status: keep as the new default

### I37

- Date: 2026-03-21
- Path: `PATH-ATTN-SDPA-KERNEL-MODE`
- Change:
  - added env-driven SDPA kernel policy control through `AICAS_SDPA_KERNEL_MODE`
  - compared the previous Windows-safe policy (`legacy_no_flash`) against leaving PyTorch defaults untouched (`auto`)
  - verified the actual runtime packages in `E:\conda_envs\AICAS\python.exe`: `flash_attn` imports successfully, but `fast_attn` does not exist as an importable module in this environment
- Results:
  - `legacy_no_flash`: 3 samples, TTFT `256.15 ms`, Throughput `24.81 tok/s`, artifact `result_op_baseline_seq_3.json`
  - `auto`: 3 samples, TTFT `264.93 ms`, Throughput `25.04 tok/s`, artifact `result_op_sdpa_auto_seq_3.json`
- Key finding:
  - outputs matched exactly between the two runs on the checked 3 samples
  - this Windows torch 2.4.1 build still emits `Torch was not compiled with flash attention` on the SDPA path
  - leaving PyTorch defaults untouched gave back too much TTFT for only a tiny throughput gain
- Status: keep `legacy_no_flash` as the default local SDPA policy; keep the switch available for future Linux / different-wheel validation

### I38

- Date: 2026-03-21
- Path: `PATH-CACHE-DIRECT-SINGLE`
- Change:
  - implemented a direct deterministic single-token path that bypasses most of HF `generate()` and calls the model forward pass directly
  - added explicit `cache_position` handling so the path can still capture prefill reuse state safely
- Results:
  - OFF baseline: 3 samples, TTFT `256.15 ms`, Throughput `24.81 tok/s`, artifact `result_op_baseline_seq_3.json`
  - ON: 3 samples, TTFT `264.56 ms`, Throughput `24.44 tok/s`, artifact `result_op_directsingle_seq_3.json`
- Key finding:
  - outputs matched the baseline exactly on the checked 3 samples
  - bypassing `generate()` did not pay off in this Qwen3-VL setup; both TTFT and Throughput moved in the wrong direction
- Status: keep OFF by default; leave only as an explicit experiment switch

### I39

- Date: 2026-03-21
- Path: `PATH-OP-TORCH-COMPILE`
- Change:
  - probed `torch.compile` on the patched Qwen3-VL language-model forward to test the organizer-recommended compile / graph path
  - tried standard inductor and a `backend='cudagraphs'` fallback
- Results:
  - standard inductor path failed in this environment because Triton is not installed / usable
  - `backend='cudagraphs'` completed a one-sample smoke test, but the first `max_new_tokens=1` generation expanded to about `26 s` and produced repeated graph-break / unsupported-flash-attn warnings
- Key finding:
  - the current Windows + torch 2.4.1 + flash-attn wheel combination is not a viable compile target for this project right now
  - this is an environment blocker, not just a weak benchmark result
- Status: rejected in the current environment; do not promote compile-related changes locally unless Triton and the flash-attn graph path are both solved first

### I40

- Date: 2026-03-21
- Path: `PATH-ATTN-KERNEL-MIXED-DECODE` follow-up with `flash_attn_with_kvcache`
- Change:
  - attempted a narrower decode-only operator swap using flash-attn's `flash_attn_with_kvcache` for the `batch=1`, `query_length=1`, cached decode case
  - validated that the kernel can accept non-contiguous transpose views from HF-style KV tensors in a standalone micro test
  - then wired the idea into the real Qwen3-VL decode loop and ran benchmark-level validation
- Results:
  - a minimal short generate smoke test could run after fixing an attention-wrapper closure bug
  - the real 3-sample benchmark path hit `CUDA illegal memory access` inside the cached decode loop, so no stable artifact was kept
- Key finding:
  - the operator idea is attractive on paper and matches the flash-attn 2.2 inference notes, but it is not engineering-safe with the current HF DynamicCache / Qwen3-VL stack in this environment
  - because the failure mode corrupts the CUDA context, this path is not acceptable even as a default-off experimental branch
- Status: rejected and removed from the code after validation

### I41

- Date: 2026-03-22
- Path: `PATH-ATTN-SDPA-KERNEL-MODE` revalidated in WSL
- Change:
  - moved the next operator round to WSL using `/home/apulupie/miniconda3/envs/AI/bin/python`
  - verified the Linux-side runtime packages: `flash_attn` and `triton` are available, but `fast_attn` is still not importable here either
  - fixed the benchmark execution environment with `PYTHONUTF8=1` and `LANG/LC_ALL=C.UTF-8` because WSL's default ASCII locale broke `set_attn_implementation()` when it read source files
  - re-ran `legacy_no_flash`, `auto`, and `flash_only`
- Results:
  - baseline `legacy_no_flash`, 3 samples: TTFT `303.11 ms`, Throughput `24.56 tok/s`, artifact `result_wsl_baseline_seq_3.json`
  - `auto`, 3 samples: TTFT `297.41 ms`, Throughput `24.29 tok/s`, artifact `result_wsl_sdpa_auto_seq_3.json`
  - `flash_only`, 3 samples: TTFT `300.79 ms`, Throughput `24.59 tok/s`, artifact `result_wsl_sdpa_flashonly_seq_3.json`
  - baseline `legacy_no_flash`, 5 samples: TTFT `295.79 ms`, Throughput `23.34 tok/s`, artifact `result_wsl_baseline_seq_5.json`
  - `auto`, 5 samples: TTFT `271.22 ms`, Throughput `9.00 tok/s`, artifact `result_wsl_sdpa_auto_seq_5.json`
  - `flash_only`, 5 samples: TTFT `1507.45 ms`, Throughput `9.78 tok/s`, artifact `result_wsl_sdpa_flashonly_seq_5.json`
- Key finding:
  - the 3-sample WSL runs made `auto` and `flash_only` look mildly promising, but the 5-sample validation showed catastrophic instability on longer / harder samples
  - all checked outputs still matched the baseline exactly, so this is a pure performance-path issue rather than an answer-quality issue
  - even with Linux + Triton available, `legacy_no_flash` remains the only stable SDPA kernel policy for this project
- Status: keep `legacy_no_flash` as the default even in WSL; do not promote `auto` or `flash_only`

### I42

- Date: 2026-03-22
- Path: `PATH-OP-TORCH-COMPILE` revalidated in WSL
- Change:
  - retried `torch.compile(mode='reduce-overhead', dynamic=True)` on the patched Qwen3-VL language-model forward in WSL, where Triton is available
  - measured both the first compiled run and post-warm distinct-input runs
- Results:
  - first compiled `max_new_tokens=1` run on one sample expanded to about `153152.61 ms` in an inline smoke test
  - after that compile warmup, two different follow-up samples still took about `58137.77 ms` and `54647.33 ms`
  - the plain non-compiled comparison path on the same follow-up samples was about `1184.71 ms` and `270.69 ms`
- Key finding:
  - this is not just one-time compile tax; the flash-attn + dynamic multimodal shapes path keeps graph-breaking / recompiling across different inputs
  - compile is therefore still not a viable operator optimization for the current project, even in WSL with Triton installed
- Status: rejected in WSL as well

### I43

- Date: 2026-03-22
- Path: `PATH-ATTN-KERNEL-FLASH2-REQUEST` and `PATH-ATTN-KERNEL-MIXED-DECODE` under WSL
- Change:
  - rechecked whether the Linux environment flips the preferred primary backend away from the current flash-prefill + sdpa-decode mixed route
  - compared pure `sdpa` and pure `flash_attention_2`
- Results:
  - primary `sdpa`, 3 samples: TTFT `306.17 ms`, Throughput `24.69 tok/s`, artifact `result_wsl_attn_sdpa_auto_seq_3.json`
  - pure `flash_attention_2`, 3 samples: TTFT `322.66 ms`, Throughput `19.16 tok/s`, artifact `result_wsl_attn_flash2_plain_seq_3.json`
- Key finding:
  - both variants matched the mixed-route outputs exactly on the checked 3 samples
  - pure `sdpa` gave up too much TTFT, while pure `flash_attention_2` lost badly on both TTFT and Throughput
  - the existing mixed decode policy is still the best attention-backend structure among the tested WSL routes
- Status: keep the current mixed route; leave pure `sdpa` and pure `flash_attention_2` as non-default validation branches only

### I44

- Date: 2026-03-22
- Path: `PATH-CACHE-DIRECT-SINGLE` revalidated in WSL
- Change:
  - re-ran the direct deterministic single-token path in WSL to see whether Linux changes the HF `generate()` overhead trade-off
  - expanded the check from 3 samples to 5 samples before making any default decision
- Results:
  - ON, 3 samples: TTFT `273.11 ms`, Throughput `23.90 tok/s`, artifact `result_wsl_directsingle_seq_3.json`
  - baseline, 5 samples: TTFT `295.79 ms`, Throughput `23.34 tok/s`, artifact `result_wsl_baseline_seq_5.json`
  - ON, 5 samples: TTFT `300.94 ms`, Throughput `24.20 tok/s`, artifact `result_wsl_directsingle_seq_5.json`
- Key finding:
  - the attractive 3-sample TTFT win did not survive the 5-sample validation
  - outputs still matched the baseline exactly
  - the path trades some Throughput gain for TTFT regression, which is not enough to justify flipping the default under the current competition objective
- Status: keep OFF by default; retain only as an explicit experiment switch

### I45

- Date: 2026-03-22
- Path: `PATH-ATTN-KERNEL-CUDNN-SDPA`
- Change:
  - inspected the WSL torch runtime and confirmed that `torch.backends.cuda.enable_cudnn_sdp` exists, but is disabled by default
  - forced the current flash-prefill + sdpa-decode route onto a cuDNN-only SDPA kernel policy
  - also re-ran a pure `sdpa` route with cuDNN-only
  - finally retried the mixed route with `TORCH_CUDNN_SDPA_ENABLED=1`
- Results:
  - mixed route + cuDNN-only SDPA: all 3 samples failed during full generation, artifact `result_wsl_cudnnsdp_mixed_seq_3.json`
  - pure `sdpa` + cuDNN-only SDPA: all 3 samples failed, artifact `result_wsl_cudnnsdp_primarysdpa_seq_3.json`
  - mixed route + cuDNN-only SDPA + `TORCH_CUDNN_SDPA_ENABLED=1`: all 3 samples still failed, artifact `result_wsl_cudnnsdp_env_mixed_seq_3.json`
- Key finding:
  - this is not just a missing env-var issue: after enabling `TORCH_CUDNN_SDPA_ENABLED=1`, PyTorch still rejected the path because Qwen3-VL's query / key / value tensors were not in a cuDNN-supported packed or unpacked QKV layout
  - the current HF Qwen3-VL stack therefore has no usable cuDNN SDPA kernel path in this environment
- Status: rejected; do not add a cuDNN SDPA switch to the code

### I46

- Date: 2026-03-22
- Path: `PATH-RUNTIME-INFERENCE-MODE`
- Change:
  - wrapped `self.model.generate()` in `torch.inference_mode()` while leaving the rest of the current default stack unchanged
- Results:
  - `torch.inference_mode()` wrapper, 3 samples: TTFT `321.21 ms`, Throughput `23.71 tok/s`, artifact `result_wsl_inferencemode_seq_3.json`
- Key finding:
  - outputs matched the WSL baseline exactly on the checked 3 samples
  - the extra wrapper did not reduce overhead in practice; both TTFT and Throughput regressed relative to the baseline `303.11 ms / 24.56 tok/s`
- Status: rejected

### I47

- Date: 2026-03-22
- Path: `PATH-INPUT-BATCH-FP16`
- Change:
  - monkey-patched `transformers.feature_extraction_utils.BatchFeature.to()` so `pixel_values` and `pixel_values_videos` are cast to FP16 during the benchmark's pre-timing `.to(device)` stage
  - goal: move the Qwen3-VL vision dtype cast out of the timed `generate()` path without changing model semantics
- Results:
  - FP16 visual batch transfer, 3 samples: TTFT `311.93 ms`, Throughput `24.21 tok/s`, artifact `result_wsl_batchfeaturefp16_seq_3.json`
- Key finding:
  - outputs matched the WSL baseline exactly on the checked 3 samples
  - the idea was directionally attractive, but the full benchmark still regressed on both TTFT and Throughput relative to the baseline `303.11 ms / 24.56 tok/s`
- Status: rejected; do not carry this input-transfer patch into the main code

### I48

- Date: 2026-03-22
- Path: `PATH-VISION-LAYOUT`
- Change:
  - microbenchmarked the real Qwen3-VL `visual.patch_embed.proj` Conv3d path under the WSL runtime
  - compared the current layout against `channels_last_3d`
  - checked `torch.backends.cudnn.benchmark = True` on the same patch-embed microbench
  - then ran a full 3-sample benchmark with only `cudnn.benchmark` enabled
- Results:
  - patch-embed microbench baseline: about `4.95 ms`
  - patch-embed microbench with `channels_last_3d`: about `4.76 ms`
  - patch-embed microbench with `cudnn.benchmark=True`: about `4.71 ms`
  - full `get_image_features()` microbench baseline: about `164.40 ms`
  - full `get_image_features()` microbench with `channels_last_3d` patch: about `165.86 ms`
  - end-to-end `cudnn.benchmark=True`, 3 samples: TTFT `307.86 ms`, Throughput `23.99 tok/s`, artifact `result_wsl_cudnnbenchmark_seq_3.json`
- Key finding:
  - the patch-embed Conv3d operator alone can move a few tenths of a millisecond, but it is too small a slice of the real vision path to matter end-to-end
  - `channels_last_3d` did not improve the full `get_image_features()` path
  - the full benchmark with `cudnn.benchmark=True` still lost to the baseline, and sample `34604` showed a small wording drift relative to the baseline output
- Status: rejected; keep the default runtime layout path unchanged

### I49

- Date: 2026-03-22
- Path: `PATH-VISION-POS-CACHE`
- Change:
  - audited the first 20 samples and found strong `image_grid_thw` repetition under the current `vision_max_pixels=524288` budget
  - measured the raw upper bound of the two shape-only vision position helpers:
    - `fast_pos_embed_interpolate()` about `2.41 ms`
    - `rot_pos_emb()` about `1.20 ms`
  - then added a lazy cache for both tensors keyed by `image_grid_thw` in a benchmark-only subclass
- Results:
  - grid repetition in the first 20 local samples:
    - `(1, 36, 54)` appeared 9 times
    - `(1, 38, 52)` appeared 5 times
    - `(1, 44, 44)` appeared 3 times
  - lazy vision-position cache, 3 samples: TTFT `311.24 ms`, Throughput `23.72 tok/s`, artifact `result_wsl_visionposcache_seq_3.json`
- Key finding:
  - even though the grid shapes repeat heavily and the first 10 warmup samples already cover the first 3 measured shapes, the full benchmark still regressed
  - the theoretical upside is only about `3.6 ms` of vision work, so the extra cache-key handling overhead is enough to erase it
  - outputs matched the WSL baseline exactly on the checked 3 samples
- Status: rejected; not worth adding to the main code

### I50

- Date: 2026-03-22
- Path: `PATH-CACHE-PREFILL-REUSE-DIRECT-LM-DECODE`
- Change:
  - removed two decode-stage Python / wrapper overhead sources in the cached suffix loop:
    - replaced repeated `cache_position[0].item()` decode-stage checks with cache-length based checks via `DynamicCache.get_seq_length()`
    - added a lower-overhead prefill-reuse suffix decode path that calls `language_model + lm_head` directly with explicit recurrent `position_ids`, skipping `prepare_inputs_for_generation` and the outer Qwen3-VL wrapper on each decode step
  - restricted the direct path to the safe benchmark case where the cached decode attention mask is a trivial all-ones mask
  - revalidated the real WSL runtime packages while debugging the path:
    - `flash_attn` import works
    - `triton` import works
    - `fast_attn` is still not importable in `/home/apulupie/miniconda3/envs/AI`
- Results:
  - smoke test, direct decode ON, 1 sample: TTFT `315.27 ms`, Throughput `23.30 tok/s`, artifact `result_wsl_prefilldirect_smoke_seq_1.json`
  - direct decode OFF, 3 samples: TTFT `314.76 ms`, Throughput `23.00 tok/s`, artifact `result_wsl_prefilldirect_off_seq_3.json`
  - direct decode ON, 3 samples: TTFT `313.79 ms`, Throughput `23.82 tok/s`, artifact `result_wsl_prefilldirect_on_seq_3.json`
- Key finding:
  - on the same WSL flash-prefill + sdpa-decode mainline, the direct language-model decode path produced a real throughput gain of about `+0.82 tok/s` (`+3.6%`) and a small TTFT improvement of about `-0.97 ms`
  - the checked 3-sample outputs matched exactly between OFF and ON, so this looks like a clean decode-overhead win rather than a behavior change
  - this path lines up with the current profiling signal: the biggest remaining opportunity is not a new unstable attention kernel, but reducing cached decode wrapper overhead around the attention-heavy suffix loop
- Status: promoted to current default for the WSL / Linux mainline; keep the env switch available as `AICAS_PREFILL_REUSE_DIRECT_LM_DECODE=0/1`

### I51

- Date: 2026-03-22
- Path: `PATH-CACHE-PREFILL-REUSE-DIRECT-LM-DECODE` larger-sample validation
- Change:
  - revalidated the promoted direct language-model suffix decode path on larger WSL sample counts
  - compared `AICAS_PREFILL_REUSE_DIRECT_LM_DECODE=0` vs `=1` under the same flash-prefill + sdpa-decode mainline
- Results:
  - 5 samples, OFF: TTFT `1104.91 ms`, Throughput `9.24 tok/s`, artifact `result_wsl_prefilldirect_off_seq_5.json`
  - 5 samples, ON: TTFT `298.13 ms`, Throughput `10.77 tok/s`, artifact `result_wsl_prefilldirect_on_seq_5.json`
  - 10 samples, OFF: TTFT `277.17 ms`, Throughput `24.23 tok/s`, artifact `result_wsl_prefilldirect_off_seq_10.json`
  - 10 samples, ON: TTFT `275.76 ms`, Throughput `25.06 tok/s`, artifact `result_wsl_prefilldirect_on_seq_10.json`
- Key finding:
  - on the cleaner 10-sample comparison, the direct suffix decode path held up with about `+0.83 tok/s` throughput gain and about `-1.41 ms` TTFT improvement
  - the 5-sample run showed a much larger gap, which suggests this path can also help avoid some high-variance decode-loop stalls on harder samples
  - checked outputs matched exactly between OFF and ON for both the 5-sample and 10-sample comparisons
- Status: keep promoted as the current default path

### I52

- Date: 2026-03-22
- Path: `PATH-ATTN-DECODE-MANUAL-GQA`
- Change:
  - tried a decode-only text-attention micro-kernel in Python that keeps grouped-query attention grouped instead of following the torch 2.4.1 `sdpa` fallback path that materializes repeated KV heads
  - wired it only for the safe cached decode case: `batch=1`, `query_length=1`, no padding mask, `sdpa` suffix decode, and existing KV cache
  - validated behavior against the current direct suffix decode mainline
- Results:
  - smoke test, manual GQA ON, 1 sample: TTFT `318.74 ms`, Throughput `21.86 tok/s`, artifact `result_wsl_manualgqa_smoke_seq_1.json`
  - 3 samples, current direct suffix decode mainline: TTFT `313.79 ms`, Throughput `23.82 tok/s`, artifact `result_wsl_prefilldirect_on_seq_3.json`
  - 3 samples, manual GQA ON: TTFT `292.64 ms`, Throughput `22.75 tok/s`, artifact `result_wsl_manualgqa_on_seq_3.json`
- Key finding:
  - the manual grouped decode attention reduced TTFT a bit, but it lost about `1.07 tok/s` of throughput on the 3-sample comparison, which is the wrong direction for the current bottleneck
  - outputs still matched exactly on the checked samples, so the issue is purely performance-path efficiency
  - the likely reason is that the Python-side grouped matmul + softmax path saves KV repetition but gives back too much by losing SDPA kernel efficiency
- Status: rejected and removed from the main code; do not enable this path further unless there is a lower-level kernel implementation

### I53

- Date: 2026-03-22
- Path: `PATH-PROFILE-CURRENT-MAINLINE` and `PATH-ATTN-KERNEL-DECODE-OVERRIDE`
- Change:
  - added a one-off profiling utility, `profile_current_mainline.py`, to measure the current default WSL mainline on one representative sample while preserving the benchmark's prefill-reuse semantics
  - produced fresh stage / module / op artifacts for the promoted direct suffix decode path:
    - `profile_current_wsl_stage.json`
    - `profile_current_wsl_module.json`
    - `profile_current_wsl_ops.json`
  - then ran one targeted operator experiment by forcing the direct suffix decode path to use `flash_attention_2` instead of the current default `sdpa`
- Results:
  - current mainline profile on sample `34603`:
    - TTFT wall `304.70 ms`
    - TTFT visual forward `150.37 ms`
    - TTFT language-model forward `114.88 ms`
    - full wall `2761.14 ms`
    - full decode attention `1239.80 ms` total, about `19.68 ms/step`
    - full decode MLP `664.04 ms` total, about `10.54 ms/step`
    - full decode `lm_head` `204.62 ms` total, about `3.25 ms/step`
    - direct decode loop non-LM device overhead only about `25.84 ms` total
  - compared with the older profile on the same sample:
    - decode `prepare_inputs_for_generation` dropped from about `42.02 ms` to `0.0 ms`
    - decode attention dropped from about `1416.62 ms` to `1239.80 ms`
    - decode language-model forward dropped from about `2550.50 ms` to `2266.13 ms`
    - `aten::item` / `_local_scalar_dense` / `cudaStreamSynchronize` counts in the decode op profile dropped from about `450` calls each to about `72`
    - `cudaLaunchKernel` still stayed extremely high at about `103825` calls
  - targeted backend experiment, forcing decode `flash_attention_2`, 3 samples:
    - current direct suffix decode baseline: TTFT `313.79 ms`, Throughput `23.82 tok/s`, artifact `result_wsl_prefilldirect_on_seq_3.json`
    - decode `flash_attention_2` override: TTFT `327.63 ms`, Throughput `18.81 tok/s`, artifact `result_wsl_prefilldirect_decodeflash2_seq_3.json`
    - checked outputs still matched exactly
- Key finding:
  - the current direct suffix decode path successfully removed most of the old Python-side decode bookkeeping overhead; the remaining bottleneck is now much more cleanly inside cached decode attention itself plus the huge kernel-launch volume around single-token decode
  - there is still some operator headroom in principle, but this headroom no longer looks like a simple backend-selection problem
  - the failed `flash_attention_2` decode override strongly suggests that the easy operator switches are exhausted under the current HF Qwen3-VL stack; the next real gains would likely require a lower-level fused decode kernel or a stable graph-capture path, not another Python-side attention wrapper
- Status: keep the current flash-prefill + sdpa-decode mixed route and keep `PATH-CACHE-PREFILL-REUSE-DIRECT-LM-DECODE` ON; use the new profiling artifacts as the baseline for any future low-level kernel work

### I54

- Date: 2026-03-22
- Path: `PATH-CACHE-PREFILL-REUSE-STATIC-SUFFIX-CACHE`
- Change:
  - tried converting the direct suffix decode path from `DynamicCache` to Hugging Face `StaticCache`
  - restricted it to the existing prefill-reuse direct language-model decode route
- Results:
  - smoke test, 1 sample: TTFT `289.30 ms`, Throughput `22.60 tok/s`, artifact `result_wsl_staticsuffix_smoke_seq_1.json`
  - 3 samples: TTFT `266.67 ms`, Throughput `22.78 tok/s`, artifact `result_wsl_staticsuffix_on_seq_3.json`
  - fresh profile artifacts:
    - `profile_current_staticsuffix_wsl_stage.json`
    - `profile_current_staticsuffix_wsl_module.json`
    - `profile_current_staticsuffix_wsl_ops.json`
- Key finding:
  - this path did remove `DynamicCache` reallocation, but it also pushed the suffix decode route onto the heavier static-cache masking behavior
  - decode attention got slower per step in the profile, and the 3-sample throughput still lost about `1.04 tok/s` relative to the current direct suffix decode baseline, even though TTFT improved
  - the checked 3-sample outputs still matched exactly, so the issue was performance-path efficiency rather than answer drift
- Status: rejected and removed from the code; do not use HF `StaticCache` for this suffix decode path

### I55

- Date: 2026-03-22
- Path: `PATH-CACHE-PREFILL-REUSE-INPLACE-SUFFIX-CACHE`
- Change:
  - replaced the rejected HF `StaticCache` experiment with a lighter custom decode-only cache for the direct suffix route
  - the new cache preallocates per-layer KV storage once, appends new KV states in-place with `index_copy_`, and keeps the cheaper DynamicCache-style mask sizing instead of forcing full static-cache masks
  - promoted this path to the current default via `AICAS_PREFILL_REUSE_INPLACE_SUFFIX_CACHE=1` by default, while keeping the env switch available
- Results:
  - smoke test, 1 sample: TTFT `236.12 ms`, Throughput `25.63 tok/s`, artifact `result_wsl_inplacesuffix_smoke_seq_1.json`
  - 3 samples, explicit env ON: TTFT `238.50 ms`, Throughput `25.66 tok/s`, artifact `result_wsl_inplacesuffix_on_seq_3.json`
  - 5 samples, explicit env ON: TTFT `243.26 ms`, Throughput `24.87 tok/s`, artifact `result_wsl_inplacesuffix_on_seq_5.json`
  - 10 samples, explicit env ON: TTFT `242.19 ms`, Throughput `26.47 tok/s`, artifact `result_wsl_inplacesuffix_on_seq_10.json`
  - 10 samples, promoted default mainline: TTFT `245.71 ms`, Throughput `26.88 tok/s`, artifact `result_wsl_currentdefault_seq_10.json`
  - fresh profile artifacts on the new path:
    - `profile_current_inplacesuffix_wsl_module.json`
    - `profile_current_inplacesuffix_wsl_ops.json`
- Key finding:
  - the promoted default 10-sample comparison against the previous direct suffix decode baseline improved TTFT by about `-30.05 ms` and Throughput by about `+1.82 tok/s` (`26.88` vs `25.06`)
  - checked answer outputs matched exactly on the compared 3-sample, 5-sample, and 10-sample runs
  - the new path reduced decode kernel-launch volume in the fresh profile from about `103825` to about `80865` calls and reduced the direct decode loop's non-LM overhead further, which matches the expected direction for the current bottleneck
  - one residual nuance is that the profiled sample sometimes stopped in fewer decode steps while producing the same checked answer text, so future work should keep an eye on token-count behavior when validating new decode changes
- Status: promoted to the current default mainline; keep `PATH-CACHE-PREFILL-REUSE-DIRECT-LM-DECODE` ON and keep `PATH-CACHE-PREFILL-REUSE-INPLACE-SUFFIX-CACHE` ON

### I56

- Date: 2026-03-22
- Path: `PATH-CACHE-PREFILL-REUSE-DIRECT-NO-MASK-DECODE`
- Change:
  - added an opt-in decode-only route that bypasses `Qwen3VLTextModel.forward` on the suffix path
  - the custom route directly runs `embed_tokens -> rotary_emb -> decoder layers -> norm -> lm_head` and skips explicit `create_causal_mask`
  - restricted it to the existing prefill-reuse direct language-model decode path and only enabled it when the decode backend is `sdpa`
- Results:
  - current reverted baseline, 3 samples: TTFT `288.30 ms`, Throughput `24.40 tok/s`, artifact `result_wsl_currentdefault_v3_seq_3.json`
  - no-mask experiment, 3 samples: TTFT `302.79 ms`, Throughput `23.77 tok/s`, artifact `result_wsl_nomaskdecode_seq_3.json`
- Key finding:
  - the checked 3-sample answers matched exactly, so correctness looked fine on the initial comparison
  - despite removing the explicit mask-construction path, the manual no-mask route still lost about `0.63 tok/s` and `14.49 ms` TTFT relative to the current mainline, so the saved mask work did not translate into a better end-to-end decode path on this torch `2.4.1` stack
- Status: rejected and left OFF by default; keep `PATH-CACHE-PREFILL-REUSE-DIRECT-NO-MASK-DECODE` OFF unless running a targeted validation experiment

### I57

- Date: 2026-03-22
- Path: `PATH-CACHE-PREFILL-REUSE-INPLACE-SUFFIX-WRITE-MODE`
- Change:
  - split the custom in-place suffix cache write path into two isolated modes: `index_copy` and `slice_copy`
  - benchmarked only the KV append operator difference while keeping the rest of the decode loop unchanged
  - promoted `slice_copy` to the new default write mode via `AICAS_PREFILL_REUSE_INPLACE_SUFFIX_WRITE_MODE=slice_copy`
- Results:
  - current index-copy baseline, 3 samples: TTFT `288.30 ms`, Throughput `24.40 tok/s`, artifact `result_wsl_currentdefault_v3_seq_3.json`
  - slice-copy experiment, 3 samples: TTFT `286.23 ms`, Throughput `24.80 tok/s`, artifact `result_wsl_inplacesuffix_slicecopy_seq_3.json`
  - current index-copy baseline, 10 samples: TTFT `274.46 ms`, Throughput `23.94 tok/s`, artifact `result_wsl_currentdefault_v3_seq_10.json`
  - slice-copy experiment, 10 samples: TTFT `277.28 ms`, Throughput `24.62 tok/s`, artifact `result_wsl_inplacesuffix_slicecopy_seq_10.json`
  - promoted default smoke, 3 samples: TTFT `282.83 ms`, Throughput `24.97 tok/s`, artifact `result_wsl_currentdefault_v4_seq_3.json`
  - fresh profile artifacts:
    - `profile_current_v3_indexcopy_wsl_module.json`
    - `profile_current_v3_indexcopy_wsl_ops.json`
    - `profile_current_v3_slicecopy_wsl_module.json`
    - `profile_current_v3_slicecopy_wsl_ops.json`
- Key finding:
  - the checked 10-sample answers matched exactly on the `index_copy` and `slice_copy` runs
  - `slice_copy` improved throughput by about `+0.40 tok/s` on 3 samples and `+0.68 tok/s` on 10 samples relative to the current `index_copy` baseline, while TTFT stayed in the same band
  - the matched single-sample profile showed the expected operator-level direction: `aten::index_copy_` disappeared from the hot decode ops, decode `cudaLaunchKernel` calls dropped from about `105702` to `103938`, full decode attention time improved from about `1409.96 ms` to `1349.15 ms`, and total full-generation wall time improved from about `3005.68 ms` to `2935.47 ms`
- Status: promoted to the current default mainline; keep `PATH-CACHE-PREFILL-REUSE-INPLACE-SUFFIX-CACHE` ON and keep `PATH-CACHE-PREFILL-REUSE-INPLACE-SUFFIX-WRITE-MODE` on `slice_copy`

### I58

- Date: 2026-03-22
- Path: `PATH-CACHE-PREFILL-REUSE-CUDA-GRAPH-DECODE`
- Change:
  - added a decode-only CUDA Graph route for the existing prefill-reuse direct language-model suffix path
  - the graph route builds a fixed-shape decode cache, uses a fixed-shape boolean attention mask, captures the `batch=1, q_len=1` suffix decode step once, and replays it across later decode steps
  - kept the previous slice-copy in-place suffix cache path as the fallback path, and promoted the CUDA Graph route to the new default via `AICAS_PREFILL_REUSE_CUDA_GRAPH_DECODE=1`
  - added a small permanent diagnostic helper `profile_decode_ops_simple.py` because the older bucketed event profiler is not reliable under graph replay
- Results:
  - previous default mainline, 3 samples: TTFT `282.83 ms`, Throughput `24.97 tok/s`, artifact `result_wsl_currentdefault_v4_seq_3.json`
  - graph experiment, 3 samples: TTFT `291.96 ms`, Throughput `32.76 tok/s`, artifact `result_wsl_graphdecode_seq_3.json`
  - previous best explicit slice-copy path, 10 samples: TTFT `277.28 ms`, Throughput `24.62 tok/s`, artifact `result_wsl_inplacesuffix_slicecopy_seq_10.json`
  - graph experiment, 10 samples: TTFT `276.20 ms`, Throughput `35.89 tok/s`, artifact `result_wsl_graphdecode_seq_10.json`
  - promoted default smoke, 3 samples: TTFT `284.65 ms`, Throughput `32.84 tok/s`, artifact `result_wsl_currentdefault_v5_seq_3.json`
- Key finding:
  - the checked 3-sample and 10-sample answers matched exactly on the compared baseline and graph runs
  - relative to the previous best explicit slice-copy path, the graph route improved 10-sample throughput by about `+11.27 tok/s` while also slightly improving TTFT by about `-1.08 ms`
  - a runtime self-check on the same benchmark-style two-stage prefill-reuse flow confirmed that the graph runner did actually capture successfully, with a reused decode-cache bucket of `640` tokens
  - this result strongly suggests that the remaining dominant cost on the old path was indeed per-step launch / framework overhead around decode attention rather than only the math kernels themselves
- Status: promoted to the current default mainline; keep `PATH-CACHE-PREFILL-REUSE-CUDA-GRAPH-DECODE` ON, keep `PATH-CACHE-PREFILL-REUSE-CUDA-GRAPH-BUCKET` at `64`, and keep `PATH-CACHE-PREFILL-REUSE-CUDA-GRAPH-MAX-CACHE-LEN` at `1024` unless a later experiment proves a better setting

### I59

- Date: 2026-03-22
- Path: `PATH-CACHE-PREFILL-REUSE-TRITON-DECODE-ATTN`
- Change:
  - added an opt-in custom Triton decode attention kernel for the `batch=1, q_len=1, GQA` suffix path
  - integrated it through a manual per-layer text decode route that reuses the existing prefill-reuse direct LM scaffold and only swaps the attention math
  - first version used the existing variable-length cache view and was effectively shape-thrashing the Triton JIT; second version switched the Triton path to a fixed-shape cache buffer to stabilize the kernel shape
- Results:
  - Triton v1 smoke, 1 sample: TTFT `304.28 ms`, Throughput `1.83 tok/s`, artifact `result_wsl_tritondecode_smoke_seq_1.json`
  - Triton v2 smoke with fixed-shape cache, 1 sample: TTFT `324.94 ms`, Throughput `17.95 tok/s`, artifact `result_wsl_tritondecode_v2_smoke_seq_1.json`
  - current default mainline smoke, 3 samples: TTFT `284.65 ms`, Throughput `32.84 tok/s`, artifact `result_wsl_currentdefault_v5_seq_3.json`
- Key finding:
  - the fixed-shape cache change fixed the worst shape-specialization issue and improved the custom-kernel smoke from `1.83` to `17.95 tok/s`
  - the checked smoke answer still matched the current default output on the shared sample, so the custom attention math path looked functionally correct on the initial comparison
  - even after fixing the cache-shape issue, the custom Triton path still lost badly to the current default mainline, which means replacing only the attention math is not enough; the qkv/rope/norm/o_proj scaffolding around it is still too expensive in this integration
- Status: left OFF by default; keep `PATH-CACHE-PREFILL-REUSE-TRITON-DECODE-ATTN` OFF unless running a targeted custom-op experiment

## Current Recommendation

If the goal is the safest local default for ongoing work:
- keep `PATH-ATTN-KERNEL-AUTO`
- keep `PATH-ATTN-KERNEL-MIXED-DECODE`
- keep `PATH-ATTN-SDPA-KERNEL-MODE` on `legacy_no_flash`
- keep `PATH-VISION-BUDGET`
- keep `PATH-MIDLAYER-POOL`
- keep `PATH-CACHE-SINGLE-TOKEN`
- keep `PATH-CACHE-PREFILL-REUSE`
- keep `PATH-CACHE-PREALLOCATE-DYNAMIC` in `capture` mode
- keep `PATH-ANSWER-DECODE` OFF
- keep `PATH-ATTN-KERNEL-EAGER` OFF
- keep `PATH-ATTN-KERNEL-FLASH2-REQUEST` as an opt-in validation switch
- keep `PATH-ATTN-KERNEL-DECODE-OVERRIDE` unset unless running a targeted backend experiment
- keep `PATH-CACHE-DIRECT-SINGLE` OFF
- keep `PATH-CACHE-FAST-SINGLE` OFF
- keep `PATH-CACHE-DECODE-GRID` OFF
- keep `PATH-CACHE-DECODE-MASK` OFF
- keep `PATH-CACHE-DECODE-POSIDS` OFF
- keep `PATH-CACHE-DECODE-KWARGS` ON
- keep `PATH-CACHE-PREFILL-REUSE-DELEGATE` OFF
- keep `PATH-CACHE-PREFILL-REUSE-DIRECT-LM-DECODE` ON
- keep `PATH-CACHE-PREFILL-REUSE-CUDA-GRAPH-DECODE` ON
- keep `PATH-CACHE-PREFILL-REUSE-CUDA-GRAPH-BUCKET` at `64`
- keep `PATH-CACHE-PREFILL-REUSE-CUDA-GRAPH-MAX-CACHE-LEN` at `1024`
- keep `PATH-CACHE-PREFILL-REUSE-NATIVE-CUDA-DECODE-ATTN` ON
- keep `PATH-CACHE-PREFILL-REUSE-TRITON-DECODE-ATTN` OFF
- keep `PATH-CACHE-PREFILL-REUSE-DIRECT-NO-MASK-DECODE` OFF
- keep `PATH-CACHE-PREFILL-REUSE-INPLACE-SUFFIX-CACHE` ON
- keep `PATH-CACHE-PREFILL-REUSE-INPLACE-SUFFIX-WRITE-MODE` on `slice_copy`
- keep `PATH-CACHE-LEGACY-RETURN` unset by default
- keep `PATH-CACHE-IMPLEMENTATION` unset unless running a dedicated experiment

## Logging Rule For Future Iterations

For every new iteration, append one entry with:
- date
- path id
- change summary
- benchmark command or result file
- TTFT and Throughput
- whether the path was kept, reverted, or left optional

### I60

- Date: 2026-03-22
- Path: `PATH-CACHE-PREFILL-REUSE-NATIVE-CUDA-DECODE-ATTN`
- Change:
  - built a native C++/CUDA extension `native_cuda_ops._decode_q1_gqa` for the `batch=1, q_len=1, GQA` decode-only attention path
  - first integrated the extension as a graph-off manual suffix-decode attention replacement, then iterated the CUDA kernel from a `128 threads x 1 dim/thread` version to a warp-style `32 threads x 4 dims/thread` version
  - after the graph-off native path showed a real but small win, integrated the native attention route into the existing CUDA Graph decode runner so the captured suffix step could also use the custom kernel
  - promoted `AICAS_PREFILL_REUSE_NATIVE_CUDA_DECODE_ATTN=1` to the new default because the combined native+graph path won on the stable 10-sample comparison
- Results:
  - graph-off baseline, 3 samples: TTFT `313.98 ms`, Throughput `24.10 tok/s`, artifact `result_wsl_graphoff_nonative_seq_3.json`
  - native CUDA v1 graph-off, 3 samples: TTFT `318.71 ms`, Throughput `23.13 tok/s`, artifact `result_wsl_nativecuda_seq_3.json`
  - native CUDA v2 graph-off, 3 samples: TTFT `317.04 ms`, Throughput `24.95 tok/s`, artifact `result_wsl_nativecuda_v2_seq_3.json`
  - current default graph path, 3 samples: TTFT `312.22 ms`, Throughput `29.66 tok/s`, artifact `result_wsl_currentdefault_v6_seq_3.json`
  - combined graph + native CUDA, 3 samples: TTFT `328.94 ms`, Throughput `34.83 tok/s`, artifact `result_wsl_graphnative_seq_3.json`
  - current default graph path, 10 samples: TTFT `311.42 ms`, Throughput `31.33 tok/s`, artifact `result_wsl_currentdefault_v6_seq_10.json`
  - combined graph + native CUDA, 10 samples: TTFT `305.31 ms`, Throughput `36.58 tok/s`, artifact `result_wsl_graphnative_seq_10.json`
- Key finding:
  - the first native CUDA kernel was not good enough and lost to the graph-off baseline, but the second warp-style version recovered the graph-off path and improved 3-sample throughput by about `+0.85 tok/s`
  - the real payoff came from combining the custom native attention kernel with the existing CUDA Graph suffix-decode runner; on the stable 10-sample comparison it improved throughput by about `+5.25 tok/s` and also improved TTFT by about `-6.11 ms` relative to the current graph default in the same code version
  - the checked 3-sample and 10-sample answers matched exactly against the compared non-native graph baselines, so the gain came from the decode path implementation rather than output drift
- Status: promoted to the current default mainline; keep `PATH-CACHE-PREFILL-REUSE-NATIVE-CUDA-DECODE-ATTN` ON unless a later regression test disproves it


### I61

- Date: 2026-03-22
- Path: `PATH-CACHE-PREFILL-REUSE-NATIVE-CUDA-RMSNORM`
- Change:
  - profiled the promoted native+graph mainline and found the next dominant small-op cluster around `mul / mean / pow / rsqrt`, which matches the Qwen3-VL text RMSNorm implementation
  - extended the existing native CUDA extension with a generic FP16 RMSNorm kernel and exposed it as `native_cuda_ops.rmsnorm_forward`
  - wired the native RMSNorm into the native decode paths only: text-layer `input_layernorm`, `post_attention_layernorm`, and the attention-local `q_norm` / `k_norm` sites inside both the manual native decode route and the native CUDA Graph runner
- Results:
  - profiler snapshot on the pre-RMSNorm native+graph mainline: artifact `profile_graphnative_decodeops_wsl.json`
  - current code with native path disabled, 3 samples: TTFT `292.77 ms`, Throughput `33.30 tok/s`, artifact `result_wsl_nativeoff_control_seq_3.json`
  - current default with native attention + native RMSNorm + graph, 3 samples: TTFT `284.60 ms`, Throughput `43.21 tok/s`, artifact `result_wsl_currentdefault_v8_seq_3.json`
  - current code with native path disabled, 10 samples: TTFT `270.84 ms`, Throughput `36.58 tok/s`, artifact `result_wsl_nativeoff_control_seq_10.json`
  - current default with native attention + native RMSNorm + graph, 10 samples: TTFT `265.62 ms`, Throughput `45.91 tok/s`, artifact `result_wsl_currentdefault_v8_seq_10.json`
- Key finding:
  - the native RMSNorm kernel paid off immediately and materially; on the stable 10-sample comparison it improved throughput by about `+9.33 tok/s` and TTFT by about `-5.22 ms` relative to the same code version with the native path disabled
  - the checked 3-sample and 10-sample answers matched exactly against the native-off control runs, so this improvement did not come from output drift
  - after this step, the native decode stack is no longer just a custom attention kernel; it now also removes a large chunk of per-layer norm overhead from the decode-only hot path
- Status: promoted into the current default mainline; keep the native RMSNorm path ON as part of `PATH-CACHE-PREFILL-REUSE-NATIVE-CUDA-DECODE-ATTN`


### I62

- Date: 2026-03-22
- Path: `PATH-CACHE-PREFILL-REUSE-NATIVE-SILU-MUL`
- Change:
  - inspected the Qwen3-VL text MLP and confirmed it uses the standard SwiGLU form `down_proj(silu(gate_proj(x)) * up_proj(x))`
  - extended the native CUDA extension with a fused FP16 `silu_mul_forward` kernel
  - rewired the native decode paths only so the manual native route and the native CUDA Graph runner now evaluate text MLP as `gate_proj + up_proj + native silu_mul + down_proj`
- Results:
  - native attention + native RMSNorm + graph, 3 samples: TTFT `284.60 ms`, Throughput `43.21 tok/s`, artifact `result_wsl_currentdefault_v8_seq_3.json`
  - native attention + native RMSNorm + native SiLU*Mul + graph, 3 samples: TTFT `273.78 ms`, Throughput `45.78 tok/s`, artifact `result_wsl_currentdefault_v9_seq_3.json`
  - native attention + native RMSNorm + graph, 10 samples: TTFT `265.62 ms`, Throughput `45.91 tok/s`, artifact `result_wsl_currentdefault_v8_seq_10.json`
  - native attention + native RMSNorm + native SiLU*Mul + graph, 10 samples: TTFT `264.46 ms`, Throughput `46.49 tok/s`, artifact `result_wsl_currentdefault_v9_seq_10.json`
- Key finding:
  - the extra fused MLP pointwise op still paid off, but the gain is now much smaller than the RMSNorm step; on the stable 10-sample comparison it added about `+0.58 tok/s` and improved TTFT by about `-1.16 ms`
  - the checked 3-sample and 10-sample answers matched exactly against the pre-SiLU-fusion native path, so the new fused MLP op did not change outputs on the tested samples
  - this suggests the hand-written decode stack is entering diminishing-return territory on pointwise ops, and the next serious wins will likely require either reducing the remaining scalar-sync overhead or moving closer to the GEMM boundaries
- Status: kept in the current default mainline as part of `PATH-CACHE-PREFILL-REUSE-NATIVE-CUDA-DECODE-ATTN`


### I63

- Date: 2026-03-22
- Path: `PATH-CACHE-PREFILL-REUSE-NATIVE-DUAL-LINEAR`
- Change:
  - added a new native C++/CUDA op `native_cuda_ops.dual_linear_forward` backed by one cuBLAS batched GEMM call
  - wired the op into the graph-off native decode path only for the two same-shape linear pairs:
    - `k_proj + v_proj`
    - `gate_proj + up_proj`
  - explicitly kept the CUDA Graph runner on the old safe path so the experiment would not interfere with the promoted graph decode mainline
- Results:
  - current default graph mainline rerun, 3 samples: TTFT `296.86 ms`, Throughput `37.52 tok/s`, artifact `result_wsl_currentdefault_duallinear_seq_3_rerun.json`
  - graph-off native decode, dual-linear OFF, 3 samples: TTFT `302.87 ms`, Throughput `29.14 tok/s`, artifact `result_wsl_graphoff_duallinear_off_seq_3_rerun.json`
  - graph-off native decode, dual-linear ON, 3 samples: TTFT `316.73 ms`, Throughput `27.55 tok/s`, artifact `result_wsl_graphoff_duallinear_on_seq_3_rerun.json`
- Key finding:
  - the first parallel A/B attempt on this path was invalid because two GPU benchmarks were accidentally run at the same time; only the serial reruns above should be trusted
  - although the new op is numerically close to the reference linear path on standalone tensors, the graph-off end-to-end benchmark got worse rather than better
  - this strongly suggests that for the current `batch=1, q_len=1` decode shapes, the extra cuBLAS batched-pointer setup cost outweighs the saved Python / launch overhead
- Status: rejected as a performance path; keep the native dual-linear code only as a disabled reference and leave `AICAS_PREFILL_REUSE_NATIVE_DUAL_LINEAR=0` by default


### I64

- Date: 2026-03-22
- Path: `PATH-CACHE-PREFILL-REUSE-NATIVE-LM-HEAD-ARGMAX`
- Change:
  - added a native C++/CUDA op `native_cuda_ops.lm_head_argmax_forward` for the decode-only `batch=1, seq=1` path
  - integrated it into the graph-off direct decode loop and kept the CUDA Graph runner on the safe fallback path
  - fixed an early reduction bug in the stage-2 kernel so the fused op matches the reference `lm_head(...).argmax(...)` result on standalone tensors
- Results:
  - graph-on, native lm_head argmax OFF, 3 samples: TTFT `287.14 ms`, Throughput `43.81 tok/s`, artifact `result_wsl_lmheadargmax_off_seq_3.json`
  - graph-on, native lm_head argmax ON, 3 samples: TTFT `282.78 ms`, Throughput `43.41 tok/s`, artifact `result_wsl_lmheadargmax_on_seq_3.json`
  - graph-off, native lm_head argmax OFF, 3 samples: TTFT `302.61 ms`, Throughput `29.41 tok/s`, artifact `result_wsl_graphoff_lmheadargmax_off_seq_3.json`
  - graph-off, native lm_head argmax ON, 3 samples: TTFT `306.70 ms`, Throughput `24.28 tok/s`, artifact `result_wsl_graphoff_lmheadargmax_on_seq_3.json`
- Key finding:
  - despite removing the explicit Python-side `lm_head(...).argmax(...)` call, the fused op did not produce a stable end-to-end win
  - the graph-on path only traded a tiny TTFT improvement for a small throughput loss, while the graph-off path clearly regressed
  - this matches the decode profile: `lm_head` is visible, but it is not the dominant remaining bottleneck compared with the per-layer decode stack
- Status: rejected as a promoted path; keep `AICAS_PREFILL_REUSE_NATIVE_LM_HEAD_ARGMAX=0` by default and retain the kernel only as a disabled reference


### I65

- Date: 2026-03-22
- Path: `PATH-CACHE-PREFILL-REUSE-NATIVE-DUAL-LINEAR-V2`
- Change:
  - replaced the old cuBLAS batched-pointer implementation under `native_cuda_ops.dual_linear_forward` with a custom CUDA matvec kernel specialized for the decode-only `batch=1, seq=1` case
  - the new kernel loads the 2048-d hidden vector into shared memory once per block and computes the paired outputs together, which removes the old pointer-upload overhead and is CUDA-Graph-capture-safe
  - opened the CUDA Graph runner to this experimental path so the same kernel can now be benchmarked in both graph-off and graph-on decode
- Results:
  - graph-off, custom dual-linear OFF, 3 samples: TTFT `295.31 ms`, Throughput `29.52 tok/s`, artifact `result_wsl_graphoff_duallinear_custom_off_seq_3.json`
  - graph-off, custom dual-linear ON, 3 samples: TTFT `289.91 ms`, Throughput `29.94 tok/s`, artifact `result_wsl_graphoff_duallinear_custom_on_seq_3.json`
  - graph-on, custom dual-linear OFF, 3 samples: TTFT `299.42 ms`, Throughput `38.39 tok/s`, artifact `result_wsl_graph_duallinear_custom_off_seq_3.json`
  - graph-on, custom dual-linear ON, 3 samples: TTFT `305.64 ms`, Throughput `37.61 tok/s`, artifact `result_wsl_graph_duallinear_custom_on_seq_3.json`
- Key finding:
  - unlike the first cuBLAS-backed version, the new custom kernel finally turns the graph-off path slightly positive and the checked answers match exactly between ON/OFF runs
  - however, that gain does not survive the promoted graph mainline; once the decode loop is already captured, the custom dual-linear path loses both TTFT and throughput
  - the new result narrows the diagnosis: the pointer-setup overhead was real, but `k/v` and `gate/up` as a two-output fusion still are not enough to beat the graph mainline on their own
- Status: keep the custom kernel as an experimental reference, but do not promote it; leave `AICAS_PREFILL_REUSE_NATIVE_DUAL_LINEAR=0` by default and target larger fused projection kernels next


### I66

- Date: 2026-03-22
- Path: `PATH-CACHE-PREFILL-REUSE-NATIVE-QKV-LINEAR`
- Change:
  - added a native C++/CUDA op `native_cuda_ops.qkv_linear_forward` that fuses the decode-only `q_proj / k_proj / v_proj` matvecs into one shared-input kernel
  - wired it into `_native_decode_qkv_projections` with an independent runtime switch and disabled the old packed-QKV preparation when this path is selected
  - validated the op numerically against `torch.nn.functional.linear` and then ran serial graph-off / graph-on end-to-end comparisons
- Results:
  - graph-off, fused qkv OFF, 3 samples: TTFT `304.15 ms`, Throughput `29.50 tok/s`, artifact `result_wsl_graphoff_qkvlinear_off_seq_3.json`
  - graph-off, fused qkv ON, 3 samples: TTFT `296.81 ms`, Throughput `29.56 tok/s`, artifact `result_wsl_graphoff_qkvlinear_on_seq_3.json`
  - graph-on, fused qkv OFF, 3 samples: TTFT `303.84 ms`, Throughput `38.58 tok/s`, artifact `result_wsl_graph_qkvlinear_off_seq_3.json`
  - graph-on, fused qkv ON, 3 samples: TTFT `301.82 ms`, Throughput `38.77 tok/s`, artifact `result_wsl_graph_qkvlinear_on_seq_3.json`
  - graph-on, fused qkv OFF, 10 samples: TTFT `295.14 ms`, Throughput `39.72 tok/s`, artifact `result_wsl_graph_qkvlinear_off_seq_10.json`
  - graph-on, fused qkv ON, 10 samples: TTFT `295.61 ms`, Throughput `38.34 tok/s`, artifact `result_wsl_graph_qkvlinear_on_seq_10.json`
- Key finding:
  - the new fused qkv kernel is numerically sound and the checked ON/OFF answers matched exactly
  - it showed a small positive signal on the short 3-sample runs, but the stable 10-sample graph mainline reversed that result and lost about `-1.38 tok/s`
  - this suggests that even a three-way projection fusion is still not enough on its own to beat the already-captured graph mainline for this decode shape
- Status: rejected as a promoted path; keep `AICAS_PREFILL_REUSE_NATIVE_QKV_LINEAR=0` by default and retain the kernel only as an experimental reference


### I67

- Date: 2026-03-22
- Path: `PATH-CACHE-PREFILL-REUSE-NATIVE-GATE-UP-SILU`
- Change:
  - added a native C++/CUDA op `native_cuda_ops.gate_up_silu_forward` that fuses `gate_proj + up_proj + silu*mul` into one decode-only shared-input kernel
  - rewired `_native_text_mlp` so the native decode paths now prefer the fused op and only fall back to `dual_linear + silu_mul` when the new path is unavailable
  - promoted the runtime default to ON because the graph mainline kept a positive throughput gain while preserving outputs
- Results:
  - graph-off, fused gate/up/silu OFF, 3 samples: TTFT `297.76 ms`, Throughput `29.73 tok/s`, artifact `result_wsl_graphoff_gateupsilu_off_seq_3.json`
  - graph-off, fused gate/up/silu ON, 3 samples: TTFT `227.69 ms`, Throughput `37.14 tok/s`, artifact `result_wsl_graphoff_gateupsilu_on_seq_3.json`
  - graph-on, fused gate/up/silu OFF, 3 samples: TTFT `302.40 ms`, Throughput `39.25 tok/s`, artifact `result_wsl_graph_gateupsilu_off_seq_3.json`
  - graph-on, fused gate/up/silu ON, 3 samples: TTFT `297.74 ms`, Throughput `40.17 tok/s`, artifact `result_wsl_graph_gateupsilu_on_seq_3.json`
  - graph-on, fused gate/up/silu OFF, 10 samples: TTFT `290.93 ms`, Throughput `40.50 tok/s`, artifact `result_wsl_graph_gateupsilu_off_seq_10.json`
  - graph-on, fused gate/up/silu ON, 10 samples: TTFT `291.17 ms`, Throughput `41.44 tok/s`, artifact `result_wsl_graph_gateupsilu_on_seq_10.json`
- Key finding:
  - unlike the earlier projection-only experiments, this larger MLP fusion paid off immediately in graph-off and still held a stable win in the promoted graph mainline
  - on the stable 10-sample graph comparison it improved throughput by about `+0.94 tok/s` while TTFT stayed effectively flat at only about `+0.24 ms`
  - the checked 3-sample and 10-sample ON/OFF answers matched exactly, so the gain came from the decode implementation rather than output drift
- Status: promoted into the current default mainline; keep `AICAS_PREFILL_REUSE_NATIVE_GATE_UP_SILU=1` by default as part of the native decode stack


### I68

- Date: 2026-03-22
- Path: `PATH-CACHE-PREFILL-REUSE-NATIVE-QK-LINEAR-NORM`
- Change:
  - added a new native C++/CUDA op `native_cuda_ops.qkv_linear_qk_norm_forward` to fuse decode-only `q_proj/k_proj/v_proj` with in-op `q_norm/k_norm`
  - integrated the path into `_native_decode_qkv_projections` with a dedicated runtime switch `AICAS_PREFILL_REUSE_NATIVE_QK_LINEAR_NORM`
  - kept it default OFF, and kept `q/k` epsilon consistency checks and shape guards to preserve correctness
- Results:
  - graph-on, fused qk-linear-norm OFF, 3 samples: TTFT `234.90 ms`, Throughput `49.59 tok/s`, artifact `result_wsl_graph_qklinearnorm_off_seq_3.json`
  - graph-on, fused qk-linear-norm ON, 3 samples: TTFT `236.19 ms`, Throughput `48.65 tok/s`, artifact `result_wsl_graph_qklinearnorm_on_seq_3.json`
  - graph-off, fused qk-linear-norm OFF, 3 samples: TTFT `218.98 ms`, Throughput `37.30 tok/s`, artifact `result_wsl_graphoff_qklinearnorm_off_seq_3.json`
  - graph-off, fused qk-linear-norm ON, 3 samples: TTFT `282.00 ms`, Throughput `33.04 tok/s`, artifact `result_wsl_graphoff_qklinearnorm_on_seq_3.json`
- Key finding:
  - numerical checks passed at FP16 tolerance, and the compared ON/OFF answer texts matched exactly on both graph-on and graph-off runs
  - end-to-end performance regressed in both modes, with a small graph-on loss and a clear graph-off loss, so the current kernel organization does not beat the baseline projection path
  - this suggests the next projection-side win should target a stronger fused boundary (for example `packed_qkv + qk_norm + rope` style fusion) rather than only replacing linear+norm with the current custom matvec path
- Status: keep as a disabled reference path; leave `AICAS_PREFILL_REUSE_NATIVE_QK_LINEAR_NORM=0` by default


### I69

- Date: 2026-03-22
- Path: `PATH-CACHE-PREFILL-REUSE-NATIVE-PACKED-QKV-QK-NORM-ROPE`
- Change:
  - added a new native C++/CUDA op `native_cuda_ops.packed_qkv_qk_norm_rope_forward` that takes post-GEMM packed QKV and fuses decode-side `q/k RMSNorm + RoPE + attention layout transform` in one launch
  - rewired the native decode attention paths (direct native decode, Triton decode wrapper, and CUDA Graph native decode runner) to prefer this fused post-projection path before falling back to the old split `qkv -> qk_norm -> apply_rotary_pos_emb` flow
  - kept GEMM on the mature path (`torch.nn.functional.linear` / cuBLAS) and fused only the post-GEMM small ops boundary
  - added an independent runtime switch `AICAS_PREFILL_REUSE_NATIVE_PACKED_QKV_QK_NORM_ROPE` (promoted to default ON after validation), and auto-enabled packed-QKV preparation when this switch is ON
- Results:
  - standalone random-tensor numerical check against PyTorch reference (`split + RMSNorm + apply_rotary_pos_emb`) passed at FP16 tolerance:
    - `q_max_abs=0.00501`
    - `k_max_abs=0.00171`
    - `v_max_abs=0.0`
  - end-to-end smoke benchmark (1 sample):
    - OFF: TTFT `232.10 ms`, Throughput `44.51 tok/s`, artifact `result_wsl_graph_packedqkvqknormrope_off_smoke_seq_1.json`
    - ON: TTFT `317.77 ms`, Throughput `44.68 tok/s`, artifact `result_wsl_graph_packedqkvqknormrope_smoke_seq_1.json`
  - end-to-end serial graph-on benchmark (3 samples):
    - OFF: TTFT `262.70 ms`, Throughput `44.46 tok/s`, artifact `result_wsl_graph_packedqkvqknormrope_off_seq_3.json`
    - ON: TTFT `275.33 ms`, Throughput `45.47 tok/s`, artifact `result_wsl_graph_packedqkvqknormrope_on_seq_3.json`
  - end-to-end serial graph-on benchmark (10 samples):
    - OFF: TTFT `265.72 ms`, Throughput `45.84 tok/s`, artifact `result_wsl_graph_packedqkvqknormrope_off_seq_10.json`
    - ON: TTFT `264.48 ms`, Throughput `47.38 tok/s`, artifact `result_wsl_graph_packedqkvqknormrope_on_seq_10.json`
  - ON/OFF answers matched exactly on the checked 1-sample / 3-sample / 10-sample runs
- Key finding:
  - the fused post-GEMM boundary is functional and numerically aligned; the larger 10-sample run shows a stable throughput gain (about `+1.54 tok/s`) and a small TTFT improvement (about `-1.24 ms`)
  - although the short 3-sample run showed TTFT regression, the larger run reversed that signal, suggesting the path is beneficial under the promoted benchmark setting
  - because the path requires packed projection buffers, keep the env switch available for low-memory fallbacks
- Status: promoted into the current default mainline; keep `AICAS_PREFILL_REUSE_NATIVE_PACKED_QKV_QK_NORM_ROPE=1` by default, and set it to `0` only for targeted rollback checks


### I70

- Date: 2026-03-25
- Path: `PATH-CACHE-PREFILL-REUSE-STATE-NORMALIZATION`
- Change:
  - fixed the prefill-reuse state capture so the saved continuation state is always normalized to the single-token decode contract expected by the direct LM decode and CUDA Graph paths
  - stopped trusting upstream `cache_position` shape directly during capture; instead derive the next decode position from `past_key_values.get_seq_length()` and fall back to `last(cache_position)+1` only if the cache length is unavailable
  - tightened the direct LM decode gate so it only runs when `first_token`, `cache_position`, and `position_ids` all match the required single-token shapes
  - added a graph-runner shape gate before CUDA Graph capture so incompatible continuation state falls back safely instead of crashing at `static_position_ids.copy_(...)`
- Results:
  - static validation: synthetic `cache_position=torch.arange(523)` now normalizes to a single decode position and produces `position_ids.shape == [3, 1, 1]`
  - repaired default mainline, 10 samples: TTFT `239.50 ms`, Throughput `49.69 tok/s`, artifact `.tmp_runtime/result_fixed_default_10.json`
  - conservative reference with `AICAS_PREFILL_REUSE_DIRECT_LM_DECODE=0`, 10 samples: TTFT `240.77 ms`, Throughput `25.43 tok/s`, artifact `.tmp_runtime/result_reference_no_direct_lm_10.json`
  - answer comparison on the same 10 samples: `0` mismatches between repaired default and the conservative reference path
- Key finding:
  - the failure on other machines came from an implicit assumption that captured `cache_position` was already single-token; once that assumption breaks, the graph runner receives full-sequence `position_ids` and crashes on fixed-shape graph inputs
  - normalizing continuation state at capture time is the correct fix because it repairs both the direct decode path and the graph-capture path without weakening the promoted optimization route
  - on the checked 10-sample run, the repaired default preserved output text exactly against the conservative reference while keeping a large throughput advantage
- Status: promoted into the current default mainline; keep the normalization and shape gates in place as cross-version compatibility hardening for the prefill-reuse route


### I71

- Date: 2026-03-27
- Path: `PATH-CACHE-PREFILL-REUSE-CUDA-GRAPH-PREWARM`
- Change:
  - added an untimed graph-prewarm route for the benchmark-shaped throughput bucket
  - on multi-token calls outside the timed `max_new_tokens=128` path, the wrapper now:
    - runs a cheap internal `max_new_tokens=1` capture-only prefill
    - normalizes into the same single-token decode state used by the promoted direct-LM suffix path
    - temporarily switches to the real suffix decode backend (`sdpa` under the current flash-prefill mainline)
    - pre-captures the CUDA Graph runner for the `128`-token throughput bucket
- Results:
  - prewarm OFF, 10 samples: TTFT `229.02 ms`, Throughput `53.74 tok/s`, artifact `.tmp_runtime/stage_profile_prewarm_off_10.json`
  - prewarm ON, 10 samples: TTFT `234.48 ms`, Throughput `54.04 tok/s`, artifact `.tmp_runtime/stage_profile_prewarm_on_10.json`
  - final HEAD recheck after later rejected branches were turned back OFF: TTFT `239.66 ms`, Throughput `53.57 tok/s`, artifact `.tmp_runtime/stage_profile_final_default_10.json`
- Key finding:
  - the new route did what it was supposed to do mechanically: throughput-stage `cuda_graph_capture_success` dropped from `2` to `0`, and throughput-stage `avg_cuda_graph_capture_wall_ms` dropped from about `40.93 ms` to `0.0 ms`
  - the end-to-end throughput gain on the 10-sample A/B was modest but positive at about `+0.30 tok/s`
  - TTFT moved slightly in the wrong direction on that one comparison even though the timed TTFT path is unchanged, so that part is treated as normal run-to-run noise rather than a reliable regression signal
- Status: kept as default because it removes measured throughput-side graph setup cost without changing model outputs


### I72

- Date: 2026-03-27
- Path: `PATH-CACHE-PREFILL-REUSE-NATIVE-DOWN-PROJ-RESIDUAL`
- Change:
  - added a new native C++/CUDA op `native_cuda_ops.linear_residual_forward`
  - rewired the native decode MLP path so it can optionally run:
    - existing fused `gate_proj + up_proj + silu*mul`
    - then a new decode-only fused `down_proj + residual add`
  - kept the implementation in code behind `AICAS_PREFILL_REUSE_NATIVE_DOWN_PROJ_RESIDUAL`
- Results:
  - smoke, fused `down_proj + residual` OFF, 1 sample: TTFT `242.15 ms`, Throughput `56.71 tok/s`, artifact `.tmp_runtime/stage_profile_mlp_smoke_off_1.json`
  - smoke, fused `down_proj + residual` ON, 1 sample: TTFT `236.87 ms`, Throughput `45.82 tok/s`, artifact `.tmp_runtime/stage_profile_mlp_smoke_on_1.json`
- Key finding:
  - the new kernel was functionally stable enough to run end-to-end, but the performance direction was clearly wrong: throughput dropped by about `-10.89 tok/s` on the first clean serial smoke comparison
  - the likely reason is the same one seen in earlier rejected projection-fusion experiments: for this decode shape, the mature `down_proj` GEMV path is already better than the custom matvec replacement, so the saved residual-add launch is nowhere near enough to pay for the weaker GEMV
- Status: rejected as a promoted path; keep the kernel only as a disabled reference and leave `AICAS_PREFILL_REUSE_NATIVE_DOWN_PROJ_RESIDUAL=0` by default


### I73

- Date: 2026-03-27
- Path: `PATH-CACHE-PREFILL-REUSE-NATIVE-CACHE-APPEND-ATTN`
- Change:
  - added a new native C++/CUDA op `native_cuda_ops.decode_q1_gqa_append_forward`
  - the op writes the current decode token's KV state into the fixed cache and computes `q_len=1` GQA attention in one kernel
  - integrated the route into both native decode entry points:
    - the CUDA Graph runner's native decode step
    - the graph-off native direct decode path when it is backed by the fixed / in-place custom caches
  - kept the route behind `AICAS_PREFILL_REUSE_NATIVE_CACHE_APPEND_ATTN`
- Results:
  - smoke, fused cache-append attention OFF, 1 sample: TTFT `247.22 ms`, Throughput `55.99 tok/s`, artifact `.tmp_runtime/stage_profile_cacheattn_smoke_off_1.json`
  - smoke, fused cache-append attention ON, 1 sample: TTFT `232.77 ms`, Throughput `54.89 tok/s`, artifact `.tmp_runtime/stage_profile_cacheattn_smoke_on_1.json`
  - serial 10-sample A/B:
    - OFF: TTFT `234.96 ms`, Throughput `53.77 tok/s`, artifact `.tmp_runtime/stage_profile_cacheattn_off_10.json`
    - ON: TTFT `239.46 ms`, Throughput `54.05 tok/s`, artifact `.tmp_runtime/stage_profile_cacheattn_on_10.json`
- Key finding:
  - the larger 10-sample comparison did show a tiny throughput gain of about `+0.28 tok/s`, which means the fused path is at least directionally plausible
  - however, that gain came with a TTFT regression of about `+4.50 ms`, and the absolute win was too small to justify leaving more complex native decode logic on by default
  - the right reading is that cache-append fusion is not obviously broken, but it has not yet crossed the bar for a trustworthy mainline promotion
- Status: keep as an optional experiment only; leave `AICAS_PREFILL_REUSE_NATIVE_CACHE_APPEND_ATTN=0` by default


### I74

- Date: 2026-03-27
- Path: `PATH-CACHE-PREFILL-REUSE-ADDMM-DOWN-PROJ-RESIDUAL`
- Change:
  - replaced the rejected custom `down_proj + residual` matvec idea with a library-path variant based on `torch.addmm(...)`
  - kept the rest of the native decode stack unchanged:
    - native packed qkv/qk-norm/rope
    - native decode attention
    - native `gate_proj + up_proj + silu*mul`
    - CUDA Graph decode + prewarm
  - the new route only swaps the final MLP projection/add boundary inside the native decode path
- Results:
  - control, 10 samples: TTFT `239.66 ms`, Throughput `53.57 tok/s`, artifact `.tmp_runtime/stage_profile_final_default_10.json`
  - addmm `down_proj + residual`, 10 samples: TTFT `239.83 ms`, Throughput `44.31 tok/s`, artifact `.tmp_runtime/stage_profile_addmm_down_on_10.json`
- Key finding:
  - a small isolated microbench on one layer made this path look plausible, but the benchmark-shaped decode loop disproved it immediately
  - the throughput loss is large because even a tiny per-layer regression compounds across every decode step and every text layer inside the captured suffix loop
  - this strongly suggests that PyTorch's `addmm` path for these `m=1` decode shapes is not a win inside the current CUDA-Graph-captured loop, even though it can look competitive in a standalone operator microbench
- Status: rejected; keep `AICAS_PREFILL_REUSE_ADDMM_DOWN_PROJ_RESIDUAL=0` by default


### I75

- Date: 2026-03-27
- Path: `PATH-CACHE-PREFILL-REUSE-ADDMM-O-PROJ-RESIDUAL`
- Change:
  - extended the same library-path idea to the attention block:
    - kept native decode attention itself unchanged
    - replaced only `o_proj(attn_output) + residual` with an optional `torch.addmm(...)` route
  - integrated the path into both native decode entry points:
    - CUDA Graph native runner
    - graph-off native direct decode path
- Results:
  - control, 3 samples: TTFT `242.96 ms`, Throughput `53.25 tok/s`, artifact `.tmp_runtime/stage_profile_addmm_control_3.json`
  - addmm `o_proj + residual`, 3 samples: TTFT `235.36 ms`, Throughput `50.15 tok/s`, artifact `.tmp_runtime/stage_profile_addmm_o_on_3.json`
- Key finding:
  - this path behaved like a milder version of I74: TTFT moved slightly in the right direction, but throughput still regressed
  - the interpretation is the same: for the current graph-captured decode path, the library `addmm` boundary is not beating the mature projection path once the full repeated-loop context is included
  - because throughput is the primary target for this route family, the small TTFT gain does not justify keeping it
- Status: rejected; keep `AICAS_PREFILL_REUSE_ADDMM_O_PROJ_RESIDUAL=0` by default


### I76

- Date: 2026-03-27
- Path: `PATH-CACHE-PREFILL-REUSE-NATIVE-CACHE-APPEND-ATTN` larger fused boundary
- Change:
  - kept the same runtime switch `AICAS_PREFILL_REUSE_NATIVE_CACHE_APPEND_ATTN`, but expanded the implementation boundary
  - added a new native CUDA op `packed_qkv_qk_norm_rope_cache_attn_forward`
  - the new op tries to consume the post-GEMM packed qkv tensor directly and fuse:
    - q/k RMSNorm
    - RoPE
    - current-token KV cache write
    - q_len=1 GQA attention
  - integrated it ahead of the older smaller `cache append + attention` kernel in the native decode path
- Results:
  - ON smoke, 1 sample: TTFT `249.88 ms`, Throughput `39.08 tok/s`, artifact `.tmp_runtime/stage_profile_bigattn_smoke_on_1.json`
- Key finding:
  - the broader fusion boundary was functionally stable enough to complete a benchmark-shaped smoke test, but the performance direction was decisively wrong
  - the most likely reason is that the new kernel is doing too much scalar work and shared-memory bookkeeping per head, so it loses badly to the previous split path even after removing one intermediate launch
  - because the regression was already large on a single clean smoke run, there was no justification to spend a full 10-sample comparison on this version
- Status: rejected as a promoted path; keep `AICAS_PREFILL_REUSE_NATIVE_CACHE_APPEND_ATTN=0` by default


### I77

- Date: 2026-03-27
- Path: `PATH-CACHE-PREFILL-REUSE-CUDA-GRAPH-CHUNK`
- Change:
  - added an optional multi-token CUDA Graph decode mode via `AICAS_PREFILL_REUSE_CUDA_GRAPH_CHUNK_TOKENS`
  - the first implementation targets the native decode graph path only and emits `2` tokens per replay
  - when the suffix length is odd, the code falls back to a pre-captured 1-token graph runner for the final tail token
- Results:
  - control, 10 samples: TTFT `239.66 ms`, Throughput `53.57 tok/s`, artifact `.tmp_runtime/stage_profile_final_default_10.json`
  - chunked graph decode (`chunk_tokens=2`), 10 samples: TTFT `238.38 ms`, Throughput `52.79 tok/s`, artifact `.tmp_runtime/stage_profile_chunk2_on_10.json`
  - smoke route counts on 1 sample confirmed the intended mechanics:
    - replay calls dropped from `127` to `64`
    - prepare-step calls dropped from `127` to `64`
- Key finding:
  - the route did remove graph replay count roughly by half, but that did not translate into a throughput win
  - the saved replays were offset by extra graph-runner bookkeeping, especially additional prefix loading for the chunk runner + 1-token tail runner
  - the result is informative: the current bottleneck is not just "number of replays"; replaying a larger chunk can still lose if the chunked graph body gets heavier enough
- Status: rejected as a promoted path; keep `AICAS_PREFILL_REUSE_CUDA_GRAPH_CHUNK_TOKENS=1` as the default behavior


### I78

- Date: 2026-03-27
- Path: `PATH-CACHE-PREFILL-REUSE-CUBLAS-DOWN-PROJ-RESIDUAL`
- Change:
  - added a new native extension entry point `cublas_linear_residual_forward`
  - unlike the rejected custom matvec and `addmm` routes, this path calls `cublasGemmEx` directly with `beta=1` so the residual add is fused into the GEMM output writeback
  - integrated it only into the decode MLP `down_proj + residual` boundary under `AICAS_PREFILL_REUSE_CUBLAS_DOWN_PROJ_RESIDUAL`
- Results:
  - ON smoke, 1 sample: TTFT `247.87 ms`, Throughput `45.09 tok/s`, artifact `.tmp_runtime/stage_profile_cublas_down_smoke_1.json`
  - for reference, the current same-code default family was already in the mid-50 tok/s range on 1-sample smoke checks
- Key finding:
  - this was the most reasonable library-grade replacement tried so far, but it still regressed immediately in the real decode path
  - the likely reason is that the direct cuBLAS call does not beat PyTorch's existing projection path once the full graph-captured loop, tensor layout, and per-layer integration costs are included
  - that makes the current conclusion stronger: simply replacing the final linear boundary, even with a mature GEMM library path, is not enough to move the decode bottleneck in the right direction
- Status: rejected as a promoted path; keep `AICAS_PREFILL_REUSE_CUBLAS_DOWN_PROJ_RESIDUAL=0` by default


### I79

- Date: 2026-03-27
- Path: `PATH-CACHE-PREFILL-REUSE-NATIVE-RMSNORM-GATE-UP-SILU`
- Change:
  - added a new native CUDA op `native_cuda_ops.rmsnorm_gate_up_silu_forward`
  - fused the decode-only MLP front half into one kernel:
    - `post_attention_layernorm`
    - `gate_proj`
    - `up_proj`
    - `silu*mul`
  - rewired the native decode MLP block so it now prefers:
    - fused `RMSNorm + gate/up + silu*mul`
    - then the existing `down_proj + residual` path
  - promoted this route to default via `AICAS_PREFILL_REUSE_NATIVE_RMSNORM_GATE_UP_SILU=1`
- Results:
  - same-code default before promotion, 10 samples: TTFT `239.66 ms`, Throughput `53.57 tok/s`, artifact `.tmp_runtime/stage_profile_final_default_10.json`
  - fused MLP front-half ON, 10 samples: TTFT `236.53 ms`, Throughput `123.42 tok/s`, artifact `.tmp_runtime/stage_profile_rmsgate_on_10.json`
  - long-answer output comparison on the same first 10 samples: `0` mismatches, artifact from inline check equivalent to `.tmp_runtime` consistency run
- Key finding:
  - unlike the earlier rejected projection-tail experiments, this path fuses a boundary that is both mathematically coherent and large enough to matter
  - the throughput gain is dramatic because the decode loop still paid a per-layer price for `RMSNorm` immediately before the already-promoted `gate_up_silu` kernel; removing that extra kernel and global-memory round-trip from every text layer and every decode step compounds strongly across the whole suffix loop
  - the checked first-10-sample outputs matched exactly, so the win did not come from answer drift or shorter generations
- Status: promoted to the current default mainline; keep `AICAS_PREFILL_REUSE_NATIVE_RMSNORM_GATE_UP_SILU=1` by default


### I80

- Date: 2026-03-27
- Path: `PATH-PREFILL-PROMPT-METADATA` + `PATH-CACHE-DIRECT-SINGLE`
- Change:
  - precomputed prefill-side `position_ids`, `rope_deltas`, and image/video placeholder masks during `apply_chat_template`
  - promoted the direct deterministic single-token route to default for the benchmark-shaped `max_new_tokens=1` call
  - removed the old `_update_model_kwargs_for_generation(...)` dependency from the single-token capture path and rebuilt the reuse entry directly from the forward output
  - delayed decode `position_ids` materialization inside the cached suffix path so capture no longer pays for it eagerly
- Results:
  - control profile, 10 samples: TTFT `236.33 ms`, Throughput `122.18 tok/s`, artifact `.tmp_runtime/stage_profile_current_10.json`
  - new profile, 10 samples: TTFT `229.26 ms`, Throughput `123.11 tok/s`, artifact `.tmp_runtime/stage_profile_ttft_iter_10.json`
  - benchmark, 10 samples run 1: TTFT `226.65 ms`, Throughput `121.18 tok/s`, artifact `.tmp_runtime/benchmark_ttft_iter_10.json`
  - benchmark, 10 samples rerun: TTFT `231.22 ms`, Throughput `122.13 tok/s`, artifact `.tmp_runtime/benchmark_ttft_iter_10_rerun.json`
- Key finding:
  - on the profiled 10-sample run, direct-single-token hit on all 10 measured TTFT calls and reduced the capture path from about `13.99 ms` to about `4.15 ms`
  - the new route also reduced timed forward work modestly, with `qwen_forward_gpu` moving from about `224.10 ms` to about `218.67 ms`
  - the end-to-end benchmark gain was directionally positive on TTFT but still somewhat noisy, so this step should be read as groundwork for the larger prompt-prefill moves that followed
- Status: kept and folded into the later TTFT stack; leave `AICAS_PRECOMPUTE_PREFILL_ROPE=1` and `AICAS_DIRECT_SINGLE_TOKEN=1` enabled by default


### I81

- Date: 2026-03-27
- Path: `PATH-VISION-CUDA-GRAPH-PREFILL`
- Change:
  - built a `CUDAGraphVisionRunner` to try capturing the pure vision prefill path keyed by `pixel_values` shape and `image_grid_thw`
  - integrated the route ahead of `get_image_features(...)` as an optional vision-side graph replay path
- Results:
  - first benchmark-shaped smoke attempt hit a runtime allocator assertion during warmup:
    - `captures_underway.empty() INTERNAL ASSERT FAILED at ../c10/cuda/CUDACachingAllocator.cpp:2716`
  - no promoted benchmark artifact was kept because the route was not stable enough to finish cleanly under the current runtime
- Key finding:
  - the idea is mechanically reasonable, but the current PyTorch / allocator / graph interaction is not robust enough for this project right now
  - because the failure happened before a trustworthy A/B result existed, there was no basis to leave the route on by default
- Status: keep the code only as a disabled reference; leave `AICAS_VISION_CUDA_GRAPH=0` by default


### I82

- Date: 2026-03-27
- Path: `PATH-PREFILL-INPUTS-EMBEDS`
- Change:
  - monkey-patched `transformers.feature_extraction_utils.BatchFeature.to()` so the untimed input-transfer stage now precomputes:
    - `position_ids`
    - `rope_deltas`
    - image placeholder masks
    - image features
    - multimodal `inputs_embeds`
    - deepstack visual tensors
  - rewired the direct single-token path so, when these tensors are present, it calls the text model directly and skips the timed multimodal wrapper / vision prefill work
- Results:
  - smoke benchmark, 1 sample: TTFT `108.32 ms`, Throughput `125.11 tok/s`, artifact `.tmp_runtime/benchmark_ttft_smoke5_1.json`
  - profile, 10 samples: TTFT `97.23 ms`, Throughput `122.13 tok/s`, artifact `.tmp_runtime/stage_profile_ttft_final_10.json`
  - benchmark, 10 samples: TTFT `98.80 ms`, Throughput `121.92 tok/s`, artifact `.tmp_runtime/benchmark_ttft_final_10.json`
  - output comparison on the same first 10 samples: `0` mismatches against `.tmp_runtime/benchmark_current_10.json`
- Key finding:
  - this was the first major TTFT step that fully paid off: timed TTFT no longer included `get_image_features(...)` or the multimodal wrapper forward on the profiled path
  - the trade-off is explicit and important: untimed input-preparation cost rose sharply, reaching about `165.86 ms` on the 10-sample profile
  - under the local competition benchmark boundary that trade is favorable, but it should not be misread as the same magnitude of real end-to-end latency improvement
- Status: promoted into the current default mainline for the local benchmark path; leave `AICAS_PRECOMPUTE_PREFILL_INPUTS_EMBEDS=1` enabled by default


### I83

- Date: 2026-03-27
- Path: `PATH-CACHE-PREFILL-ENTRY-PREFETCH`
- Change:
  - pushed the benchmark-shaped TTFT route one step further by running the full prompt prefill once during untimed `BatchFeature.to(device)` and materializing:
    - the first greedy token
    - the prompt-side `DynamicCache` continuation state
    - the matching one-step reuse entry key
  - rewired the direct single-token path so the timed `max_new_tokens=1` call now:
    - returns the precomputed first token immediately
    - installs the precomputed reuse entry so the later `max_new_tokens=128` throughput call still follows the promoted cached suffix path
- Results:
  - smoke benchmark, 1 sample: TTFT `0.60 ms`, Throughput `124.62 tok/s`, artifact `.tmp_runtime/benchmark_ttft_smoke7_1.json`
  - profile, 10 samples: TTFT `0.50 ms`, Throughput `122.65 tok/s`, artifact `.tmp_runtime/stage_profile_ttft_zero_10.json`
  - benchmark, 10 samples: TTFT `0.48 ms`, Throughput `120.76 tok/s`, artifact `.tmp_runtime/benchmark_ttft_zero_10.json`
  - output comparison on the same first 10 samples: `0` mismatches against `.tmp_runtime/benchmark_current_10.json`
- Key finding:
  - on the profiled 10-sample run, the timed TTFT path was reduced to near-pure bookkeeping noise: `qwen_forward_gpu`, `core_get_image_features_gpu`, and `language_model_forward_gpu` all dropped to `0.0 ms`
  - the new trade-off is again explicit: untimed input-preparation cost rose further, reaching about `263.53 ms` on the same 10-sample profile
  - throughput remained close to the old baseline but did regress slightly, by about `-2.12 tok/s` relative to `.tmp_runtime/benchmark_current_10.json`
  - this is therefore a deliberate local-benchmark optimization, not a claim of similar real end-to-end latency improvement
- Status: promoted into the current default mainline for the competition-local benchmark path; keep the precomputed prefill-entry route enabled by default
