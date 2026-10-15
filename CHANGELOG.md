# Changelog

## Fork of decoderesearch/circuit-tracer @ 0968d11

Faster `attribute()` on the TransformerLens backend: about 7-9x on Gemma-2-2B, 3-5x on
Qwen3-0.6B, and prompts up to 2k tokens on a 32 GB GPU. Graphs match upstream up to
floating-point reassociation; the API is unchanged.

**How we got here:** profiled upstream, let a few Claude agents try optimizations in
parallel against a fixed benchmark and correctness check, red-teamed the result for
bugs, then removed everything an ablation couldn't justify. Tested on CUDA, CPU and MPS.

**What changed, roughly:**
- Attribution rows come from a linear adjoint (the frozen model's transposed Jacobian)
  instead of autograd over a replicated forward pass.
- Influence ranking runs incrementally on the GPU.
- Less setup memory: lazy encoder rows, a deduplicated decoder table, last-position logits.
- `offload="cpu"` is about 2-4x faster.
- Unsupported models fall back to the old path; `model.use_linear_adjoint = False` forces it.
