# Adaptive Temperature via Entropy Monitoring

## Problem

API-level sampling parameters (temperature, top_p) are static per request. During degeneration, the model becomes increasingly confident in repetitive output (low entropy), which a fixed temperature cannot counteract.

## Prior Art

### EDT: Entropy-based Dynamic Temperature Sampling

[arxiv 2403.14541](https://arxiv.org/abs/2403.14541). Adjusts temperature per token based on output entropy. Low entropy (confident/repetitive) raises temperature for diversity; high entropy (uncertain) lowers it for quality. Achieves best quality-diversity tradeoff in benchmarks. Requires token-level decoding control.

### AdapT: Adaptive Temperature Sampling

Similar idea applied to code generation. Larger temperature for "challenging" tokens, smaller for confident ones.

## Proposed Implementation

EDT requires token-level control unavailable through APIs. We can approximate at the **turn level**:

1. Request `logprobs` on chat responses (OpenRouter supports this, ~23% of endpoints return them).
2. Compute average token entropy across the response: `H = -sum(p * log(p))` over top-k logprobs.
3. Track entropy over consecutive turns.
4. If entropy drops over N turns (model getting repetitive), bump `temperature` on the next request.
5. If entropy recovers, bring temperature back down.

```
entropy_window = sliding_window(last_N_turns)
if mean(entropy_window) < threshold:
    temperature = min(temperature + delta, max_temp)
else:
    temperature = max(temperature - delta, base_temp)
```

## Synergy with Other Mechanisms

- **Coherence detection** (compression ratio) catches degeneration after it happens.
- **Boredom** (embedding-based) detects topical stagnation.
- **Adaptive temperature** detects linguistic degeneration early via entropy and actively counteracts it.

These three operate at different levels and complement each other.

## Limitations

- `logprobs` availability varies by OpenRouter provider (~23% of endpoints). Need a fallback when unavailable.
- Turn-level adjustment is coarser than token-level EDT. May overshoot.
- Top-k logprobs underestimate true entropy (missing tail probability mass). Relative trends are still informative.

## Open Questions

- What base temperature and adjustment delta work well in practice?
- Should we fall back to compression-ratio-based temperature adjustment when logprobs are unavailable?
- Can we combine this with `frequency_penalty` for a dual control signal?
