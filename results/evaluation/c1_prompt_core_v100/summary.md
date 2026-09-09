# Prompt-specific fixed-subspace residual core diagnostic

12 fixed LongBench prompts, all36 layers;32 fit Q and16 disjoint diagnostic Q per prompt.
Frozen full-K C1-V96 FP16 V100 memory-efficient SDPA prefill. All routing diagnostic arms FP32; FP64 core solve.
Page32/B2048, pinned page0, non-sink Page-Fisher. Not generation accuracy.

```json
{
  "identity": {
    "train_loss": 0.560271431089813,
    "diagnostic_loss": 0.5559454365183727,
    "mass_recall": 0.9533094492597027,
    "non_sink_mass_recall": 0.9370972346841882,
    "page_overlap": 0.8508382726598669
  },
  "diagonal": {
    "train_loss": 0.4877598704660822,
    "diagnostic_loss": 0.5521125130015184,
    "mass_recall": 0.9532749421109825,
    "non_sink_mass_recall": 0.9370046657479381,
    "page_overlap": 0.8505805686668113
  },
  "full": {
    "train_loss": 0.3324518734662327,
    "diagnostic_loss": 0.684017693572235,
    "mass_recall": 0.9516227166138749,
    "non_sink_mass_recall": 0.9345884368573294,
    "page_overlap": 0.8369479709201388
  }
}
```
