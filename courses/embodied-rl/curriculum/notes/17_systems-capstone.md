# 18 · From readable RL to a defensible research experiment

### Sample efficiency and wall-clock efficiency are different axes
A method can use fewer environment transitions yet take longer because each update is expensive. Another can learn faster in wall time by increasing simulation throughput while consuming more data. On real robots, resets, human assistance, and control latency add costs that a simulated transitions-per-second chart does not capture.

FastTD3 and FlashSAC study specific scalable learning recipes, including algorithmic and systems choices. Running ordinary TD3 or SAC on a Colab GPU does not reproduce them. Small state networks often gain little from a GPU when a CPU simulator supplies one transition at a time. Accelerator benefits are more plausible with larger batches, many environments, image encoders, or larger policies; measure the actual bottleneck.

EXPO-FT adds a specific form of off-policy adaptation around pretrained action generation. Real-Time EXPO-FT additionally addresses asynchronous timing and alignment. A slow model may generate a chunk based on an older observation while a faster process edits or selects actions using newer information. Queue prefixes, already executed actions, and timestamps become part of the learning and control design. A stale but high-scoring action can still be wrong for the current state.

### Make the capstone answer one question
Choose a single task and embodiment, then compare one mechanism at a time. For example: does success-guided reset sampling improve held-out dual-wrist reaching success on AgileX at the same transition budget? Or: does torso control reduce unreachable-goal failures on R1 Pro compared with arm-only control? Do not simultaneously change the reward model, policy architecture, reset distribution, observation modality, and optimizer and then call the result a PPO-versus-SAC comparison.

Define training seeds, evaluation seeds, budgets, assistance accounting, stopping rules, and metrics before running. Save versions, configuration, raw logs, checkpoints, and failure videos. Separate an integration smoke test from a converged policy and a converged policy from a statistically supported comparison. A policy's last checkpoint is not automatically its best checkpoint; selecting one on evaluation data can bias the reported result.

Use a hierarchy of compute budgets: seconds for algebra and gradient checks, minutes for update plumbing, longer controlled runs for learning curves, then repeated seeds for claims. Mac CPU covers the conceptual and small state-policy path. Colab CUDA is an optional execution path, not a guaranteed GPU allocation or proof of faster training. The complete course remains useful even when no GPU is assigned.

### Derivation: count what a training budget actually contains
If N environments each collect H steps, a rollout contains $NH$ transitions. With E epochs and minibatch size B, PPO performs approximately $E\lceil NH/B\rceil$ minibatch steps. For replay methods define
$$UTD=\frac{\text{critic optimizer steps}}{\text{new environment transitions}}.$$
Actor and temperature steps can have separate ratios. Report all relevant conventions.

Wall time is not simply a sum when pipelines overlap. In a sequential loop it includes environment, inference, update, reset, rendering, and logging time. In an asynchronous pipeline, throughput is limited by the bottleneck and synchronization overhead. Measure per-component timings and end-to-end time instead of extrapolating a kernel speedup into a learning speedup.

### The same rollout, very different optimization work
Eight environments times 256 steps give 2048 transitions. Four epochs with minibatches of 256 give 32 minibatch optimizer steps. Changing to ten epochs yields 80 optimizer steps on the same transitions. In SAC, collecting 2048 transitions with UTD 1 gives 2048 critic steps. These raw counts do not prove either method is better because batch size, network size, and gradient structure also differ.

The systems sketch lets you change environment and update time. It reports a toy sequential throughput calculation explicitly as a calculation, not a measured FastTD3 or FlashSAC result.



## Your experiment

Write a complete pre-run experiment card, execute one CPU baseline, and inspect all recorded metrics. Generate a matched-budget seed sweep command. Profile one loop without rendering and another with recording. Identify exactly which additional components a faithful FastTD3, FlashSAC, EXPO-FT, or Real-Time EXPO-FT reproduction still needs.

## A failure to watch for

A fast simulator benchmark and a successful learning experiment are different kinds of evidence.

## Explain it back

What is the smallest convincing result for your first contribution?

<details><summary>Reasoning</summary>

One clear hypothesis, a reproducible baseline, a controlled change, held-out evaluation, repeated training seeds, honest uncertainty, and a documented failure analysis.

</details>

## Primary sources

- https://younggyo.me/fast_td3/
- https://holiday-robot.github.io/FlashSAC/
- https://pd-perry.github.io/expo-ft/
- https://pd-perry.github.io/real-time-expo-ft/
- https://research.google.com/colaboratory/faq.html
