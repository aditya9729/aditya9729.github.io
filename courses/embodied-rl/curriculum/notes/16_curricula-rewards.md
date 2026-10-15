# 17 · What you practice, what you reward, what you retain

### Three interventions at three different places in the loop
Success-guided sampling changes which configurations are practiced. A learned reward model changes the learning signal assigned to outcomes or progress. Continual learning changes how new training affects previously acquired behavior. These interventions can be combined, but they should not be confused or evaluated with the same single metric.

SGS, presented in A Balanced Data Diet, uses success information over configurations to guide sampling. A finite set of reset seeds makes that idea inspectable in this course. It is not replay prioritization: we choose the next environment configuration rather than the next stored transition minibatch. Histories are initially uncertain and become policy-dependent as learning progresses. A target success band is a sampling preference, not a direct estimate of learning progress.

Learned rewards introduce another estimator between observation and policy update. SARM2 estimates structured task progress/value with stage-related machinery; its released progress-like outputs should not be casually treated as calibrated immediate RL rewards. RARM uses reference-conditioned progress and confidence/high-water-style gating in its own method. Both require data and model assumptions absent from a hand-written distance reward.

### Progress scores are vulnerable to temporal mistakes
If a model predicts progress 0.7 at every frame, summing that absolute score rewards waiting forever unless the horizon or other terms intervene. Differences measure change, while discounted potential shaping uses a specific gamma-weighted difference. Repeated advance–retreat cycles can exploit poorly designed progress rewards. Confidence gating and high-water logic can help with some patterns while introducing history dependence that the observation may need to expose.

ACCRUE addresses continual off-policy training of robot foundation models with mechanisms for retaining prior behavior while acquiring current tasks. A simple output-distillation exercise can illustrate retention pressure, but it is not the full actor/critic, memory, representation, and task-transition system. Retention is measured by re-evaluating prior tasks, not by observing a smaller distillation loss.

The first contribution-grade study here is deliberately narrow: uniform versus SGS-shaped reset sampling on one embodiment, one task, one fixed reward, and a matched transition budget. Evaluate on the original held-out reset distribution. A curriculum that improves success only on its preferred training resets has not established broader task competence.

### Derivation: reset weights, reward shaping, and retention
An illustrative SGS-shaped weight over estimated success p is
$$w(p)=(p+\epsilon)^{\kappa p_*}(1-p+\epsilon)^{\kappa(1-p_*)},\qquad
P(i)=\operatorname{softmax}_i(\log w_i/T).$$
The notebook implements this isolated mechanism, not the complete paper protocol. A potential-based reward term is
$$F(s,s')=\gamma\Phi(s')-\Phi(s).$$
Its discounted trajectory sum telescopes, leaving boundary terms under the required assumptions. Raw $\Phi(s')$ or an undiscounted difference is not automatically equivalent.

A retention surrogate can penalize discrepancy between current and frozen actor outputs on retained inputs. Its coefficient trades adaptation against preservation; output similarity is only a proxy for retained task success and may miss distribution shifts.

### A reward loop that pays for standing still
Suppose absolute progress stays at 0.7 for ten steps. Summing it yields reward 7 despite no progress. A raw difference gives zero after the first change. With gamma 0.99 and a constant potential 0.7, the potential term is -0.007 each step, and the full discounted sum must be interpreted with its boundary terms—not as an arbitrary penalty invented by the shaping theorem.

For the sampler, compare reset success estimates [0.05,0.5,0.95]. With a target near 0.5, middle configurations can receive more mass. If all configurations fail and histories are identical, the sampler cannot infer which failures are learnable without additional information.



## Your experiment

Use actual native success histories in the existing robot reset wrapper, then analyze synthetic histories to isolate the weighting rule. Demonstrate an absolute-progress reward loop and derive the potential telescoping identity numerically. Design a retention evaluation matrix with before/after success on every old task. Keep real model training gaps explicit.

## A failure to watch for

A learned progress estimate is not automatically an immediate reward, and a sampling change is not an algorithm improvement under every evaluation distribution.

## Explain it back

Why must evaluation resets remain separate from success-guided training resets?

<details><summary>Reasoning</summary>

Otherwise the method can look better by choosing easier or preferred configurations rather than improving behavior on the task distribution we intended to measure.

</details>

## Primary sources

- https://sgs-rl.github.io/
- https://accrue-rl.github.io/
- https://qianzhong-chen.github.io/sarm2.github.io/
- https://rarm-robotics.github.io/
