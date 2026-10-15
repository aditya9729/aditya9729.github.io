# 12 · Offline data, imitation, and residual post-training

### Reusing data is not the same as solving offline RL
An off-policy method can reuse transitions collected by another behavior policy. Offline RL removes the ability to collect corrective data during training. That changes the problem substantially: an actor can choose actions unsupported by the dataset, and a critic may confidently extrapolate to them. A replay-capable algorithm is not automatically a reliable offline algorithm.

Behavior cloning instead learns to predict demonstrated actions. Its supervised objective does not require a reward, but it can compound errors when the learned policy visits states unlike the demonstrations. A strong demonstration policy is often a useful starting point for robot learning. RL then addresses a different question: can interaction improve the behavior according to a specified reward and evaluation protocol?

A residual architecture freezes a base policy and learns bounded corrections. This can restrict exploration around a useful behavior and preserve much of its competence. It can also restrict the solution space: if the base pushes into an obstacle and the residual is too small, the learner cannot escape. Clipping may saturate the combined command, making the residual's nominal authority misleading.

Action chunks complicate composition. A base predicts several future commands, while observations continue changing. A residual may be applied per step, per chunk, or through an edit policy. Those choices define different control timing and credit assignment. A state-based one-step residual in this course is a mechanism experiment; it is not a full VLA or ResFiT implementation.

Evaluation must account for human resets, interventions, demonstrations, and reward annotations. Report autonomous success separately from success with assistance. Demonstrations are data, not free experience that disappears from a sample-efficiency comparison. A method trained with a useful initialization and one trained from scratch are answering different experimental questions unless that advantage is explicitly part of the benchmark.

### Derivation: two objectives and one composition rule
Behavior cloning minimizes
$$L_{BC}(\theta)=-\mathbb E_{(o,a)\sim\mathcal D_{demo}}\log\pi_\theta(a\mid o).$$
It does not contain a Bellman target. Residual control composes
$$a=\operatorname{clip}(\pi_{base}(o)+\beta\,\pi_{res}(o),a_{min},a_{max}).$$
Freezing the base means its parameters are not optimized. The residual environment includes the base policy in its action transformation. If the base has hidden temporal state or chunk queues, that state may also need to be part of the residual agent's observation to preserve the intended Markov property.

Offline critic extrapolation appears in a target such as $r+\gamma Q(s',\mu(s'))$ when $\mu(s')$ lies outside the data's action support. Reusing the Bellman equation does not certify that extrapolated value. Conservative or behavior-constrained methods introduce additional design choices; they are outside the small residual implementation here.

### Does the residual have enough authority?
If the base outputs +0.8, residual range is [-1,1], and beta is 0.1, the combined action can only lie in [0.7,0.9] before clipping. It cannot reverse direction. With beta 0.4 it can lie in [0.4,1.2], but the upper range saturates at 1. The useful authority is asymmetric around the base action.

Compute this interval joint by joint before attributing a failure to SAC. An impossible correction is an interface limitation, not merely poor optimization.



## Your experiment

Build a frozen-base residual action wrapper, verify zero residual reproduces the base command, and run actual SAC updates in the induced environment. Plot saturation frequency. Design a comparison with the same base, interaction budget, reset assistance, and fixed autonomous success metric.

## A failure to watch for

Off-policy online learning and offline RL differ in whether unsupported predictions can be corrected by new interaction.

## Explain it back

Why can a bounded residual preserve competence yet prevent a necessary recovery?

<details><summary>Reasoning</summary>

It constrains exploration around the base behavior. That same constraint can exclude actions required to undo a base-policy mistake.

</details>

## Primary sources

- https://arxiv.org/abs/2005.01643
- https://residual-offpolicy-rl.github.io/
