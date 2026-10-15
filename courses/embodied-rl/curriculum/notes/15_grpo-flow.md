# 16 · Grouped policy updates and flow-action policies

### Do not treat every new acronym as a new branch of RL
GRPO changes how relative outcomes provide a policy-learning signal; flow-based action policies change how actions or chunks are represented and sampled. These are different axes. Before asking whether either improves a robot, identify the data distribution, action representation, gradient estimator, and evaluation task.

GRPO's original setting groups multiple outputs for the same input context and derives relative reward information without a separately learned value baseline in that formulation. In robotics, “same context” is demanding: object pose, robot posture, contacts, hidden state, and randomness can all differ across resets. Grouping arbitrary episodes may mostly compare task difficulty instead of action quality.

Standardizing group rewards yields zero signal when every outcome is identical. Sparse tasks where every rollout fails can therefore provide no relative learning signal. Increasing group size costs more trajectories; critic-free does not mean experience-free. Group normalization also changes reward scaling and can affect how different prompt or task groups contribute to an update.

### A flow actor changes the optimization surface
A flow-matching policy generates actions by integrating a learned velocity field from noise toward an action sample. That can represent complex distributions, including multimodal demonstrations. But differentiating a Q-guided objective through generated actions raises optimization and stability questions. A stronger critic gradient is not automatically a better policy update.

QF3 studies a particular way to filter Q guidance in a flow-policy setting while retaining a flow-matching anchor. The earlier appendix has an isolated clipped-gradient exercise. The main course does not disguise a Gaussian SAC actor as a flow actor. A faithful QF3 experiment would require the actual flow parameterization, integration scheme, training data, objective, and critic recipe.

Use a component table when reading new methods: policy representation, value estimator, data source, update rule, regularizers, and systems implementation. That table often reveals which earlier chapter you need to revisit. It also prevents an experiment from changing five mechanisms and attributing the result to one named method.

### Derivation: group normalization is only one component
For returns $R_1,\dots,R_G$ from one declared context, a common grouped signal is
$$\hat A_i=\frac{R_i-\bar R}{\sqrt{G^{-1}\sum_j(R_j-\bar R)^2}+\epsilon}.$$
This normalization alone is not a complete GRPO implementation. The policy ratio, log-probability accounting, KL/reference choices, and trajectory/step objective must also be specified. Continuous actions need valid action densities and consistent stored actions.

For a flow sketch, $x_t=(1-t)z+ta$, with target velocity $a-z$. Flow matching fits a vector field to such targets. A Q-guided term introduces an additional gradient through the predicted action construction. Clipping a guidance component can suppress that path outside a declared interval, but does not fix a wrong critic.

### Grouping can remove the wrong thing
Returns [0,0,1,1] have mean 0.5 and population standard deviation 0.5, giving approximately [-1,-1,1,1]. Returns [0,0,0,0] give zeros. Now imagine the first two episodes were impossible resets and the last two easy resets. The same numbers no longer isolate policy quality at a shared context.

In the exercise, first calculate the arithmetic, then document what makes a robotic reset comparable. This is the distinction between implementing a formula and implementing a valid learning experiment.


### Sampling better actions versus changing the policy
There are two distinct places to spend computation. At training time, optimize policy parameters so useful actions are more likely later. At decision time, draw several candidates from a fixed policy and select or reweight them. They can share a reward or critic while remaining different algorithms. A stronger candidate selector does not by itself show that the underlying policy learned.

For a prior policy p and a candidate distribution q, a KL-regularized improvement problem has the form
$$\max_q\mathbb E_q[Q(a)]-\beta D_{KL}(q\Vert p),
\qquad q^*(a)\propto p(a)\exp(Q(a)/\beta).$$
This differs from the uniform-reference finite-action entropy example in SAC: the original policy now appears as a nonuniform prior. If candidate actions are sampled from p, normalized exponential Q weights approximate this tilt. The approximation depends on proposal coverage and critic quality. No reweighting can discover a successful mode that never appears among candidates.

The effective sample size of normalized weights is $1/\sum_i w_i^2$. When one candidate has almost all the mass, ESS approaches 1 even if many candidates were generated. Lower temperature can therefore increase preference for high-Q candidates while decreasing effective coverage. Use log-sum-exp stabilization before exponentiating; otherwise large values can overflow.

For the robot experiment, sample bounded commands or short action chunks from one frozen base policy, score them with a fixed critic, and compare one sample against best-of-N and weighted selection. Account for inference latency, executed transitions, collision constraints and autonomous success. A scalar Q can overvalue an out-of-distribution candidate, so best-of-N can amplify critic errors as well as find useful actions.

Feng's Beyond RL article motivates this sampling perspective and discusses sequential sampling in language reasoning. Robotics adds dynamics, partial observation, contact and execution costs. The notebook's finite candidate-weight exercise demonstrates the probability arithmetic; it is not a validated robot TSMC implementation. A simulator or world model would also need reliable branching and rollout scoring before a trajectory-particle planner could be assessed.


## Your experiment

Implement robust group normalization and an exact differentiable bandit policy-ratio update. Test equal-reward groups and negative rewards. Inspect the flow-gradient clipping appendix, identify which paths are detached, and write a faithful-implementation checklist for QF3. No validated robotic GRPO or full QF3 reproduction is claimed.

## A failure to watch for

A critic-free baseline can save model computation while requiring many costly robot trajectories.

## Explain it back

Why is grouping arbitrary robot episodes an unreliable analogue of grouping responses to one prompt?

<details><summary>Reasoning</summary>

Different resets and hidden physical states change task difficulty and outcomes. The relative signal may compare contexts instead of decisions within a context.

</details>

## Primary sources

- https://arxiv.org/abs/2402.03300
- https://qf3-rl.github.io/
- https://shengyu-feng.github.io/blog/2026/08/27/beyond-RL/
