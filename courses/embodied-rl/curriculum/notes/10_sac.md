# 11 · SAC: optimize reward and uncertainty together

### Exploration becomes part of the objective
Adding random noise to a deterministic action is one way to explore. SAC instead learns a stochastic policy under an entropy-regularized objective. It values reward and, according to a temperature coefficient, the spread of its action distribution. This is not a claim that more randomness is always better. Entropy is useful only relative to a task, action representation, and learning stage.

The critic predicts a soft return that includes the future entropy contribution. The actor trades off a Q estimate against its own log probability. Two critics and a slowly moving target critic help stabilize estimation. Unlike TD3, the common modern SAC formulation does not require a separate target actor: next actions are sampled from the current stochastic actor inside a detached target calculation.

### Two different gradient estimators in the same course
Earlier REINFORCE treated sampled actions as fixed and differentiated their log probabilities. SAC can reparameterize a Gaussian: sample noise independent of the parameters, then express the action as a differentiable function of mean and scale. This lets the actor receive a pathwise derivative through Q with respect to its sampled action. It still does not differentiate the simulator.

Robot commands are bounded, so the Gaussian sample is transformed by tanh and then scaled to native action limits. The log-density correction is essential: squeezing probability mass through tanh changes density. Summing the correction over joints gives the joint log probability. Near saturation, numerical cancellation is a practical concern, which is why the code uses a stable log-Jacobian expression.

Temperature can be fixed or learned. Automatic tuning uses a target entropy convention. Differential entropy can be negative, and changing action units changes its value. A target such as minus action dimension is a common heuristic in normalized coordinates, not a physical law or a guaranteed best exploration level for every embodiment.

Watch entropy, alpha, action saturation, critic values, and actual evaluation behavior together. A high-entropy policy may waste actions; a low-entropy policy may collapse prematurely. FlashSAC changes systems and optimization design at scale; calling this small implementation FlashSAC would conceal the hard parts the paper contributes.

### Derivation: soft target, actor, and temperature
The soft objective augments discounted reward with $\alpha\mathcal H(\pi(\cdot\mid s))$. In the twin-Q formulation,
$$y=r+\gamma(1-d)[\min_i Q_{\bar w_i}(s',a')-\alpha\log\pi_\theta(a'\mid s')],\quad a'\sim\pi_\theta.$$
Minimize the detached-target critic regression. For the actor, reparameterize $a=\tanh(\mu_\theta(s)+\sigma_\theta(s)\odot\epsilon)$ and minimize
$$L_\pi=\mathbb E[\alpha\log\pi_\theta(a\mid s)-\min_iQ_{w_i}(s,a)].$$
Hold critic parameters fixed while retaining derivatives through their action input. With $\alpha=\exp\beta$, the code uses the usual log-temperature surrogate
$$L_\beta=-\mathbb E[\beta\operatorname{stopgrad}(\log\pi_\theta(a\mid s)+\mathcal H_{target})].$$
State the sign convention: target entropy is the desired value of $-\mathbb E\log\pi$ and may be negative for a continuous density.

### The entropy term changes the target
Take reward 1, gamma 0.99, target Q minimum 2, alpha 0.2, and next-action log density -1.5. The soft continuation is $2-0.2(-1.5)=2.3$, giving target 3.277. Removing entropy gives 2.98. At true termination, both targets are exactly 1.

Now increase alpha in the sketch and observe the actor's preference across a narrow and broad action distribution. This is a conceptual objective comparison, not an animation of online training.


### The question SAC actually answers
Imagine the Franka wrist is just left of its goal. Three normalized commands have estimated long-term values 0, 1 and 2. Should the policy always choose the third? If those estimates were exact and exploration had no value, yes. During learning, the estimates are uncertain and the policy still needs useful experience. SAC formalizes a particular compromise: maximize expected return plus an entropy bonus. Entropy describes the policy's spread; it is not an uncertainty estimate of the critic, and it is not a collision-avoidance mechanism.

First solve a smaller problem exactly. Fix one state, treat its Q values as constants, and optimize a categorical distribution over candidate actions:
$$\max_{p}\sum_a p_a Q_a-\alpha\sum_a p_a\log p_a,
\quad\sum_a p_a=1,\quad\alpha>0.$$
Introduce a multiplier for the sum constraint. Differentiating with respect to each probability gives $Q_a-\alpha(1+\log p_a)+\lambda=0$. Normalize the exponentials to obtain
$$p_a^*=\frac{\exp(Q_a/\alpha)}{\sum_b\exp(Q_b/\alpha)}.$$
At alpha 1 and Q values [0,1,2], the probabilities are approximately [0.090,0.245,0.665]. At alpha 0.2 they are [0.000045,0.006693,0.993262]. The interactive sketch above performs this exact finite-action calculation. A robot SAC actor is usually a squashed Gaussian fitted through gradients; it cannot represent every possible continuous soft-optimal distribution. In particular, a single Gaussian may poorly represent two distinct ways around an obstacle.

### From a soft value to the critic target
For a fixed stochastic policy, define its soft state value as
$$V^\pi(s)=\mathbb E_{a\sim\pi}[Q^\pi(s,a)-\alpha\log\pi(a\mid s)].$$
The action value is immediate reward plus discounted next soft value. Sample one next action instead of integrating every possible command. The result is the target in the previous derivation. It uses the next state and a newly sampled next action, not the stored action from the current transition. Modern SAC estimates that continuation with the smaller of two target critics. Taking a minimum reduces a useful source of positive estimation bias; it can also introduce underestimation, and does not certify either critic as correct.

The replay record is `(observation, action, reward, next_observation, terminal)`. A batch with B=64 and the 7D Franka action interface has observations [64,21], actions [64,7], rewards [64], and each critic output [64]. AgileX uses 12D actions and the fixed-base R1 Pro uses 18D. Keep scalar outputs squeezed consistently: subtracting [64,1] from [64] broadcasts into [64,64], yielding a finite but completely wrong loss.

Two online Q networks consume the concatenated observation and action. Two target Q networks have the same architecture but are updated by averaging parameters. The actor maps an observation to a mean vector and a log-standard-deviation vector. A separate scalar log alpha has its own optimizer. There is no target actor in this implementation.

### Why sampling must preserve the actor gradient
Write a Gaussian sample as $u=\mu_\theta(s)+\sigma_\theta(s)\odot\epsilon$, with standard-normal noise independent of theta, then set $a=\tanh u$. For the Q part of the actor loss, the chain rule contains
$$-\frac{\partial Q_w(s,a)}{\partial a}
\frac{\partial\tanh u}{\partial u}
\frac{\partial u}{\partial\theta}.$$
That path asks which nearby command the critic predicts will be better. Freezing the critic weights preserves this action derivative; evaluating Q under `no_grad()` removes it. These operations are not interchangeable. A detached action sample also severs the path. PyTorch's `rsample()` or the explicit noise construction preserves it. The simulator is absent from this computation graph: its experience trained the critic earlier.

The log-probability term also carries a total derivative through the reparameterized sample. This differs from REINFORCE, where the action is held fixed while differentiating its log probability. Both can be valid estimators for their objectives; mixing their code patterns silently changes the estimator.

### The tanh correction is a change of variables
The policy density must describe the action actually given to the normalized environment. Inverting tanh yields the change-of-variables formula
$$\log\pi(a\mid s)=\sum_j\left[\log\mathcal N(u_j;\mu_j,\sigma_j^2)-\log(1-\tanh^2 u_j)\right].$$
There is one density per transition after summing across joints. Near the action limits, compute the log-Jacobian with the stable identity $\log(1-\tanh^2u)=2(\log2-u-\operatorname{softplus}(-2u))$. Directly subtracting a saturated tanh square from 1 can produce log zero. Multiplying normalized actions by a constant physical scale adds a constant density correction; the course learns and tunes temperature in normalized coordinates so the convention stays consistent.

### Follow one update in the code
1. Sample a batch uniformly from replay. Compute alpha from log alpha and detach it for the actor and critic updates.
2. Inside `torch.no_grad()`, sample next actions with the current actor. Evaluate both target critics. Construct the soft Bellman target and mask continuation only at true termination. A time-limit truncation still bootstraps from the final pre-reset observation.
3. Evaluate both online critics on the stored actions. Backpropagate their squared errors against the fixed target and step the critic optimizer. A target must not move merely to reduce its own regression error.
4. Freeze online critic parameters. Sample new current-state actions with reparameterization. Minimize alpha times log density minus the smaller online Q. Step only the actor optimizer; then restore critic parameter gradients.
5. Detach the sampled log density when updating log alpha. This optimizer controls temperature, not the actor. With desired entropy h, its log-temperature surrogate has derivative $\widehat H-h$. If observed entropy is below h, gradient descent increases log alpha and gives entropy a larger weight. The usual h=-action dimension is a heuristic for a continuous normalized density; negative differential entropy is possible.
6. Slowly move target critic weights toward online weights. With tau 0.005, one update moves a target 0.5% of the gap. These are parameter averages, not an extra fitted value network.

### A numerical minibatch you can audit
For one nonterminal sample, reward 1, gamma 0.99, minimum next Q 2, next log density -1.5 and alpha 0.2 produce y=3.277. If current Q predictions are 2.5 and 3.0, the sum of squared errors is $0.777^2+0.277^2=0.680458$. Their scalar derivatives are -1.554 and -0.554. Gradient descent therefore pushes both predictions upward on this sample. Other batch samples and shared network parameters can oppose that direction.

For an actor sample with current log density -0.7 and minimum Q 2.4, its scalar loss contribution is -2.54. The negative number is not itself a failure or a success score. If a small action change raises Q by 0.1 while leaving log density fixed, the loss falls by 0.1. If it raises log density by 0.5 while leaving Q fixed, the loss rises by 0.1. The actual update combines both derivatives across a batch.

### Diagnose before increasing the training budget
Track independent evaluation returns, held-out success, Q magnitudes, alpha, action saturation and each loss. Check that target critic parameters have no gradient, while actor parameters receive gradients through Q. Remove the tanh correction as a controlled failure case and inspect how action saturation and temperature change. Change reward scale without adjusting temperature and predict the balance shift first. A critic trained on replayed actions can be wrong on actor-selected actions; replay reuse is sample efficient but does not remove distribution shift.

Finally run the SB3 SAC baseline on the same environment. Match transitions, reset seeds, network size, replay warmup, update-to-data ratio and evaluation definition. Different defaults mean two curves are a framework comparison, not an isolated test of your update equation. The course runner records explicit configuration and produces a real rollout; short smoke budgets verify execution rather than convergence.


## Your experiment

Verify the tanh density correction and finite gradients near saturation. Execute separate critic, actor, and alpha updates on real replay data. Confirm target gradients are absent while actor gradients through Q remain present. Compare fixed and learned alpha only after the base implementation passes these checks.

## A failure to watch for

For SAC, an action sample needs rsample-style reparameterization during the pathwise actor update.

## Explain it back

Why is the Gaussian log probability before tanh not the final action log probability?

<details><summary>Reasoning</summary>

A nonlinear transformation changes density through its Jacobian determinant. Ignoring that term changes the entropy objective and the actor gradient.

</details>

## Primary sources

- https://arxiv.org/abs/1812.05905
- https://spinningup.openai.com/en/latest/algorithms/sac.html
