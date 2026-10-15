# 08 · PPO: the four cases behind clipping

### Why not take many ordinary gradient steps on one rollout?
After a rollout, its actions were sampled by the old policy. Changing the policy makes those samples progressively less representative. Ordinary repeated policy-gradient updates ignore that shift. PPO uses an action likelihood ratio and a clipped surrogate to limit incentives for repeatedly reinforcing the same samples.

The ratio compares the probability density of the same stored action under the current and old policies at the same stored observation. It does not compare a newly sampled action with the old one, and it is not the reward. Store the old log probability when collecting the rollout. Recomputing it with the updated policy destroys the reference and makes the ratio appear to be one.

### Clipping is asymmetric because advantage has a sign
A positive advantage asks to increase an action's probability. Beyond ratio $1+\epsilon$, the clipped surrogate stops rewarding further increases for that sample. A negative advantage asks to decrease probability. Below $1-\epsilon$, it stops rewarding further decreases. Moves in the harmful direction remain penalized. Taking the minimum of clipped and unclipped terms creates this asymmetry.

This is not a hard constraint that keeps every ratio inside the interval. Other samples, shared parameters, and the critic or entropy term can still move a probability. PPO clipping also does not guarantee monotonic improvement in a real neural-network implementation. Measure approximate KL, clip fraction, entropy, value error, and policy performance. Early stopping on excessive KL is a useful separate convention, not the same operation as clipping.

For a bounded continuous policy, use a consistent density for the stored action. Our actor samples an unconstrained Gaussian latent u, maps it through tanh, and includes the change-of-variables correction. The notebook stores u and evaluates its transformed density consistently. Simply clipping sampled Gaussian actions at the actuator bounds and then computing an unclipped Gaussian density produces a different distribution with boundary mass.

Finally, rollout length, epochs, minibatches, and parallel environments are separate controls. A large number of optimizer epochs on a tiny rollout can overfit stale advantages. More parallel environments changes data diversity and reset statistics; it does not by itself change PPO into an off-policy method. The notebook's minimal trainer uses one environment for clarity, while larger-scale reference systems are studied separately.

### Derivation: the surrogate and its cases
Let $\rho_t(\theta)=\exp[\log\pi_\theta(a_t\mid s_t)-\log\pi_{old}(a_t\mid s_t)]$. PPO-Clip maximizes
$$L^{clip}=\mathbb E_t\left[\min\left(\rho_t\hat A_t,\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t\right)\right].$$
If $\hat A>0$, the sample objective is $\hat A\min(\rho,1+\epsilon)$. If $\hat A<0$, it is $\hat A\max(\rho,1-\epsilon)$. Derive those two expressions before looking at the plot.

The actor loss is the negative mean of this objective. The value loss is a separate regression. A common approximate KL statistic is $\mathbb E[(\rho-1)-\log\rho]$ on old-policy samples; it is a diagnostic estimate, not a per-state exact KL constraint.

For $a=\tanh u$, use $\log\pi(a\mid s)=\log\mathcal N(u;\mu,\sigma)-\sum_i\log(1-\tanh^2u_i)$. Stable code computes the log Jacobian through softplus rather than subtracting nearly equal floating-point numbers near saturation.

### Calculate both signs
With epsilon 0.2, ratio 1.5, and advantage +2, the unclipped term is 3 and the clipped term is 2.4, so the objective is 2.4. With advantage -2 at the same ratio, the terms are -3 and -2.4, and the minimum is -3: the harmful increase remains penalized.

Now use ratio 0.5. Positive advantage gives 1, while negative advantage gives -1.6 after clipping. Plot all four cases. If your implementation always clips the ratio before multiplication and omits the minimum, it is wrong in two regions.


### Why PPO needs both an old policy and a current policy
Suppose an action produced a better outcome than the value baseline expected. REINFORCE says to increase its probability. If we repeatedly optimize the same rollout, however, that action was sampled by an older policy. PPO keeps its old log probability and measures the current-to-old probability ratio. The old policy supplies the sampling distribution; the current policy is the object being changed. These are different roles even when both started with identical parameters.

The importance-ratio identity can translate an expectation over old-policy actions to current-policy actions at a fixed state, provided the old policy has support. It does not correct the entire changing state-visitation distribution. PPO's clipped surrogate is therefore a local training approximation, not an exact guarantee of improved return.

### Derive the clipping rule by protecting the incentive
For positive advantage, increasing the ratio improves the sampled surrogate, but PPO stops rewarding that increase above 1+epsilon. For negative advantage, decreasing the ratio helps, but its benefit stops below 1-epsilon. The minimum of the unclipped and clipped objectives implements both rules. The minimum, rather than an absolute clamp on every gradient, is the important detail.

With epsilon 0.2 and advantage +2, ratio 1.5 gives candidates 3 and 2.4: choose 2.4 and the actor gradient through this sample is flat. At ratio 0.7, choose 1.4: the update still encourages recovery of the action probability. With advantage -2, ratio 1.5 chooses -3 and strongly penalizes making a bad action more likely. At ratio 0.7, choose -1.6 and stop rewarding further suppression. Predict each case in the sketch before touching its sign selector.

### One rollout, then several optimizer passes
The actor maps [B,observation dimension] to a distribution over actions. A value network maps the same observations to [B]. During collection, save observations, actions (or their pre-tanh samples), old log probabilities, rewards, values, final next values, terminal flags and truncation flags. These saved quantities are detached. After the rollout, compute GAE once with the collected value estimates; create fixed value targets and usually normalize the batch advantages.

For each optimization epoch, reevaluate the saved actions under the changing actor, compute log-probability differences, exponentiate, and apply the clipped objective. Do not resample actions during this ratio calculation. Their advantage labels belong to the actions actually taken. Fit the value network separately to the detached targets. The compact teaching loop uses the complete rollout per epoch; SB3 additionally provides minibatching and configurable schedules.

GAE uses a terminal-only bootstrap mask in each TD residual and an episode-boundary mask when propagating the trace. A timeout allows a final value estimate but must not pass advantage credit into the next reset episode. In a Gaussian policy, sum the action-coordinate log densities before calculating the ratio. Otherwise you optimize independent per-joint ratios instead of the probability ratio of the joint command.

### What you should see after one step
On a batch containing only a positively advantaged unsaturated action, its log probability should initially increase. The value prediction should move toward its fixed target. Neither old log probabilities nor advantage labels should receive gradients. After multiple epochs, inspect approximate KL and the fraction of samples outside the clipping interval. Clipping the sampled surrogate does not impose a hard KL limit; a high learning rate can still move the policy a long way.

For the lab, compare one, four and twelve optimization epochs at a matched environment budget. Predict why extra epochs may initially improve data reuse and later cause excessive policy drift. Use untouched evaluation resets. Report success and return, not just a falling actor loss: the surrogate's numeric value changes when advantages are normalized and is not a direct task score.


## Your experiment

Implement the clipped objective and test all four sign/direction cases. Differentiate the objective with respect to the ratio to inspect where gradients become zero. Execute a real PPO rollout/update in Reacher using the from-scratch trainer, then alter epochs or clipping one at a time. Treat the short run as an implementation check, not a convergence claim.

## A failure to watch for

PPO clipping removes an incentive; it does not impose an inviolable trust region.

## Explain it back

Why must old log probabilities remain fixed while optimizing a rollout?

<details><summary>Reasoning</summary>

They define the sampling-policy reference in the importance ratio. Updating both numerator and denominator together removes the correction and invalidates the intended surrogate.

</details>

## Primary sources

- https://arxiv.org/abs/1707.06347
- https://spinningup.openai.com/en/latest/algorithms/ppo.html
