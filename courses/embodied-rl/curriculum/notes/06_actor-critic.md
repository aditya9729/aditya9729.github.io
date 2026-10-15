# 07 · Actor–critic and the advantage estimate

### The critic predicts; the actor decides
REINFORCE waited for a sampled return to decide whether an action was useful. An actor–critic uses a learned prediction to reduce that burden. The actor parameterizes the behavior policy. The critic estimates value. The critic's error changes the actor's learning signal, so a bad critic can make a reasonable actor update go in the wrong direction.

Actor–critic is a family, not one on-policy algorithm. A2C and PPO use on-policy value estimates; DDPG, TD3, and SAC are off-policy actor–critics using action-value functions. In this lesson we first build a synchronous on-policy actor–critic so the advantage calculation remains visible. The concrete implementation uses a separate actor and value network, avoiding accidental shared-gradient effects.

An advantage asks whether taking an action is better than the policy's usual outcome in that state. We cannot generally compute Q minus V exactly, so we estimate it. A one-step TD residual is one estimate. A long reward-to-go minus V is another. Their errors have different time structure. GAE combines TD residuals across a range of horizons.

### Unroll the recursion before writing a loop
GAE is often introduced as a mysterious backward loop. Instead, write its first terms: current residual, gamma-lambda times the next residual, then another power for the following residual. This is a discounted sum of prediction surprises. When lambda is zero, only one residual remains. When lambda approaches one, more of the sampled future contributes. An imperfect critic and finite rollout boundaries still affect the estimator.

Two boundaries require two masks. A true terminal prevents bootstrapping from the next state. A timeout may permit that bootstrap but must stop GAE's recurrence from crossing into the new episode. If both are collapsed into one done flag in every line, at least one of those meanings is lost.

Normalize advantages only after constructing the batch. Normalization is a numerical convention that changes gradient scale, not a proof of unbiasedness for every estimator. Critic targets must be detached; actor advantages must also be detached. Watch both gradient norms, value error, and explained variance, while remembering that a well-fitted value predictor for a bad policy can coexist with poor task success.

### Derivation: from TD residuals to GAE
Define $\delta_t=r_t+\gamma(1-d_t)V(s_{t+1})-V(s_t)$. Then
$$\hat A_t^{GAE}=\sum_{l\ge0}(\gamma\lambda)^l\delta_{t+l}.$$
In a finite buffer with reset indicator $b_t=d_t\lor\mathrm{truncated}_t$, implement
$$\hat A_t=\delta_t+\gamma\lambda(1-b_t)\hat A_{t+1},\qquad \hat R_t=\hat A_t+V(s_t).$$
The actor minimizes $-\mathbb E[\log\pi_\theta(a_t\mid s_t)\operatorname{stopgrad}(\hat A_t)]$; the critic minimizes $\mathbb E[(V_w(s_t)-\operatorname{stopgrad}\hat R_t)^2]$. These two losses train different predictions and should not share unintended targets.

Expanding the two-step expression shows telescoping cancellation of intermediate value terms when lambda is one and boundaries are handled consistently. With lambda below one, those intermediate value predictions remain part of the weighted estimate.

### Three surprises, one advantage
Suppose TD residuals are [1, -0.5, 2], gamma is 0.9, and lambda is 0.8. Then $\hat A_0=1+0.72(-0.5)+0.72^2(2)=1.6768$. With lambda zero, it would be 1. With a reset after the first transition, it must also be 1 regardless of the later episode's residuals.

Use the sketch's trace-length control to see each residual's weight. The colors represent contributions to a sum, not a guarantee that every future action caused the current outcome.


### Separate the prediction from the decision
The critic estimates the return expected before choosing the sampled action. The actor needs to know whether that action did better than expected. The TD residual combines immediate reward, a next-state value prediction and the current prediction. A positive residual is evidence favoring the sampled action, subject to critic error. It is not a ground-truth advantage label.

The value network maps [B,O] to [B]. The stochastic actor maps [B,O] to a joint command distribution. Detach the advantage before weighting log probabilities: the actor loss should change the action distribution, not lower its own label by changing the critic. Fit the critic separately against detached return targets. A shared feature trunk is possible, but the course uses separate networks so these gradient routes remain visible.

With reward 1, gamma 0.9, current value 2 and next value 3, the TD residual is 1.7. If next residual is -0.5 and lambda is 0.8, a two-step trace produces advantage $1.7+0.9(0.8)(-0.5)=1.34$. Longer traces move toward Monte Carlo information; shorter traces rely more strongly on the critic. Lambda does not independently guarantee lower total error.

At a timeout, use the final next observation to bootstrap, but stop the trace before the reset episode. At true termination, remove continuation as well. Inspect these two masks separately. This lesson's AC uses one on-policy rollout and one actor pass; PPO changes the update objective and reuses the rollout for several passes. TD3 and SAC also belong to the actor–critic family, with off-policy Q critics instead of this state-value estimator.


## Your experiment

Implement the backward GAE recursion and compare it against an explicit weighted sum on a no-reset sequence. Add terminal and timeout cases with hand-calculated expected targets. Run the from-scratch AC trainer on actual Reacher transitions and verify both actor and critic parameters change.

## A failure to watch for

A critic is an estimator with its own failures; a small value loss alone is not a robot success metric.

## Explain it back

Which mask differs between the TD bootstrap and the GAE recurrence at a timeout?

<details><summary>Reasoning</summary>

Bootstrapping may continue from the final timeout state, but recurrence stops before the next reset episode. A true terminal stops both.

</details>

## Primary sources

- https://arxiv.org/abs/1506.02438
- https://arxiv.org/abs/1602.01783
