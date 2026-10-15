# 06 · How a sampled action creates a policy gradient

### The environment does not give you a differentiable loss
In supervised learning, a label gives a direct error for a prediction. In ordinary model-free RL, MuJoCo returns a reward after an action; autograd does not automatically pass through the simulator. REINFORCE estimates how changing action probabilities changes expected return, without differentiating the environment's dynamics.

The intuition is to increase the probability of actions on better trajectories. But stating that intuition is not enough: an action may be followed by a lucky event, and long trajectories mix useful and useless decisions. A sampled return weights a stochastic gradient estimate. The update becomes reliable only by averaging appropriately and managing variance.

The log-derivative trick turns a derivative of a trajectory probability into that probability times a derivative of its log. This lets us sample trajectories from the current policy instead of enumerating them. When environment dynamics do not depend on policy parameters, their terms vanish from the log-probability derivative. They still matter through which trajectories occur; they have not disappeared from the RL problem.

### Credit assignment and causality
Reward received before an action cannot be caused by that action. Replacing a whole-trajectory return with reward-to-go removes irrelevant past rewards from its weight. A state-dependent baseline can remove another source of variance without changing the expected gradient, provided it does not introduce action dependence into that score-function term. A learned value function is a useful baseline, not an external judge that knows the correct answer.

REINFORCE is on-policy in its basic form. The trajectories must come from the policy whose log probabilities define the estimator. Reusing old data without correction changes the sampling distribution and can bias the update. Importance ratios address part of that mismatch; PPO will use them locally rather than treating arbitrary stale data as interchangeable.

Finally, a scalar loss is only a convenient device for autograd. We minimize minus log probability times a detached return or advantage. The negative sign converts ascent on return into descent on a loss. If you let the return or fitted baseline carry an unintended gradient into the actor, you are no longer implementing the estimator derived here.

### Derivation: the score-function identity
For a finite-horizon undiscounted objective, start with $J=\int p_\theta(\tau)R(\tau)d\tau$:
$$\nabla J=\mathbb E_{\tau\sim p_\theta}[R(\tau)\nabla\log p_\theta(\tau)]
=\mathbb E\left[\sum_t\nabla\log\pi_\theta(a_t\mid s_t)R(\tau)\right].$$
Causality permits reward-to-go in place of the whole return. For a discounted finite-horizon objective $G_0$, the exact trajectory expression includes $\gamma^tG_t$ in each time-t contribution. Many practical implementations use a discounted state-visitation convention or a surrogate without that outer factor; do not mix the two derivations silently.

A baseline vanishes in expectation because
$$\sum_a\pi(a\mid s)\nabla\log\pi(a\mid s)b(s)=b(s)\nabla\sum_a\pi(a\mid s)=0.$$
Thus a useful implementation minimizes $-\sum_t\log\pi_\theta(a_t\mid s_t)\operatorname{stopgrad}(G_t-b(s_t))$, with the objective/discount convention stated explicitly.

### A two-action policy you can solve exactly
Let $p=\operatorname{sigmoid}(\theta)$ choose an action worth 2; the other is worth -1. Then $J=2p-(1-p)=3p-1$ and $dJ/d\theta=3p(1-p)$. At $p=0.5$, the exact derivative is 0.75.

Sample actions and average $r(a)\nabla\log\pi(a)$. It should approach 0.75 as the batch grows. Subtract an action-independent baseline and inspect the sample variance. A poor baseline can increase variance; “baseline” is not a universal variance-reduction guarantee for every choice.


### The environment gives rewards, not a gradient
For a trajectory, differentiate its probability with the log-derivative identity. Environment transition probabilities contain no policy parameters and disappear from the derivative. What remains is a sum of action log-probability gradients weighted by return. Causality lets us omit rewards from before the action; changing an action cannot alter a reward that already happened.

The notebook uses an undiscounted episodic CartPole objective with gamma=1. Its actor maps four observations to two logits. Sample each action, retain its differentiable log probability, and collect an episode. Compute reward-to-go backward, detach these labels, and minimize negative log probability times return. The action itself is a categorical sample and is not differentiated through. Optimizer descent on the negative objective implements return ascent.

If an action has probability 0.25 and return 2, its own-logit ascent contribution is 1.5. A state-dependent baseline does not change the expected score-function gradient because the policy score has zero conditional expectation. It can greatly reduce variance. An action-dependent baseline needs additional correction; subtracting an arbitrary learned Q from every action's label is not the same argument.

Long episodes and sparse success produce noisy returns. Replacing Monte Carlo returns with a bootstrapped critic reduces variance while introducing approximation bias. This is the conceptual move into actor–critic. Compare repeated seed runs or gradient estimates; one lucky high-return episode is not a dependable improvement claim.


## Your experiment

Verify the bandit gradient analytically, with autograd, and by Monte Carlo. Then execute a short REINFORCE run in Gymnasium CartPole. Explain the sign of each update and graph returns across episodes. The short run demonstrates the estimator; extend it only after verifying the mechanics.

## A failure to watch for

An action-dependent baseline cannot be inserted into the simple proof without additional correction.

## Explain it back

Why does model-free policy gradient not require the derivative of the transition function?

<details><summary>Reasoning</summary>

The score-function estimator differentiates the action probability terms of sampled trajectories. Environment outcomes enter the return weights and sampling distribution rather than an autograd path through dynamics.

</details>

## Primary sources

- https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html
- https://www.cs.ualberta.ca/~sutton/williams-92.pdf
