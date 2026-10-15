# 01 · The mathematics behind a policy

### A distribution is a way to act, not just a way to describe uncertainty
Imagine a Franka wrist above a cube. One policy always commands the same small motion. Another assigns probabilities to several motions and samples one. Neither policy knows the reward of an untried action. A stochastic policy lets us explore while retaining a mathematical description of exactly how likely each sampled action was. That description becomes the handle used by a policy-gradient update.

Before learning an algorithm, separate three objects: an outcome, its probability, and its value. If a grasp succeeds with probability 0.7 and earns 10, its expected reward is 7; no individual attempt earns 7. An expectation averages over possible outcomes. An empirical mean averages actual samples. The two become close under suitable sampling assumptions, but a small sample can be misleading. Ten successful grasps do not prove certainty.

For discrete actions, a network outputs logits. A softmax converts them into nonnegative numbers that sum to one. Adding the same constant to all logits leaves those probabilities unchanged. For continuous commands, a diagonal Gaussian uses a mean vector and one standard deviation per action coordinate. Its density can exceed one because a density is not the probability of an exact action. Probabilities come from integrals over regions. This matters when comparing log probabilities between discrete and continuous policies.

### Why logarithms appear in an action update
The probability of a trajectory multiplies many terms. Logarithms turn that product into a sum, which is numerically and algebraically easier to differentiate. A derivative asks how a small parameter change affects a quantity. A gradient collects those derivatives for all parameters. PyTorch computes gradients of a scalar loss; it does not know that the loss is an RL objective. A sign error can therefore train a perfectly functional network to make good actions less likely.

We will use vectors for observations and actions, batches for independent samples, and a leading time axis for rollouts. Write tensor shapes before implementing any equation. A Gaussian log density initially has one value per action dimension. Summing over the action axis produces one log probability per transition. Averaging instead changes its scale with the number of joints and silently changes learning behavior between Franka and a bimanual robot.

### Derivation: from logits to a useful derivative
For logits $z_i$, define $p_i=\exp(z_i)/\sum_j\exp(z_j)$. Then
$$\log p_i=z_i-\log\sum_j\exp(z_j),\qquad
\frac{\partial\log p_i}{\partial z_k}=\mathbf1[i=k]-p_k.$$
The derivative is positive for the sampled action's own logit and negative for alternatives. Multiplying it by a positive advantage pushes probability toward the sampled action; a negative advantage reverses that direction.

For one Gaussian coordinate,
$$\log p(a)=-\frac{(a-\mu)^2}{2\sigma^2}-\log\sigma-\tfrac12\log(2\pi),\qquad
\frac{\partial\log p(a)}{\partial\mu}=\frac{a-\mu}{\sigma^2}.$$
Treat the sampled action as fixed in a score-function update. A reparameterized SAC update is different: $a$ is a differentiable function of the policy parameters. We will make that distinction explicit later.

### Work a gradient by hand
Let the probability of moving right be $p=0.25$. For a sampled right action, the derivative of its log probability with respect to its own logit is $1-p=0.75$. If its advantage is $+2$, the ascent contribution is $+1.5$. If the advantage is $-2$, it becomes $-1.5$. The environment did not supply either gradient: it supplied outcomes from which we built the weighting signal.

In the interactive sketch, change the Gaussian mean and spread. Follow what changes: the likely action, the probability mass inside a region, and the density at a point. These are three different quantities.



## Your experiment

Implement softmax without calling softmax, compare its analytic derivative with autograd, and verify that joint log probabilities sum over the action axis. Then use finite differences to check the Gaussian mean derivative. Repeat with a standard deviation of 0.1 and explain the larger gradient magnitude.

## A failure to watch for

A probability density is not a probability, and an action-dimension mean is not a joint log probability.

## Explain it back

If you double the number of independent action coordinates, what happens to the joint log probability?

<details><summary>Reasoning</summary>

It is the sum of all coordinate log densities; its magnitude commonly grows with dimension. Entropy and temperature conventions must account for that dimension.

</details>

## Primary sources

- https://spinningup.openai.com/en/latest/spinningup/rl_intro.html
- https://pytorch.org/docs/stable/distributions.html
