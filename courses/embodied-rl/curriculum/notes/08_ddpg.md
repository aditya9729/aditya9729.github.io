# 09 · A critic that teaches continuous actions

### A maximum over actions is easy until actions are vectors
DQN takes a maximum over a short list of discrete actions. A robot arm has a continuous action space, so finding an exact maximizing action inside every critic target is usually impractical. A deterministic actor approximates that optimization: it maps the observation directly to a candidate command, and a critic supplies a differentiable score.

The critic learns from replayed environment transitions. The actor does not copy their recorded actions; it asks the critic to score the actions it currently proposes at those replayed observations. This separates a behavior action stored in the buffer from the actor action used for an update. Confusing them can turn the actor loss into a constant with respect to actor parameters.

DDPG combines this deterministic policy-gradient idea with a neural critic, replay, and slowly updated targets. Exploration noise is added during data collection because a deterministic actor otherwise repeats its command for a repeated observation. That noise is distinct from TD3's later target smoothing, which is added only while constructing critic targets.

### Follow the gradient through the action, not the simulator
Freeze critic parameters during an actor step, but retain the derivative of Q with respect to its action input. The gradient flows through Q into the actor output and then into actor parameters. Wrapping the entire Q evaluation in no_grad would destroy that path. Freezing parameters and disabling all autograd are different operations.

This update optimizes the critic's current estimate. If the critic invents a sharp high-value ridge outside the data distribution, the actor may exploit it. A high predicted Q can therefore accompany poor environment return. Rendering rollouts and comparing critic predictions with realized outcomes are part of debugging, not cosmetic extras.

Action normalization is another source of hidden mistakes. We let the actor output normalized values in [-1,1] and let an environment wrapper map them to native bounds. Critics see the same normalized coordinate convention in data collection and actor updates. If replay stores radians but the actor produces normalized values, the critic is scoring a different action space from the one you think you trained.

### Derivation: the deterministic actor gradient
For critic $Q_w$ and actor $\mu_\theta$, optimize the replay-state surrogate
$$J_\mu(\theta)=\mathbb E_{s\sim\mathcal D}[Q_w(s,\mu_\theta(s))].$$
The chain rule gives
$$\nabla_\theta J_\mu=\mathbb E[\nabla_a Q_w(s,a)|_{a=\mu_\theta(s)}\nabla_\theta\mu_\theta(s)].$$
The critic target uses slowly moving copies:
$$y=r+\gamma(1-d)Q_{\bar w}(s',\mu_{\bar\theta}(s')).$$
Update targets by $\bar w\leftarrow(1-\tau)\bar w+\tau w$. Here tau is a mixing coefficient, not time and not a trajectory. Small tau slows the target's response but does not make its estimates correct.

### A critic gradient you can draw
Let $Q(a)=-(a-0.4)^2$ and actor output a. Then $dQ/da=-2(a-0.4)$. At a=-0.2 the gradient is +1.2, pointing toward 0.4. If the critic's peak is wrong, the actor will still follow it. The update has no independent access to the true best action.

Add a narrow spurious bump near 0.9 and inspect how gradient ascent can chase it. This is why critic error is amplified by optimization rather than behaving like harmless prediction noise.


### Why continuous control introduces an actor
DQN can enumerate two CartPole actions and take a maximum. A continuous 18-coordinate R1 command has infinitely many choices, so brute-force maximization is not the same simple operation. DDPG amortizes this search with a neural actor. Given a state, the actor predicts a command intended to maximize the critic's estimated long-term return.

Replay supplies stored transitions for critic regression. A target actor proposes the next action, a target critic evaluates it, and both computations are detached. The online actor then changes its command using the online critic's derivative with respect to action. This derivative answers a local question under the critic's approximation; it is not a derivative through MuJoCo or a guarantee that moving the real wrist will increase return.

For $Q(a)=-(a-0.4)^2$, the derivative at a=0 is +0.8. Gradient ascent pushes the actor toward positive commands. If a faulty learned critic instead places its peak at -0.7, the same mechanism confidently moves the actor the wrong way. This example explains the need for TD3's estimation controls better than memorizing three extra components.

The course treats DDPG as the derivation bridge and implements the TD3 descendant for longer runs. The SB3 runner exposes DDPG for a library baseline. Use the same normalized action contract and evaluate with exploration noise disabled. Keep this distinction explicit when reporting which algorithms you implemented yourself.


## Your experiment

Differentiate a hand-written Q(a) through a tiny deterministic actor. Compare critic-parameter freezing with no_grad and explain why only one preserves actor gradients. Then inspect the normalization wrapper used by the real Gymnasium trainer. DDPG is a derivation bridge here; the longer runnable deterministic trainer is TD3 in the next chapter.

## A failure to watch for

Do not detach the actor action before passing it to the critic during the actor update.

## Explain it back

Why can a deterministic actor improve its predicted Q while becoming worse in the environment?

<details><summary>Reasoning</summary>

The actor optimizes an imperfect critic and may exploit estimation errors or extrapolation away from replay coverage.

</details>

## Primary sources

- https://arxiv.org/abs/1509.02971
- https://proceedings.mlr.press/v32/silver14.html
