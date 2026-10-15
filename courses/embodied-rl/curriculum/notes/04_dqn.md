# 05 · From a Q-table to a deep Q-network

### The neural network changes representation, not the question
A Q-table stores one prediction per state–action pair. That works when the state set is small. A robot observation is a vector with many possible values, so a table becomes impractical. A neural Q-function shares parameters across observations and can generalize: nearby inputs may receive related predictions. It can also generalize incorrectly.

Q-learning is off-policy because its target follows a greedy next action even when data were collected by an exploratory behavior policy. SARSA uses the next action actually sampled from the behavior policy. This difference is about the target policy, not merely whether you have a replay buffer. An on-policy method can reuse a small recent batch; an off-policy method can learn from fresh data.

Deep Q-learning adds stabilizing machinery around a neural bootstrap. Replay reduces temporal correlation within a minibatch and lets past transitions contribute again. A target network changes slowly, reducing how rapidly the regression labels move. Neither makes the labels ground truth. The network can still bootstrap from its own errors, especially away from the data distribution.

### Follow one minibatch all the way through
For a discrete action space, a Q-network outputs one value per action. Gather the value of the action that was actually taken. Construct the target using the next observation, a terminal mask, and the target network's best next value. Regress the gathered prediction toward the detached target. Update the target network on a specified schedule.

The maximization can overestimate values because it tends to select actions with positive estimation error. Double DQN separates action selection with the online network from evaluation with the target network. Later, TD3 will use two critics and a minimum in continuous control. These ideas address related estimation problems but are not identical algorithms.

DQN does not directly solve an arbitrary continuous robot action space. If each of seven joints has eleven discrete choices, the Cartesian product has $11^7$ combinations before adding a gripper. An actor network that optimizes continuous actions becomes useful. We learn DQN first because it exposes replay, targets, bootstrapping, and distribution shift in a form we can inspect exactly.

### Derivation: a sampled Bellman optimality target
Tabular Q-learning updates
$$Q(s,a)\leftarrow Q(s,a)+\eta[r+\gamma(1-d)\max_{a'}Q(s',a')-Q(s,a)].$$
Replace the table by $Q_w$ and use a separate target parameter vector $\bar w$:
$$y=r+\gamma(1-d)\max_{a'}Q_{\bar w}(s',a'),\qquad
L(w)=\mathbb E_{(s,a,r,s',d)\sim\mathcal D}[\ell(Q_w(s,a),\operatorname{stopgrad}y)].$$
The notebook uses a Huber loss to reduce the influence of large residuals. This changes the penalty shape, not the Bellman target. For Double DQN, use $a^*=\arg\max_a Q_w(s',a)$ and evaluate $Q_{\bar w}(s',a^*)$.

### Watch a maximum amplify noise
Suppose two equally good actions have true value 3. One noisy critic predicts [2, 4] and another sample predicts [4, 2]. Each estimate is unbiased action by action across the two samples, but the maximum is always 4. The selection operation created an upward bias.

Replay has a separate issue: sampling an action from a transition is not choosing a new action for the past state. That action is part of the fixed data. Keep it separate from the next action used only inside the target.


### Turn a Bellman backup into supervised regression
A neural Q network outputs one value per discrete action: CartPole observations [B,4] become [B,2]. A replay entry records the action actually executed. Use `gather` to select that action's prediction; do not average or regress every action against the same label. A separate target network constructs the next-state bootstrap label under no-grad. The only supervised label here is a noisy, moving estimate of return.

With reward 1, gamma 0.9, next target values [2,5] and no true termination, the label is 5.5. If the current prediction for the recorded action is 3, the residual prediction-minus-target is -2.5. Huber loss has a linear tail, so its scalar derivative here is -1 instead of the squared loss derivative -5. That limits one sample's influence without making its target correct.

Exploration comes from an epsilon-greedy behavior policy during collection, not from the regression loss. Replay mixes older experiences and reduces immediate temporal correlation, but a small buffer can still be narrow. A delayed target copy reduces feedback speed, but introduces a stale target. These are engineering tradeoffs to examine rather than magic stability switches.

Standard DQN both chooses and evaluates a maximizing next action using the target network. Double DQN instead selects with the online network and evaluates that selected action with the target network. This reduces one selection-bias mechanism. The course's editable function implements standard DQN; do not relabel it Double DQN. For robot joint commands, continuous actor–critics avoid an exponentially growing discrete action grid.


## Your experiment

Collect actual CartPole transitions, store a replay minibatch, and execute a DQN gradient step. Inspect shapes [B,obs], [B], and [B,actions]. Verify target-network gradients remain absent. Compare max-target and Double-DQN targets on a constructed critic disagreement case. Long training is optional; passing this check does not mean CartPole is solved.

## A failure to watch for

A replay buffer does not by itself make every learning objective off-policy or correct.

## Explain it back

Why does discretizing each joint independently make ordinary DQN awkward for robot arms?

<details><summary>Reasoning</summary>

A flat action-value output for every joint combination grows exponentially with the number of joints. Structured factorizations require extra assumptions; continuous actors provide another route.

</details>

## Primary sources

- https://arxiv.org/abs/1312.5602
- https://arxiv.org/abs/1509.06461
