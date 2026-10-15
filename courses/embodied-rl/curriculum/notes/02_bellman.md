# 03 · Why the Bellman equation is a recursion

### A value is a prediction, not a reward
Imagine four symbolic task stages: approach, aligned, grasped, and completed. This is a deliberately small decision model, not a substitute for a physical robot. We use it because we can solve every value exactly and check every later estimator against a known answer.

The value of being aligned is the expected remaining return if you follow a specified policy. The action-value of closing the gripper is the expected return after committing to that action and then following the policy. The difference between those predictions is an advantage. All three depend on the reward, discount, transition dynamics, and policy. None is a physical property of the robot.

The central trick is to split a long future into one immediate reward and another prediction. If you already knew the value of the next state, you would not need to enumerate all later branches again. Bellman's equation expresses that self-consistency. Dynamic programming repeatedly enforces it when the transition model is available.

### Averaging and maximizing answer different questions
Policy evaluation asks how good a fixed policy is. It averages action values using that policy's probabilities. Optimality asks how good the best achievable behavior is and uses a maximum over actions. Replacing an expectation with a maximum changes the question. This distinction will return when we compare SARSA, Q-learning, deterministic actor–critics, and SAC.

Value iteration starts with a guess, backs up one-step predictions, and repeats. In a discounted finite MDP, the Bellman optimality operator is a contraction in the maximum norm. Informally, if two guesses differ by at most ten everywhere, their backed-up guesses differ by at most gamma times ten. Repeating this shrinkage gives a unique fixed point. This guarantee depends on the setting; do not transfer it blindly to a neural network trained with stale replay and a moving target.

Terminal states need an explicit convention. We give a reward on entering completion and make completion absorbing with zero future reward. If you accidentally keep returning the success reward while the terminal state self-loops, its value becomes $10/(1-\gamma)$ instead of zero after completion. The math is not wrong; the model you wrote describes a different reward process.

### Derivation: pull one reward out of the sum
Start with $G_t=r_t+\gamma G_{t+1}$. Take conditional expectations:
$$V^\pi(s)=\sum_a\pi(a\mid s)\sum_{s'}P(s'\mid s,a)\left[r(s,a,s')+\gamma V^\pi(s')\right].$$
Fix the first action to obtain
$$Q^\pi(s,a)=\sum_{s'}P(s'\mid s,a)\left[r(s,a,s')+\gamma\sum_{a'}\pi(a'\mid s')Q^\pi(s',a')\right].$$
For optimal values, replace the next-policy expectation with the best next action. In a finite action space,
$$V^*(s)=\max_a\sum_{s'}P(s'\mid s,a)[r+\gamma V^*(s')].$$
For a fixed policy in matrix form, $V=r_\pi+\gamma P_\pi V$, so $V=(I-\gamma P_\pi)^{-1}r_\pi$ when the inverse exists. Iterative evaluation and this linear solve should agree numerically.

### Back up a two-branch action
Suppose an action earns $-0.5$, reaches a next state with value 8 with probability 0.75, and reaches value 0 otherwise. With gamma 0.9, its action value is $-0.5+0.9(0.75\cdot8+0.25\cdot0)=4.9$. If another action has value 4.5, a greedy policy chooses the first. A fixed policy choosing each action half the time instead has state value 4.7.

The interactive tree lets you change the next-state values and branch probability. Watch the expectation before pressing the optimality switch. That switch changes the action aggregation, not the physical transition probabilities.



## Your experiment

Build transition tensor P[s,a,s′] and reward matrix R[s,a] for the four-stage model. Run value iteration and policy evaluation. Solve the fixed policy by a linear system and compare. Deliberately put a repeated terminal reward in the model, predict the new value, and repair it.

## A failure to watch for

The Bellman expectation equation evaluates a policy; the optimality equation searches over policies.

## Explain it back

Why does the tabular contraction argument not prove that a neural Q-network will converge?

<details><summary>Reasoning</summary>

Projection into a restricted function class, stochastic updates, off-policy sampling, and moving targets change the operator being implemented. The exact discounted tabular contraction no longer settles the question.

</details>

## Primary sources

- https://www.davidsilver.uk/teaching/
- http://incompleteideas.net/book/the-book-2nd.html
