"""Original, source-linked long-form curriculum. Content is shared by site and notebooks."""
LESSONS=[]
def add(slug,title,unit,sketch,story,math,worked,experiment,pitfall,question,answer,refs):
    LESSONS.append(dict(slug=slug,title=title,unit=unit,sketch=sketch,story=story.strip(),math=math.strip(),worked=worked.strip(),experiment=experiment.strip(),pitfall=pitfall,question=question,answer=answer,refs=refs,minutes=90))

add('probability','The mathematics behind a policy','I · Foundations','distribution',r'''
### A distribution is a way to act, not just a way to describe uncertainty
Imagine a Franka wrist above a cube. One policy always commands the same small motion. Another assigns probabilities to several motions and samples one. Neither policy knows the reward of an untried action. A stochastic policy lets us explore while retaining a mathematical description of exactly how likely each sampled action was. That description becomes the handle used by a policy-gradient update.

Before learning an algorithm, separate three objects: an outcome, its probability, and its value. If a grasp succeeds with probability 0.7 and earns 10, its expected reward is 7; no individual attempt earns 7. An expectation averages over possible outcomes. An empirical mean averages actual samples. The two become close under suitable sampling assumptions, but a small sample can be misleading. Ten successful grasps do not prove certainty.

For discrete actions, a network outputs logits. A softmax converts them into nonnegative numbers that sum to one. Adding the same constant to all logits leaves those probabilities unchanged. For continuous commands, a diagonal Gaussian uses a mean vector and one standard deviation per action coordinate. Its density can exceed one because a density is not the probability of an exact action. Probabilities come from integrals over regions. This matters when comparing log probabilities between discrete and continuous policies.

### Why logarithms appear in an action update
The probability of a trajectory multiplies many terms. Logarithms turn that product into a sum, which is numerically and algebraically easier to differentiate. A derivative asks how a small parameter change affects a quantity. A gradient collects those derivatives for all parameters. PyTorch computes gradients of a scalar loss; it does not know that the loss is an RL objective. A sign error can therefore train a perfectly functional network to make good actions less likely.

We will use vectors for observations and actions, batches for independent samples, and a leading time axis for rollouts. Write tensor shapes before implementing any equation. A Gaussian log density initially has one value per action dimension. Summing over the action axis produces one log probability per transition. Averaging instead changes its scale with the number of joints and silently changes learning behavior between Franka and a bimanual robot.
''',r'''
### Derivation: from logits to a useful derivative
For logits $z_i$, define $p_i=\exp(z_i)/\sum_j\exp(z_j)$. Then
$$\log p_i=z_i-\log\sum_j\exp(z_j),\qquad
\frac{\partial\log p_i}{\partial z_k}=\mathbf1[i=k]-p_k.$$
The derivative is positive for the sampled action's own logit and negative for alternatives. Multiplying it by a positive advantage pushes probability toward the sampled action; a negative advantage reverses that direction.

For one Gaussian coordinate,
$$\log p(a)=-\frac{(a-\mu)^2}{2\sigma^2}-\log\sigma-\tfrac12\log(2\pi),\qquad
\frac{\partial\log p(a)}{\partial\mu}=\frac{a-\mu}{\sigma^2}.$$
Treat the sampled action as fixed in a score-function update. A reparameterized SAC update is different: $a$ is a differentiable function of the policy parameters. We will make that distinction explicit later.
''',r'''
### Work a gradient by hand
Let the probability of moving right be $p=0.25$. For a sampled right action, the derivative of its log probability with respect to its own logit is $1-p=0.75$. If its advantage is $+2$, the ascent contribution is $+1.5$. If the advantage is $-2$, it becomes $-1.5$. The environment did not supply either gradient: it supplied outcomes from which we built the weighting signal.

In the interactive sketch, change the Gaussian mean and spread. Follow what changes: the likely action, the probability mass inside a region, and the density at a point. These are three different quantities.
''',r'''Implement softmax without calling softmax, compare its analytic derivative with autograd, and verify that joint log probabilities sum over the action axis. Then use finite differences to check the Gaussian mean derivative. Repeat with a standard deviation of 0.1 and explain the larger gradient magnitude.''','A probability density is not a probability, and an action-dimension mean is not a joint log probability.','If you double the number of independent action coordinates, what happens to the joint log probability?','It is the sum of all coordinate log densities; its magnitude commonly grows with dimension. Entropy and temperature conventions must account for that dimension.',['https://spinningup.openai.com/en/latest/spinningup/rl_intro.html','https://pytorch.org/docs/stable/distributions.html'])

add('mdp','From a robot task to an RL objective','I · Foundations','return',r'''
### Start with a task you can define precisely
“Make the robot useful” is not an MDP specification. “Move the Franka wrist into a target region within a fixed control budget” is closer. Define what the policy sees, what it commands, how the world changes, how reward is measured, and what ends an episode. These decisions determine the learning problem before you choose PPO or SAC.

A state contains the information required to predict the next-state distribution given an action. An observation is what the policy actually receives. A single camera image usually omits velocity, occluded object pose, and contact state. Calling that image a state does not make the problem Markov. History, recurrent memory, or a state estimator may be needed. A policy that uses exact simulator object poses is solving a different observation problem from one using head and wrist cameras.

The action is equally important. A scalar can mean a joint torque, a joint-position increment, an end-effector increment, or a desired base velocity. The same numeric vector has no meaning without units, frame, scale, and timing. We keep these interfaces explicit in the Franka, AgileX, and R1 Pro labs, but learning the simulator itself is not the curriculum's goal.

### Reward now versus behavior over time
The reward is an immediate scalar; the return is a sum over future rewards. A reaching reward can favor being close right now while a later success reward favors completing a grasp. RL optimizes an expectation of return under the trajectories generated by its policy. It does not directly optimize the phrase you use to describe the task.

Discounting changes the relative importance of time. At 20 Hz, a discount of 0.99 has an approximate effective scale of 100 steps, or five seconds. That is a useful intuition, not a hard cutoff: later rewards still contribute. If you double control frequency while keeping the same per-step discount and reward, you change the time-based objective. For continuous-time discount rate $\kappa$, use $\gamma=\exp(-\kappa\Delta t)$ when translating between control rates.

Separate termination from truncation. Dropping an object may be a terminal failure by task definition. Stopping a data collector after 200 steps may only truncate a continuing task. A truly finite-horizon deadline is different and requires a time-aware state. Later chapters will show why confusing these cases corrupts value targets.
''',r'''
### Derivation: an objective over trajectories
An MDP is specified by $(\mathcal S,\mathcal A,P,r,\rho_0,\gamma)$. For trajectory $\tau$,
$$p_\theta(\tau)=\rho_0(s_0)\prod_{t=0}^{T-1}\pi_\theta(a_t\mid s_t)P(s_{t+1}\mid s_t,a_t).$$
Define reward $r_t$ after action $a_t$ and return
$$G_t=\sum_{k=t}^{T-1}\gamma^{k-t}r_k,\qquad J(\theta)=\mathbb E_{\tau\sim p_\theta}[G_0].$$
The expectation depends on the policy twice in ordinary language: it chooses actions and thereby changes which future states are visited. Differentiating only the immediate reward ignores that distributional effect.

For the effective horizon intuition, $\sum_{k\ge0}\gamma^k=1/(1-\gamma)$ when $0\le\gamma<1$. This is a sum of discount weights, not a guarantee that all task dependencies fit inside that horizon.
''',r'''
### A small trajectory with a surprising sign
Suppose a reach–grasp–lift attempt receives $[-1,-1,10]$ and $\gamma=0.9$. Then $G_0=-1-0.9+8.1=6.2$, $G_1=-1+9=8$, and $G_2=10$. The early negative rewards do not imply the early actions should be discouraged: their downstream outcome may be worthwhile.

Now delay the success by two extra penalty steps. The return falls even though the binary task success stays one. The algorithm is learning a preference for faster completion. That may be desirable, but you should know it is part of the objective rather than an accidental algorithm property.
''',r'''Compute returns backward from a reward list. Sweep gamma and control dt in the timeline sketch. Write a one-page MDP specification for Franka reaching with exact observation fields, command units, reset distribution, reward terms, and stopping rules. Identify one observation aliasing case where two identical images require different actions.''','Binary success, shaped reward, and discounted return measure different things.','Does a higher shaped return prove that the real task succeeds more often?','No. Shaping may reward partial progress or exploit unintended behavior. Keep a separate, fixed evaluation success metric.',['https://spinningup.openai.com/en/latest/spinningup/rl_intro.html','https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/'])

add('bellman','Why the Bellman equation is a recursion','I · Foundations','bellman',r'''
### A value is a prediction, not a reward
Imagine four symbolic task stages: approach, aligned, grasped, and completed. This is a deliberately small decision model, not a substitute for a physical robot. We use it because we can solve every value exactly and check every later estimator against a known answer.

The value of being aligned is the expected remaining return if you follow a specified policy. The action-value of closing the gripper is the expected return after committing to that action and then following the policy. The difference between those predictions is an advantage. All three depend on the reward, discount, transition dynamics, and policy. None is a physical property of the robot.

The central trick is to split a long future into one immediate reward and another prediction. If you already knew the value of the next state, you would not need to enumerate all later branches again. Bellman's equation expresses that self-consistency. Dynamic programming repeatedly enforces it when the transition model is available.

### Averaging and maximizing answer different questions
Policy evaluation asks how good a fixed policy is. It averages action values using that policy's probabilities. Optimality asks how good the best achievable behavior is and uses a maximum over actions. Replacing an expectation with a maximum changes the question. This distinction will return when we compare SARSA, Q-learning, deterministic actor–critics, and SAC.

Value iteration starts with a guess, backs up one-step predictions, and repeats. In a discounted finite MDP, the Bellman optimality operator is a contraction in the maximum norm. Informally, if two guesses differ by at most ten everywhere, their backed-up guesses differ by at most gamma times ten. Repeating this shrinkage gives a unique fixed point. This guarantee depends on the setting; do not transfer it blindly to a neural network trained with stale replay and a moving target.

Terminal states need an explicit convention. We give a reward on entering completion and make completion absorbing with zero future reward. If you accidentally keep returning the success reward while the terminal state self-loops, its value becomes $10/(1-\gamma)$ instead of zero after completion. The math is not wrong; the model you wrote describes a different reward process.
''',r'''
### Derivation: pull one reward out of the sum
Start with $G_t=r_t+\gamma G_{t+1}$. Take conditional expectations:
$$V^\pi(s)=\sum_a\pi(a\mid s)\sum_{s'}P(s'\mid s,a)\left[r(s,a,s')+\gamma V^\pi(s')\right].$$
Fix the first action to obtain
$$Q^\pi(s,a)=\sum_{s'}P(s'\mid s,a)\left[r(s,a,s')+\gamma\sum_{a'}\pi(a'\mid s')Q^\pi(s',a')\right].$$
For optimal values, replace the next-policy expectation with the best next action. In a finite action space,
$$V^*(s)=\max_a\sum_{s'}P(s'\mid s,a)[r+\gamma V^*(s')].$$
For a fixed policy in matrix form, $V=r_\pi+\gamma P_\pi V$, so $V=(I-\gamma P_\pi)^{-1}r_\pi$ when the inverse exists. Iterative evaluation and this linear solve should agree numerically.
''',r'''
### Back up a two-branch action
Suppose an action earns $-0.5$, reaches a next state with value 8 with probability 0.75, and reaches value 0 otherwise. With gamma 0.9, its action value is $-0.5+0.9(0.75\cdot8+0.25\cdot0)=4.9$. If another action has value 4.5, a greedy policy chooses the first. A fixed policy choosing each action half the time instead has state value 4.7.

The interactive tree lets you change the next-state values and branch probability. Watch the expectation before pressing the optimality switch. That switch changes the action aggregation, not the physical transition probabilities.
''',r'''Build transition tensor P[s,a,s′] and reward matrix R[s,a] for the four-stage model. Run value iteration and policy evaluation. Solve the fixed policy by a linear system and compare. Deliberately put a repeated terminal reward in the model, predict the new value, and repair it.''','The Bellman expectation equation evaluates a policy; the optimality equation searches over policies.','Why does the tabular contraction argument not prove that a neural Q-network will converge?','Projection into a restricted function class, stochastic updates, off-policy sampling, and moving targets change the operator being implemented. The exact discounted tabular contraction no longer settles the question.',['https://www.davidsilver.uk/teaching/','http://incompleteideas.net/book/the-book-2nd.html'])

add('td','Learning values from experience','I · Foundations','td',r'''
### You rarely know the transition table
Dynamic programming assumed a model of what every action would do. On a robot, you usually collect transitions instead: observation, action, reward, next observation, and stopping flags. Value learning then becomes a prediction problem with unusual targets. The label may contain another prediction from the same learner.

Monte Carlo waits until an outcome is available and regresses a value estimate toward a sampled return. It uses a long observed future rather than a value estimate at the next state. That can avoid bootstrapping bias from an imperfect value model, but the return may vary dramatically across attempts. A handoff can fail late for reasons unrelated to an earlier good approach, making the early-state target noisy.

One-step temporal-difference learning instead uses reward plus the next-state estimate. It can update after a single transition. Its TD residual is a discrepancy between an old prediction and a one-step target, not necessarily a task failure. A positive TD residual says the outcome was better than that value prediction expected. It does not say the absolute reward was positive.

### Bias–variance is a design choice with a time axis
An n-step target uses several actual rewards and then bootstraps. Longer targets rely less on a near-term value model but expose the estimate to more rollout variability. Lambda returns combine multiple horizons with geometric weights. GAE will later use the same idea for advantage estimation rather than just value prediction.

This is where episode semantics stop being bookkeeping. A rollout buffer may end while the task continues; bootstrap from the final state in that case. After any reset, stop temporal credit from leaking into the next episode. Those are different masks. A vector environment may automatically return a reset observation instead of the pre-reset final observation; inspect its API before constructing a target.

The value loss normally treats its bootstrap target as fixed during that gradient step. This is a semi-gradient update. Backpropagating through the next-state prediction would optimize a different residual objective. In code, put target construction under no_grad or explicitly detach it. Naming a variable target does not stop autograd.
''',r'''
### Derivation: three targets for one prediction
For a value estimate $V_w$, define
$$y_t^{MC}=G_t,\quad y_t^{TD}=r_t+\gamma(1-d_t)V_w(s_{t+1}),\quad
\delta_t=y_t^{TD}-V_w(s_t).$$
Here $d_t$ marks true termination. Minimize $\frac12(V_w(s_t)-\operatorname{stopgrad}(y_t))^2$. Its gradient is $(V_w-y_t)\nabla_wV_w$.

The n-step target, before any terminal boundary, is
$$y_t^{(n)}=\sum_{k=0}^{n-1}\gamma^k r_{t+k}+\gamma^n V_w(s_{t+n}).$$
For an infinite continuing sequence, the lambda return is $(1-\lambda)\sum_{n\ge1}\lambda^{n-1}y_t^{(n)}$. Finite episodes require the remaining return mass and stopping conditions to be handled explicitly; do not truncate the infinite formula without adjustment.
''',r'''
### A negative reward can be a positive surprise
Let $V(s)=4$, $V(s')=6$, reward $-1$, and gamma 0.9. The TD target is 4.4 and the residual is $+0.4$. With tabular step size 0.25, the updated value is $4+0.25\cdot0.4=4.1$. If the transition is terminal, the target becomes $-1$, the residual becomes $-5$, and the update goes the other way.

If the transition only times out, using the terminal target would discard the predicted continuation and shift the target by 5.4. That is a modeling error, not a random numerical fluctuation.
''',r'''Compute MC, TD, and n-step targets on the same trajectory. Add independent reward noise and estimate their empirical variance over many repeats. Construct a two-episode buffer and test that lambda recursion never crosses the reset. Inspect target tensors and assert that they do not require gradients.''','A reset boundary must stop recurrence even when its final state remains eligible for bootstrapping.','Why can a value target be detached even though it contains the current value network?','The intended semi-gradient update treats the bootstrap label as fixed for this update. Differentiating through it would implement a different optimization problem.',['https://www.davidsilver.uk/teaching/','https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/'])

add('dqn','From a Q-table to a deep Q-network','II · Deep value learning','replay',r'''
### The neural network changes representation, not the question
A Q-table stores one prediction per state–action pair. That works when the state set is small. A robot observation is a vector with many possible values, so a table becomes impractical. A neural Q-function shares parameters across observations and can generalize: nearby inputs may receive related predictions. It can also generalize incorrectly.

Q-learning is off-policy because its target follows a greedy next action even when data were collected by an exploratory behavior policy. SARSA uses the next action actually sampled from the behavior policy. This difference is about the target policy, not merely whether you have a replay buffer. An on-policy method can reuse a small recent batch; an off-policy method can learn from fresh data.

Deep Q-learning adds stabilizing machinery around a neural bootstrap. Replay reduces temporal correlation within a minibatch and lets past transitions contribute again. A target network changes slowly, reducing how rapidly the regression labels move. Neither makes the labels ground truth. The network can still bootstrap from its own errors, especially away from the data distribution.

### Follow one minibatch all the way through
For a discrete action space, a Q-network outputs one value per action. Gather the value of the action that was actually taken. Construct the target using the next observation, a terminal mask, and the target network's best next value. Regress the gathered prediction toward the detached target. Update the target network on a specified schedule.

The maximization can overestimate values because it tends to select actions with positive estimation error. Double DQN separates action selection with the online network from evaluation with the target network. Later, TD3 will use two critics and a minimum in continuous control. These ideas address related estimation problems but are not identical algorithms.

DQN does not directly solve an arbitrary continuous robot action space. If each of seven joints has eleven discrete choices, the Cartesian product has $11^7$ combinations before adding a gripper. An actor network that optimizes continuous actions becomes useful. We learn DQN first because it exposes replay, targets, bootstrapping, and distribution shift in a form we can inspect exactly.
''',r'''
### Derivation: a sampled Bellman optimality target
Tabular Q-learning updates
$$Q(s,a)\leftarrow Q(s,a)+\eta[r+\gamma(1-d)\max_{a'}Q(s',a')-Q(s,a)].$$
Replace the table by $Q_w$ and use a separate target parameter vector $\bar w$:
$$y=r+\gamma(1-d)\max_{a'}Q_{\bar w}(s',a'),\qquad
L(w)=\mathbb E_{(s,a,r,s',d)\sim\mathcal D}[\ell(Q_w(s,a),\operatorname{stopgrad}y)].$$
The notebook uses a Huber loss to reduce the influence of large residuals. This changes the penalty shape, not the Bellman target. For Double DQN, use $a^*=\arg\max_a Q_w(s',a)$ and evaluate $Q_{\bar w}(s',a^*)$.
''',r'''
### Watch a maximum amplify noise
Suppose two equally good actions have true value 3. One noisy critic predicts [2, 4] and another sample predicts [4, 2]. Each estimate is unbiased action by action across the two samples, but the maximum is always 4. The selection operation created an upward bias.

Replay has a separate issue: sampling an action from a transition is not choosing a new action for the past state. That action is part of the fixed data. Keep it separate from the next action used only inside the target.
''',r'''Collect actual CartPole transitions, store a replay minibatch, and execute a DQN gradient step. Inspect shapes [B,obs], [B], and [B,actions]. Verify target-network gradients remain absent. Compare max-target and Double-DQN targets on a constructed critic disagreement case. Long training is optional; passing this check does not mean CartPole is solved.''','A replay buffer does not by itself make every learning objective off-policy or correct.','Why does discretizing each joint independently make ordinary DQN awkward for robot arms?','A flat action-value output for every joint combination grows exponentially with the number of joints. Structured factorizations require extra assumptions; continuous actors provide another route.',['https://arxiv.org/abs/1312.5602','https://arxiv.org/abs/1509.06461'])

add('reinforce','How a sampled action creates a policy gradient','III · Policy optimization','policy',r'''
### The environment does not give you a differentiable loss
In supervised learning, a label gives a direct error for a prediction. In ordinary model-free RL, MuJoCo returns a reward after an action; autograd does not automatically pass through the simulator. REINFORCE estimates how changing action probabilities changes expected return, without differentiating the environment's dynamics.

The intuition is to increase the probability of actions on better trajectories. But stating that intuition is not enough: an action may be followed by a lucky event, and long trajectories mix useful and useless decisions. A sampled return weights a stochastic gradient estimate. The update becomes reliable only by averaging appropriately and managing variance.

The log-derivative trick turns a derivative of a trajectory probability into that probability times a derivative of its log. This lets us sample trajectories from the current policy instead of enumerating them. When environment dynamics do not depend on policy parameters, their terms vanish from the log-probability derivative. They still matter through which trajectories occur; they have not disappeared from the RL problem.

### Credit assignment and causality
Reward received before an action cannot be caused by that action. Replacing a whole-trajectory return with reward-to-go removes irrelevant past rewards from its weight. A state-dependent baseline can remove another source of variance without changing the expected gradient, provided it does not introduce action dependence into that score-function term. A learned value function is a useful baseline, not an external judge that knows the correct answer.

REINFORCE is on-policy in its basic form. The trajectories must come from the policy whose log probabilities define the estimator. Reusing old data without correction changes the sampling distribution and can bias the update. Importance ratios address part of that mismatch; PPO will use them locally rather than treating arbitrary stale data as interchangeable.

Finally, a scalar loss is only a convenient device for autograd. We minimize minus log probability times a detached return or advantage. The negative sign converts ascent on return into descent on a loss. If you let the return or fitted baseline carry an unintended gradient into the actor, you are no longer implementing the estimator derived here.
''',r'''
### Derivation: the score-function identity
For a finite-horizon undiscounted objective, start with $J=\int p_\theta(\tau)R(\tau)d\tau$:
$$\nabla J=\mathbb E_{\tau\sim p_\theta}[R(\tau)\nabla\log p_\theta(\tau)]
=\mathbb E\left[\sum_t\nabla\log\pi_\theta(a_t\mid s_t)R(\tau)\right].$$
Causality permits reward-to-go in place of the whole return. For a discounted finite-horizon objective $G_0$, the exact trajectory expression includes $\gamma^tG_t$ in each time-t contribution. Many practical implementations use a discounted state-visitation convention or a surrogate without that outer factor; do not mix the two derivations silently.

A baseline vanishes in expectation because
$$\sum_a\pi(a\mid s)\nabla\log\pi(a\mid s)b(s)=b(s)\nabla\sum_a\pi(a\mid s)=0.$$
Thus a useful implementation minimizes $-\sum_t\log\pi_\theta(a_t\mid s_t)\operatorname{stopgrad}(G_t-b(s_t))$, with the objective/discount convention stated explicitly.
''',r'''
### A two-action policy you can solve exactly
Let $p=\operatorname{sigmoid}(\theta)$ choose an action worth 2; the other is worth -1. Then $J=2p-(1-p)=3p-1$ and $dJ/d\theta=3p(1-p)$. At $p=0.5$, the exact derivative is 0.75.

Sample actions and average $r(a)\nabla\log\pi(a)$. It should approach 0.75 as the batch grows. Subtract an action-independent baseline and inspect the sample variance. A poor baseline can increase variance; “baseline” is not a universal variance-reduction guarantee for every choice.
''',r'''Verify the bandit gradient analytically, with autograd, and by Monte Carlo. Then execute a short REINFORCE run in Gymnasium CartPole. Explain the sign of each update and graph returns across episodes. The short run demonstrates the estimator; extend it only after verifying the mechanics.''','An action-dependent baseline cannot be inserted into the simple proof without additional correction.','Why does model-free policy gradient not require the derivative of the transition function?','The score-function estimator differentiates the action probability terms of sampled trajectories. Environment outcomes enter the return weights and sampling distribution rather than an autograd path through dynamics.',['https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html','https://www.cs.ualberta.ca/~sutton/williams-92.pdf'])

add('actor-critic','Actor–critic and the advantage estimate','III · Policy optimization','gae',r'''
### The critic predicts; the actor decides
REINFORCE waited for a sampled return to decide whether an action was useful. An actor–critic uses a learned prediction to reduce that burden. The actor parameterizes the behavior policy. The critic estimates value. The critic's error changes the actor's learning signal, so a bad critic can make a reasonable actor update go in the wrong direction.

Actor–critic is a family, not one on-policy algorithm. A2C and PPO use on-policy value estimates; DDPG, TD3, and SAC are off-policy actor–critics using action-value functions. In this lesson we first build a synchronous on-policy actor–critic so the advantage calculation remains visible. The concrete implementation uses a separate actor and value network, avoiding accidental shared-gradient effects.

An advantage asks whether taking an action is better than the policy's usual outcome in that state. We cannot generally compute Q minus V exactly, so we estimate it. A one-step TD residual is one estimate. A long reward-to-go minus V is another. Their errors have different time structure. GAE combines TD residuals across a range of horizons.

### Unroll the recursion before writing a loop
GAE is often introduced as a mysterious backward loop. Instead, write its first terms: current residual, gamma-lambda times the next residual, then another power for the following residual. This is a discounted sum of prediction surprises. When lambda is zero, only one residual remains. When lambda approaches one, more of the sampled future contributes. An imperfect critic and finite rollout boundaries still affect the estimator.

Two boundaries require two masks. A true terminal prevents bootstrapping from the next state. A timeout may permit that bootstrap but must stop GAE's recurrence from crossing into the new episode. If both are collapsed into one done flag in every line, at least one of those meanings is lost.

Normalize advantages only after constructing the batch. Normalization is a numerical convention that changes gradient scale, not a proof of unbiasedness for every estimator. Critic targets must be detached; actor advantages must also be detached. Watch both gradient norms, value error, and explained variance, while remembering that a well-fitted value predictor for a bad policy can coexist with poor task success.
''',r'''
### Derivation: from TD residuals to GAE
Define $\delta_t=r_t+\gamma(1-d_t)V(s_{t+1})-V(s_t)$. Then
$$\hat A_t^{GAE}=\sum_{l\ge0}(\gamma\lambda)^l\delta_{t+l}.$$
In a finite buffer with reset indicator $b_t=d_t\lor\mathrm{truncated}_t$, implement
$$\hat A_t=\delta_t+\gamma\lambda(1-b_t)\hat A_{t+1},\qquad \hat R_t=\hat A_t+V(s_t).$$
The actor minimizes $-\mathbb E[\log\pi_\theta(a_t\mid s_t)\operatorname{stopgrad}(\hat A_t)]$; the critic minimizes $\mathbb E[(V_w(s_t)-\operatorname{stopgrad}\hat R_t)^2]$. These two losses train different predictions and should not share unintended targets.

Expanding the two-step expression shows telescoping cancellation of intermediate value terms when lambda is one and boundaries are handled consistently. With lambda below one, those intermediate value predictions remain part of the weighted estimate.
''',r'''
### Three surprises, one advantage
Suppose TD residuals are [1, -0.5, 2], gamma is 0.9, and lambda is 0.8. Then $\hat A_0=1+0.72(-0.5)+0.72^2(2)=1.6768$. With lambda zero, it would be 1. With a reset after the first transition, it must also be 1 regardless of the later episode's residuals.

Use the sketch's trace-length control to see each residual's weight. The colors represent contributions to a sum, not a guarantee that every future action caused the current outcome.
''',r'''Implement the backward GAE recursion and compare it against an explicit weighted sum on a no-reset sequence. Add terminal and timeout cases with hand-calculated expected targets. Run the from-scratch AC trainer on actual Reacher transitions and verify both actor and critic parameters change.''','A critic is an estimator with its own failures; a small value loss alone is not a robot success metric.','Which mask differs between the TD bootstrap and the GAE recurrence at a timeout?','Bootstrapping may continue from the final timeout state, but recurrence stops before the next reset episode. A true terminal stops both.',['https://arxiv.org/abs/1506.02438','https://arxiv.org/abs/1602.01783'])

add('ppo','PPO: the four cases behind clipping','III · Policy optimization','ppo',r'''
### Why not take many ordinary gradient steps on one rollout?
After a rollout, its actions were sampled by the old policy. Changing the policy makes those samples progressively less representative. Ordinary repeated policy-gradient updates ignore that shift. PPO uses an action likelihood ratio and a clipped surrogate to limit incentives for repeatedly reinforcing the same samples.

The ratio compares the probability density of the same stored action under the current and old policies at the same stored observation. It does not compare a newly sampled action with the old one, and it is not the reward. Store the old log probability when collecting the rollout. Recomputing it with the updated policy destroys the reference and makes the ratio appear to be one.

### Clipping is asymmetric because advantage has a sign
A positive advantage asks to increase an action's probability. Beyond ratio $1+\epsilon$, the clipped surrogate stops rewarding further increases for that sample. A negative advantage asks to decrease probability. Below $1-\epsilon$, it stops rewarding further decreases. Moves in the harmful direction remain penalized. Taking the minimum of clipped and unclipped terms creates this asymmetry.

This is not a hard constraint that keeps every ratio inside the interval. Other samples, shared parameters, and the critic or entropy term can still move a probability. PPO clipping also does not guarantee monotonic improvement in a real neural-network implementation. Measure approximate KL, clip fraction, entropy, value error, and policy performance. Early stopping on excessive KL is a useful separate convention, not the same operation as clipping.

For a bounded continuous policy, use a consistent density for the stored action. Our actor samples an unconstrained Gaussian latent u, maps it through tanh, and includes the change-of-variables correction. The notebook stores u and evaluates its transformed density consistently. Simply clipping sampled Gaussian actions at the actuator bounds and then computing an unclipped Gaussian density produces a different distribution with boundary mass.

Finally, rollout length, epochs, minibatches, and parallel environments are separate controls. A large number of optimizer epochs on a tiny rollout can overfit stale advantages. More parallel environments changes data diversity and reset statistics; it does not by itself change PPO into an off-policy method. The notebook's minimal trainer uses one environment for clarity, while larger-scale reference systems are studied separately.
''',r'''
### Derivation: the surrogate and its cases
Let $\rho_t(\theta)=\exp[\log\pi_\theta(a_t\mid s_t)-\log\pi_{old}(a_t\mid s_t)]$. PPO-Clip maximizes
$$L^{clip}=\mathbb E_t\left[\min\left(\rho_t\hat A_t,\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t\right)\right].$$
If $\hat A>0$, the sample objective is $\hat A\min(\rho,1+\epsilon)$. If $\hat A<0$, it is $\hat A\max(\rho,1-\epsilon)$. Derive those two expressions before looking at the plot.

The actor loss is the negative mean of this objective. The value loss is a separate regression. A common approximate KL statistic is $\mathbb E[(\rho-1)-\log\rho]$ on old-policy samples; it is a diagnostic estimate, not a per-state exact KL constraint.

For $a=\tanh u$, use $\log\pi(a\mid s)=\log\mathcal N(u;\mu,\sigma)-\sum_i\log(1-\tanh^2u_i)$. Stable code computes the log Jacobian through softplus rather than subtracting nearly equal floating-point numbers near saturation.
''',r'''
### Calculate both signs
With epsilon 0.2, ratio 1.5, and advantage +2, the unclipped term is 3 and the clipped term is 2.4, so the objective is 2.4. With advantage -2 at the same ratio, the terms are -3 and -2.4, and the minimum is -3: the harmful increase remains penalized.

Now use ratio 0.5. Positive advantage gives 1, while negative advantage gives -1.6 after clipping. Plot all four cases. If your implementation always clips the ratio before multiplication and omits the minimum, it is wrong in two regions.
''',r'''Implement the clipped objective and test all four sign/direction cases. Differentiate the objective with respect to the ratio to inspect where gradients become zero. Execute a real PPO rollout/update in Reacher using the from-scratch trainer, then alter epochs or clipping one at a time. Treat the short run as an implementation check, not a convergence claim.''','PPO clipping removes an incentive; it does not impose an inviolable trust region.','Why must old log probabilities remain fixed while optimizing a rollout?','They define the sampling-policy reference in the importance ratio. Updating both numerator and denominator together removes the correction and invalidates the intended surrogate.',['https://arxiv.org/abs/1707.06347','https://spinningup.openai.com/en/latest/algorithms/ppo.html'])

add('ddpg','A critic that teaches continuous actions','IV · Off-policy control','critic',r'''
### A maximum over actions is easy until actions are vectors
DQN takes a maximum over a short list of discrete actions. A robot arm has a continuous action space, so finding an exact maximizing action inside every critic target is usually impractical. A deterministic actor approximates that optimization: it maps the observation directly to a candidate command, and a critic supplies a differentiable score.

The critic learns from replayed environment transitions. The actor does not copy their recorded actions; it asks the critic to score the actions it currently proposes at those replayed observations. This separates a behavior action stored in the buffer from the actor action used for an update. Confusing them can turn the actor loss into a constant with respect to actor parameters.

DDPG combines this deterministic policy-gradient idea with a neural critic, replay, and slowly updated targets. Exploration noise is added during data collection because a deterministic actor otherwise repeats its command for a repeated observation. That noise is distinct from TD3's later target smoothing, which is added only while constructing critic targets.

### Follow the gradient through the action, not the simulator
Freeze critic parameters during an actor step, but retain the derivative of Q with respect to its action input. The gradient flows through Q into the actor output and then into actor parameters. Wrapping the entire Q evaluation in no_grad would destroy that path. Freezing parameters and disabling all autograd are different operations.

This update optimizes the critic's current estimate. If the critic invents a sharp high-value ridge outside the data distribution, the actor may exploit it. A high predicted Q can therefore accompany poor environment return. Rendering rollouts and comparing critic predictions with realized outcomes are part of debugging, not cosmetic extras.

Action normalization is another source of hidden mistakes. We let the actor output normalized values in [-1,1] and let an environment wrapper map them to native bounds. Critics see the same normalized coordinate convention in data collection and actor updates. If replay stores radians but the actor produces normalized values, the critic is scoring a different action space from the one you think you trained.
''',r'''
### Derivation: the deterministic actor gradient
For critic $Q_w$ and actor $\mu_\theta$, optimize the replay-state surrogate
$$J_\mu(\theta)=\mathbb E_{s\sim\mathcal D}[Q_w(s,\mu_\theta(s))].$$
The chain rule gives
$$\nabla_\theta J_\mu=\mathbb E[\nabla_a Q_w(s,a)|_{a=\mu_\theta(s)}\nabla_\theta\mu_\theta(s)].$$
The critic target uses slowly moving copies:
$$y=r+\gamma(1-d)Q_{\bar w}(s',\mu_{\bar\theta}(s')).$$
Update targets by $\bar w\leftarrow(1-\tau)\bar w+\tau w$. Here tau is a mixing coefficient, not time and not a trajectory. Small tau slows the target's response but does not make its estimates correct.
''',r'''
### A critic gradient you can draw
Let $Q(a)=-(a-0.4)^2$ and actor output a. Then $dQ/da=-2(a-0.4)$. At a=-0.2 the gradient is +1.2, pointing toward 0.4. If the critic's peak is wrong, the actor will still follow it. The update has no independent access to the true best action.

Add a narrow spurious bump near 0.9 and inspect how gradient ascent can chase it. This is why critic error is amplified by optimization rather than behaving like harmless prediction noise.
''',r'''Differentiate a hand-written Q(a) through a tiny deterministic actor. Compare critic-parameter freezing with no_grad and explain why only one preserves actor gradients. Then inspect the normalization wrapper used by the real Gymnasium trainer. DDPG is a derivation bridge here; the longer runnable deterministic trainer is TD3 in the next chapter.''','Do not detach the actor action before passing it to the critic during the actor update.','Why can a deterministic actor improve its predicted Q while becoming worse in the environment?','The actor optimizes an imperfect critic and may exploit estimation errors or extrapolation away from replay coverage.',['https://arxiv.org/abs/1509.02971','https://proceedings.mlr.press/v32/silver14.html'])

add('td3','TD3: why two critics are only the first change','IV · Off-policy control','twins',r'''
### Three changes address three coupled failure modes
TD3 extends the deterministic actor–critic pattern with clipped double Q targets, delayed actor/target updates, and target-policy smoothing. Learning only “use two critics” misses most of the algorithm. Each change addresses a different way that an actor can overfit critic errors.

Two critics are trained on the same replay data with separate parameters. Their minimum is used in the target to reduce the tendency to bootstrap from an overoptimistic prediction. The critics are correlated estimators, not independent votes. Taking a minimum can introduce underestimation; it is a conservative bias tradeoff rather than an unbiased truth detector.

The actor is updated less frequently than the critics. This gives value estimates more opportunity to respond before the actor moves again. The target actor and target critics typically update on that delayed schedule too. Be precise about whether a reported “update ratio” counts critic optimizer steps, actor optimizer steps, or total transitions. They are not interchangeable quantities.

Target-policy smoothing adds clipped noise to the target actor's action before evaluating the target critics. It encourages values that are less dependent on a single sharp action spike. It is not the same noise used for exploration in the environment. A target-noise setting that is sensible in normalized [-1,1] units may be nonsensical if accidentally interpreted as radians or actuator force.

### What you should log besides reward
Inspect each critic's scale, their disagreement, target values, and action saturation. A falling critic loss may only mean the online critics match a bad bootstrap target. A small actor loss can mean Q values are large, not that physical behavior is good. Compare predicted returns with measured returns on fixed evaluation resets.

The original code in this course makes the two schedules and two noise sources explicit. It uses compact MLPs and a single Gymnasium environment to make updates inspectable. FastTD3's scale and systems choices are a later research layer; this implementation is not a reproduction of its distributional critics, massive parallelism, or performance claims.
''',r'''
### Derivation: the clipped target
Construct the noisy bounded target action
$$\tilde a=\operatorname{clip}(\mu_{\bar\theta}(s')+\operatorname{clip}(\epsilon,-c,c),-1,1),\quad \epsilon\sim\mathcal N(0,\sigma^2I).$$
Then
$$y=r+\gamma(1-d)\min_{i=1,2}Q_{\bar w_i}(s',\tilde a),\qquad
L_Q=\sum_i\mathbb E[(Q_{w_i}(s,a)-y)^2].$$
Every d_delay critic steps, optimize $L_\mu=-\mathbb E[Q_{w_1}(s,\mu_\theta(s))]$ and Polyak-update targets. The delay symbol here is not the terminal mask d. The actor usually uses Q1, not the minimum; follow the algorithm rather than making every twin operation a minimum by habit.
''',r'''
### A single transition, two estimates
With reward -1, gamma 0.99, and target critics [6, 4], the nonterminal target is $-1+0.99\cdot4=2.96$. Using the optimistic maximum gives 4.94 instead. If the transition is truly terminal, both collapse to -1.

If both critics overestimate the same action by two because their training data miss that region, their minimum is still wrong. The sketch's error-correlation control illustrates why diversity matters and why two networks do not guarantee coverage.
''',r'''Hand-calculate targets, inspect the delayed actor-update schedule, and execute the from-scratch TD3 updates on actual Reacher data. Set delay to one or target noise to zero as separate ablations. Keep the same data budget and evaluation resets; do not claim a multi-seed result from one short run.''','Exploration noise and target smoothing are applied in different places for different purposes.','Does taking the smaller of two critics guarantee a lower bound on the true Q?','No. Both estimators may be wrong in the same direction. The minimum changes bias and often reduces overestimation; it is not a certified bound.',['https://arxiv.org/abs/1802.09477','https://spinningup.openai.com/en/latest/algorithms/td3.html'])

add('sac','SAC: optimize reward and uncertainty together','IV · Off-policy control','entropy',r'''
### Exploration becomes part of the objective
Adding random noise to a deterministic action is one way to explore. SAC instead learns a stochastic policy under an entropy-regularized objective. It values reward and, according to a temperature coefficient, the spread of its action distribution. This is not a claim that more randomness is always better. Entropy is useful only relative to a task, action representation, and learning stage.

The critic predicts a soft return that includes the future entropy contribution. The actor trades off a Q estimate against its own log probability. Two critics and a slowly moving target critic help stabilize estimation. Unlike TD3, the common modern SAC formulation does not require a separate target actor: next actions are sampled from the current stochastic actor inside a detached target calculation.

### Two different gradient estimators in the same course
Earlier REINFORCE treated sampled actions as fixed and differentiated their log probabilities. SAC can reparameterize a Gaussian: sample noise independent of the parameters, then express the action as a differentiable function of mean and scale. This lets the actor receive a pathwise derivative through Q with respect to its sampled action. It still does not differentiate the simulator.

Robot commands are bounded, so the Gaussian sample is transformed by tanh and then scaled to native action limits. The log-density correction is essential: squeezing probability mass through tanh changes density. Summing the correction over joints gives the joint log probability. Near saturation, numerical cancellation is a practical concern, which is why the code uses a stable log-Jacobian expression.

Temperature can be fixed or learned. Automatic tuning uses a target entropy convention. Differential entropy can be negative, and changing action units changes its value. A target such as minus action dimension is a common heuristic in normalized coordinates, not a physical law or a guaranteed best exploration level for every embodiment.

Watch entropy, alpha, action saturation, critic values, and actual evaluation behavior together. A high-entropy policy may waste actions; a low-entropy policy may collapse prematurely. FlashSAC changes systems and optimization design at scale; calling this small implementation FlashSAC would conceal the hard parts the paper contributes.
''',r'''
### Derivation: soft target, actor, and temperature
The soft objective augments discounted reward with $\alpha\mathcal H(\pi(\cdot\mid s))$. In the twin-Q formulation,
$$y=r+\gamma(1-d)[\min_i Q_{\bar w_i}(s',a')-\alpha\log\pi_\theta(a'\mid s')],\quad a'\sim\pi_\theta.$$
Minimize the detached-target critic regression. For the actor, reparameterize $a=\tanh(\mu_\theta(s)+\sigma_\theta(s)\odot\epsilon)$ and minimize
$$L_\pi=\mathbb E[\alpha\log\pi_\theta(a\mid s)-\min_iQ_{w_i}(s,a)].$$
Hold critic parameters fixed while retaining derivatives through their action input. With $\alpha=\exp\beta$, the code uses the usual log-temperature surrogate
$$L_\beta=-\mathbb E[\beta\operatorname{stopgrad}(\log\pi_\theta(a\mid s)+\mathcal H_{target})].$$
State the sign convention: target entropy is the desired value of $-\mathbb E\log\pi$ and may be negative for a continuous density.
''',r'''
### The entropy term changes the target
Take reward 1, gamma 0.99, target Q minimum 2, alpha 0.2, and next-action log density -1.5. The soft continuation is $2-0.2(-1.5)=2.3$, giving target 3.277. Removing entropy gives 2.98. At true termination, both targets are exactly 1.

Now increase alpha in the sketch and observe the actor's preference across a narrow and broad action distribution. This is a conceptual objective comparison, not an animation of online training.
''',r'''Verify the tanh density correction and finite gradients near saturation. Execute separate critic, actor, and alpha updates on real replay data. Confirm target gradients are absent while actor gradients through Q remain present. Compare fixed and learned alpha only after the base implementation passes these checks.''','For SAC, an action sample needs rsample-style reparameterization during the pathwise actor update.','Why is the Gaussian log probability before tanh not the final action log probability?','A nonlinear transformation changes density through its Jacobian determinant. Ignoring that term changes the entropy objective and the actor gradient.',['https://arxiv.org/abs/1812.05905','https://spinningup.openai.com/en/latest/algorithms/sac.html'])

add('offline','Offline data, imitation, and residual post-training','V · Robot learning','dataset',r'''
### Reusing data is not the same as solving offline RL
An off-policy method can reuse transitions collected by another behavior policy. Offline RL removes the ability to collect corrective data during training. That changes the problem substantially: an actor can choose actions unsupported by the dataset, and a critic may confidently extrapolate to them. A replay-capable algorithm is not automatically a reliable offline algorithm.

Behavior cloning instead learns to predict demonstrated actions. Its supervised objective does not require a reward, but it can compound errors when the learned policy visits states unlike the demonstrations. A strong demonstration policy is often a useful starting point for robot learning. RL then addresses a different question: can interaction improve the behavior according to a specified reward and evaluation protocol?

A residual architecture freezes a base policy and learns bounded corrections. This can restrict exploration around a useful behavior and preserve much of its competence. It can also restrict the solution space: if the base pushes into an obstacle and the residual is too small, the learner cannot escape. Clipping may saturate the combined command, making the residual's nominal authority misleading.

Action chunks complicate composition. A base predicts several future commands, while observations continue changing. A residual may be applied per step, per chunk, or through an edit policy. Those choices define different control timing and credit assignment. A state-based one-step residual in this course is a mechanism experiment; it is not a full VLA or ResFiT implementation.

Evaluation must account for human resets, interventions, demonstrations, and reward annotations. Report autonomous success separately from success with assistance. Demonstrations are data, not free experience that disappears from a sample-efficiency comparison. A method trained with a useful initialization and one trained from scratch are answering different experimental questions unless that advantage is explicitly part of the benchmark.
''',r'''
### Derivation: two objectives and one composition rule
Behavior cloning minimizes
$$L_{BC}(\theta)=-\mathbb E_{(o,a)\sim\mathcal D_{demo}}\log\pi_\theta(a\mid o).$$
It does not contain a Bellman target. Residual control composes
$$a=\operatorname{clip}(\pi_{base}(o)+\beta\,\pi_{res}(o),a_{min},a_{max}).$$
Freezing the base means its parameters are not optimized. The residual environment includes the base policy in its action transformation. If the base has hidden temporal state or chunk queues, that state may also need to be part of the residual agent's observation to preserve the intended Markov property.

Offline critic extrapolation appears in a target such as $r+\gamma Q(s',\mu(s'))$ when $\mu(s')$ lies outside the data's action support. Reusing the Bellman equation does not certify that extrapolated value. Conservative or behavior-constrained methods introduce additional design choices; they are outside the small residual implementation here.
''',r'''
### Does the residual have enough authority?
If the base outputs +0.8, residual range is [-1,1], and beta is 0.1, the combined action can only lie in [0.7,0.9] before clipping. It cannot reverse direction. With beta 0.4 it can lie in [0.4,1.2], but the upper range saturates at 1. The useful authority is asymmetric around the base action.

Compute this interval joint by joint before attributing a failure to SAC. An impossible correction is an interface limitation, not merely poor optimization.
''',r'''Build a frozen-base residual action wrapper, verify zero residual reproduces the base command, and run actual SAC updates in the induced environment. Plot saturation frequency. Design a comparison with the same base, interaction budget, reset assistance, and fixed autonomous success metric.''','Off-policy online learning and offline RL differ in whether unsupported predictions can be corrected by new interaction.','Why can a bounded residual preserve competence yet prevent a necessary recovery?','It constrains exploration around the base behavior. That same constraint can exclude actions required to undo a base-policy mistake.',['https://arxiv.org/abs/2005.01643','https://residual-offpolicy-rl.github.io/'])

add('franka','Franka: turn an algorithm into a robot experiment','V · Robot learning','robot',r'''
### Put the RL questions first
Our single-arm embodiment is the actual Franka Emika Panda model, not a two-link drawing renamed Franka. Its model supplies geometry, inertias, joints, actuator conventions, and limits. The course adds a small reaching MDP around it so an actor–critic update can be traced from a bounded command to a measured wrist displacement and reward.

This exercise is deliberately reaching, not a claim of solved grasping. Reaching lets us isolate policy learning, observation frames, exploration, and evaluation before adding sparse contact rewards. The earlier robosuite Panda Lift notebook remains available for a native manipulation task. A reaching policy and a lifting policy should never share a success label merely because they use the same robot model.

### Write a control contract before training
The native Menagerie arm actuators accept target joint angles; its shared gripper control has a different scale. Our learning wrapper exposes normalized **joint-target increments** for seven arm joints and holds the gripper at a stated setting. This is not a torque policy and not the native [0,255] gripper convention. The wrapper is part of the MDP.

The target is sampled from the neighborhood of an attainable wrist configuration. We generate a candidate configuration, compute its forward kinematics, and then reset the robot to its start configuration. That makes the target kinematically reachable under the model, but it does not guarantee a collision-free or dynamically feasible trajectory. Goals near the start are a curriculum for debugging, not a general manipulation benchmark.

Observations include named joint positions/velocities and wrist-to-goal error. MuJoCo's full generalized state may also contain object coordinates; never assume the first few qpos entries are all you need. Use named joint indices and declare all quantities' frames. A Cartesian error in world coordinates must not silently become a base-frame error when you later enable a mobile base.

Measure learning using held-out reset seeds, distance over time, command magnitude, and success under an explicit tolerance/hold rule. Render representative failures. Compare PPO and SAC with matched transitions and multiple training seeds before making algorithm claims. One short finite-gradient test establishes integration, not task competence.
''',r'''
### Derivation: the learning wrapper defines the action semantics
For normalized action $a_t\in[-1,1]^7$, define
$$q^{target}_{t+1}=\operatorname{clip}(q_t+\Delta q_{max}a_t,q_{min},q_{max}).$$
The model's servo dynamics then attempt to follow that target. This is not the kinematic update $q_{t+1}=q_t+\Delta q_{max}a_t$: actual q evolves through dynamics.

The course reaching reward uses distance and command effort, with a separate held-success condition. If a progress term is added, distinguish raw distance decrease $d_t-d_{t+1}$ from policy-invariant potential shaping $\gamma\Phi(s_{t+1})-\Phi(s_t)$. The latter requires consistent discount and terminal treatment; arbitrary progress shaping does not inherit the guarantee.
''',r'''
### Same action number, different meanings
An action value of 0.5 with a maximum joint increment of 0.04 radians requests a 0.02-radian target shift for that joint. It does not command a 0.5-radian jump or 0.5 newton-metres. The measured movement may be smaller because of inertia, servo behavior, contact, and the control interval.

Before training, step +0.5 and -0.5 separately from the same reset and plot both the requested target and measured joint position. That plot is a better interface check than seeing the robot move vaguely in the right direction.
''',r'''Load the pinned Panda asset, print named joints and controls, construct the reaching environment, and run environment checks. Execute at least one from-scratch PPO update on real robot transitions. Then compare actor outputs, target angles, measured joint angles, and reward components. Optional longer runs remain explicitly separate from correctness checks.''','A native gripper scale and an arm angle scale cannot share an undocumented normalization rule.','Why does a goal generated by forward kinematics not prove the reaching episode is easy or safe?','It proves a configuration can place the wrist there in the model. It does not prove a collision-free path, dynamic feasibility under controls, or real-robot safety.',['https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda','https://robosuite.ai/docs/modules/controllers.html'])

add('agilex','AgileX: learn coordination, not two isolated reaches','V · Robot learning','bimanual',r'''
### Use the vendor embodiment, not a similar-looking ALOHA
ALOHA names a family of bimanual systems. The DeepMind ALOHA 2 model is not interchangeable with AgileX's hardware. This course uses the actual MJCF from AgileX's published mobile_aloha_sim legacy source, pinned to a specific commit. Its arm geometry, joint naming, gripper actuation, and surrounding scene are retained.

Despite the repository name, this particular MJCF has a fixed chassis. We do not claim mobile AgileX control from it. The model also includes scene objects, so qpos contains more than the arm joints. Its four finger actuator coordinates are slide distances, while twelve arm coordinates are angles. A flat numeric action array is not a sufficient description of that interface.

The first bimanual learning problem coordinates both wrists toward a paired goal. A central actor receives both arms' state and emits one concatenated action. We add a relative-wrist term to make the coordination objective explicit. This is an introductory coordination task, not shared-object grasping. Physical object manipulation requires relevant contacts and a task definition such as lift, handoff, or insertion.

### How to tell whether the arms are actually coordinating
Two independent policies can each reduce a local wrist error while producing an incompatible relative pose. A central policy may learn useful dependencies, but centralized input alone does not guarantee coordination. Measure both per-arm error and a relative-geometry metric. If the task uses a shared object, log object pose/tilt, both grasp states, and success, not only wrist proximity.

Exploration also scales with action dimension. Independent Gaussian noise in twelve joints can produce much larger aggregate movement than in seven joints with the same per-coordinate standard deviation. Compare total action norm, per-joint scales, and end-effector motion before transferring a single-arm exploration setting. Entropy coefficients and target entropy conventions need the same care.

Ablations should distinguish evaluation-time command masking from retraining a policy under a restricted action space. The first measures sensitivity of a learned policy to an intervention. The second measures what a different learner can achieve under a different control contract. Neither should be mislabeled as a physically welded arm unless the simulator model was changed accordingly.
''',r'''
### Derivation: local errors and a coupling term
Let $e_L=x_L-g_L$, $e_R=x_R-g_R$, and desired relative vector $d^*=g_L-g_R$. One coordination loss is
$$\ell=\|e_L\|^2+\|e_R\|^2+\lambda\|(x_L-x_R)-d^*\|^2.$$
The relative term couples both arms' gradients even without a shared object. In a contact task the dynamics add coupling as well. These are distinct sources of dependence.

A central policy writes $\pi_\theta(a_L,a_R\mid o_L,o_R,g)$. A diagonal Gaussian has conditionally independent sampling noise given its means, yet each mean can depend on both arms' observations. Conditional noise factorization therefore does not imply two independent controllers.
''',r'''
### A coordinated target can expose local success inflation
Suppose each wrist is within 4 cm of its goal but the errors point in opposite directions. Their relative error may be 8 cm. Reporting only mean wrist error hides that mismatch. With a shared tray, such disagreement can also create tilt or internal contact forces.

Use the sketch to move one wrist while keeping the other fixed. Inspect both local errors and relative error. It is a geometry illustration; actual trajectories are generated separately by the MuJoCo notebook.
''',r'''Load official AgileX geometry, inspect named joint indexing and finger units, and step the paired-goal environment. Compare joint action norms with Franka. Execute an optimizer update, then design a two-arm observation/action ablation with separate per-arm, relative-pose, and task-success metrics.''','Do not label DeepMind ALOHA 2 as AgileX or infer a working mobile base from a repository name.','Can a diagonal-Gaussian centralized policy still coordinate the arms?','Yes. Its action means can depend on both arms and the shared objective. Conditional independence of sampling noise is not independence of the decision functions.',['https://github.com/agilexrobotics/mobile_aloha_sim','https://github.com/agilexrobotics/aloha-agilex'])

add('r1-pro','R1 Pro: coordinate the torso, arms, and base honestly','V · Robot learning','whole',r'''
### Whole-body RL is a problem formulation, not a robot label
An R1 Pro has more ways to place its wrists than a fixed single arm. Torso posture can change reach and clearance. Base motion can move both wrists together and alter the coordinate frames. A policy that controls these degrees of freedom under one objective can learn coordination, but the word whole-body does not specify whether the low-level controller uses IK, impedance, joint targets, or direct torques.

We use the actual official R1 Pro URDF-derived geometry and inertial model. The published ManipSim asset fixes the chassis and has no MuJoCo actuators after import. We therefore distinguish three layers: the vendor model, course-defined joint servos, and an optional ideal planar-base extension. The last two are teaching adaptations, not factory-calibrated actuator or wheel dynamics.

The default R1 lab uses four torso joints, both seven-joint arms, and held grippers. Its optional planar extension adds x, y, and yaw generalized coordinates driven by declared idealized controls. This permits a controlled coordination experiment on the actual upper-body geometry while making the missing wheel steering/rolling model explicit. It cannot establish terrain navigation, slip robustness, or hardware deployment performance.

### Redundancy is useful only with an objective
Many configurations can achieve similar wrist positions. The learner needs preferences among them: posture costs, action limits, torso excursion, base movement cost, collision constraints, or task-specific success. A reward that only measures wrist error may encourage awkward or high-effort postures. A large action penalty can instead discourage the base even when moving it is necessary. Reward weights are part of the hypothesis.

A Jacobian explains local kinematic influence. Stack the two wrists' Jacobians and inspect columns for base, torso, and arms. A singular-value calculation describes local motion authority under the chosen units and scaling, not a global ability to complete a contact task. Constraints and contact forces change what is dynamically feasible.

The curriculum's capstone separates arm-only, torso-plus-arms, and ideal-base-plus-upper-body action spaces. Train each independently and also test evaluation-time masks on a full policy. Match goals, transitions, and resets. Report failure causes instead of averaging away base saturation, unreachable targets, or one-arm collapse.
''',r'''
### Derivation: a structured body action
Partition $q=[q_b,q_t,q_L,q_R]$. Locally,
$$\begin{bmatrix}\dot x_L\\\dot x_R\end{bmatrix}=
\begin{bmatrix}J_{Lb}&J_{Lt}&J_{LL}&0\\J_{Rb}&J_{Rt}&0&J_{RR}\end{bmatrix}
\begin{bmatrix}\dot q_b\\\dot q_t\\\dot q_L\\\dot q_R\end{bmatrix}.$$
The shared base and torso columns move both wrist tasks. Independent arm commands affect only their own kinematic branches, although contacts may dynamically couple the entire system.

A possible reward is $-\|e_L\|-\|e_R\|-\lambda_b\|a_b\|^2-\lambda_t\|a_t\|^2-\lambda_a\|a_a\|^2$. These are normalized command costs, not mechanical work. Mechanical work requires actuator forces and velocities with consistent mappings and units.
''',r'''
### The same wrist displacement has several explanations
If both wrists need a 10 cm translation, moving the idealized base can reduce both errors together. If the goals require opposite translations, the base alone cannot solve them. A torso rotation can help one goal while worsening the other. The actor must learn these dependencies from the observation and reward; concatenating commands does not supply them automatically.

Vary the base cost in the sketch, then predict whether the preferred motion strategy changes. Test the prediction in the notebook using the explicitly named planar-base extension rather than claiming a vendor wheel-controller result.
''',r'''Compile the official model, list the imported joints, add documented course PD control, and verify finite stepping. Compare fixed-chassis upper-body control with the optional ideal planar base. Compute actual MuJoCo wrist Jacobians and inspect body-part columns. Execute a short PPO update; long coordination training remains a separate experiment.''','A converted URDF is geometry and dynamics data, not a validated RL environment or a calibrated mobile controller.','What does the optional planar-base experiment establish, and what does it leave unresolved?','It studies centralized coordination with an idealized planar base and actual upper-body geometry. It does not validate R1 steering, rolling, slip, terrain contact, real actuator response, or hardware safety.',['https://github.com/OpenGalaxea/GalaxeaManipSim','https://docs.galaxea-ai.com/Guide/R1Pro/isaacsim/r1pro_isaacsim_guide/'])

add('grpo-flow','Grouped policy updates and flow-action policies','VI · Research literacy','group',r'''
### Do not treat every new acronym as a new branch of RL
GRPO changes how relative outcomes provide a policy-learning signal; flow-based action policies change how actions or chunks are represented and sampled. These are different axes. Before asking whether either improves a robot, identify the data distribution, action representation, gradient estimator, and evaluation task.

GRPO's original setting groups multiple outputs for the same input context and derives relative reward information without a separately learned value baseline in that formulation. In robotics, “same context” is demanding: object pose, robot posture, contacts, hidden state, and randomness can all differ across resets. Grouping arbitrary episodes may mostly compare task difficulty instead of action quality.

Standardizing group rewards yields zero signal when every outcome is identical. Sparse tasks where every rollout fails can therefore provide no relative learning signal. Increasing group size costs more trajectories; critic-free does not mean experience-free. Group normalization also changes reward scaling and can affect how different prompt or task groups contribute to an update.

### A flow actor changes the optimization surface
A flow-matching policy generates actions by integrating a learned velocity field from noise toward an action sample. That can represent complex distributions, including multimodal demonstrations. But differentiating a Q-guided objective through generated actions raises optimization and stability questions. A stronger critic gradient is not automatically a better policy update.

QF3 studies a particular way to filter Q guidance in a flow-policy setting while retaining a flow-matching anchor. The earlier appendix has an isolated clipped-gradient exercise. The main course does not disguise a Gaussian SAC actor as a flow actor. A faithful QF3 experiment would require the actual flow parameterization, integration scheme, training data, objective, and critic recipe.

Use a component table when reading new methods: policy representation, value estimator, data source, update rule, regularizers, and systems implementation. That table often reveals which earlier chapter you need to revisit. It also prevents an experiment from changing five mechanisms and attributing the result to one named method.
''',r'''
### Derivation: group normalization is only one component
For returns $R_1,\dots,R_G$ from one declared context, a common grouped signal is
$$\hat A_i=\frac{R_i-\bar R}{\sqrt{G^{-1}\sum_j(R_j-\bar R)^2}+\epsilon}.$$
This normalization alone is not a complete GRPO implementation. The policy ratio, log-probability accounting, KL/reference choices, and trajectory/step objective must also be specified. Continuous actions need valid action densities and consistent stored actions.

For a flow sketch, $x_t=(1-t)z+ta$, with target velocity $a-z$. Flow matching fits a vector field to such targets. A Q-guided term introduces an additional gradient through the predicted action construction. Clipping a guidance component can suppress that path outside a declared interval, but does not fix a wrong critic.
''',r'''
### Grouping can remove the wrong thing
Returns [0,0,1,1] have mean 0.5 and population standard deviation 0.5, giving approximately [-1,-1,1,1]. Returns [0,0,0,0] give zeros. Now imagine the first two episodes were impossible resets and the last two easy resets. The same numbers no longer isolate policy quality at a shared context.

In the exercise, first calculate the arithmetic, then document what makes a robotic reset comparable. This is the distinction between implementing a formula and implementing a valid learning experiment.
''',r'''Implement robust group normalization and an exact differentiable bandit policy-ratio update. Test equal-reward groups and negative rewards. Inspect the flow-gradient clipping appendix, identify which paths are detached, and write a faithful-implementation checklist for QF3. No validated robotic GRPO or full QF3 reproduction is claimed.''','A critic-free baseline can save model computation while requiring many costly robot trajectories.','Why is grouping arbitrary robot episodes an unreliable analogue of grouping responses to one prompt?','Different resets and hidden physical states change task difficulty and outcomes. The relative signal may compare contexts instead of decisions within a context.',['https://arxiv.org/abs/2402.03300','https://qf3-rl.github.io/'])

add('curricula-rewards','What you practice, what you reward, what you retain','VI · Research literacy','curriculum',r'''
### Three interventions at three different places in the loop
Success-guided sampling changes which configurations are practiced. A learned reward model changes the learning signal assigned to outcomes or progress. Continual learning changes how new training affects previously acquired behavior. These interventions can be combined, but they should not be confused or evaluated with the same single metric.

SGS, presented in A Balanced Data Diet, uses success information over configurations to guide sampling. A finite set of reset seeds makes that idea inspectable in this course. It is not replay prioritization: we choose the next environment configuration rather than the next stored transition minibatch. Histories are initially uncertain and become policy-dependent as learning progresses. A target success band is a sampling preference, not a direct estimate of learning progress.

Learned rewards introduce another estimator between observation and policy update. SARM2 estimates structured task progress/value with stage-related machinery; its released progress-like outputs should not be casually treated as calibrated immediate RL rewards. RARM uses reference-conditioned progress and confidence/high-water-style gating in its own method. Both require data and model assumptions absent from a hand-written distance reward.

### Progress scores are vulnerable to temporal mistakes
If a model predicts progress 0.7 at every frame, summing that absolute score rewards waiting forever unless the horizon or other terms intervene. Differences measure change, while discounted potential shaping uses a specific gamma-weighted difference. Repeated advance–retreat cycles can exploit poorly designed progress rewards. Confidence gating and high-water logic can help with some patterns while introducing history dependence that the observation may need to expose.

ACCRUE addresses continual off-policy training of robot foundation models with mechanisms for retaining prior behavior while acquiring current tasks. A simple output-distillation exercise can illustrate retention pressure, but it is not the full actor/critic, memory, representation, and task-transition system. Retention is measured by re-evaluating prior tasks, not by observing a smaller distillation loss.

The first contribution-grade study here is deliberately narrow: uniform versus SGS-shaped reset sampling on one embodiment, one task, one fixed reward, and a matched transition budget. Evaluate on the original held-out reset distribution. A curriculum that improves success only on its preferred training resets has not established broader task competence.
''',r'''
### Derivation: reset weights, reward shaping, and retention
An illustrative SGS-shaped weight over estimated success p is
$$w(p)=(p+\epsilon)^{\kappa p_*}(1-p+\epsilon)^{\kappa(1-p_*)},\qquad
P(i)=\operatorname{softmax}_i(\log w_i/T).$$
The notebook implements this isolated mechanism, not the complete paper protocol. A potential-based reward term is
$$F(s,s')=\gamma\Phi(s')-\Phi(s).$$
Its discounted trajectory sum telescopes, leaving boundary terms under the required assumptions. Raw $\Phi(s')$ or an undiscounted difference is not automatically equivalent.

A retention surrogate can penalize discrepancy between current and frozen actor outputs on retained inputs. Its coefficient trades adaptation against preservation; output similarity is only a proxy for retained task success and may miss distribution shifts.
''',r'''
### A reward loop that pays for standing still
Suppose absolute progress stays at 0.7 for ten steps. Summing it yields reward 7 despite no progress. A raw difference gives zero after the first change. With gamma 0.99 and a constant potential 0.7, the potential term is -0.007 each step, and the full discounted sum must be interpreted with its boundary terms—not as an arbitrary penalty invented by the shaping theorem.

For the sampler, compare reset success estimates [0.05,0.5,0.95]. With a target near 0.5, middle configurations can receive more mass. If all configurations fail and histories are identical, the sampler cannot infer which failures are learnable without additional information.
''',r'''Use actual native success histories in the existing robot reset wrapper, then analyze synthetic histories to isolate the weighting rule. Demonstrate an absolute-progress reward loop and derive the potential telescoping identity numerically. Design a retention evaluation matrix with before/after success on every old task. Keep real model training gaps explicit.''','A learned progress estimate is not automatically an immediate reward, and a sampling change is not an algorithm improvement under every evaluation distribution.','Why must evaluation resets remain separate from success-guided training resets?','Otherwise the method can look better by choosing easier or preferred configurations rather than improving behavior on the task distribution we intended to measure.',['https://sgs-rl.github.io/','https://accrue-rl.github.io/','https://qianzhong-chen.github.io/sarm2.github.io/','https://rarm-robotics.github.io/'])

add('systems-capstone','From readable RL to a defensible research experiment','VI · Research literacy','systems',r'''
### Sample efficiency and wall-clock efficiency are different axes
A method can use fewer environment transitions yet take longer because each update is expensive. Another can learn faster in wall time by increasing simulation throughput while consuming more data. On real robots, resets, human assistance, and control latency add costs that a simulated transitions-per-second chart does not capture.

FastTD3 and FlashSAC study specific scalable learning recipes, including algorithmic and systems choices. Running ordinary TD3 or SAC on a Colab GPU does not reproduce them. Small state networks often gain little from a GPU when a CPU simulator supplies one transition at a time. Accelerator benefits are more plausible with larger batches, many environments, image encoders, or larger policies; measure the actual bottleneck.

EXPO-FT adds a specific form of off-policy adaptation around pretrained action generation. Real-Time EXPO-FT additionally addresses asynchronous timing and alignment. A slow model may generate a chunk based on an older observation while a faster process edits or selects actions using newer information. Queue prefixes, already executed actions, and timestamps become part of the learning and control design. A stale but high-scoring action can still be wrong for the current state.

### Make the capstone answer one question
Choose a single task and embodiment, then compare one mechanism at a time. For example: does success-guided reset sampling improve held-out dual-wrist reaching success on AgileX at the same transition budget? Or: does torso control reduce unreachable-goal failures on R1 Pro compared with arm-only control? Do not simultaneously change the reward model, policy architecture, reset distribution, observation modality, and optimizer and then call the result a PPO-versus-SAC comparison.

Define training seeds, evaluation seeds, budgets, assistance accounting, stopping rules, and metrics before running. Save versions, configuration, raw logs, checkpoints, and failure videos. Separate an integration smoke test from a converged policy and a converged policy from a statistically supported comparison. A policy's last checkpoint is not automatically its best checkpoint; selecting one on evaluation data can bias the reported result.

Use a hierarchy of compute budgets: seconds for algebra and gradient checks, minutes for update plumbing, longer controlled runs for learning curves, then repeated seeds for claims. Mac CPU covers the conceptual and small state-policy path. Colab CUDA is an optional execution path, not a guaranteed GPU allocation or proof of faster training. The complete course remains useful even when no GPU is assigned.
''',r'''
### Derivation: count what a training budget actually contains
If N environments each collect H steps, a rollout contains $NH$ transitions. With E epochs and minibatch size B, PPO performs approximately $E\lceil NH/B\rceil$ minibatch steps. For replay methods define
$$UTD=\frac{\text{critic optimizer steps}}{\text{new environment transitions}}.$$
Actor and temperature steps can have separate ratios. Report all relevant conventions.

Wall time is not simply a sum when pipelines overlap. In a sequential loop it includes environment, inference, update, reset, rendering, and logging time. In an asynchronous pipeline, throughput is limited by the bottleneck and synchronization overhead. Measure per-component timings and end-to-end time instead of extrapolating a kernel speedup into a learning speedup.
''',r'''
### The same rollout, very different optimization work
Eight environments times 256 steps give 2048 transitions. Four epochs with minibatches of 256 give 32 minibatch optimizer steps. Changing to ten epochs yields 80 optimizer steps on the same transitions. In SAC, collecting 2048 transitions with UTD 1 gives 2048 critic steps. These raw counts do not prove either method is better because batch size, network size, and gradient structure also differ.

The systems sketch lets you change environment and update time. It reports a toy sequential throughput calculation explicitly as a calculation, not a measured FastTD3 or FlashSAC result.
''',r'''Write a complete pre-run experiment card, execute one CPU baseline, and inspect all recorded metrics. Generate a matched-budget seed sweep command. Profile one loop without rendering and another with recording. Identify exactly which additional components a faithful FastTD3, FlashSAC, EXPO-FT, or Real-Time EXPO-FT reproduction still needs.''','A fast simulator benchmark and a successful learning experiment are different kinds of evidence.','What is the smallest convincing result for your first contribution?','One clear hypothesis, a reproducible baseline, a controlled change, held-out evaluation, repeated training seeds, honest uncertainty, and a documented failure analysis.',['https://younggyo.me/fast_td3/','https://holiday-robot.github.io/FlashSAC/','https://pd-perry.github.io/expo-ft/','https://pd-perry.github.io/real-time-expo-ft/','https://research.google.com/colaboratory/faq.html'])
