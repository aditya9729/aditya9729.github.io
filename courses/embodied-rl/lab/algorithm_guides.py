"""Original explanations connecting objectives, tensors, gradients and experiments."""

GUIDES={
'sac':r'''
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
''',
'ppo':r'''
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
''',
'td3':r'''
### Start from the weakness of a deterministic actor
DDPG learns an actor that maximizes a learned Q function. If the critic has a spurious peak, the actor has every incentive to find it. Executing the proposed action may reveal that it is poor, but bootstrapped learning can reinforce the mistaken estimate before enough correcting experience arrives. TD3 combines three interventions aimed at that feedback loop. They are complementary design choices, not a proof that estimation error disappears.

First, two online critics learn from the same replay and target; use the smaller of their target predictions for bootstrapping. Second, add small clipped noise to the target actor's command before evaluating continuation. This discourages target values supported only by an extremely narrow action peak. Third, update the actor less frequently than the critics, allowing additional value-fitting steps between policy changes. Correlated critics can still make correlated mistakes.

### Keep two different noises separate
Exploration noise is applied to actions executed by the behavior policy during data collection. It changes the experience stored in replay. Target-policy smoothing noise is sampled inside a detached Bellman target. It changes the continuation estimate used for critic regression. Neither noise is an entropy objective. TD3's learned policy remains deterministic, although its training-time behavior policy is noisy.

### Read the network and optimizer contract
The actor consumes observations [B,O] and returns bounded commands [B,A]. Each critic consumes observations and actions and returns [B]. There are target copies of both the actor and critics. A replay minibatch supplies stored commands, rewards, next observations and true-terminal flags. For the Franka A=7; the exact same update code accepts AgileX A=12 when all shapes are consistent.

Under no-grad, calculate the next target action, add target noise, clip to normalized limits, and evaluate the target critic minimum. Form the bootstrapped target and fit both online critics to it. Every second critic update in this implementation, freeze critic parameters and minimize negative Q1 evaluated at the current actor's action. Use Q1 for the policy gradient; the target minimum remains part of the critic target. The actor's gradient must traverse Q1's action input. Update the target actor and target critics only on these delayed steps.

### Work the target before running it
Take reward -0.1, gamma 0.99, and target predictions 4 and 6 after smoothing. The target is 3.86. Using only the optimistic critic gives 5.84. At true termination the target is -0.1 regardless of both critics. A 3.86 target is a regression label, not evidence that the robot will achieve that return: it inherits the accuracy and support of the bootstrapped estimates.

Set delay to 1, then remove smoothing, then replace the target minimum by Q1, one controlled change at a time. Keep reward, reset distribution, transition count and evaluation seeds fixed. Compare critic disagreement, evaluation return and action saturation. A single run can illustrate a failure; multiple training seeds are needed to claim a reliable algorithmic effect.
''',
'ddpg':r'''
### Why continuous control introduces an actor
DQN can enumerate two CartPole actions and take a maximum. A continuous 18-coordinate R1 command has infinitely many choices, so brute-force maximization is not the same simple operation. DDPG amortizes this search with a neural actor. Given a state, the actor predicts a command intended to maximize the critic's estimated long-term return.

Replay supplies stored transitions for critic regression. A target actor proposes the next action, a target critic evaluates it, and both computations are detached. The online actor then changes its command using the online critic's derivative with respect to action. This derivative answers a local question under the critic's approximation; it is not a derivative through MuJoCo or a guarantee that moving the real wrist will increase return.

For $Q(a)=-(a-0.4)^2$, the derivative at a=0 is +0.8. Gradient ascent pushes the actor toward positive commands. If a faulty learned critic instead places its peak at -0.7, the same mechanism confidently moves the actor the wrong way. This example explains the need for TD3's estimation controls better than memorizing three extra components.

The course treats DDPG as the derivation bridge and implements the TD3 descendant for longer runs. The SB3 runner exposes DDPG for a library baseline. Use the same normalized action contract and evaluate with exploration noise disabled. Keep this distinction explicit when reporting which algorithms you implemented yourself.
''',
'dqn':r'''
### Turn a Bellman backup into supervised regression
A neural Q network outputs one value per discrete action: CartPole observations [B,4] become [B,2]. A replay entry records the action actually executed. Use `gather` to select that action's prediction; do not average or regress every action against the same label. A separate target network constructs the next-state bootstrap label under no-grad. The only supervised label here is a noisy, moving estimate of return.

With reward 1, gamma 0.9, next target values [2,5] and no true termination, the label is 5.5. If the current prediction for the recorded action is 3, the residual prediction-minus-target is -2.5. Huber loss has a linear tail, so its scalar derivative here is -1 instead of the squared loss derivative -5. That limits one sample's influence without making its target correct.

Exploration comes from an epsilon-greedy behavior policy during collection, not from the regression loss. Replay mixes older experiences and reduces immediate temporal correlation, but a small buffer can still be narrow. A delayed target copy reduces feedback speed, but introduces a stale target. These are engineering tradeoffs to examine rather than magic stability switches.

Standard DQN both chooses and evaluates a maximizing next action using the target network. Double DQN instead selects with the online network and evaluates that selected action with the target network. This reduces one selection-bias mechanism. The course's editable function implements standard DQN; do not relabel it Double DQN. For robot joint commands, continuous actor–critics avoid an exponentially growing discrete action grid.
''',
'reinforce':r'''
### The environment gives rewards, not a gradient
For a trajectory, differentiate its probability with the log-derivative identity. Environment transition probabilities contain no policy parameters and disappear from the derivative. What remains is a sum of action log-probability gradients weighted by return. Causality lets us omit rewards from before the action; changing an action cannot alter a reward that already happened.

The notebook uses an undiscounted episodic CartPole objective with gamma=1. Its actor maps four observations to two logits. Sample each action, retain its differentiable log probability, and collect an episode. Compute reward-to-go backward, detach these labels, and minimize negative log probability times return. The action itself is a categorical sample and is not differentiated through. Optimizer descent on the negative objective implements return ascent.

If an action has probability 0.25 and return 2, its own-logit ascent contribution is 1.5. A state-dependent baseline does not change the expected score-function gradient because the policy score has zero conditional expectation. It can greatly reduce variance. An action-dependent baseline needs additional correction; subtracting an arbitrary learned Q from every action's label is not the same argument.

Long episodes and sparse success produce noisy returns. Replacing Monte Carlo returns with a bootstrapped critic reduces variance while introducing approximation bias. This is the conceptual move into actor–critic. Compare repeated seed runs or gradient estimates; one lucky high-return episode is not a dependable improvement claim.
''',
'actor-critic':r'''
### Separate the prediction from the decision
The critic estimates the return expected before choosing the sampled action. The actor needs to know whether that action did better than expected. The TD residual combines immediate reward, a next-state value prediction and the current prediction. A positive residual is evidence favoring the sampled action, subject to critic error. It is not a ground-truth advantage label.

The value network maps [B,O] to [B]. The stochastic actor maps [B,O] to a joint command distribution. Detach the advantage before weighting log probabilities: the actor loss should change the action distribution, not lower its own label by changing the critic. Fit the critic separately against detached return targets. A shared feature trunk is possible, but the course uses separate networks so these gradient routes remain visible.

With reward 1, gamma 0.9, current value 2 and next value 3, the TD residual is 1.7. If next residual is -0.5 and lambda is 0.8, a two-step trace produces advantage $1.7+0.9(0.8)(-0.5)=1.34$. Longer traces move toward Monte Carlo information; shorter traces rely more strongly on the critic. Lambda does not independently guarantee lower total error.

At a timeout, use the final next observation to bootstrap, but stop the trace before the reset episode. At true termination, remove continuation as well. Inspect these two masks separately. This lesson's AC uses one on-policy rollout and one actor pass; PPO changes the update objective and reuses the rollout for several passes. TD3 and SAC also belong to the actor–critic family, with off-policy Q critics instead of this state-value estimator.
''',
'grpo-flow':r'''
### Sampling better actions versus changing the policy
There are two distinct places to spend computation. At training time, optimize policy parameters so useful actions are more likely later. At decision time, draw several candidates from a fixed policy and select or reweight them. They can share a reward or critic while remaining different algorithms. A stronger candidate selector does not by itself show that the underlying policy learned.

For a prior policy p and a candidate distribution q, a KL-regularized improvement problem has the form
$$\max_q\mathbb E_q[Q(a)]-\beta D_{KL}(q\Vert p),
\qquad q^*(a)\propto p(a)\exp(Q(a)/\beta).$$
This differs from the uniform-reference finite-action entropy example in SAC: the original policy now appears as a nonuniform prior. If candidate actions are sampled from p, normalized exponential Q weights approximate this tilt. The approximation depends on proposal coverage and critic quality. No reweighting can discover a successful mode that never appears among candidates.

The effective sample size of normalized weights is $1/\sum_i w_i^2$. When one candidate has almost all the mass, ESS approaches 1 even if many candidates were generated. Lower temperature can therefore increase preference for high-Q candidates while decreasing effective coverage. Use log-sum-exp stabilization before exponentiating; otherwise large values can overflow.

For the robot experiment, sample bounded commands or short action chunks from one frozen base policy, score them with a fixed critic, and compare one sample against best-of-N and weighted selection. Account for inference latency, executed transitions, collision constraints and autonomous success. A scalar Q can overvalue an out-of-distribution candidate, so best-of-N can amplify critic errors as well as find useful actions.

Feng's Beyond RL article motivates this sampling perspective and discusses sequential sampling in language reasoning. Robotics adds dynamics, partial observation, contact and execution costs. The notebook's finite candidate-weight exercise demonstrates the probability arithmetic; it is not a validated robot TSMC implementation. A simulator or world model would also need reliable branching and rollout scoring before a trajectory-particle planner could be assessed.
'''
}

RECIPES={
'sac':[
('Replay','Read a stored batch of observations, commands, rewards, final next observations and true-terminal flags. This is off-policy experience.','No gradients enter replay.'),
('Soft target','Sample next actions, evaluate the target critic minimum, subtract alpha × next log density, and bootstrap.','Everything in this target is detached.'),
('Critics','Regress both online Q predictions for stored actions toward the soft target.','Only online critic weights change.'),
('Actor','Resample current actions and minimize alpha × log density − min Q.','Freeze Q weights; keep the derivative through its action input.'),
('Temperature','Compare sampled differential entropy with the target entropy.','Detach log density; update only log alpha.'),
('Target copies','Average target critic parameters toward online critics.','No optimizer or backpropagation in this averaging step.')],
'ppo':[
('Collect','Execute the frozen behavior policy and save actions, values and old log probabilities.','Stored rollout tensors are fixed during optimization.'),
('Advantages','Compute TD residuals, GAE and value targets with explicit episode masks.','Advantage and return labels are detached.'),
('Ratio','Reevaluate the same saved actions under the current policy.','New log probabilities carry actor gradients.'),
('Clip','Take the minimum of ordinary and clipped advantage-weighted ratios.','Useful probability changes lose incentive past the clip boundary.'),
('Optimize','Take several actor and value steps, then discard the rollout.','Track KL and clip fraction; collect fresh data next.')],
'td3':[
('Replay','Sample real stored transitions.','No gradients through data.'),
('Smooth target','Target actor + clipped noise → target critic minimum → Bellman label.','Detach the entire label.'),
('Two critics','Fit both online Q estimates on recorded actions.','Critic optimizer steps every update.'),
('Delayed actor','Every second update, maximize Q1 for the actor command.','Q1 action derivative remains live; its weights are frozen.'),
('Slow copies','Update target actor and critic parameters.','Averaging runs only on delayed actor-update steps.')]
}

PATHS={
'probability':('Probability → gradients','Compute a joint log density and check its derivative before touching a return.'),
'mdp':('Task → trajectory objective','Specify observation, action units, rewards and stopping rules before choosing an optimizer.'),
'bellman':('Return → recursion','Condition on the first transition, then replace a future trajectory by its expected value.'),
'td':('Observed reward → bootstrapped target','Separate target construction, prediction and the residual that changes the estimate.'),
'dqn':('Replay → target → Q regression','Select the executed action prediction, hold the target fixed and fit a neural value.'),
'reinforce':('Trajectory → weighted log probability','Use reward-to-go as a fixed weight on the sampled action score.'),
'actor-critic':('Prediction → advantage → policy change','Fit a value baseline, estimate advantage and weight the actor score.'),
'ppo':('Rollout → ratio → clipped improvement','Reuse a short rollout while checking how far its policy probabilities move.'),
'ddpg':('Critic surface → actor action','Use an action derivative to replace explicit continuous-action maximization.'),
'td3':('Replay → conservative target → delayed actor','Separate value fitting, target smoothing and policy improvement.'),
'sac':('Replay → soft values → stochastic actor','Learn a bounded distribution by balancing predicted return and entropy.'),
'offline':('Demonstrations → support → online correction','Distinguish imitation labels from Q estimates and new interaction.'),
'franka':('7 commands → wrist motion → return','Keep action units and reset semantics fixed when changing algorithms.'),
'agilex':('12 commands → coordinated wrists','Ask whether a joint policy improves a coupled objective, rather than only two independent reaches.'),
'r1-pro':('Torso + arms → shared reachability','Test the contribution of torso control with matched reset sets and budgets.'),
'grpo-flow':('Candidate group → relative signal','Distinguish sampling, ranking and parameter optimization.'),
'curricula-rewards':('Success history → training distribution','Change task sampling while holding evaluation conditions fixed.'),
'systems-capstone':('Hypothesis → budget → evidence','Measure learning and latency separately; attribute a change through controlled comparisons.')
}

def enrich(lessons):
    for l in lessons:
        title,description=PATHS[l['slug']]
        l['learning_goal']=description
        l['update_title']=title
        l['guide']=GUIDES.get(l['slug'],'')
        l['recipe']=RECIPES.get(l['slug'],[])
    # Added sources support only the original supplemental explanations.
    next(l for l in lessons if l['slug']=='grpo-flow')['refs'].append('https://shengyu-feng.github.io/blog/2026/08/27/beyond-RL/')
