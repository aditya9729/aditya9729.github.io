# 15 · R1 Pro: coordinate the torso, arms, and base honestly

### Whole-body RL is a problem formulation, not a robot label
An R1 Pro has more ways to place its wrists than a fixed single arm. Torso posture can change reach and clearance. Base motion can move both wrists together and alter the coordinate frames. A policy that controls these degrees of freedom under one objective can learn coordination, but the word whole-body does not specify whether the low-level controller uses IK, impedance, joint targets, or direct torques.

We use the actual official R1 Pro URDF-derived geometry and inertial model. The published ManipSim asset fixes the chassis and has no MuJoCo actuators after import. We therefore distinguish three layers: the vendor model, course-defined joint servos, and an optional ideal planar-base extension. The last two are teaching adaptations, not factory-calibrated actuator or wheel dynamics.

The default R1 lab uses four torso joints, both seven-joint arms, and held grippers. Its optional planar extension adds x, y, and yaw generalized coordinates driven by declared idealized controls. This permits a controlled coordination experiment on the actual upper-body geometry while making the missing wheel steering/rolling model explicit. It cannot establish terrain navigation, slip robustness, or hardware deployment performance.

### Redundancy is useful only with an objective
Many configurations can achieve similar wrist positions. The learner needs preferences among them: posture costs, action limits, torso excursion, base movement cost, collision constraints, or task-specific success. A reward that only measures wrist error may encourage awkward or high-effort postures. A large action penalty can instead discourage the base even when moving it is necessary. Reward weights are part of the hypothesis.

A Jacobian explains local kinematic influence. Stack the two wrists' Jacobians and inspect columns for base, torso, and arms. A singular-value calculation describes local motion authority under the chosen units and scaling, not a global ability to complete a contact task. Constraints and contact forces change what is dynamically feasible.

The curriculum's capstone separates arm-only, torso-plus-arms, and ideal-base-plus-upper-body action spaces. Train each independently and also test evaluation-time masks on a full policy. Match goals, transitions, and resets. Report failure causes instead of averaging away base saturation, unreachable targets, or one-arm collapse.

### Derivation: a structured body action
Partition $q=[q_b,q_t,q_L,q_R]$. Locally,
$$\begin{bmatrix}\dot x_L\\\dot x_R\end{bmatrix}=
\begin{bmatrix}J_{Lb}&J_{Lt}&J_{LL}&0\\J_{Rb}&J_{Rt}&0&J_{RR}\end{bmatrix}
\begin{bmatrix}\dot q_b\\\dot q_t\\\dot q_L\\\dot q_R\end{bmatrix}.$$
The shared base and torso columns move both wrist tasks. Independent arm commands affect only their own kinematic branches, although contacts may dynamically couple the entire system.

A possible reward is $-\|e_L\|-\|e_R\|-\lambda_b\|a_b\|^2-\lambda_t\|a_t\|^2-\lambda_a\|a_a\|^2$. These are normalized command costs, not mechanical work. Mechanical work requires actuator forces and velocities with consistent mappings and units.

### The same wrist displacement has several explanations
If both wrists need a 10 cm translation, moving the idealized base can reduce both errors together. If the goals require opposite translations, the base alone cannot solve them. A torso rotation can help one goal while worsening the other. The actor must learn these dependencies from the observation and reward; concatenating commands does not supply them automatically.

Vary the base cost in the sketch, then predict whether the preferred motion strategy changes. Test the prediction in the notebook using the explicitly named planar-base extension rather than claiming a vendor wheel-controller result.



## Your experiment

Compile the official model, list the imported joints, add documented course PD control, and verify finite stepping. Compare fixed-chassis upper-body control with the optional ideal planar base. Compute actual MuJoCo wrist Jacobians and inspect body-part columns. Execute a short PPO update; long coordination training remains a separate experiment.

## A failure to watch for

A converted URDF is geometry and dynamics data, not a validated RL environment or a calibrated mobile controller.

## Explain it back

What does the optional planar-base experiment establish, and what does it leave unresolved?

<details><summary>Reasoning</summary>

It studies centralized coordination with an idealized planar base and actual upper-body geometry. It does not validate R1 steering, rolling, slip, terrain contact, real actuator response, or hardware safety.

</details>

## Primary sources

- https://github.com/OpenGalaxea/GalaxeaManipSim
- https://docs.galaxea-ai.com/Guide/R1Pro/isaacsim/r1pro_isaacsim_guide/
