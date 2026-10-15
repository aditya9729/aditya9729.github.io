# 14 · AgileX: learn coordination, not two isolated reaches

### Use the vendor embodiment, not a similar-looking ALOHA
ALOHA names a family of bimanual systems. The DeepMind ALOHA 2 model is not interchangeable with AgileX's hardware. This course uses the actual MJCF from AgileX's published mobile_aloha_sim legacy source, pinned to a specific commit. Its arm geometry, joint naming, gripper actuation, and surrounding scene are retained.

Despite the repository name, this particular MJCF has a fixed chassis. We do not claim mobile AgileX control from it. The model also includes scene objects, so qpos contains more than the arm joints. Its four finger actuator coordinates are slide distances, while twelve arm coordinates are angles. A flat numeric action array is not a sufficient description of that interface.

The first bimanual learning problem coordinates both wrists toward a paired goal. A central actor receives both arms' state and emits one concatenated action. We add a relative-wrist term to make the coordination objective explicit. This is an introductory coordination task, not shared-object grasping. Physical object manipulation requires relevant contacts and a task definition such as lift, handoff, or insertion.

### How to tell whether the arms are actually coordinating
Two independent policies can each reduce a local wrist error while producing an incompatible relative pose. A central policy may learn useful dependencies, but centralized input alone does not guarantee coordination. Measure both per-arm error and a relative-geometry metric. If the task uses a shared object, log object pose/tilt, both grasp states, and success, not only wrist proximity.

Exploration also scales with action dimension. Independent Gaussian noise in twelve joints can produce much larger aggregate movement than in seven joints with the same per-coordinate standard deviation. Compare total action norm, per-joint scales, and end-effector motion before transferring a single-arm exploration setting. Entropy coefficients and target entropy conventions need the same care.

Ablations should distinguish evaluation-time command masking from retraining a policy under a restricted action space. The first measures sensitivity of a learned policy to an intervention. The second measures what a different learner can achieve under a different control contract. Neither should be mislabeled as a physically welded arm unless the simulator model was changed accordingly.

### Derivation: local errors and a coupling term
Let $e_L=x_L-g_L$, $e_R=x_R-g_R$, and desired relative vector $d^*=g_L-g_R$. One coordination loss is
$$\ell=\|e_L\|^2+\|e_R\|^2+\lambda\|(x_L-x_R)-d^*\|^2.$$
The relative term couples both arms' gradients even without a shared object. In a contact task the dynamics add coupling as well. These are distinct sources of dependence.

A central policy writes $\pi_\theta(a_L,a_R\mid o_L,o_R,g)$. A diagonal Gaussian has conditionally independent sampling noise given its means, yet each mean can depend on both arms' observations. Conditional noise factorization therefore does not imply two independent controllers.

### A coordinated target can expose local success inflation
Suppose each wrist is within 4 cm of its goal but the errors point in opposite directions. Their relative error may be 8 cm. Reporting only mean wrist error hides that mismatch. With a shared tray, such disagreement can also create tilt or internal contact forces.

Use the sketch to move one wrist while keeping the other fixed. Inspect both local errors and relative error. It is a geometry illustration; actual trajectories are generated separately by the MuJoCo notebook.



## Your experiment

Load official AgileX geometry, inspect named joint indexing and finger units, and step the paired-goal environment. Compare joint action norms with Franka. Execute an optimizer update, then design a two-arm observation/action ablation with separate per-arm, relative-pose, and task-success metrics.

## A failure to watch for

Do not label DeepMind ALOHA 2 as AgileX or infer a working mobile base from a repository name.

## Explain it back

Can a diagonal-Gaussian centralized policy still coordinate the arms?

<details><summary>Reasoning</summary>

Yes. Its action means can depend on both arms and the shared objective. Conditional independence of sampling noise is not independence of the decision functions.

</details>

## Primary sources

- https://github.com/agilexrobotics/mobile_aloha_sim
- https://github.com/agilexrobotics/aloha-agilex
