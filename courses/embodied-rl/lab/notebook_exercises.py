"""Executable worked cases. These are experiments, not mirrors of implementation tests."""
EXERCISES={
'probability':[
'''logits=torch.tensor([0.,0.,0.],requires_grad=True)
p=torch.softmax(logits,0); lp=p[1].log(); lp.backward()
expected=torch.tensor([0.,1.,0.])-p.detach()
print('probabilities:',p.detach(),'score gradient:',logits.grad)
assert torch.allclose(logits.grad,expected)
mean=torch.tensor(.2,requires_grad=True);std=.5;fixed_action=torch.tensor(.6)
Normal(mean,std).log_prob(fixed_action).backward()
print('Gaussian analytic / autograd:',(.6-.2)/std**2,mean.grad.item())
assert abs(mean.grad.item()-1.6)<1e-5''',
'''actor=core.SquashedActor(3,7)
obs=torch.zeros(4,3);u,a,lp=actor.sample(obs)
print('obs, latent, action, joint logp shapes:',obs.shape,u.shape,a.shape,lp.shape)
assert lp.shape==(4,)
assert torch.isfinite(actor.log_prob(obs,torch.full((4,7),10.))).all()'''],
'mdp':[
'''rewards=[-1.,-1.,10.]
print('returns:',core.discounted_returns(rewards,.9))
assert np.allclose(core.discounted_returns(rewards,.9),[6.2,8,10])
for dt in [.05,.025]:
    gamma=np.exp(-.2*dt)
    print('dt',dt,'gamma',gamma,'one-second discount',gamma**(1/dt))''',
'''import gymnasium as gym
env=gym.make('Reacher-v5',max_episode_steps=4)
obs,_=env.reset(seed=7)
print('observation:',obs.shape,'actions:',env.action_space)
for _ in range(4):obs,r,terminal,timeout,info=env.step(np.zeros(2,np.float32))
print('terminal / timeout:',terminal,timeout)
assert not terminal and timeout
env.close()'''],
'bellman':[
'''P,R=core.grasp_mdp();V,Q,history=core.value_iteration(P,R,.9)
print('optimal values:',V,'greedy actions:',Q.argmax(1))
policy=np.full((4,2),.5)
Ppi=np.einsum('sa,sak->sk',policy,P);rpi=(policy*R).sum(1)
exact=np.linalg.solve(np.eye(4)-.9*Ppi,rpi)
v=np.zeros(4)
for _ in range(400):v,_=core.bellman_backup(P,R,v,.9,policy)
print('fixed policy linear solve:',exact)
assert np.allclose(v,exact)
assert V[-1]==0.
plt.figure(figsize=(7,3));plt.plot(np.array(history));plt.xlabel('Bellman iteration');plt.ylabel('Value by state');plt.show()'''],
'td':[
'''v=4.;vn=6.;r=-1.;gamma=.9
target=r+gamma*vn
print('nonterminal target / TD residual:',target,target-v)
print('terminal target:',r,'timeout target:',target)
assert np.isclose(v+.25*(target-v),4.1)
rng=np.random.default_rng(7);noise=rng.normal(0,1,(2000,10))
mc=noise@(.9**np.arange(10));td=noise[:,0]
print('illustrative noisy-return variance:',mc.var(),'one-reward variance:',td.var())
plt.figure(figsize=(7,3));plt.hist(mc,bins=35,alpha=.5,label='10-step noisy sum');plt.hist(td,bins=35,alpha=.5,label='one reward');plt.legend();plt.show()''',
'''r=torch.tensor([1.,2.,100.]);v=torch.zeros(3);nv=torch.tensor([3.,4.,5.])
te=torch.tensor([False,False,True]);tr=torch.tensor([False,True,False])
A,returns=core.gae(r,v,nv,te,tr,gamma=.9,lam=.8)
print('advantages with timeout boundary:',A)
assert torch.allclose(A[1],torch.tensor(5.6))
assert not returns.requires_grad'''],
'dqn':[
'''import gymnasium as gym
env=gym.make('CartPole-v1',max_episode_steps=20);o,_=env.reset(seed=7)
rows=[];env.action_space.seed(7)
for _ in range(64):
    a=env.action_space.sample();n,r,te,tr,_=env.step(a);rows.append((o,a,r,n,te));o=n
    if te or tr:o,_=env.reset()
env.close()
batch=[torch.as_tensor(np.array(x),dtype=torch.float32) for x in zip(*rows)]
q=core.mlp(4,2);target=copy.deepcopy(q);opt=torch.optim.Adam(q.parameters(),lr=.001)
loss=core.dqn_step(q,target,opt,batch)
print('DQN loss on actual transitions:',loss)
assert all(p.grad is None for p in target.parameters())''',
'''errors=np.array([[2.,4.],[4.,2.]])
print('actionwise mean:',errors.mean(0),'mean selected max:',errors.max(1).mean())
assert errors.max(1).mean()>errors.mean()
# Your turn: construct separate online selection and target evaluation arrays.'''],
'reinforce':[
'''theta=torch.tensor(0.,requires_grad=True);p=theta.sigmoid();J=3*p-1;J.backward()
print('exact bandit policy gradient:',theta.grad.item());assert theta.grad.item()==.75
torch.manual_seed(7);a=torch.bernoulli(torch.full((100000,),.5));reward=3*a-1
score=a-.5
print('Monte Carlo gradient:',(reward*score).mean().item())
print('Variance, no baseline / baseline .5:',(reward*score).var().item(),((reward-.5)*score).var().item())''',
'''policy,episode_returns=core.reinforce_cartpole(episodes=12,device=DEVICE)
print('Executed REINFORCE episode returns:',episode_returns)
plt.figure(figsize=(7,3));plt.plot(episode_returns,'o-');plt.xlabel('Episode');plt.ylabel('Return');plt.title('Short execution run, not a solved-task claim');plt.show()'''],
'actor-critic':[
'''delta=torch.tensor([1.,-.5,2.]);v=torch.zeros(3);nv=torch.zeros(3)
no=torch.zeros(3,dtype=torch.bool)
A,_=core.gae(delta,v,nv,no,no,gamma=.9,lam=.8)
explicit=1+.72*(-.5)+.72**2*2
print('recursive / explicit:',A[0].item(),explicit)
assert abs(A[0].item()-explicit)<1e-6
reset=torch.tensor([True,False,False]);A2,_=core.gae(delta,v,nv,no,reset,.9,.8)
assert A2[0]==1.''',
'''import gymnasium as gym
env=gym.make('Reacher-v5',max_episode_steps=32)
actor,run=core.train_continuous(env,'AC',steps=128,rollout=64,device=DEVICE)
print(run);assert run['optimizer_updates']>0;env.close()'''],
'ppo':[
'''ratios=torch.tensor([1.5,1.5,.5,.5],requires_grad=True)
advantages=torch.tensor([2.,-2.,2.,-2.])
obj=torch.minimum(ratios*advantages,ratios.clamp(.8,1.2)*advantages)
obj.sum().backward()
print('objectives:',obj.detach(),'derivatives:',ratios.grad)
assert torch.allclose(obj.detach(),torch.tensor([2.4,-3.,1.,-1.6]))
assert torch.allclose(ratios.grad,torch.tensor([0.,-2.,2.,0.]))''',
'''# Wire your editable function definition into the original training loop.
core.ppo_objective=ppo_objective
import gymnasium as gym
env=gym.make('Reacher-v5',max_episode_steps=32)
actor,run=core.train_continuous(env,'PPO',steps=256,rollout=64,device=DEVICE)
print(run);assert run['optimizer_updates']==16;env.close()'''],
'ddpg':[
'''theta=torch.tensor(-.2,requires_grad=True)
Q=-(theta-.4)**2;(-Q).backward()
print('loss gradient:',theta.grad.item(),'Q ascent direction:',-theta.grad.item())
assert np.isclose(-theta.grad.item(),1.2)
actor=core.DeterministicActor(3,2);critic=core.TwinQ(3,2)
for p in critic.parameters():p.requires_grad_(False)
loss=-critic(torch.zeros(4,3),actor(torch.zeros(4,3)))[0].mean()
loss.backward()
assert any(p.grad is not None for p in actor.parameters())
assert all(p.grad is None for p in critic.parameters())
print('Frozen critic parameters retained the action-gradient path.')'''],
'td3':[
'''target=-1+.99*min(6.,4.)
print('TD3 target:',target);assert np.isclose(target,2.96)
import gymnasium as gym
core.td3_update=td3_update
env=gym.make('Reacher-v5',max_episode_steps=32)
actor,run=core.train_continuous(env,'TD3',steps=160,warmup=32,device=DEVICE)
print(run);assert run['optimizer_updates']==128;env.close()'''],
'sac':[
'''target=1+.99*(2-.2*(-1.5))
print('soft Bellman target:',target);assert np.isclose(target,3.277)
actor=core.SquashedActor(3,2);o=torch.zeros(5,3)
u,a,lp=actor.sample(o,True);loss=(.2*lp-a.sum(-1)).mean();loss.backward()
assert all(torch.isfinite(p.grad).all() for p in actor.parameters() if p.grad is not None)
print('Reparameterized squashed policy gradients are finite.')''',
'''import gymnasium as gym
core.sac_update=sac_update
env=gym.make('Reacher-v5',max_episode_steps=32)
actor,run=core.train_continuous(env,'SAC',steps=160,warmup=32,device=DEVICE)
print(run);assert run['optimizer_updates']==128;env.close()'''],
'offline':[
'''base=.8;beta=.1
print('residual action interval:',base-beta,base+beta)
assert base-beta>0
obs=torch.randn(32,4);demo=torch.tanh(torch.randn(32,2))
policy=core.DeterministicActor(4,2);opt=torch.optim.Adam(policy.parameters(),lr=.001)
for _ in range(10):
    loss=(policy(obs)-demo).square().mean();opt.zero_grad();loss.backward();opt.step()
print('Supervised action regression loss:',loss.item(),'This is BC arithmetic, not RL.')''',
'''from physics_lab import ResidualAction
import gymnasium as gym
class FrozenBase:
    def predict(self,o,deterministic=True):return np.zeros(2,np.float32),None
env=ResidualAction(gym.make('Reacher-v5',max_episode_steps=32),FrozenBase(),scale=.2)
actor,run=core.train_continuous(env,'SAC',steps=128,warmup=32,device=DEVICE)
print('Residual SAC updates:',run['optimizer_updates']);env.close()'''],
'franka':[
'''ensure_robot_assets()
from embodiment_env import RobotReach
env=RobotReach('franka',horizon=24);obs,info=env.reset(seed=7)
print('Robot joint names:',env.joint_names,'observation:',obs.shape,'actions:',env.action_space.shape)
print('Native actuator ranges:',env.model.actuator_ctrlrange)
print('Control contract:',info['control_note'])
actor,run=core.train_continuous(env,'PPO',steps=64,rollout=32,device=DEVICE)
print(run);env.close()'''],
'agilex':[
'''ensure_robot_assets()
from embodiment_env import RobotReach
env=RobotReach('agilex',horizon=24);obs,info=env.reset(seed=7)
print('Actual vendor joints:',env.joint_names)
print('Full model nq:',env.model.nq,'policy action dim:',env.action_space.shape)
print('Fixed chassis; native finger controls are meters, arm controls radians.')
actor,run=core.train_continuous(env,'PPO',steps=64,rollout=32,device=DEVICE)
print(run);env.close()'''],
'r1-pro':[
'''ensure_robot_assets()
from embodiment_env import RobotReach
import mujoco
for planar in (False,True):
    env=RobotReach('r1-pro',planar_base=planar,horizon=16);obs,info=env.reset(seed=7)
    print('Ideal planar extension:',planar,'action dim:',env.action_space.shape,'contract:',info['control_note'])
    print('Structured command slices:',env.action_parts)
    Jp=np.zeros((3,env.model.nv));Jr=np.zeros((3,env.model.nv))
    mujoco.mj_jacBody(env.model,env.data,Jp,Jr,env.eef_ids[0])
    print('Wrist translational Jacobian singular values:',np.linalg.svd(Jp[:,env.vidx],compute_uv=False))
    actor,run=core.train_continuous(env,'PPO',steps=32,rollout=16,device=DEVICE)
    print('Real model updates:',run['optimizer_updates']);env.close()'''],
'grpo-flow':[
'''r=torch.tensor([0.,0.,1.,1.]);A=(r-r.mean())/(r.std(unbiased=False)+1e-8)
print('group advantages:',A);assert torch.allclose(A,torch.tensor([-1.,-1.,1.,1.]))
equal=torch.zeros(4);assert ((equal-equal.mean())/(equal.std(unbiased=False)+1e-8)==0).all()
theta=torch.tensor(0.,requires_grad=True);oldp=torch.tensor(.5);newp=theta.sigmoid()
actions=torch.tensor([0.,1.,0.,1.]);returns=torch.tensor([-1.,2.,-1.,2.])
advantages=(returns-returns.mean())/(returns.std(unbiased=False)+1e-8)
new_logp=actions*newp.log()+(1-actions)*(1-newp).log()
old_logp=torch.full((4,),float(np.log(.5)))
loss,ratio=core.ppo_objective(new_logp,old_logp,advantages);loss.backward()
print('Grouped bandit clipped-ratio gradient:',theta.grad.item(),'Not robotic GRPO.')''',
'''delta=torch.tensor([-2.,-.2,.2,2.],requires_grad=True)
guided=delta.clamp(-.5,.5);guided.sum().backward()
print('Isolated clipped guidance derivative:',delta.grad)
assert torch.equal(delta.grad,torch.tensor([0.,1.,1.,0.]))
print('This is only one gradient mechanism, not a QF3 flow actor.')'''],
'curricula-rewards':[
'''p=np.array([.05,.5,.95]);kappa=4.;target=.5
w=(p+1e-3)**(kappa*target)*(1-p+1e-3)**(kappa*(1-target));w/=w.sum()
print('SGS-shaped reset weights:',w);assert w.argmax()==1
phi=np.array([0.,.2,.7,.7]);gamma=.9
F=gamma*phi[1:]-phi[:-1]
total=F@(gamma**np.arange(len(F)))
boundary=-phi[0]+gamma**(len(phi)-1)*phi[-1]
print('discounted potential sum / boundary term:',total,boundary)
assert np.isclose(total,boundary)
print('Absolute progress .7 repeated ten steps pays:',10*.7)''',
'''ensure_robot_assets()
from physics_lab import SuccessGuidedReset
from embodiment_env import RobotReach
env=SuccessGuidedReset(RobotReach('franka',horizon=4),seeds=range(100,104))
for ep in range(4):
    o,_=env.reset()
    for _ in range(4):o,r,te,tr,info=env.step(np.zeros(env.action_space.shape,np.float32))
print('Actual robot success histories:',env.history,'weights:',env.probabilities())
env.close()'''],
'systems-capstone':[
'''N,H,E,B=8,256,4,256
transitions=N*H;updates=E*int(np.ceil(transitions/B))
print('rollout transitions:',transitions,'PPO minibatch steps:',updates)
assert transitions==2048 and updates==32
experiment={'hypothesis':'Torso control improves held-out paired-wrist reach success',
 'embodiment':'official R1 Pro geometry + course controls',
 'comparison':['arms only','torso + arms'], 'training_seeds':[7,11,23],
 'evaluation_seeds':list(range(10001,10021)), 'transitions_per_seed':100000,
 'metrics':['success with declared hold threshold','per-wrist distance','command effort','failure category'],
 'fixed':['reward','goal distribution','observation','controller','control frequency'],
 'boundary':'No validated wheel dynamics or hardware claim'}
print(json.dumps(experiment,indent=2))''']}

EDITABLE={'bellman':'bellman_backup','td':'discounted_returns','dqn':'dqn_step','actor-critic':'gae','ppo':'ppo_objective','td3':'td3_update','sac':'sac_update','reinforce':'reinforce_cartpole'}
