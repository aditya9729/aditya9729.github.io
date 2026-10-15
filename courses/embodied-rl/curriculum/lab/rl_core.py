"""Readable RL algorithms for Gymnasium. CPU first; CUDA is an explicit option.

Original educational implementations. No simulator assets, SB3 algorithms, or
paper-specific throughput recipes are hidden inside these functions.
"""
from __future__ import annotations
import argparse, copy, json, math, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal, Categorical

def mlp(inputs, outputs, width=64):
    return nn.Sequential(nn.Linear(inputs,width),nn.Tanh(),nn.Linear(width,width),nn.Tanh(),nn.Linear(width,outputs))

def discounted_returns(rewards, gamma=.99, bootstrap=0.):
    out=np.zeros(len(rewards),np.float64);future=float(bootstrap)
    for t in reversed(range(len(rewards))):
        future=float(rewards[t])+gamma*future;out[t]=future
    return out

def bellman_backup(P,R,V,gamma=.9,policy=None):
    """P[s,a,s'] and expected immediate reward R[s,a]; absorbing terminals explicit."""
    Q=R+gamma*np.einsum('sak,k->sa',P,V)
    return (Q.max(1) if policy is None else (policy*Q).sum(1)),Q

def value_iteration(P,R,gamma=.9,tol=1e-10):
    V=np.zeros(len(P));history=[V.copy()]
    for _ in range(10000):
        new,Q=bellman_backup(P,R,V,gamma)
        history.append(new.copy())
        if np.max(np.abs(new-V))<tol:return new,Q,history
        V=new
    raise RuntimeError('Check discount and transition matrix.')

def grasp_mdp():
    """Symbolic four-state decision problem, NOT a physical robot simulation.

    State 0 approach, 1 aligned, 2 grasped, 3 absorbing completion. Two actions:
    0 advance carefully, 1 rush. Goal-entry reward +10, no repeated terminal reward.
    """
    P=np.zeros((4,2,4));R=np.zeros((4,2))
    P[0,0,1]=1;R[0,0]=-.5
    P[0,1,1]=.5;P[0,1,0]=.5;R[0,1]=-.1
    P[1,0,2]=1;R[1,0]=-.5
    P[1,1,2]=.3;P[1,1,0]=.7;R[1,1]=-.1
    P[2,:,3]=1;R[2,:]=10
    P[3,:,3]=1
    return P,R

@torch.no_grad()
def gae(rewards,values,next_values,terminated,truncated,gamma=.99,lam=.95):
    """Inputs [time] or [time,env]. Separate bootstrap and recursion masks."""
    advantage=torch.zeros_like(rewards);running=torch.zeros_like(rewards[-1])
    for t in reversed(range(len(rewards))):
        delta=rewards[t]+gamma*(~terminated[t]).float()*next_values[t]-values[t]
        running=delta+gamma*lam*(~(terminated[t]|truncated[t])).float()*running
        advantage[t]=running
    return advantage,advantage+values

class SquashedActor(nn.Module):
    def __init__(self,obs_dim,act_dim):
        super().__init__();self.net=mlp(obs_dim,2*act_dim);self.act_dim=act_dim
    def distribution(self,obs):
        mean,logstd=self.net(obs).chunk(2,-1)
        return Normal(mean,logstd.clamp(-5,2).exp())
    @staticmethod
    def log_jacobian(u):
        return 2*(math.log(2)-u-F.softplus(-2*u))
    def log_prob(self,obs,u):
        return (self.distribution(obs).log_prob(u)-self.log_jacobian(u)).sum(-1)
    def sample(self,obs,reparameterize=False):
        d=self.distribution(obs);u=d.rsample() if reparameterize else d.sample()
        return u,u.tanh(),(d.log_prob(u)-self.log_jacobian(u)).sum(-1)
    def forward(self,obs):return self.distribution(obs).mean.tanh()

class DeterministicActor(nn.Module):
    def __init__(self,obs_dim,act_dim):super().__init__();self.net=mlp(obs_dim,act_dim)
    def forward(self,obs):return self.net(obs).tanh()

class TwinQ(nn.Module):
    def __init__(self,obs_dim,act_dim):
        super().__init__();self.q1=mlp(obs_dim+act_dim,1);self.q2=mlp(obs_dim+act_dim,1)
    def forward(self,o,a):
        x=torch.cat([o,a],-1);return self.q1(x).squeeze(-1),self.q2(x).squeeze(-1)

class Replay:
    def __init__(self,obs_dim,act_dim,capacity=100000):
        self.capacity=capacity;self.pos=0;self.size=0
        self.arrays={k:np.zeros((capacity,*shape),np.float32) for k,shape in
                     [('obs',(obs_dim,)),('action',(act_dim,)),('reward',()),('next_obs',(obs_dim,)),('terminal',())]}
    def add(self,obs,action,reward,next_obs,terminal):
        for k,v in zip(self.arrays,(obs,action,reward,next_obs,terminal)):self.arrays[k][self.pos]=v
        self.pos=(self.pos+1)%self.capacity;self.size=min(self.size+1,self.capacity)
    def sample(self,batch=64,device='cpu',rng=None):
        rng=np.random.default_rng() if rng is None else rng
        ids=rng.integers(self.size,size=batch)
        return {k:torch.as_tensor(v[ids],device=device) for k,v in self.arrays.items()}

@torch.no_grad()
def polyak(target,source,tau=.005):
    for t,s in zip(target.parameters(),source.parameters()):t.lerp_(s,tau)

def ppo_objective(logp,old_logp,advantages,clip=.2):
    ratio=(logp-old_logp).exp()
    objective=torch.minimum(ratio*advantages,ratio.clamp(1-clip,1+clip)*advantages)
    return -objective.mean(),ratio

def td3_update(actor,critic,target_actor,target_critic,actor_opt,critic_opt,batch,update,
               gamma=.99,tau=.005,delay=2,target_noise=.2,noise_clip=.5):
    o,a,r,n,d=(batch[k] for k in ('obs','action','reward','next_obs','terminal'))
    with torch.no_grad():
        noise=(torch.randn_like(a)*target_noise).clamp(-noise_clip,noise_clip)
        next_action=(target_actor(n)+noise).clamp(-1,1)
        q1,q2=target_critic(n,next_action)
        target=r+gamma*(1-d)*torch.minimum(q1,q2)
    q1,q2=critic(o,a);loss=F.mse_loss(q1,target)+F.mse_loss(q2,target)
    critic_opt.zero_grad();loss.backward();critic_opt.step()
    actor_loss=None
    if update%delay==0:
        for p in critic.parameters():p.requires_grad_(False)
        actor_loss=-critic(o,actor(o))[0].mean()
        actor_opt.zero_grad();actor_loss.backward();actor_opt.step()
        for p in critic.parameters():p.requires_grad_(True)
        polyak(target_actor,actor,tau);polyak(target_critic,critic,tau)
    return {'critic_loss':float(loss.detach()),'actor_loss':None if actor_loss is None else float(actor_loss.detach())}

def sac_update(actor,critic,target_critic,actor_opt,critic_opt,log_alpha,alpha_opt,batch,
               gamma=.99,tau=.005,target_entropy=None):
    o,a,r,n,d=(batch[k] for k in ('obs','action','reward','next_obs','terminal'))
    alpha=log_alpha.exp().detach()
    with torch.no_grad():
        _,na,nlp=actor.sample(n,True);q1,q2=target_critic(n,na)
        target=r+gamma*(1-d)*(torch.minimum(q1,q2)-alpha*nlp)
    q1,q2=critic(o,a);qloss=F.mse_loss(q1,target)+F.mse_loss(q2,target)
    critic_opt.zero_grad();qloss.backward();critic_opt.step()
    for p in critic.parameters():p.requires_grad_(False)
    _,sample,lp=actor.sample(o,True);q1,q2=critic(o,sample)
    aloss=(alpha*lp-torch.minimum(q1,q2)).mean()
    actor_opt.zero_grad();aloss.backward();actor_opt.step()
    for p in critic.parameters():p.requires_grad_(True)
    target_entropy=-actor.act_dim if target_entropy is None else target_entropy
    alpha_loss=-(log_alpha*(lp.detach()+target_entropy)).mean()
    alpha_opt.zero_grad();alpha_loss.backward();alpha_opt.step()
    polyak(target_critic,critic,tau)
    return {'critic_loss':float(qloss.detach()),'actor_loss':float(aloss.detach()),'alpha':float(log_alpha.exp().detach())}

class NormalizedAction:
    """Factory returns a real Gymnasium ActionWrapper; policy actions stay [-1,1]."""
    @staticmethod
    def wrap(env):
        import gymnasium as gym
        class Wrapper(gym.ActionWrapper):
            def __init__(self,e):
                super().__init__(e);self.low=e.action_space.low;self.high=e.action_space.high
                assert np.isfinite(self.low).all() and np.isfinite(self.high).all()
                self.action_space=gym.spaces.Box(-1,1,e.action_space.shape,np.float32)
            def action(self,a):return self.low+(np.clip(a,-1,1)+1)*.5*(self.high-self.low)
        return Wrapper(env)

def train_continuous(env,algo='PPO',steps=4096,seed=7,device='cpu',rollout=256,warmup=128):
    """Real Gymnasium loop; observation normalization intentionally left as an exercise.

    Saves raw pre-reset next observations and masks only true terminal transitions
    in Bellman targets. No automatic reset / vector API ambiguity is hidden here.
    """
    torch.manual_seed(seed);rng=np.random.default_rng(seed);torch.set_num_threads(1)
    env=NormalizedAction.wrap(env);env.action_space.seed(seed)
    obs,_=env.reset(seed=seed);od=len(obs);ad=env.action_space.shape[0]
    actor=(DeterministicActor(od,ad) if algo=='TD3' else SquashedActor(od,ad)).to(device)
    opt=torch.optim.Adam(actor.parameters(),lr=3e-4)
    log=[];diagnostics=[];episode_return=0.;updates=0;began=time.perf_counter()
    if algo in ('AC','PPO'):
        value=mlp(od,1).to(device);vopt=torch.optim.Adam(value.parameters(),lr=3e-4)
        for start in range(0,steps,rollout):
            data={k:[] for k in ('obs','u','lp','v','nv','r','te','tr')}
            for _ in range(min(rollout,steps-start)):
                o=torch.as_tensor(obs,dtype=torch.float32,device=device)
                with torch.no_grad():u,a,lp=actor.sample(o);v=value(o).squeeze(-1)
                final,r,te,tr,info=env.step(a.cpu().numpy())
                with torch.no_grad():nv=value(torch.as_tensor(final,dtype=torch.float32,device=device)).squeeze(-1)
                for k,x in zip(data,(o,u,lp,v,nv,torch.tensor(r,dtype=torch.float32,device=device),torch.tensor(te,device=device),torch.tensor(tr,device=device))):data[k].append(x.detach().clone())
                obs=final;episode_return+=r
                if te or tr:
                    log.append({'step':start+len(data['r']),'return':episode_return});episode_return=0.;obs,_=env.reset()
            b={k:torch.stack(v) for k,v in data.items()}
            advantages,returns=gae(b['r'],b['v'],b['nv'],b['te'],b['tr'],lam=0. if algo=='AC' else .95)
            adv=(advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-8)
            for _ in range(4 if algo=='PPO' else 1):
                lp=actor.log_prob(b['obs'],b['u'])
                loss,_=ppo_objective(lp,b['lp'],adv) if algo=='PPO' else (-(lp*adv).mean(),None)
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(actor.parameters(),.5);opt.step()
                vloss=F.mse_loss(value(b['obs']).squeeze(-1),returns)
                vopt.zero_grad();vloss.backward();nn.utils.clip_grad_norm_(value.parameters(),.5);vopt.step();updates+=1
            with torch.no_grad():
                new_lp=actor.log_prob(b['obs'],b['u']);log_ratio=new_lp-b['lp'];ratio=log_ratio.exp()
                diagnostics.append({'step':start+len(b['r']),'actor_loss':float(loss.detach()),'value_loss':float(vloss.detach()),
                    'approx_kl':float(((ratio-1)-log_ratio).mean()),'clip_fraction':float(((ratio-1).abs()>.2).float().mean())})
    elif algo in ('TD3','SAC'):
        critic=TwinQ(od,ad).to(device);target=copy.deepcopy(critic);qopt=torch.optim.Adam(critic.parameters(),lr=3e-4)
        target_actor=copy.deepcopy(actor);replay=Replay(od,ad);log_alpha=nn.Parameter(torch.tensor(0.,device=device));alpha_opt=torch.optim.Adam([log_alpha],lr=3e-4)
        for t in range(steps):
            with torch.no_grad():
                if t<warmup:a=env.action_space.sample()
                elif algo=='SAC':a=actor.sample(torch.as_tensor(obs,dtype=torch.float32,device=device))[1].cpu().numpy()
                else:a=np.clip(actor(torch.as_tensor(obs,dtype=torch.float32,device=device)).cpu().numpy()+rng.normal(0,.1,ad),-1,1)
            final,r,te,tr,info=env.step(a);replay.add(obs,a,r,final,te);obs=final;episode_return+=r
            if te or tr:log.append({'step':t+1,'return':episode_return});episode_return=0.;obs,_=env.reset()
            if t>=warmup:
                batch=replay.sample(64,device,rng);updates+=1
                if algo=='SAC':last=sac_update(actor,critic,target,opt,qopt,log_alpha,alpha_opt,batch)
                else:last=td3_update(actor,critic,target_actor,target,opt,qopt,batch,updates)
                if updates==1 or updates%100==0 or t==steps-1:diagnostics.append({'step':t+1,**last})
    else:raise ValueError(algo)
    return actor,{'algorithm':algo,'steps':steps,'optimizer_updates':updates,
        'critic_optimizer_steps':updates,'actor_optimizer_steps':updates//2 if algo=='TD3' else updates,
        'temperature_optimizer_steps':updates if algo=='SAC' else 0,
        'episodes':log,'diagnostics':diagnostics,'seconds':time.perf_counter()-began,'seed':seed,'device':device}

def dqn_step(q,target,opt,batch,gamma=.99):
    o,a,r,n,d=batch
    with torch.no_grad():y=r+gamma*(1-d)*target(n).max(-1).values
    prediction=q(o).gather(1,a.long()[:,None]).squeeze(1)
    loss=F.smooth_l1_loss(prediction,y)
    opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(q.parameters(),10.);opt.step()
    return float(loss.detach())

def reinforce_cartpole(episodes=12,seed=7,device='cpu'):
    import gymnasium as gym
    torch.manual_seed(seed);env=gym.make('CartPole-v1',max_episode_steps=200)
    policy=mlp(4,2).to(device);opt=torch.optim.Adam(policy.parameters(),lr=.003);logs=[]
    for ep in range(episodes):
        obs,_=env.reset(seed=seed+ep);lps=[];rewards=[]
        for _ in range(200):
            d=Categorical(logits=policy(torch.as_tensor(obs,dtype=torch.float32,device=device)))
            a=d.sample();obs,r,te,tr,_=env.step(a.item());lps.append(d.log_prob(a));rewards.append(r)
            if te or tr:break
        G=torch.as_tensor(discounted_returns(rewards,gamma=1.),dtype=torch.float32,device=device)
        # No learned baseline in this first implementation; keep the estimator explicit.
        loss=-(torch.stack(lps)*G).sum()
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(policy.parameters(),10.);opt.step()
        logs.append(sum(rewards))
    env.close();return policy,logs

def evaluate_continuous(actor,env,seeds=range(10001,10006)):
    device=next(actor.parameters()).device;env=NormalizedAction.wrap(env);rows=[]
    for seed in seeds:
        o,_=env.reset(seed=seed);ret=0.;done=False
        while not done:
            with torch.no_grad():a=actor(torch.as_tensor(o,dtype=torch.float32,device=device)).cpu().numpy()
            o,r,te,tr,info=env.step(a);ret+=r;done=te or tr
        rows.append({'seed':seed,'return':ret,'success':info.get('is_success')})
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--algo',choices=['AC','PPO','TD3','SAC'],default='PPO')
    p.add_argument('--env',default='Reacher-v5');p.add_argument('--steps',type=int,default=10000)
    p.add_argument('--seed',type=int,default=7);p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--out',default='my-runs');args=p.parse_args()
    import gymnasium as gym
    env=gym.make(args.env);actor,history=train_continuous(env,args.algo,args.steps,args.seed,args.device)
    env.close();history['evaluation']=evaluate_continuous(actor,gym.make(args.env))
    out=Path(args.out);out.mkdir(exist_ok=True,parents=True);stem=out/f'{args.algo.lower()}_{args.seed}'
    torch.save(actor.state_dict(),stem.with_suffix('.pt'));stem.with_suffix('.json').write_text(json.dumps(history,indent=2)+'\n')
    print(json.dumps(history,indent=2))
