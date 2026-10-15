"""Physics-backed course: native Gymnasium and robosuite 1.5.2 via Gymnasium.

Run with MUJOCO_GL=egl on headless Linux with EGL, or your platform's GL backend.
No simulator assets or third-party source are redistributed in the course pack.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import time
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch

TASKS = {
    'reacher': {'backend':'gymnasium', 'id':'Reacher-v5'},
    'pusher': {'backend':'gymnasium', 'id':'Pusher-v5'},
    'lift': {'backend':'robosuite', 'id':'Lift', 'robots':'Panda'},
    'bimanual': {'backend':'robosuite', 'id':'TwoArmLift', 'robots':['Panda','Panda']},
    'mobile': {'backend':'robosuite', 'id':'TwoArmLift', 'robots':'Tiago'},
}

class SuiteGymEnv(gym.Env):
    """State-only adapter; timeout != terminal, object-state is privileged.

    Uses robosuite's BASIC composite controller. Arm action units are normalized
    OSC pose increments, not direct torques. Mobile actions additionally command
    torso/head/base. Action splits are read from the constructed controller.
    """
    metadata = {'render_modes':['rgb_array'], 'render_fps':20}

    def __init__(self, task='lift', render_mode=None, horizon=200,
                 reward_shaping=True, seed=7, width=480, height=360):
        import robosuite as suite
        from robosuite.controllers import load_composite_controller_config
        if task not in ('lift','bimanual','mobile'):
            raise ValueError(task)
        self.task, self.render_mode = task, render_mode
        self.width, self.height = width, height
        self._renderer=None
        cfg=TASKS[task]
        robot_names=cfg['robots'] if isinstance(cfg['robots'],list) else [cfg['robots']]
        configs=[load_composite_controller_config(robot=r) for r in robot_names]
        if any(c['type']!='BASIC' for c in configs):
            raise RuntimeError('These course tasks require robot-specific BASIC configs.')
        kwargs=dict(env_name=cfg['id'], robots=cfg['robots'], controller_configs=configs,
                    has_renderer=False, has_offscreen_renderer=False,
                    use_camera_obs=False, use_object_obs=True, control_freq=20,
                    horizon=horizon, ignore_done=False, hard_reset=False,
                    reward_shaping=reward_shaping, seed=seed)
        if task=='bimanual': kwargs['env_configuration']='opposed'
        self.suite=suite.make(**kwargs)
        obs=self.suite.reset()
        self.keys=['object-state']+[f'robot{i}_proprio-state' for i in range(len(self.suite.robots))]
        sample=self._flatten(obs)
        low,high=self.suite.action_spec
        self.action_space=gym.spaces.Box(low.astype(np.float32),high.astype(np.float32))
        self.observation_space=gym.spaces.Box(-np.inf,np.inf,sample.shape,np.float32)
        self.action_parts={}
        offset=0
        for i,robot in enumerate(self.suite.robots):
            for part,(start,end) in robot._action_split_indexes.items():
                self.action_parts[f'robot{i}/{part}']=(offset+start,offset+end)
            offset+=robot.action_dim
        self.steps=0

    def _flatten(self,obs):
        return np.concatenate([np.asarray(obs[k]).ravel() for k in self.keys]).astype(np.float32)

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        if seed is not None:
            # robosuite 1.5 uses Generator objects, not np.random.seed alone.
            self.suite.seed=seed
            self.suite.rng=np.random.default_rng(seed)
            sampler=getattr(self.suite,'placement_initializer',None)
            if sampler is not None:
                sampler.rng=self.suite.rng
                for child in getattr(sampler,'samplers',{}).values():
                    child.rng=self.suite.rng
        self.steps=0
        obs=self.suite.reset()
        return self._flatten(obs),self._info()

    def _info(self):
        return {'is_success':bool(self.suite._check_success()),
                'contacts':int(self.suite.sim.data.ncon), 'steps':self.steps}

    def step(self,action):
        action=np.asarray(action,dtype=np.float32)
        if action.shape!=self.action_space.shape or not np.isfinite(action).all():
            raise ValueError('Expected one finite bounded action vector.')
        obs,reward,done,info=self.suite.step(np.clip(action,self.action_space.low,self.action_space.high))
        self.steps+=1
        # robosuite's done is its fixed horizon, NOT task success.
        info.update(self._info())
        return self._flatten(obs),float(reward),False,bool(done),info

    def render(self):
        if self.render_mode!='rgb_array':
            raise RuntimeError('Construct with render_mode="rgb_array" first.')
        import mujoco
        if self._renderer is None:
            self._renderer=mujoco.Renderer(self.suite.sim.model._model,height=self.height,width=self.width)
        camera=mujoco.MjvCamera()
        camera.lookat[:]=[0,0,.8]
        camera.distance=3.2 if self.task=='mobile' else 2.8
        camera.azimuth=135;camera.elevation=-22
        self._renderer.update_scene(self.suite.sim.data._data,camera=camera)
        return self._renderer.render().copy()

    def close(self):
        if self._renderer is not None: self._renderer.close()
        self.suite.close()

class ActionParts(gym.ActionWrapper):
    """Expose selected controller parts only; other commands are zero.

    This masks COMMANDS. It does not weld/immobilize joints or the physical base.
    Zero absolute torso/head targets may not mean hold-current; inspect configs.
    """
    def __init__(self,env,include=None):
        super().__init__(env)
        parts=env.unwrapped.action_parts
        selected=list(parts) if include is None else list(include)
        unknown=set(selected)-set(parts)
        if unknown: raise ValueError(f'Unknown parts {unknown}; available: {parts}')
        self.indices=np.array([i for name in selected for i in range(*parts[name])],dtype=int)
        self.action_space=gym.spaces.Box(env.action_space.low[self.indices],env.action_space.high[self.indices])
    def action(self,action):
        full=np.zeros(self.env.action_space.shape,np.float32)
        full[self.indices]=action
        return full

class ResidualAction(gym.ActionWrapper):
    """Frozen base policy plus bounded residual, not an exact ResFiT reproduction."""
    def __init__(self,env,base_policy,scale=.2):
        super().__init__(env)
        self.base_policy,self.scale,self.last_obs=base_policy,float(scale),None
    def reset(self,**kwargs):
        self.last_obs,info=self.env.reset(**kwargs)
        return self.last_obs,info
    def action(self,residual):
        base,_=self.base_policy.predict(self.last_obs,deterministic=True)
        return np.clip(base+self.scale*np.asarray(residual),self.env.action_space.low,self.env.action_space.high)
    def step(self,action):
        self.last_obs,reward,terminated,truncated,info=self.env.step(self.action(action))
        return self.last_obs,reward,terminated,truncated,info

class SuccessGuidedReset(gym.Wrapper):
    """SGS-shaped finite reset sampler applied to actual robot episodes.

    Isolated adaptation, not a reproduction of the SGS paper. The underlying
    task must expose info['is_success']. Evaluation should use the unwrapped,
    original reset distribution, not this curriculum's training seeds.
    """
    def __init__(self,env,seeds=range(100,108),target=.5,kappa=4.,temperature=1.):
        super().__init__(env)
        self.seeds=list(seeds);self.history=[[] for _ in self.seeds]
        self.target,self.kappa,self.temperature=target,kappa,temperature
        self.active=None;self.ever_success=False;self.sampler_rng=np.random.default_rng(7)
    def probabilities(self):
        p=np.array([np.mean(h[-20:]) if h else .5 for h in self.history])
        logits=(self.kappa*self.target*np.log(p+1e-3)+
                self.kappa*(1-self.target)*np.log(1-p+1e-3))/self.temperature
        weights=np.exp(logits-logits.max())
        return weights/weights.sum()
    def reset(self,*,seed=None,options=None):
        if seed is not None: self.sampler_rng=np.random.default_rng(seed)
        self.active=int(self.sampler_rng.choice(len(self.seeds),p=self.probabilities()))
        self.ever_success=False
        return self.env.reset(seed=self.seeds[self.active],options=options)
    def step(self,action):
        obs,r,terminated,truncated,info=self.env.step(action)
        if 'is_success' not in info: raise ValueError('SGS needs an explicit task success metric.')
        self.ever_success |= bool(info['is_success'])
        if terminated or truncated: self.history[self.active].append(float(self.ever_success))
        return obs,r,terminated,truncated,info

def make_env(task='reacher',seed=7,render_mode=None,horizon=None,reward_shaping=True):
    cfg=TASKS[task]
    if cfg['backend']=='gymnasium':
        kwargs={'render_mode':render_mode}
        if render_mode=='rgb_array': kwargs.update(width=480,height=360)
        if horizon is not None: kwargs['max_episode_steps']=horizon
        env=gym.make(cfg['id'],**kwargs)
    else:
        env=SuiteGymEnv(task,render_mode,horizon or 200,reward_shaping,seed)
    env.action_space.seed(seed)
    return env

def build_agent(algo,env,seed=7):
    """Reference implementations; source-inspection exercises live in notebook 05."""
    from stable_baselines3 import PPO,A2C,TD3,SAC
    from stable_baselines3.common.noise import NormalActionNoise
    common=dict(policy='MlpPolicy',env=env,seed=seed,device='cpu',verbose=0,
                learning_rate=3e-4,gamma=.99,policy_kwargs={'net_arch':[128,128]})
    if algo=='PPO': return PPO(**common,n_steps=1024,batch_size=256,n_epochs=10,gae_lambda=.95)
    if algo=='A2C': return A2C(**common,n_steps=32,gae_lambda=.95)
    off=dict(buffer_size=200_000,learning_starts=1000,batch_size=256,train_freq=1,gradient_steps=1,tau=.005)
    if algo=='SAC': return SAC(**common,**off,ent_coef='auto')
    if algo=='TD3':
        noise=NormalActionNoise(np.zeros(env.action_space.shape),.1*np.ones(env.action_space.shape))
        return TD3(**common,**off,action_noise=noise,policy_delay=2,target_policy_noise=.2,target_noise_clip=.5)
    raise ValueError(algo)

def evaluate(policy,task,seeds=range(10001,10011),horizon=None,env_factory=None):
    """Identical reset seeds for comparisons; no cross-environment return ranking."""
    env=env_factory() if env_factory else make_env(task,horizon=horizon)
    rows=[]
    for seed in seeds:
        obs,info=env.reset(seed=int(seed));done=False;ret=0.;steps=0;success=False;contacts=0;dist=[]
        env.action_space.seed(int(seed))
        while not done:
            action=env.action_space.sample() if policy is None else policy.predict(obs,deterministic=True)[0]
            obs,r,terminated,truncated,info=env.step(action)
            ret+=r;steps+=1;done=terminated or truncated
            success |= bool(info.get('is_success',False));contacts=max(contacts,info.get('contacts',0))
            if task=='reacher': dist.append(float(np.linalg.norm(obs[-2:])))
        row=dict(seed=int(seed),return_=ret,steps=steps,ever_success=success,max_contacts=contacts)
        if dist: row.update(mean_distance=float(np.mean(dist)),final_distance=dist[-1])
        rows.append(row)
    env.close()
    returns=np.array([r['return_'] for r in rows])
    return {'episodes':rows,'return_mean':float(returns.mean()),'return_std':float(returns.std()),
            'ever_success_rate':float(np.mean([r['ever_success'] for r in rows])) if task not in ('reacher','pusher') else None}

def train(algo='PPO',task='reacher',steps=100_000,seed=7,out='physics-runs'):
    from stable_baselines3.common.monitor import Monitor
    torch.set_num_threads(1)
    folder=Path(out);folder.mkdir(parents=True,exist_ok=True)
    prefix=folder/f'{task}_{algo.lower()}_{seed}'
    env=Monitor(make_env(task,seed),filename=str(prefix))
    agent=build_agent(algo,env,seed)
    started=time.perf_counter();agent.learn(total_timesteps=steps)
    seconds=time.perf_counter()-started
    agent.save(str(prefix));env.close()
    result={'algorithm':algo,'task':task,'seed':seed,'requested_steps':steps,
            'actual_steps':agent.num_timesteps,'seconds':seconds,
            'versions':{p:importlib.metadata.version(p) for p in ['gymnasium','mujoco','robosuite','stable-baselines3','torch']},
            'evaluation':evaluate(agent,task)}
    prefix.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='evaluation'}),flush=True)
    return agent,result

def record(task,path,policy=None,seed=10001,steps=160):
    import imageio.v2 as imageio
    env=make_env(task,seed,render_mode='rgb_array')
    obs,_=env.reset(seed=seed);env.action_space.seed(seed)
    frames=[env.render()]
    for _ in range(steps):
        a=env.action_space.sample() if policy is None else policy.predict(obs,deterministic=True)[0]
        obs,_,terminated,truncated,_=env.step(a);frames.append(env.render())
        if terminated or truncated: break
    fps=env.metadata.get('render_fps',20)
    imageio.mimsave(path,frames,fps=fps)
    env.close()
    return {'file':str(path),'frames':len(frames),'fps':fps,'policy':'random' if policy is None else 'learned'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--algo',choices=['A2C','PPO','TD3','SAC'],default='PPO')
    p.add_argument('--task',choices=TASKS,default='reacher');p.add_argument('--steps',type=int,default=100_000)
    p.add_argument('--seed',type=int,default=7);p.add_argument('--out',default='physics-runs')
    args=p.parse_args();train(args.algo,args.task,args.steps,args.seed,args.out)
