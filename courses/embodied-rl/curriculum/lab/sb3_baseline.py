"""Inspectable library baseline for the same Gymnasium / robot reaching tasks."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import gymnasium as gym
import torch
from stable_baselines3 import PPO,SAC,TD3,A2C,DDPG,DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from rl_core import NormalizedAction

def make_env(name):
    if name in ('franka','agilex','r1-pro','r1-pro-planar'):
        from embodiment_env import RobotReach
        return RobotReach('r1-pro' if name=='r1-pro-planar' else name,planar_base=name=='r1-pro-planar')
    return gym.make(name)

def run(algo='SAC',env_name='Reacher-v5',steps=10000,seed=7,out='runs/sb3',eval_episodes=5):
    torch.set_num_threads(1)
    root=Path(out)/f'{algo}-{env_name}-seed{seed}';root.mkdir(parents=True,exist_ok=True)
    env=make_env(env_name)
    continuous=isinstance(env.action_space,gym.spaces.Box)
    if algo in ('SAC','TD3','DDPG') and not continuous:
        env.close();raise ValueError(f'{algo} needs continuous actions. Use Reacher-v5 or a robot task.')
    if algo=='DQN' and not isinstance(env.action_space,gym.spaces.Discrete):
        env.close();raise ValueError('DQN needs discrete actions; use CartPole-v1.')
    if continuous:env=NormalizedAction.wrap(env)
    env=Monitor(env,str(root/'train.monitor.csv'))
    common=dict(policy='MlpPolicy',env=env,seed=seed,device='cpu',verbose=0,policy_kwargs=dict(net_arch=[64,64]))
    if algo in ('SAC','TD3','DDPG'):
        common.update(buffer_size=max(10000,steps),learning_starts=128,batch_size=64,train_freq=1,gradient_steps=1,learning_rate=3e-4)
        if algo in ('TD3','DDPG'):common['action_noise']=NormalActionNoise(np.zeros(env.action_space.shape[0]),.1*np.ones(env.action_space.shape[0]))
    elif algo=='PPO':common.update(n_steps=256,batch_size=64,n_epochs=4,learning_rate=3e-4)
    elif algo=='A2C':common.update(n_steps=32,learning_rate=3e-4)
    else:common.update(learning_starts=128,batch_size=64,buffer_size=max(10000,steps))
    model=globals()[algo](**common);began=time.perf_counter()
    try:
        model.learn(total_timesteps=steps);model.save(root/'policy')
    finally:env.close()
    evaluation=make_env(env_name)
    if continuous:evaluation=NormalizedAction.wrap(evaluation)
    returns=[];success=[]
    try:
        for ep in range(eval_episodes):
            obs,_=evaluation.reset(seed=10000+seed+ep);total=0.;hit=False
            for _ in range(1000):
                action,_=model.predict(obs,deterministic=True)
                obs,reward,terminal,truncated,info=evaluation.step(action);total+=float(reward)
                hit=hit or bool(info.get('is_success',False))
                if terminal or truncated:break
            returns.append(total);success.append(hit)
    finally:evaluation.close()
    import stable_baselines3
    report=dict(algorithm=algo,environment=env_name,requested_transitions=steps,actual_transitions=model.num_timesteps,
                seed=seed,eval_seed_start=10000+seed,evaluation_returns=returns,
                task_success=success if env_name in ('franka','agilex','r1-pro','r1-pro-planar') else None,
                seconds=time.perf_counter()-began,device='cpu',sb3_version=stable_baselines3.__version__,
                torch_version=torch.__version__,network=[64,64],scope='Baseline execution; a short run is not convergence')
    (root/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--algo',choices=['PPO','SAC','TD3','A2C','DDPG','DQN'],default='SAC')
    p.add_argument('--env',default='Reacher-v5');p.add_argument('--steps',type=int,default=10000)
    p.add_argument('--seed',type=int,default=7);p.add_argument('--out',default='runs/sb3');p.add_argument('--eval-episodes',type=int,default=5)
    a=p.parse_args()
    if a.steps<1 or a.eval_episodes<1:p.error('steps and eval-episodes must be positive')
    run(a.algo,a.env,a.steps,a.seed,a.out,a.eval_episodes)
