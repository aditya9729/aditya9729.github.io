"""Check imports, one real Gymnasium transition and optional SB3 model creation."""
import argparse,json,sys,platform
from pathlib import Path
import numpy as np,torch,gymnasium as gym,mujoco
p=argparse.ArgumentParser();p.add_argument('--profile',choices=['core','sb3'],default='core');a=p.parse_args()
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'lab'))
import rl_core
env=gym.make('Reacher-v5');obs,_=env.reset(seed=7)
obs,reward,te,tr,_=env.step(env.action_space.sample());assert np.isfinite(obs).all();env.close()
report=dict(platform=platform.platform(),python=sys.version.split()[0],torch=torch.__version__,
            gymnasium=gym.__version__,mujoco=mujoco.__version__,cuda_available=torch.cuda.is_available(),real_environment_step=True)
if a.profile=='sb3':
    import stable_baselines3
    from stable_baselines3 import SAC
    env=gym.make('Reacher-v5');model=SAC('MlpPolicy',env,device='cpu',buffer_size=100,policy_kwargs={'net_arch':[16,16]});env.close()
    report['stable_baselines3']=stable_baselines3.__version__
print(json.dumps(report,indent=2))
