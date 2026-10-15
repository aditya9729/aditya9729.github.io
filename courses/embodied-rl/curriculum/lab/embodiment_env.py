"""Gymnasium reaching tasks on real vendor robot geometry.

Franka/AgileX keep native actuators. R1 Pro adds COURSE PD torque control to a
passive URDF import. Optional planar coordinates are an idealized teaching
extension, not R1 steering/rolling dynamics. No hardware deployment.
"""
from __future__ import annotations
from pathlib import Path
import os
import xml.etree.ElementTree as ET
import gymnasium as gym
import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ASSETS=Path(os.environ.get('RL_ROBOT_ASSETS',str(ROOT/'vendor'/'embodiments')))

class RobotReach(gym.Env):
    metadata={'render_modes':['rgb_array'],'render_fps':20}
    def __init__(self,robot='franka',planar_base=False,render_mode=None,horizon=100):
        self.robot=robot;self.planar_base=bool(planar_base);self.render_mode=render_mode;self.horizon=horizon;self.renderer=None
        if planar_base and robot!='r1-pro':raise ValueError('The idealized planar extension is R1 Pro only.')
        paths={'franka':'franka_panda/scene.xml','agilex':'agilex_aloha/aloha_v1.xml','r1-pro':'galaxea_r1_pro/r1_pro_passive.xml'}
        self.path=ASSETS/paths[robot]
        if not self.path.exists():raise FileNotFoundError('Run: python lab/setup_robot_assets.py (pinned upstream assets, about 151 MiB).')
        self.control_note='native actuator model'
        if robot=='r1-pro':
            doc=ET.parse(self.path).getroot()
            doc.find('compiler').set('meshdir',str((self.path.parent/'meshes').resolve()))
            option=doc.find('option')
            if option is None:option=ET.SubElement(doc,'option')
            option.set('timestep','.002');option.set('integrator','implicitfast')
            # Add only explicit, course-defined torque motors. Preserve source geometry/inertias.
            acts=ET.SubElement(doc,'actuator')
            names=[j.get('name') for j in doc.findall('.//joint') if j.get('name')]
            if planar_base:
                base=doc.find("worldbody/body[@name='base_link']")
                if base is None:raise RuntimeError('Expected preserved official base_link.')
                for name,kind,axis,ran in [('course_base_x','slide','1 0 0','-.6 .6'),('course_base_y','slide','0 1 0','-.6 .6'),('course_base_yaw','hinge','0 0 1','-1 1')]:
                    ET.SubElement(base,'joint',name=name,type=kind,axis=axis,range=ran,damping='50',armature='1')
                names=['course_base_x','course_base_y','course_base_yaw']+names
            for name in names:ET.SubElement(acts,'motor',name='course_'+name,joint=name,ctrllimited='true',ctrlrange='-100 100')
            self.model=mujoco.MjModel.from_xml_string(ET.tostring(doc,encoding='unicode'))
            self.control_note='course PD torque control; not factory calibrated'+('; ideal planar base, not wheel dynamics' if planar_base else '; source chassis fixed')
        else:self.model=mujoco.MjModel.from_xml_path(str(self.path))
        self.data=mujoco.MjData(self.model)
        if robot=='franka':
            self.joint_names=[f'joint{i}' for i in range(1,8)];self.eef_names=['hand'];self.act_ids=np.arange(7)
        elif robot=='agilex':
            self.joint_names=[f'{side}_joint{i}' for side in ['fl','fr'] for i in range(1,7)]
            self.eef_names=['fl_link6','fr_link6'];self.act_ids=np.array([self.model.actuator(n).id for n in self.joint_names])
        else:
            self.joint_names=([f'course_base_{s}' for s in ['x','y','yaw']] if planar_base else [])+[f'torso_joint{i}' for i in range(1,5)]+[f'{s}_arm_joint{i}' for s in ['left','right'] for i in range(1,8)]
            self.eef_names=['left_gripper_link','right_gripper_link'];self.act_ids=np.array([self.model.actuator('course_'+n).id for n in self.joint_names])
        self.joint_ids=np.array([self.model.joint(n).id for n in self.joint_names])
        if robot=='franka':self.action_parts={'arm':(0,7)}
        elif robot=='agilex':self.action_parts={'left_arm':(0,6),'right_arm':(6,12)}
        else:
            offset=3 if planar_base else 0
            self.action_parts=({'base':(0,3)} if planar_base else {})
            self.action_parts.update(torso=(offset,offset+4),left_arm=(offset+4,offset+11),right_arm=(offset+11,offset+18))
        self.qidx=self.model.jnt_qposadr[self.joint_ids];self.vidx=self.model.jnt_dofadr[self.joint_ids]
        self.eef_ids=[self.model.body(n).id for n in self.eef_names]
        self.low=self.model.jnt_range[self.joint_ids,0].copy();self.high=self.model.jnt_range[self.joint_ids,1].copy()
        if robot=='agilex':
            # Native joint ranges are broad; target limits are narrower. Use their intersection.
            self.low=np.maximum(self.low,self.model.actuator_ctrlrange[self.act_ids,0]);self.high=np.minimum(self.high,self.model.actuator_ctrlrange[self.act_ids,1])
        self.delta=np.full(len(self.qidx),.035)
        if planar_base:self.delta[:3]=[.012,.012,.025]
        self.action_space=gym.spaces.Box(-1,1,(len(self.qidx),),np.float32)
        self.observation_space=gym.spaces.Box(-np.inf,np.inf,(2*len(self.qidx)+6*len(self.eef_ids)+1,),np.float32)
        self.substeps=max(1,round(.05/self.model.opt.timestep));self.dt=self.substeps*self.model.opt.timestep
        self.metadata={**self.metadata,'render_fps':round(1/self.dt)}
        self.goal=np.zeros((len(self.eef_ids),3));self.age=0;self.hold=0

    def wrists(self):return self.data.xpos[self.eef_ids].copy()
    def observation(self):
        return np.concatenate([self.data.qpos[self.qidx],self.data.qvel[self.vidx],self.wrists().ravel(),(self.goal-self.wrists()).ravel(),[self.hold/5]]).astype(np.float32)
    def _info(self):
        err=np.linalg.norm(self.goal-self.wrists(),axis=1)
        return {'distance':err.copy(),'is_success':self.hold>=5,'contacts':int(self.data.ncon),'control_note':self.control_note}
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed);mujoco.mj_resetData(self.model,self.data)
        if self.robot=='franka' and self.model.nkey:mujoco.mj_resetDataKeyframe(self.model,self.data,0)
        elif self.robot=='agilex':
            home=np.array([0.,1.,1.2,-.4,0.,0.]*2);self.data.qpos[self.qidx]=home
            self.data.ctrl[self.act_ids]=home
        else:
            home=np.zeros(len(self.qidx));home=np.clip(home,self.low+.02,self.high-.02)
            self.data.qpos[self.qidx]=home
            for name in ['left_gripper_finger_joint1','left_gripper_finger_joint2','right_gripper_finger_joint1','right_gripper_finger_joint2']:
                self.data.qpos[self.model.joint(name).qposadr]=.025
        start=np.clip(self.data.qpos[self.qidx].copy()+self.np_random.uniform(-.02,.02,len(self.qidx)),self.low+.005,self.high-.005)
        self.data.qpos[self.qidx]=start;mujoco.mj_forward(self.model,self.data)
        candidate=np.clip(start+self.np_random.uniform(-.15,.15,len(start)),self.low+.01,self.high-.01)
        if self.planar_base:candidate[:3]=np.clip(start[:3]+self.np_random.uniform(-.08,.08,3),self.low[:3],self.high[:3])
        self.data.qpos[self.qidx]=candidate;mujoco.mj_forward(self.model,self.data);self.goal=self.wrists()
        self.data.qpos[self.qidx]=start;self.data.qvel[:]=0;mujoco.mj_forward(self.model,self.data)
        self.age=0;self.hold=0
        return self.observation(),self._info()

    def step(self,action):
        action=np.asarray(action,np.float64)
        if action.shape!=self.action_space.shape or not np.isfinite(action).all():raise ValueError('One finite normalized action vector required.')
        action=np.clip(action,-1,1);target=np.clip(self.data.qpos[self.qidx]+self.delta*action,self.low,self.high)
        if self.robot!='r1-pro':self.data.ctrl[self.act_ids]=target
        for _ in range(self.substeps):
            if self.robot=='r1-pro':
                desired=self.data.qpos.copy();desired[self.qidx]=target
                for name in ['left_gripper_finger_joint1','left_gripper_finger_joint2','right_gripper_finger_joint1','right_gripper_finger_joint2']:desired[self.model.joint(name).qposadr]=.025
                for ai in range(self.model.nu):
                    ji=self.model.actuator_trnid[ai,0];qi=self.model.jnt_qposadr[ji];vi=self.model.jnt_dofadr[ji]
                    is_base=self.model.joint(ji).name.startswith('course_base')
                    kp,kd=(300.,70.) if is_base else (70.,12.)
                    force=kp*(desired[qi]-self.data.qpos[qi])-kd*self.data.qvel[vi]+self.data.qfrc_bias[vi]
                    self.data.ctrl[ai]=np.clip(force,-100,100)
            mujoco.mj_step(self.model,self.data)
        self.age+=1;dist=np.linalg.norm(self.goal-self.wrists(),axis=1)
        self.hold=self.hold+1 if dist.max()<.035 else 0
        reward=-float(dist.mean())-.002*float(np.square(action).mean())
        return self.observation(),reward,bool(self.hold>=5),bool(self.age>=self.horizon and self.hold<5),self._info()

    def render(self):
        if self.render_mode!='rgb_array':raise RuntimeError('Set render_mode="rgb_array" when constructing.')
        if self.renderer is None:self.renderer=mujoco.Renderer(self.model,height=480,width=640)
        cam=mujoco.MjvCamera();cam.lookat[:]=[.1,0,.7 if self.robot!='franka' else .3];cam.distance=2.8 if self.robot!='franka' else 1.8;cam.azimuth=130;cam.elevation=-20
        self.renderer.update_scene(self.data,camera=cam)
        for g in self.goal:
            scene=self.renderer.scene
            if scene.ngeom<scene.maxgeom:
                mujoco.mjv_initGeom(scene.geoms[scene.ngeom],mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.025,.025,.025]),g,np.eye(3).ravel(),np.array([.9,.55,.1,.8]));scene.ngeom+=1
        return self.renderer.render().copy()
    def close(self):
        if self.renderer is not None:self.renderer.close()
