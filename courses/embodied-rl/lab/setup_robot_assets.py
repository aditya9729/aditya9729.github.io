"""Download exact upstream model revisions, preserving licenses. Requires git.

No access tokens or login. Fetches only declared model subtrees using sparse Git.
The R1 XML import is passive; learning controls are defined in embodiment_env.py.
"""
from pathlib import Path
import os
import shutil, subprocess, tempfile, xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1];DEST=Path(os.environ.get('RL_ROBOT_ASSETS',str(ROOT/'vendor'/'embodiments')))
SOURCES=[
 ('franka_panda','https://github.com/google-deepmind/mujoco_menagerie','62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35','franka_emika_panda'),
 ('agilex_aloha','https://github.com/agilexrobotics/mobile_aloha_sim','2843ff11d2695c1563a1c9847f632aa9734f5bbc','aloha_mujoco/aloha/meshes_mujoco'),
 ('galaxea_r1_pro','https://github.com/OpenGalaxea/GalaxeaManipSim','abe7f5161eeaa150e6eaffdf443af5df7f23f356','galaxea_sim/assets/r1_pro')]
def main():
    DEST.mkdir(parents=True,exist_ok=True)
    for name,url,sha,subtree in SOURCES:
        path=DEST/name
        marker=path/'.course-source-commit'
        required=path/'meshes_mujoco'/'table.STL' if name=='agilex_aloha' else path/('scene.xml' if name=='franka_panda' else 'robot.urdf')
        if marker.exists() and marker.read_text().strip()==sha and required.exists():print('Already prepared',name);continue
        with tempfile.TemporaryDirectory(prefix='rl-model-') as temp:
            def git(*args):subprocess.run(['git',*args],cwd=temp,check=True)
            git('init','-q');git('remote','add','origin',url)
            git('sparse-checkout','init','--cone');git('sparse-checkout','set',subtree)
            git('fetch','--depth','1','--filter=blob:none','origin',sha);git('checkout','--detach','FETCH_HEAD')
            source=Path(temp)/subtree
            if name=='agilex_aloha':
                # Upstream XML uses meshdir='./meshes_mujoco'; retain its relative layout.
                shutil.copytree(source,path,dirs_exist_ok=True)
            else:shutil.copytree(source,path,dirs_exist_ok=True)
            license_file=Path(temp)/'LICENSE'
            if license_file.exists():shutil.copy2(license_file,path/'LICENSE')
            marker.write_text(sha+'\n')
        print('Downloaded',name,sha)
    import mujoco
    original=DEST/'galaxea_r1_pro'/'robot.urdf'
    tree=ET.parse(original);root=tree.getroot();extension=ET.SubElement(root,'mujoco')
    ET.SubElement(extension,'compiler',meshdir='meshes',discardvisual='false',fusestatic='false')
    generated=original.with_name('r1_pro_mujoco_import.urdf');tree.write(generated,encoding='utf-8',xml_declaration=True)
    model=mujoco.MjModel.from_xml_path(str(generated));mujoco.mj_saveLastXML(str(original.with_name('r1_pro_passive.xml')),model)
    print('Passive R1 Pro import:',model.nq,'qpos;',model.nu,'native actuators')
    for file in [DEST/'franka_panda'/'scene.xml',DEST/'agilex_aloha'/'aloha_v1.xml',original.with_name('r1_pro_passive.xml')]:
        checked=mujoco.MjModel.from_xml_path(str(file));print('Verified downloaded model:',file.name,'nq',checked.nq,'nu',checked.nu)
if __name__=='__main__':main()
