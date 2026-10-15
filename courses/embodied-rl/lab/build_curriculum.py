"""Build long-form site content and executable curriculum notebooks from one source."""
from pathlib import Path
import os, sys, re, json, inspect, html, shutil, textwrap,base64,zlib
import nbformat as nbf
from markdown_it import MarkdownIt
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
from curriculum_content import LESSONS
from algorithm_guides import enrich
from notebook_exercises import EXERCISES,EDITABLE
import rl_core
enrich(LESSONS)

PROJECT=Path(__file__).resolve().parents[1]
OUT=PROJECT/'curriculum' if (PROJECT/'index.html').exists() else PROJECT/'dist'/'curriculum'
NBS=OUT/'notebooks';LAB=OUT/'lab'
for d in [OUT,NBS,LAB,OUT/'notes',OUT/'figures']:d.mkdir(parents=True,exist_ok=True)
for name in ['rl_core.py','course_figures.py','embodiment_env.py','setup_robot_assets.py','physics_lab.py','sb3_baseline.py']:
    shutil.copy2(PROJECT/'lab'/name,LAB/name)
for name in ['assets_manifest.json','PROVENANCE.md']:
    if (PROJECT/'vendor'/'embodiments'/name).exists():shutil.copy2(PROJECT/'vendor'/'embodiments'/name,OUT/name)

BOOT=r'''
import sys, os, subprocess, zipfile, json, math, copy, inspect
from pathlib import Path
IN_COLAB='google.colab' in sys.modules
if IN_COLAB:
    # Keep Colab's existing CUDA-enabled Torch; install only course essentials.
    subprocess.check_call([sys.executable,'-m','pip','install','-q','gymnasium[mujoco]==1.4.0','mujoco==3.3.7','matplotlib','imageio','imageio-ffmpeg'])
    os.environ.setdefault('MUJOCO_GL','egl')
elif sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
    os.environ.setdefault('MUJOCO_GL','egl')
def find_root():
    return next((p for p in [Path.cwd(),*Path.cwd().parents] if (p/'lab'/'rl_core.py').exists()),None)
ROOT=find_root()
if ROOT is None:
    # The notebook carries a compressed snapshot of its own editable course modules.
    # No private Site authentication, ZIP upload, GitHub runtime fetch or secret is needed.
    import base64,zlib
    ROOT=Path.cwd()/'Embodied_RL_Notebook';(ROOT/'lab').mkdir(parents=True,exist_ok=True)
    sources=json.loads(zlib.decompress(base64.b64decode('__MODULE_SNAPSHOT__')))
    for name,source in sources.items():
        if Path(name).name!=name or not name.endswith('.py'):raise ValueError('Invalid embedded module')
        (ROOT/'lab'/name).write_text(source)
    os.chdir(ROOT)
    print('Restored editable course modules to',ROOT/'lab')
sys.path.insert(0,str(ROOT/'lab'))
import numpy as np, torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal,Categorical
import matplotlib
matplotlib.use('module://matplotlib_inline.backend_inline')
import matplotlib.pyplot as plt
from IPython.display import display,Markdown,Code,Image
from IPython.core.pylabtools import select_figure_formats
select_figure_formats(get_ipython(),{'png'})
import rl_core as core
from rl_core import mlp,polyak,discounted_returns
from course_figures import sketch
torch.set_num_threads(1);torch.manual_seed(7)
USE_GPU=False  # Set True in Colab if CUDA is actually allocated; CPU is the baseline.
DEVICE='cuda' if USE_GPU and torch.cuda.is_available() else 'cpu'
print('Python',sys.version.split()[0],'Torch',torch.__version__,'selected device:',DEVICE)
print('CPU exercises were executed on Linux. Mac CPU / Colab CUDA paths are provided, not remotely verified.')
def ensure_robot_assets():
    import embodiment_env
    if not (embodiment_env.ASSETS/'franka_panda'/'scene.xml').exists():
        print('Downloading pinned upstream robot assets (~151 MiB) and preparing the passive R1 import.')
        subprocess.check_call([sys.executable,str(ROOT/'lab'/'setup_robot_assets.py')])
    assert (embodiment_env.ASSETS/'galaxea_r1_pro'/'r1_pro_passive.xml').exists()
'''
snapshot={p.name:p.read_text() for p in LAB.glob('*.py')}
BOOT=BOOT.replace('__MODULE_SNAPSHOT__',base64.b64encode(zlib.compress(json.dumps(snapshot).encode(),9)).decode())

md=MarkdownIt('commonmark',{'html':False}).enable('table')
def render_markdown(text):
    maths=[]
    def protect(m):
        block=m.group(1) is not None;source=m.group(1) if block else m.group(2)
        token=f'MATHPLACEHOLDER{len(maths)}END'
        maths.append((token,source,block));return '\n\n'+token+'\n\n' if block else token
    protected=re.sub(r'\$\$(.*?)\$\$|\$([^$\n]+)\$',protect,text,flags=re.S)
    result=md.render(protected)
    for token,source,block in maths:
        element=f'<'+('div' if block else 'span')+f' class="math {"display" if block else "inline"}">{html.escape(source)}</'+('div' if block else 'span')+'>'
        result=result.replace('<p>'+token+'</p>',element).replace(token,element)
    return result

site=[]
for i,l in enumerate(LESSONS):
    name=f'{i:02d}_{l["slug"].replace("-","_")}.ipynb';l['notebook']=name
    full=f'# {i+1:02d} · {l["title"]}\n\n{l["story"]}\n\n{l["math"]}\n\n{l["worked"]}\n\n{l["guide"]}\n\n## Your experiment\n\n{l["experiment"]}\n\n## A failure to watch for\n\n{l["pitfall"]}\n\n## Explain it back\n\n{l["question"]}\n\n<details><summary>Reasoning</summary>\n\n{l["answer"]}\n\n</details>\n\n## Primary sources\n\n'+''.join(f'- {x}\n' for x in l['refs'])
    (OUT/'notes'/f'{i:02d}_{l["slug"]}.md').write_text(full)
    item={**l,'storyHTML':render_markdown(l['story']),'mathHTML':render_markdown(l['math']),'workedHTML':render_markdown(l['worked']),'guideHTML':render_markdown(l['guide']),'word_count':len(full.split()),'code':inspect.getsource(getattr(rl_core,EDITABLE[l['slug']])) if l['slug'] in EDITABLE else EXERCISES[l['slug']][0]}
    site.append(item)
    cells=[nbf.v4.new_markdown_cell(f'# {i+1:02d} · {l["title"]}\n\nAn independent course by Aditya Gudal.\n\n{l["unit"]} · 60–120 minutes including exercises.\n\nRead the explanation, work the numbers, edit the code, and explain a failure.\n\n**Runtime:** Run the setup cell in Colab or an installed local Python environment. This notebook embeds its editable Python modules; no curriculum ZIP upload is required. CPU is the default. Robot chapters download pinned source assets separately. See `INSTALL.md` for reproducible local setup and optional Stable Baselines3.'),nbf.v4.new_code_cell(BOOT.strip()),nbf.v4.new_markdown_cell(l['story']),nbf.v4.new_code_cell(f"fig=sketch('{l['slug']}');display(fig);plt.close(fig)"),nbf.v4.new_markdown_cell(l['math']),nbf.v4.new_markdown_cell(l['worked'])]
    if l['guide']:cells.append(nbf.v4.new_markdown_cell('## Follow the algorithm\n\n'+l['guide']))
    if l['slug']=='sac':
        cells.append(nbf.v4.new_code_cell("# Exact finite-action entropy improvement; Gaussian robot SAC is an approximation.\nq=torch.tensor([0.,1.,2.]);alpha=1.\nprob=torch.softmax(q/alpha,dim=0)\nentropy=-(prob*prob.log()).sum()\nassert torch.allclose(prob,torch.tensor([.09003057,.24472847,.66524096]),atol=1e-6)\nprint('Probabilities:',prob,'entropy:',float(entropy))\nassert abs((2.5-3.277)**2+(3.-3.277)**2-.680458)<1e-9\nprint('Worked critic squared-error sum: 0.680458')"))
    if l['slug']=='grpo-flow':
        cells.append(nbf.v4.new_code_cell("# KL tilt of a nonuniform prior; this is a finite sampling exercise.\nprior=torch.tensor([.6,.3,.1]);q=torch.tensor([0.,1.,2.]);beta=.5\nweights=torch.softmax(prior.log()+q/beta,dim=0)\ness=1/weights.square().sum()\nassert torch.isfinite(weights).all() and abs(float(weights.sum())-1)<1e-6\nprint('Tilted probabilities:',weights,'ESS:',float(ess))"))
    if l['slug'] in EDITABLE:
        name_fn=EDITABLE[l['slug']]
        cells.append(nbf.v4.new_markdown_cell('## The editable implementation\n\nThis is original course code, not a hidden optimizer call. Modify it, rerun it, and connect it to the trainer where indicated. Full source and collection loops are in `lab/rl_core.py`.'))
        cells.append(nbf.v4.new_code_cell(inspect.getsource(getattr(rl_core,name_fn))+f'\n\n# Connect this editable definition to the shared trainer / experiment.\ncore.{name_fn}={name_fn}\n'))
    for snippet in EXERCISES[l['slug']]:cells.append(nbf.v4.new_code_cell(snippet))
    if l['slug'] in ('franka','agilex','r1-pro'):
        file={'franka':'franka.png','agilex':'agilex.png','r1-pro':'r1-pro.png'}[l['slug']]
        cells.append(nbf.v4.new_code_cell(f"image_path=ROOT/'assets'/'{file}'\nif image_path.exists():\n    display(Image(filename=str(image_path)))\nelse:\n    from embodiment_env import RobotReach\n    preview=RobotReach('{l['slug']}',render_mode='rgb_array')\n    preview.reset(seed=7);plt.imshow(preview.render());plt.axis('off');display(plt.gcf());plt.close();preview.close()\nprint('Actual source-geometry MuJoCo render; the short learning run above is not a solved-task claim.')"))
    cells.extend([nbf.v4.new_markdown_cell('## Your experiment\n\n'+l['experiment']+'\n\nRecord your prediction, exact change, measured result, and failure explanation. Longer training is optional and must use a declared budget.'),nbf.v4.new_code_cell("# Your lab record: edit these fields after making one controlled change.\nlab_record={'prediction':'','change':'','result':'','explanation':''}\nprint(lab_record)"),nbf.v4.new_markdown_cell('## Explain it back\n\n'+l['question']+'\n\n<details><summary>Compare your reasoning</summary>\n\n'+l['answer']+'\n\n</details>\n\n## Failure to watch for\n\n'+l['pitfall']+'\n\n## Primary sources\n\n'+''.join(f'- {x}\n' for x in l['refs']))])
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'},'colab':{'name':l['notebook'],'provenance':[]}})
    nbf.validate(nb);nbf.write(nb,NBS/l['notebook'])
(OUT/'content.json').write_text(json.dumps(site,ensure_ascii=False,indent=2)+'\n')
(OUT/'content.js').write_text('window.CURRICULUM='+json.dumps(site,ensure_ascii=False,separators=(',',':'))+';\n')

if '--execute' in sys.argv:
    os.environ['RL_ROBOT_ASSETS']=str(PROJECT/'vendor'/'embodiments')
    os.chdir(OUT);shell=InteractiveShell.instance();report={}
    for l in site:
        p=NBS/l['notebook'];nb=nbf.read(p,as_version=4);shell.reset(new_session=False);count=0
        for cell in nb.cells:
            if cell.cell_type!='code':continue
            count+=1
            with capture_output() as captured:result=shell.run_cell(cell.source,store_history=False)
            if result.error_before_exec or result.error_in_exec:
                print(captured.stdout,captured.stderr,flush=True)
                raise RuntimeError(f'{p.name} cell {count}: {result.error_before_exec or result.error_in_exec}')
            outputs=[]
            if captured.stdout:outputs.append(nbf.v4.new_output('stream',name='stdout',text=captured.stdout))
            if captured.stderr:outputs.append(nbf.v4.new_output('stream',name='stderr',text=captured.stderr))
            for output in captured.outputs:outputs.append(nbf.v4.new_output('display_data',data=output.data,metadata=output.metadata))
            cell.execution_count=count;cell.outputs=outputs
        nbf.validate(nb);nbf.write(nb,p);report[p.name]=count
        print('EXECUTED',p.name,count,'code cells',flush=True)
    (OUT/'execution.json').write_text(json.dumps({'runtime':'Linux Python3.12 CPU; in-process IPython; actual MuJoCo stepping','separate_jupyter_kernel':'Host socket restrictions blocked kernel launch','notebooks':report,'total_code_cells':sum(report.values()),'mac_native_executed':False,'colab_cuda_executed':False},indent=2)+'\n')
print('Built',len(site),'chapters;',sum(l['word_count'] for l in site),'words',flush=True)
