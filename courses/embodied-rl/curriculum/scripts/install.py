"""Create an isolated CPU course environment. Run with --dry-run to inspect."""
import argparse,json,os,platform,shlex,subprocess,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',choices=['core','sb3'],default='sb3')
    p.add_argument('--python',default=sys.executable,help='Python 3.11 or 3.12 executable')
    p.add_argument('--venv',default='.venv');p.add_argument('--dry-run',action='store_true')
    p.add_argument('--skip-smoke',action='store_true');a=p.parse_args()
    root=Path(__file__).resolve().parents[1];venv=(root/a.venv).resolve()
    exe=venv/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    commands=[[a.python,'-m','venv',str(venv)],[str(exe),'-m','pip','install','--upgrade','pip'],
              [str(exe),'-m','pip','install','-r',str(root/'requirements.txt')]]
    if a.profile=='sb3':commands.append([str(exe),'-m','pip','install','-r',str(root/'requirements-sb3.txt')])
    if not a.skip_smoke:commands.append([str(exe),str(root/'scripts'/'doctor.py'),'--profile',a.profile])
    for command in commands:
        print(shlex.join(command),flush=True)
        if not a.dry_run:subprocess.check_call(command,cwd=root)
    if not a.dry_run:
        freeze=subprocess.check_output([str(exe),'-m','pip','freeze'],text=True)
        (venv/'course-freeze.txt').write_text(freeze)
        print('Installed. Exact resolved versions: '+str(venv/'course-freeze.txt'))
        print('Launch: '+shlex.join([str(exe),'-m','jupyter','lab',str(root/'notebooks'/'00_probability.ipynb')]))

if __name__=='__main__':main()
