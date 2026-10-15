(()=>{
 'use strict';
 const $=id=>document.getElementById(id),lessons=window.CURRICULUM,config=window.COURSE_CONFIG;
 const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 let state={complete:[],notes:{},runtime:'mac'},index=0;
 try{const saved=JSON.parse(localStorage.getItem('embodied-rl-curriculum-v3')||'{}');state={...state,...saved};if(!Array.isArray(state.complete))state.complete=[];if(!state.notes||typeof state.notes!=='object')state.notes={};}catch{}
 const persist=()=>{try{localStorage.setItem('embodied-rl-curriculum-v3',JSON.stringify(state));}catch{}};
 function nav(){
  const query=$('chapter-search').value.trim().toLowerCase();let unit='';
  $('chapter-list').innerHTML=lessons.map((l,i)=>{
   if(query&&!`${l.title} ${l.unit} ${l.story}`.toLowerCase().includes(query))return '';
   const heading=l.unit!==unit?`<div class="unit-heading">${esc(l.unit)}</div>`:'';unit=l.unit;
   return heading+`<button class="chapter-link ${i===index?'active':''}" data-chapter="${i}" ${i===index?'aria-current="page"':''}><span>${state.complete.includes(l.slug)?'✓':String(i+1).padStart(2,'0')}</span>${esc(l.title)}</button>`;
  }).join('');
  document.querySelectorAll('[data-chapter]').forEach(b=>b.addEventListener('click',()=>select(Number(b.dataset.chapter),true)));
  const n=state.complete.filter(slug=>lessons.some(l=>l.slug===slug)).length;
  $('progress-label').textContent=`${n} / ${lessons.length} studied`;$('progress-bar').style.width=100*n/lessons.length+'%';
 }
 function renderMath(){
  if(!window.katex)return;
  document.querySelectorAll('.math').forEach(el=>{try{window.katex.render(el.textContent,el,{displayMode:el.classList.contains('display'),throwOnError:true,strict:'warn',trust:false});}catch(e){el.classList.add('math-error');el.setAttribute('title','Math rendering unavailable: '+e.message);}});
 }
 function select(i,scroll=false){
  index=Math.max(0,Math.min(i,lessons.length-1));const l=lessons[index];
  $('unit-label').textContent=l.unit;$('chapter-title').textContent=l.title;
  $('learning-goal').textContent=l.learning_goal;
  $('chapter-position').textContent=`CHAPTER ${String(index+1).padStart(2,'0')} / ${lessons.length}`;
  $('reading-time').textContent=`${Math.max(3,Math.ceil(l.word_count/200))} min reading + derivation`;
  $('intuition').innerHTML=l.storyHTML;$('derivation').innerHTML=l.mathHTML;$('worked').innerHTML=l.workedHTML;
  $('algorithm-guide').innerHTML=l.guideHTML;$('update-title').textContent=l.update_title;
  $('algorithm-guide').hidden=!l.guideHTML;setupTrace(l);
  $('chapter-code').textContent=l.code;$('copy-code').textContent='Copy code';$('experiment-text').textContent=l.experiment;$('pitfall-text').textContent=l.pitfall;
  $('check-question').textContent=l.question;$('check-answer').textContent=l.answer;$('check-answer').hidden=true;$('answer-toggle').textContent='Compare your reasoning ↓';$('answer-toggle').setAttribute('aria-expanded','false');
  $('lab-notes').value=state.notes[l.slug]||'';$('mark-complete').checked=state.complete.includes(l.slug);
  $('chapter-notebook').href='curriculum/notebooks/'+l.notebook;
  $('hero-notebook').href=$('chapter-notebook').href;
  const colab=`https://colab.research.google.com/github/${config.repository}/blob/${config.ref}/${config.notebookPath}${l.notebook}`;
  $('chapter-colab').href=colab;$('hero-colab').href=colab;
  const algo={dqn:'DQN',reinforce:'PPO','actor-critic':'A2C',ppo:'PPO',ddpg:'DDPG',td3:'TD3',sac:'SAC'}[l.slug]||'SAC';
  const env=l.slug==='dqn'?'CartPole-v1':['franka','agilex','r1-pro'].includes(l.slug)?l.slug:'Reacher-v5';
  $('baseline-command').textContent=`python3 scripts/install.py --profile sb3 --python python3.11\n.venv/bin/python lab/sb3_baseline.py --algo ${algo} --env ${env} --steps 10000 --seed 7`;
  $('chapter-notes').href=`curriculum/notes/${String(index).padStart(2,'0')}_${l.slug}.md`;
  $('primary-sources').innerHTML=l.refs.map(url=>`<a href="${esc(url)}" target="_blank" rel="noopener noreferrer">${esc(url.replace(/^https?:\/\//,''))} ↗</a>`).join('');
  $('previous-chapter').disabled=index===0;$('next-chapter').disabled=index===lessons.length-1;
  nav();setupSketch(l);renderMath();
  $('course-navigation').classList.remove('open');$('menu-toggle').setAttribute('aria-expanded','false');
  history.replaceState(null,'','#'+l.slug);
  if(scroll)$('reading').scrollIntoView({behavior:window.matchMedia?.('(prefers-reduced-motion: reduce)').matches?'auto':'smooth',block:'start'});
 }
 function setupTrace(l){
  const steps=l.recipe||[];$('update-trace').hidden=!steps.length;
  $('update-steps').innerHTML=steps.map((s,i)=>`<button class="trace-tab" id="trace-tab-${i}" role="tab" aria-controls="trace-panel" aria-selected="${i===0}" tabindex="${i===0?0:-1}" data-step="${i}"><span>${i+1}</span>${esc(s[0])}</button>`).join('');
  function show(i,focus=false){
   const step=steps[i];$('trace-position').textContent=`STAGE ${i+1} / ${steps.length}`;
   $('trace-title').textContent=step[0];$('trace-description').textContent=step[1];$('trace-gradient').textContent=step[2];
   $('trace-panel').setAttribute('aria-labelledby',`trace-tab-${i}`);
   document.querySelectorAll('[data-step]').forEach((b,j)=>{b.setAttribute('aria-selected',String(j===i));b.setAttribute('tabindex',j===i?'0':'-1');if(focus&&j===i)b.focus();});
  }
  document.querySelectorAll('[data-step]').forEach(b=>{
   b.addEventListener('click',()=>show(Number(b.dataset.step)));
   b.addEventListener('keydown',e=>{const i=Number(b.dataset.step);let next;
    if(e.key==='ArrowRight')next=(i+1)%steps.length;else if(e.key==='ArrowLeft')next=(i-1+steps.length)%steps.length;else if(e.key==='Home')next=0;else if(e.key==='End')next=steps.length-1;
    if(next!==undefined){e.preventDefault();show(next,true);}
   });
  });if(steps.length)show(0);
 }
 const green='#3c6046',orange='#b66c43',blue='#67899a',purple='#8576a0';
 const text=(x,y,s,size=14,fill='#42543d',anchor='middle')=>`<text x="${x}" y="${y}" text-anchor="${anchor}" font-family="system-ui,sans-serif" font-size="${size}" fill="${fill}">${esc(s)}</text>`;
 const line=(x,y,x2,y2,col=green)=>`<path d="M${x} ${y} L${x2} ${y2}" stroke="${col}" stroke-width="2" fill="none" stroke-linecap="round"/>`;
 const box=(x,y,w,h,label)=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="10" fill="#f9faf2" stroke="${green}" stroke-width="1.6"/>`+text(x+w/2,y+h/2+5,label);
 function axes(xlabel,ylabel){return line(62,270,755,270,'#8e9b81')+line(62,270,62,35,'#8e9b81')+text(408,318,xlabel,13)+text(65,20,ylabel,12,'#63735b','start');}
 function path(points,color=green,width=3){return `<path d="${points.map(([x,y],i)=>(i?'L':'M')+x.toFixed(2)+' '+y.toFixed(2)).join(' ')}" fill="none" stroke="${color}" stroke-width="${width}"/>`;}
 const range=(label,min,max,step,value)=>`<label for="sketch-value">${label}<input id="sketch-value" type="range" min="${min}" max="${max}" step="${step}" value="${value}"><output id="sketch-number">${value}</output></label>`;
 function setupSketch(l){
  $('robot-image').hidden=true;$('concept-svg').hidden=false;
  const kind=l.sketch;
  let control=range('Discount γ',0,.99,.01,.9),title='One step, many possible futures';
  if(['distribution','policy'].includes(kind)){control=range('Gaussian spread σ',.1,1.5,.05,.5);title='An action is a sample, not the whole policy';}
  else if(kind==='entropy'){control=range('Temperature α',.1,3,.05,1);title='Same Q values. A different action distribution.';}
  else if(kind==='bellman'){control=range('Branch probability p',0,1,.05,.75);title='A future summarized by one backup';}
  else if(['td','gae'].includes(kind)){control=range('Trace parameter λ',0,1,.05,.8);title='Credit as a weighted sum of surprises';}
  else if(kind==='ppo'){control=range('Probability ratio ρ',.1,2,.05,1.5)+'<label for="sketch-sign">Advantage<select id="sketch-sign"><option value="2">+2 · better than expected</option><option value="-2">−2 · worse than expected</option></select></label>';title='Four cases. One minimum.';}
  else if(['twins','replay','critic'].includes(kind)){control=range('Estimation noise scale',0,.6,.02,.2);title=kind==='replay'?'The maximum can prefer positive error':'The actor optimizes an estimate';}
  else if(kind==='dataset'){control=range('Residual authority β',0,1,.05,.2);title='A correction cannot exceed its authority';}
  else if(['robot','bimanual','whole'].includes(kind)){control='<span>Actual vendor geometry · MuJoCo-rendered image</span>';title='Keep the embodiment and control contract explicit';}
  else if(kind==='group'){control='<label for="group-pattern">Group outcomes<select id="group-pattern"><option value="mixed">0, 0, 1, 1</option><option value="equal">0, 0, 0, 0</option><option value="graded">1, 2, 3, 4</option></select></label>';title='Relative to which other attempts?';}
  else if(kind==='curriculum'){control=range('Target success p*',.1,.9,.05,.5);title='Choose what to practice next';}
  else if(kind==='systems'){control=range('Update time per transition (ms)',.1,10,.1,2);title='A GPU cannot remove every bottleneck';}
  $('sketch-title').textContent=title;$('sketch-controls').innerHTML=control;
  $('sketch-controls').querySelectorAll('input,select').forEach(el=>{el.addEventListener('input',()=>drawSketch(l));el.addEventListener('change',()=>drawSketch(l));});
  drawSketch(l);
 }
 function drawSketch(l){
  const kind=l.sketch,v=Number($('sketch-value')?.value??.9);if($('sketch-number'))$('sketch-number').textContent=v.toFixed(2);
  let svg='',result='',caption='Original numerical illustration. This is not a simulator rollout or a measured training curve.';
  if(kind==='entropy'){
   const q=[0,1,2],raw=q.map(x=>Math.exp((x-2)/v)),sum=raw.reduce((a,b)=>a+b),prob=raw.map(x=>x/sum);
   const H=-prob.reduce((a,p)=>a+p*Math.log(p),0),EQ=prob.reduce((a,p,i)=>a+p*q[i],0);
   svg=line(65,270,750,270,'#9fb8c4');
   prob.forEach((p,i)=>{const x=180+220*i,h=p*205;svg+=`<rect x="${x-55}" y="${270-h}" width="110" height="${h}" rx="5" fill="${i===2?green:blue}"/>`+text(x,251-h,(100*p).toFixed(2)+'%',16)+text(x,298,`command ${i+1} · Q = ${q[i]}`,13);});
   result=`π*(a) ∝ exp(Q(a)/α) → [${prob.map(p=>p.toFixed(4)).join(', ')}]\nExpected Q = ${EQ.toFixed(3)} · entropy = ${H.toFixed(3)} nats\nReward + entropy objective = ${(EQ+v*H).toFixed(3)}`;
   caption='Exact optimizer over three fixed action candidates. The continuous SAC actor approximates a soft policy through a parameterized distribution; these bars are not a learned robot rollout.';
  }else if(['distribution','policy'].includes(kind)){
   svg=axes('Pre-tanh Gaussian coordinate u','Density');
   const pts=[];for(let i=0;i<=300;i++){const x=-3+6*i/300,p=Math.exp(-.5*(x/v)**2)/(v*Math.sqrt(2*Math.PI));pts.push([62+693*i/300,270-p*58]);}
   svg+=path(pts)+line(408.5,270,408.5,45,'#bfc9b0')+text(408.5,292,'mean 0',12);
   const H=.5*Math.log(2*Math.PI*Math.E*v*v);
   result=`Gaussian entropy = ½ log(2πeσ²) = ${H.toFixed(3)} nats\nThis is the pre-tanh scalar Gaussian; transformed action entropy differs.`;
  }else if(kind==='return'){
   const rewards=[-1,-1,10],weights=rewards.map((r,i)=>r*v**i);svg=line(80,100,730,100,'#899c76');
   for(let i=0;i<3;i++){const x=130+i*260;svg+=`<circle cx="${x}" cy="100" r="9" fill="${green}"/>`+text(x,65,'step '+i,14)+box(x-78,135,156,58,'reward '+rewards[i])+text(x,226,`γ^${i} × r = ${weights[i].toFixed(3)}`,14);}
   result=`G₀ = −1 − γ + 10γ² = ${weights.reduce((a,b)=>a+b,0).toFixed(3)}\nThe success event is the same; its discounted value changes.`;
  }else if(kind==='bellman'){
   svg=box(34,125,135,60,'state s')+box(294,45,170,64,'next value 8')+box(294,213,170,64,'next value 0');
   svg+=line(170,150,293,80)+line(170,155,293,245)+text(215,91,`p = ${v.toFixed(2)}`,13)+text(212,227,`1−p = ${(1-v).toFixed(2)}`,13);
   svg+=line(465,78,596,145)+line(465,245,596,173)+box(598,127,168,67,'one Q estimate');
   result=`Q(s,a) = −0.5 + 0.9[p × 8 + (1−p) × 0] = ${(-.5+.9*v*8).toFixed(3)}\nA fixed-policy state value still averages across its actions.`;
  }else if(['td','gae'].includes(kind)){
   const deltas=[1,-.5,2],contrib=deltas.map((d,i)=>d*(.9*v)**i);svg=line(70,190,750,190,'#899c76');
   for(let i=0;i<3;i++){const x=160+240*i,h=contrib[i]*63;svg+=`<rect x="${x-50}" y="${h>=0?190-h:190}" width="100" height="${Math.abs(h)}" rx="4" fill="${h>=0?green:orange}"/>`+text(x,290,`δ${i} = ${deltas[i]}`,13)+text(x,h>=0?180-h:215+Math.abs(h),contrib[i].toFixed(3),13);}
   result=`Â₀ = 1 + (0.9λ)(−0.5) + (0.9λ)²(2) = ${contrib.reduce((a,b)=>a+b,0).toFixed(4)}\nA reset after the first transition cuts off both later terms.`;
  }else if(kind==='ppo'){
   const A=Number($('sketch-sign').value),f=r=>Math.min(r*A,Math.min(1.2,Math.max(.8,r))*A),yy=z=>155-z*29,xx=r=>62+(r-.1)/1.9*693;
   svg=axes('New / old probability ratio','Sample objective (maximize)');
   svg+=line(xx(.8),35,xx(.8),270,'#aebd98')+line(xx(1.2),35,xx(1.2),270,'#aebd98');
   svg+=path(Array.from({length:160},(_,i)=>{const r=.1+1.9*i/159;return [xx(r),yy(f(r))];}),A>0?green:orange);
   svg+=`<circle cx="${xx(v)}" cy="${yy(f(v))}" r="7" fill="${purple}"/>`+text(xx(.8),292,'.8',12)+text(xx(1.2),292,'1.2',12);
   const gradient=A>0?(v>1.2?0:A):(v<.8?0:A);
   result=`A = ${A}; ρA = ${(v*A).toFixed(3)}; clipped term = ${(Math.min(1.2,Math.max(.8,v))*A).toFixed(3)}\nminimum = ${f(v).toFixed(3)}; derivative w.r.t. ρ = ${gradient} away from kinks.`;
  }else if(['twins','replay','critic'].includes(kind)){
   svg=axes('Continuous action a','Illustrative value');let truth=[],q1=[],q2=[],minimum=[];
   for(let i=0;i<=250;i++){const a=-1+2*i/250,t=-((a-.3)**2),b=t+v*(Math.sin(9*a)+.75),c=t+v*(Math.cos(9*a)+.5),x=62+693*i/250;truth.push([x,135-t*60]);q1.push([x,135-b*60]);q2.push([x,135-c*60]);minimum.push([x,135-Math.min(b,c)*60]);}
   svg+=path(truth,'#a3ae97',2)+path(q1,orange,2)+path(q2,blue,2)+path(minimum,green,3)+text(130,42,'min(Q₁,Q₂)',12,green,'start')+text(330,42,'Q₁',12,orange,'start')+text(430,42,'Q₂',12,blue,'start');
   result='Two estimates can both be wrong. Their minimum is not a certified lower bound.\nThese curves are constructed to expose selection error, not learned robot critics.';
  }else if(kind==='dataset'){
   const lo=.8-v,hi=.8+v;svg=line(75,160,745,160,'#a7b795')+text(75,195,'−1',14)+text(745,195,'+1',14);
   const x=a=>75+(a+1)/2*670;svg+=`<rect x="${x(Math.max(-1,lo))}" y="133" width="${Math.max(1,x(Math.min(1,hi))-x(Math.max(-1,lo)))}" height="54" rx="6" fill="#b8c98f"/>`+line(x(.8),105,x(.8),215,green)+text(x(.8),88,'base +0.8',14);
   result=`Before clipping: [${lo.toFixed(2)}, ${hi.toFixed(2)}]\nAfter clipping: [${Math.max(-1,lo).toFixed(2)}, ${Math.min(1,hi).toFixed(2)}] · Can reverse direction? ${lo<0?'Yes':'No'}`;
  }else if(['robot','bimanual','whole'].includes(kind)){
   const map={robot:['franka',7,'native joint-target actuator model'],bimanual:['agilex',12,'official vendor model, fixed chassis'],whole:['r1-pro',18,'course PD control; optional ideal planar extension adds 3 commands']},[robot,dim,note]=map[kind];
   $('concept-svg').hidden=true;$('robot-image').hidden=false;$('robot-image').src=`curriculum/assets/${robot}.png`;
   result=`${dim} normalized arm/upper-body commands in the default reaching task.\n${note}.`;
   caption='Actual source geometry rendered in MuJoCo. Model/control integration was tested; no solved grasping or hardware transfer is implied.';
  }else if(kind==='group'){
   const rewards=$('group-pattern').value==='equal'?[0,0,0,0]:$('group-pattern').value==='graded'?[1,2,3,4]:[0,0,1,1];
   const mean=rewards.reduce((a,b)=>a+b)/4,std=Math.sqrt(rewards.reduce((s,r)=>s+(r-mean)**2,0)/4),A=rewards.map(r=>(r-mean)/(std+1e-8));svg=line(55,165,750,165,'#a6b493');
   A.forEach((a,i)=>{const x=135+i*175,h=a*75;svg+=`<rect x="${x-42}" y="${h>=0?165-h:165}" width="84" height="${Math.max(1,Math.abs(h))}" rx="4" fill="${a>=0?green:orange}"/>`+text(x,298,`R${i+1}=${rewards[i]}`,13)+text(x,h>=0?151-h:190+Math.abs(h),a.toFixed(2),13);});
   result=`mean = ${mean.toFixed(3)}; population std = ${std.toFixed(3)}\nNormalized group signal = [${A.map(a=>a.toFixed(2)).join(', ')}]`;
   caption='Group-normalization arithmetic only, not a full GRPO trainer. The attempts must have a valid shared-context interpretation.';
  }else if(kind==='curriculum'){
   const p=[.05,.2,.35,.5,.65,.8,.95],raw=p.map(p=>(p+.001)**(4*v)*(1-p+.001)**(4*(1-v))),sum=raw.reduce((a,b)=>a+b),weights=raw.map(w=>w/sum);svg=line(60,270,750,270,'#a4b390');
   weights.forEach((w,i)=>{const x=100+i*100,h=w*630;svg+=`<rect x="${x-27}" y="${270-h}" width="54" height="${h}" rx="4" fill="${green}"/>`+text(x,295,p[i].toFixed(2),12)+text(x,257-h,(w*100).toFixed(1)+'%',12);});
   result=`Target success = ${v.toFixed(2)}; highest sampling mass at p = ${p[weights.indexOf(Math.max(...weights))].toFixed(2)}.\nSynthetic success estimates isolate the weighting rule; notebook histories come from actual robot episodes.`;
  }else if(kind==='systems'){
   const env=1.,infer=.5,total=env+infer+v;svg=box(90,104,190,78,'environment 1 ms')+box(305,104,190,78,'inference 0.5 ms')+box(520,104,190,78,`update ${v.toFixed(1)} ms`)+line(280,142,304,142)+line(495,142,519,142);
   result=`Sequential toy cost = ${total.toFixed(2)} ms / transition\nCalculated throughput ≈ ${(1000/total).toFixed(1)} transitions/s. Not a measured GPU result.`;
   caption='A bottleneck calculation, not a FastTD3/FlashSAC benchmark. Asynchronous overlap requires a different timing model.';
  }
  $('concept-svg').innerHTML=svg;$('sketch-result').textContent=result;$('sketch-caption').textContent=caption;
 }
 function runtime(mode){
  state.runtime=mode;persist();$('runtime-label').textContent=mode==='colab'?'Colab / optional CUDA':mode==='rlinf'?'Linux GPU track':'CPU-capable';
  document.querySelectorAll('[data-runtime]').forEach(b=>{const active=b.dataset.runtime===mode;b.classList.toggle('selected',active);b.setAttribute('aria-pressed',String(active));});
  const instructions={
   mac:`<p class="section-lead">Download and extract the course pack. Use Python 3.11 or 3.12 supported by your Mac. The installer creates an isolated environment, checks real environment stepping and records exact resolved versions.</p><pre>python3 scripts/install.py --profile sb3 --python python3.11\n.venv/bin/python -m jupyter lab notebooks/00_probability.ipynb\n\n# Once, for the actual robot models:\n.venv/bin/python lab/setup_robot_assets.py</pre><p class="small-note">Use <code>--profile core</code> to skip SB3 or <code>--dry-run</code> to inspect commands first. Use native macOS rendering; headless Linux EGL settings are not Mac settings.</p>`,
   colab:`<ol><li>Use “Open this lesson in Colab” above. It identifies the exact chapter notebook in the public GitHub review branch.</li><li>Run the first setup cell. The notebook restores its own editable course modules; no ZIP upload is required.</li><li>Keep CPU for the small exercises. Set <code>USE_GPU=True</code> only if you allocated a GPU and CUDA is available.</li><li>Robot labs separately fetch about 151 MiB of pinned assets. Save notebooks and outputs before your session ends.</li></ol><pre>DEVICE = 'cuda' if USE_GPU and torch.cuda.is_available() else 'cpu'</pre><p class="small-note">Colab may ask you to sign in and confirm execution of a third-party notebook. GPU availability varies.</p>`,
   sb3:`<p class="section-lead">Use Stable Baselines3 as an independent reference after you can explain the original update. The runner supports PPO, A2C, DQN, DDPG, TD3 and SAC, and saves a model, training CSV and held-out evaluation JSON.</p><pre>python3 scripts/install.py --profile sb3 --python python3.11\n.venv/bin/python lab/sb3_baseline.py --algo SAC --env Reacher-v5 --steps 10000 --seed 7\n\n# After fetching the models:\n.venv/bin/python lab/sb3_baseline.py --algo PPO --env franka --steps 10000 --seed 7</pre><p class="small-note">Match actual transition counts and evaluation seeds. On-policy rollout collection can exceed the requested count. The 10,000-step example is an initial experiment, not a convergence guarantee.</p>`,
   lerobot:`<p class="section-lead">Use LeRobot when you need demonstration datasets and pretrained-policy workflows. Keep it in a separate environment. The script pins an upstream source commit and uses Python 3.12 and the training extra.</p><pre>bash scripts/install-lerobot.sh\n# Separate environment: lerobot-course/.venv\n# Check ffmpeg / TorchCodec platform instructions in INSTALL.md.</pre><p class="small-note">This does not install a Galaxea hardware interface, calibrate a robot or download pretrained weights. <a href="https://huggingface.co/docs/lerobot/installation" target="_blank" rel="noopener noreferrer">Official installation guidance ↗</a></p>`,
   rlinf:`<p class="section-lead">Move to RLinf for advanced distributed VLA/environment experiments on a suitable Linux GPU server. The pinned source script follows the upstream OpenVLA + ManiSkill/LIBERO installation combination.</p><pre>bash scripts/install-rlinf.sh\n# Separate workbench: rlinf-course/\n# Review upstream hardware and model/environment requirements first.</pre><p class="small-note">This is a provided setup path, not a tested cluster run or an R1 Pro integration. <a href="https://rlinf.readthedocs.io/en/latest/rst_source/start/installation.html" target="_blank" rel="noopener noreferrer">Official installation guidance ↗</a></p>`
  };$('runtime-instructions').innerHTML=instructions[mode]||instructions.mac;
 }
 $('chapter-search').addEventListener('input',nav);
 $('previous-chapter').addEventListener('click',()=>select(index-1,true));$('next-chapter').addEventListener('click',()=>select(index+1,true));
 $('lab-notes').addEventListener('input',()=>{state.notes[lessons[index].slug]=$('lab-notes').value;persist();});
 $('mark-complete').addEventListener('change',()=>{const slug=lessons[index].slug;state.complete=state.complete.filter(s=>s!==slug);if($('mark-complete').checked)state.complete.push(slug);persist();nav();});
 $('answer-toggle').addEventListener('click',()=>{const show=$('check-answer').hidden;$('check-answer').hidden=!show;$('answer-toggle').setAttribute('aria-expanded',String(show));$('answer-toggle').textContent=show?'Hide the reasoning ↑':'Compare your reasoning ↓';});
 $('copy-code').addEventListener('click',async()=>{try{await navigator.clipboard.writeText($('chapter-code').textContent);$('copy-code').textContent='Copied ✓';}catch{$('copy-code').textContent='Select and copy the code below';}});
 $('export-notes').addEventListener('click',()=>{const blob=new Blob([JSON.stringify({course:'Embodied RL curriculum',...state},null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='embodied-rl-lab-notes.json';a.click();URL.revokeObjectURL(url);});
 $('runtime-open').addEventListener('click',()=>$('runtime').scrollIntoView({behavior:'smooth'}));
 $('source-link').href=config.sourceURL;
 $('menu-toggle').addEventListener('click',()=>{const open=$('menu-toggle').getAttribute('aria-expanded')!=='true';$('menu-toggle').setAttribute('aria-expanded',String(open));$('course-navigation').classList.toggle('open',open);});
 document.addEventListener('keydown',e=>{if(e.key==='Escape'){$('course-navigation').classList.remove('open');$('menu-toggle').setAttribute('aria-expanded','false');}});
 document.querySelectorAll('[data-runtime]').forEach(b=>b.addEventListener('click',()=>runtime(b.dataset.runtime)));
 document.querySelectorAll('[data-jump]').forEach(a=>a.addEventListener('click',e=>{e.preventDefault();select(lessons.findIndex(l=>l.slug===a.dataset.jump),true);}));
 window.addEventListener('hashchange',()=>{const i=lessons.findIndex(l=>l.slug===location.hash.slice(1));if(i>=0&&i!==index)select(i,true);});
 const evidence=window.COURSE_EVIDENCE||{};
 $('evidence-content').innerHTML=`<p><strong>${lessons.length} long-form chapters</strong> with ${lessons.reduce((s,l)=>s+l.word_count,0).toLocaleString()} words of notes, worked examples, original mathematical sketches, and 18 companion notebooks.</p><p><strong>${evidence.total_code_cells??'See execution report'} default code cells executed on Linux CPU.</strong> The tests include actual Gymnasium transitions, original optimizer updates, and vendor-geometry MuJoCo robot tasks. Small runs verify implementation; they are not research-grade performance comparisons.</p><p>Original code covers returns, Bellman backups, DQN updates, REINFORCE, AC/GAE, PPO, TD3, and SAC. DDPG is a derivation bridge. Research mechanisms are labeled separately from complete paper reproductions. Browser layout QA was unavailable.</p><a href="curriculum/execution.json" download>Notebook execution report ↗</a>`;
 runtime(['mac','colab','sb3','lerobot','rlinf'].includes(state.runtime)?state.runtime:'mac');const initial=lessons.findIndex(l=>l.slug===location.hash.slice(1));select(initial>=0?initial:0);
})();
