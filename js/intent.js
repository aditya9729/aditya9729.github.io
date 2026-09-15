/*
 * The Glasshouse: a deliberately hand-wired interpretability sketch.
 * No language model, learned features, model traces or hidden-state extraction.
 * Human intent and the agents' incentive structure are independent inputs.
 * The intervention removes only the authored intent features; all other inputs
 * and all action-score weights remain fixed for the paired comparison.
 */
'use strict';
window.LAB = window.LAB || {};

LAB.IntentExperiment = (() => {
  const A = LAB.Art;
  const COLORS = {
    ink: '#e9f3f0', dim: '#93b0bf', line: '#345168', dark: '#0b1729',
    human: '#f7c585', fair: '#95ebda', fast: '#b9acff', joint: '#95ebda',
    own: '#f2aac9', split: '#95ebda', relay: '#b9acff', race: '#f2aac9',
  };
  const ACTIONS = ['split', 'relay', 'race'];
  const ALLOCATION = {split: [1, 1], relay: [0, 2], race: [2, 0]};

  /** Evaluate the toy circuit; values are authored utilities, not probabilities. */
  function evaluate(goal, regime, muted = false) {
    if (!['fair', 'fast'].includes(goal) || !['cooperative', 'competitive'].includes(regime)) {
      throw new TypeError('Unknown human intent or agent incentive structure.');
    }
    const features = {
      fair: !muted && goal === 'fair' ? 3.8 : 0,
      fast: !muted && goal === 'fast' ? 3.8 : 0,
      joint: regime === 'cooperative' ? 2 : 0,
      own: regime === 'competitive' ? 3.2 : 0.6,
    };
    const scores = {
      split: features.fair + features.joint,
      relay: features.fast + features.joint,
      race: features.own + features.fast * 0.4,
    };
    // Stable tie-break order is part of the toy, not an inferred preference.
    const choice = ACTIONS.reduce((best, a) => scores[a] > scores[best] ? a : best, 'split');
    return {goal, regime, muted: Boolean(muted), features, scores, choice,
      allocation: [...ALLOCATION[choice]]};
  }

  function build(g) {
    let goal = 'fair', regime = 'cooperative', muted = false;
    let testedIntervention = false;
    const seen = new Set(['fair/cooperative']);
    const current = () => evaluate(goal, regime, muted);
    const intact = () => evaluate(goal, regime, false);
    const describe = a => ({split: 'split the cells', relay: 'relay to B', race: 'keep both with A'}[a]);

    function controls() {
      const button = (label, fn, extras = {}) => ({type: 'button', label, fn, ...extras});
      g.setActions([
        button(`Intent: ${goal}`, () => {goal = goal === 'fair' ? 'fast' : 'fair'; changed('goal');},
          {ariaLabel: `Change human intent. Currently ${goal === 'fair' ? 'share fairly' : 'finish quickly'}.`}),
        button(`Agents: ${regime === 'cooperative' ? 'co-op' : 'versus'}`,
          () => {regime = regime === 'cooperative' ? 'competitive' : 'cooperative'; changed('regime');},
          {ariaLabel: `Switch agent incentives. Currently ${regime}.`}),
        button(muted ? 'Restore intent' : 'Mute intent', () => {
          muted = !muted;
          if (muted) testedIntervention = true;
          changed('intervention');
        }, {key: true, pressed: muted, ariaLabel: 'Mute human-intent features in the illustrative circuit'}),
      ]);
    }

    function changed(reason) {
      // Discovery requires all four intact comparisons, not simply click count.
      if (!muted) seen.add(`${goal}/${regime}`);
      controls();
      const a = intact(), b = current();
      if (muted) {
        g.say(a.choice === b.choice
          ? `Intent muted. The scores change; the choice stays ${b.choice}.`
          : `Same scene, intent muted: ${a.choice} → ${b.choice}.`);
      } else {
        g.say(`${goal === 'fair' ? 'Share fairly' : 'Finish quickly'} · ${regime} → ${describe(b.choice)}.`);
      }
      if (seen.size === 4 && testedIntervention) {
        // Keep the useful intervention result instead of replacing it with confetti copy.
        g.finish(g.message);
      }
    }
    controls();
    g.say('Two cells. B delivers faster. Try fair vs fast.');
    g.snapshot = () => ({goal, regime, muted, testedIntervention,
      seen: [...seen], intact: intact(), current: current()});

    function trace(c, points, color, amount, t, selected = false, cut = false) {
      c.save();
      c.setLineDash(cut ? [4, 5] : []);
      A.line(c, points, amount ? `${color}${selected ? 'd9' : '68'}` : '#294050', selected ? 2.3 : 1.4);
      c.setLineDash([]);
      if (amount && !cut) {
        const phase = (t * 0.45 + amount * 0.07) % 1;
        const lengths = points.slice(1).map((p,i) => Math.hypot(p[0]-points[i][0],p[1]-points[i][1]));
        let distance = phase * lengths.reduce((a,b) => a+b,0);
        for(let i=0;i<lengths.length;i++) {
          if(distance<=lengths[i] || i===lengths.length-1) {
            const p=points[i],q=points[i+1],k=distance/(lengths[i]||1);
            A.circle(c,p[0]+(q[0]-p[0])*k,p[1]+(q[1]-p[1])*k,3,color);
            break;
          }
          distance-=lengths[i];
        }
      }
      c.restore();
    }

    function cell(c, x, y, color, ghost = false, s = 1) {
      A.R(c, x - 5*s, y - 8*s, 10*s, 15*s, ghost ? '#344956' : color);
      A.R(c, x - 2*s, y - 10*s, 4*s, 2*s, ghost ? '#344956' : color);
      if (!ghost) {
        A.R(c, x - 3*s, y - 5*s, 6*s, 3*s, '#1f4353');
        A.R(c, x - 3*s, y + 1*s, 6*s, 3*s, '#1f4353');
      }
    }

    function outcome(g, c, x, y, width, height, state, title, enabled) {
      const compact = g.W < 500;
      const textSize = compact ? 15 : 12;
      A.R(c, x, y, width, height, enabled ? '#1b3844' : '#1b2a3e');
      A.R(c, x + 1, y + 1, width - 2, height - 2, '#0a1828');
      A.text(c, title, x + width/2, y + 17, enabled ? COLORS.ink : COLORS.dim, textSize, 'center');
      if(height<64) {
        const cy = y + height - 12;
        for(let i=0;i<2;i++) {
          const xx=x+width*(i?.67:.22);
          A.text(c,i?'B':'A',xx-13,cy+4,i?COLORS.fair:COLORS.human,15,'center');
          if(!state) A.text(c,'?',xx+9,cy+4,COLORS.dim,15,'center');
          else if(!state.allocation[i]) A.text(c,'–',xx+9,cy+4,COLORS.dim,15,'center');
          else for(let j=0;j<state.allocation[i];j++) cell(c,xx+7+j*13,cy,COLORS[state.choice],false,.85);
        }
        return;
      }
      const botY = y + height - 14;
      const ax = x + width*.28, bx = x + width*.72;
      const s = Math.min(1.25, Math.max(0.65, (height - 35)/25));
      A.robot(c, ax, botY, s, COLORS.human);
      A.robot(c, bx, botY, s, COLORS.fair);
      A.text(c, 'A', ax - 20, botY - 4, COLORS.human, textSize, 'center');
      A.text(c, 'B', bx - 20, botY - 4, COLORS.fair, textSize, 'center');
      if (state) {
        state.allocation.forEach((n, i) => {
          const xx = i ? bx : ax;
          for (let j = 0; j < n; j++) cell(c, xx + 15 + j*14, botY - 10*s, COLORS[state.choice], false, s*.8);
        });
        // A tiny arrow shows the extra handoff in the relay action.
        if (state.choice === 'relay') {
          A.line(c, [[ax+13,botY-8],[bx-17,botY-8]], COLORS.relay);
          A.poly(c, [[bx-14,botY-8],[bx-20,botY-12],[bx-20,botY-4]], COLORS.relay);
        }
      } else {
        A.text(c, '?', x+width/2, botY-5, COLORS.dim, 22, 'center');
      }
    }

    // A simpler diagram on very short phone / landscape canvases: no tiny
    // activation numbers and no overlapping robot sprites. It computes the same circuit.
    function drawCompact(c,t,model,baseline) {
      const w=g.W,h=g.H;
      A.R(c,0,0,w,h,'#0a1426');
      const y0=20,step=(h*.57-20)/3;
      const features={fair:[w*.49,y0],fast:[w*.49,y0+step],joint:[w*.49,y0+step*2],own:[w*.49,y0+step*3]};
      const inputs={human:[w*.135,y0+step*.45],agents:[w*.135,y0+step*2.5]};
      const outputs={split:[w*.865,y0],relay:[w*.865,y0+step*1.5],race:[w*.865,y0+step*3]};
      for(const f of ['fair','fast']) trace(c,[inputs.human,features[f]],COLORS.human,model.features[f],t,false,muted);
      for(const f of ['joint','own']) trace(c,[inputs.agents,features[f]],COLORS[f],model.features[f],t);
      for(const [f,a,weight] of [['fair','split',1],['fast','relay',1],['fast','race',.4],['joint','split',1],['joint','relay',1],['own','race',1]]){
        trace(c,[features[f],outputs[a]],COLORS[a],model.features[f]*weight,t,model.choice===a);
      }
      for(const [id,pt] of Object.entries(inputs)) {
        const col=id==='human'?COLORS.human:COLORS[regime==='cooperative'?'joint':'own'];
        g.card(c,pt[0]-w*.11,pt[1]-12,w*.22,24,col,'#132237');
        A.text(c,id==='human'?goal.toUpperCase():regime==='cooperative'?'CO-OP':'VERSUS',pt[0],pt[1]+5,col,17,'center');
      }
      for(const [id,pt] of Object.entries(features)) {
        const col=model.features[id]?COLORS[id]:COLORS.dim;
        g.card(c,pt[0]-31,pt[1]-10,62,20,col,'#132237');
        A.text(c,({joint:'TEAM',own:'SELF'}[id]||id).toUpperCase(),pt[0],pt[1]+5,col,15,'center');
        if(muted&&['fair','fast'].includes(id))A.line(c,[[pt[0]-31,pt[1]+10],[pt[0]+31,pt[1]-10]],COLORS.own,2);
      }
      for(const [id,pt] of Object.entries(outputs)){
        const col=id===model.choice?COLORS[id]:COLORS.dim;
        g.card(c,pt[0]-w*.11,pt[1]-11,w*.22,22,col,id===model.choice?'#233c48':'#0a192b');
        A.text(c,id.toUpperCase(),pt[0],pt[1]+5,col,17,'center');
      }
      const panelY=h*.66,panelH=h-panelY-20;
      outcome(g,c,10,panelY,(w-30)/2,panelH,baseline,'INTACT',!muted);
      outcome(g,c,(w+10)/2,panelY,(w-30)/2,panelH,muted?model:null,muted?'MUTED':'COMPARE?',muted);
      A.text(c,'HAND-WIRED CIRCUIT',10,h-5,COLORS.dim,12);
      A.text(c,`${seen.size}/4 + ${testedIntervention?1:0}/1`,w-10,h-5,COLORS.fair,12,'right');
    }

    g.draw = (c, t) => {
      const w = g.W, h = g.H, small = w < 500;
      const model = current(), baseline = intact();
      if(h<230){drawCompact(c,t,model,baseline);return;}
      A.R(c, 0, 0, w, h, '#0a1426');
      for (let x=16; x<w; x+=30) A.line(c, [[x,0],[x,h]], '#8ccdd60b');
      for (let y=12; y<h; y+=30) A.line(c, [[0,y],[w,y]], '#8ccdd60b');
      A.glow(c, w*.5, h*.28, w*.36, muted ? COLORS.own : COLORS.fair, .065);
      const fs = small ? 15 : 12;
      const head = small ? 14 : 10;
      A.text(c, 'HUMAN / AGENTS', w*.145, 19, COLORS.dim, head, 'center');
      A.text(c, 'TOY FEATURES', w*.49, 19, COLORS.dim, head, 'center');
      A.text(c, 'CHOICE', w*.855, 19, COLORS.dim, head, 'center');

      const top = 35, bottom = h*.59;
      const span = Math.max(74, bottom-top);
      const inputs = {human: [w*.145, top+span*.23], agents: [w*.145, top+span*.77]};
      const features = {
        fair: [w*.40, top+span*.22], fast: [w*.56, top+span*.22],
        joint: [w*.40, top+span*.76], own: [w*.56, top+span*.76],
      };
      const outputs = {
        split: [w*.855, top+span*.09], relay: [w*.855, top+span*.47], race: [w*.855, top+span*.85],
      };
      for (const feature of ['fair','fast']) {
        const from=inputs.human,to=features[feature];
        const path=feature==='fast' ? [from,[w*.30,from[1]-29],[to[0],from[1]-29],to] : [from,to];
        trace(c,path,COLORS.human,model.features[feature],t,false,muted);
      }
      for (const feature of ['joint','own']) trace(c, [inputs.agents,features[feature]], COLORS[feature], model.features[feature], t);
      const edges = [['fair','split',1],['fast','relay',1],['fast','race',.4],['joint','split',1],['joint','relay',1],['own','race',1]];
      for (const [f, action, weight] of edges) {
        const from=features[f],to=outputs[action];
        const path=f==='fair' ? [from,[from[0]+14,from[1]-26],[w*.70,from[1]-26],to] : [from,to];
        trace(c,path,COLORS[action],model.features[f]*weight,t,model.choice===action);
      }

      const inputW = w*.22, inputH = Math.min(39,span*.32);
      for (const [id, point] of Object.entries(inputs)) {
        const col = id==='human' ? COLORS.human : COLORS[regime==='cooperative'?'joint':'own'];
        g.card(c, point[0]-inputW/2,point[1]-inputH/2,inputW,inputH,muted&&id==='human'?COLORS.line:col,'#142337');
        A.text(c, id==='human' ? goal.toUpperCase() : regime==='cooperative'?'CO-OP':'VERSUS',
          point[0],point[1]+5,col,small?17:16,'center');
      }
      const radius = Math.min(w*.045,span*.16);
      for (const [id, point] of Object.entries(features)) {
        const active = model.features[id]>0, col = COLORS[id];
        A.glow(c,...point,radius*2,col,active?.13:.025);
        A.circle(c,...point,radius,active?col:COLORS.line);
        A.circle(c,...point,radius-2,'#132237');
        // Magnitude is the actual authored feature activation, not decoration.
        A.text(c,model.features[id].toFixed(1),point[0],point[1]+4,active?col:COLORS.dim,small?15:12,'center');
        A.text(c,id.toUpperCase(),point[0],point[1]+radius+15,active?col:COLORS.dim,fs,'center');
        if (muted && ['fair','fast'].includes(id)) {
          A.line(c,[[point[0]-radius,point[1]-radius],[point[0]+radius,point[1]+radius]],COLORS.own,2);
        }
      }
      for (const [id, point] of Object.entries(outputs)) {
        const selected = id===model.choice, col = COLORS[id], bw=w*.22, bh=Math.min(35,span*.27);
        g.card(c,point[0]-bw/2,point[1]-bh/2,bw,bh,selected?col:COLORS.line,selected?'#233548':'#0c192b');
        A.text(c,id.toUpperCase(),point[0]-bw*.39,point[1]+5,selected?col:COLORS.dim,fs);
        A.text(c,model.scores[id].toFixed(1),point[0]+bw*.4,point[1]+5,selected?col:COLORS.dim,fs,'right');
        if (selected) A.poly(c,[[point[0]-bw/2-9,point[1]-4],[point[0]-bw/2-3,point[1]],[point[0]-bw/2-9,point[1]+4]],col);
      }
      const panelY = Math.max(h*.66,bottom+20), panelH = h-panelY-23;
      outcome(g,c,14,panelY,(w-42)/2,panelH,baseline,'INTACT',!muted);
      outcome(g,c,(w+14)/2,panelY,(w-42)/2,panelH,muted?model:null,muted?'INTENT MUTED':'MUTE TO COMPARE',muted);
      A.text(c,'FIXED SCENE · HAND-WIRED CIRCUIT',13,h-6,COLORS.dim,small?12:10);
      A.text(c,`${seen.size}/4 + ${testedIntervention?1:0}/1`,w-13,h-6,COLORS.fair,small?12:10,'right');
    };
  }

  return Object.freeze({build, evaluate});
})();
