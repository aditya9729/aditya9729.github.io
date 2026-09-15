/* Original procedural pixel art. No remote images, fonts, engine or asset packs. */
'use strict';
window.LAB = window.LAB || {};
LAB.Art = (()=>{
 const P = {
  hub:['#090e25','#34345c','#a97065','#ffc777','#254450'],
  coast:['#0b1c36','#24576a','#409196','#78e6ee','#2c5461'],
  aurora:['#0d1835','#2e4a66','#718294','#adf6d4','#23555a'],
  city:['#100c28','#392553','#513a75','#c399ff','#2e345e'],
  terminal:['#061b26','#163c4a','#2b6261','#8fffb9','#244854'],
  rain:['#090f29','#212e58','#494b77','#8ed3ff','#203b51'],
  arcade:['#190d32','#3c225d','#8d3e79','#ffbc72','#593969'],
  desert:['#251938','#794c59','#d79a66','#ffdb8d','#675771'],
  orbital:['#080e2b','#26234b','#63445b','#ffb983','#383e69'],
  vault:['#081b24','#1b4046','#40666c','#8de7cb','#254d49'],
  garden:['#10252d','#2b5957','#749178','#a5f7bf','#34694f'],
  ruins:['#112737','#376273','#7eaaa2','#b3f1ca','#36585b'],
  underwater:['#071b37','#17436a','#266f84','#7aecdb','#17526b'],
  moon:['#15142f','#39345b','#846786','#ffc2d9','#564e72'],
  dream:['#141638','#534075','#9d6b96','#abedec','#624b88'],
  glacier:['#0c2542','#436b89','#aed4d7','#d0fff7','#407b96'],
  workshop:['#23172a','#63404a','#b88463','#ffc69c','#715658'],
  furnace:['#1b1426','#593443','#bd7651','#ffc668','#7c4147'],
  nebula:['#161034','#41265f','#725093','#dfb3ff','#38365e'],
  memory:['#101c34','#344b6b','#6b779b','#99bbed','#46567a'],
  automata:['#181f2f','#384b55','#7f8a74','#dddf9f','#455d5d'],
  glasshouse:['#11142f','#3c3e65','#9a88ae','#a8f3de','#395b67'],
  cosmos:['#071525','#213e58','#547c8c','#94f2d4','#2c566a']
 };
 const rand = seed => { let n=seed|0; return ()=>{n^=n<<13;n^=n>>>17;n^=n<<5;return (n>>>0)/4294967296;};};
 const hash=s=>[...s].reduce((v,c)=>(v*31+c.charCodeAt(0))|0,123);
 const R=(c,x,y,w,h,col)=>{c.fillStyle=col;c.fillRect(Math.round(x),Math.round(y),Math.max(1,Math.round(w)),Math.max(1,Math.round(h)));};
 const line=(c,pts,col,width=1)=>{c.strokeStyle=col;c.lineWidth=width;c.beginPath();pts.forEach((p,i)=>i?c.lineTo(...p):c.moveTo(...p));c.stroke();};
 function poly(c,pts,col){c.fillStyle=col;c.beginPath();pts.forEach((p,i)=>i?c.lineTo(...p):c.moveTo(...p));c.closePath();c.fill();}
 function circle(c,x,y,r,col){c.fillStyle=col;c.beginPath();c.arc(Math.round(x),Math.round(y),r,0,Math.PI*2);c.fill();}
 function glow(c,x,y,r,col,alpha=.22){c.save();c.globalAlpha=alpha;let g=c.createRadialGradient(x,y,0,x,y,r);g.addColorStop(0,col);g.addColorStop(1,'transparent');c.fillStyle=g;c.fillRect(x-r,y-r,r*2,r*2);c.restore();}
 function text(c,txt,x,y,col='#dbe6eb',size=7,align='left'){c.font=`bold ${size}px "Courier New", monospace`;c.fillStyle=col;c.textAlign=align;c.fillText(txt,Math.round(x),Math.round(y));c.textAlign='left';}
 function stars(c,w,h,rng){for(let i=0;i<w*h/380;i++){let x=rng()*w,y=rng()*h*.75;R(c,x,y,rng()>.94?2:1,1,['#afbfd1','#717693','#d2b096'][i%3]);}}
 function planet(c,x,y,r,col,rng,ring=false){glow(c,x,y,r*1.8,col,.13);c.save();c.beginPath();c.arc(x,y,r,0,7);c.clip();let g=c.createLinearGradient(x-r,y-r,x+r,y+r);g.addColorStop(0,col);g.addColorStop(.4,col);g.addColorStop(1,'#26203f');c.fillStyle=g;c.fillRect(x-r,y-r,r*2,r*2);for(let i=0;i<r*8;i++){let px=x-r+rng()*r*2,py=y-r+rng()*r*2;R(c,px,py,2+rng()*7,1+rng()*3,rng()>.5?'#ffdfa11a':'#31244635');}c.restore();if(ring){c.strokeStyle=col+'80';c.lineWidth=3;c.beginPath();c.ellipse(x,y,r*1.7,r*.28,-.3,0,7);c.stroke();}}
 function hills(c,w,h,rng,col,y,amp){const points=[[0,h],[0,y]];for(let x=0;x<=w+15;x+=15)points.push([x,y-Math.sin(x*.012)*amp*.3-rng()*amp]);points.push([w,h]);poly(c,points,col);}
 function tower(c,x,y,w,h,col,light,rng){R(c,x,y-h,w,h,col);R(c,x+3,y-h-3,w-6,4,col);if(rng()>.6){R(c,x+w*.52,y-h-17,1,18,col);R(c,x+w*.52,y-h-17,2,2,light);}for(let j=y-h+8;j<y-3;j+=8)for(let i=x+4;i<x+w-3;i+=7){let a=rng();R(c,i,j,2,3,a>.59?light:(a>.3?'#384552':'#222c46'));}R(c,x+w-2,y-h,2,h,'#04091c44');}
 function buildings(c,w,h,rng,y,p,layer){for(let x=-10;x<w;x+=17+rng()*38){let bw=14+rng()*35,bh=20+rng()*(layer?80:55);tower(c,x,y,bw,bh,layer?'#172338':'#29324c',layer?'#f5ba76':'#7d7494',rng);}}
 function tree(c,x,y,s,col,rng){R(c,x-2*s,y-25*s,4*s,28*s,'#394550');for(let j=0;j<4;j++){let yy=y-(j*8+14)*s;poly(c,[[x,yy-12*s],[x-(17-j*2)*s,yy+7*s],[x+(16-j*2)*s,yy+7*s]],col);}for(let i=0;i<12;i++)R(c,x+(rng()-.5)*24*s,y-(12+rng()*28)*s,2*s,s,'#a3d6a715');}
 function arch(c,x,y,w,h,col){R(c,x,y-h,w,h,col);R(c,x+7,y-h+9,w-14,h-8,'#0a1823');R(c,x+4,y-h-3,w-8,5,col);for(let yy=y-h;yy<y;yy+=8){R(c,x,yy,w,1,'#96b9a51a');}R(c,x-3,y,8,3,col);R(c,x+w-5,y,8,3,col);}
 function makeBackground(biome,w,h,seed=42){
  const c=document.createElement('canvas');c.width=w;c.height=h;const x=c.getContext('2d');const p=P[biome]||P.hub,rng=rand(hash(biome)+seed);const ground=Math.round(h*.79);
  let sky=x.createLinearGradient(0,0,0,h);sky.addColorStop(0,p[0]);sky.addColorStop(.64,p[1]);sky.addColorStop(1,p[2]);x.fillStyle=sky;x.fillRect(0,0,w,h);stars(x,w,h,rng);
  // Distant atmosphere, kept free of baked-in lettering.
  for(let i=0;i<15;i++){let xx=rng()*w,yy=h*(.12+rng()*.53);glow(x,xx,yy,35+rng()*65,p[2],.08);for(let j=0;j<4;j++)R(x,xx+j*10,yy+rng()*6,18+rng()*25,1,'#dcbcd10c');}
  if(['hub','moon','orbital','dream','nebula','arcade','cosmos'].includes(biome)){planet(x,w*.73,h*.27,Math.min(68,w*.13),p[2],rng,['orbital','nebula','cosmos'].includes(biome));if(biome==='dream')planet(x,w*.22,h*.34,16,p[3],rng,true);}
  if(['coast','aurora','glacier'].includes(biome)){
   for(let j=0;j<5;j++){let pts=[];for(let xx=-5;xx<w;xx+=8)pts.push([xx,h*.25+Math.sin(xx*.012+j*.7)*h*.065+j*7]);line(x,pts,['#51a9ad24','#7df4c924','#7ab7d828'][j%3],7);}
   planet(x,w*.82,h*.25,17,'#c2ece5',rng);
  }
  if(['desert','workshop','furnace'].includes(biome))planet(x,w*.72,h*.34,39,p[3],rng);
  hills(x,w,h,rng,p[1],h*.61,45);hills(x,w,h,rng,p[4],h*.68,29);
  if(['hub','city','terminal','rain','arcade'].includes(biome)){
   buildings(x,w,h,rng,h*.69,p,0);buildings(x,w,h,rng,h*.76,p,1);
   // Observatory dome and antennas.
   let cx=w*.67,cy=h*.61;circle(x,cx,cy,27,'#19243c');R(x,cx-28,cy,56,24,'#172136');line(x,[[cx-25,cy],[cx-18,cy-18],[cx,cy-27],[cx+18,cy-18],[cx+25,cy]],'#736073',2);R(x,cx-2,cy-48,2,23,'#998095');R(x,cx-16,cy+5,5,9,p[3]);R(x,cx+7,cy+5,5,9,p[3]);
   for(let i=0;i<8;i++){let xx=rng()*w;tree(x,xx,ground,.45+rng()*.5,'#193d40',rng);}
  } else if(['coast','aurora'].includes(biome)){
   R(x,0,ground-45,w,48,'#244e63');for(let j=0;j<18;j++){let yy=ground-44+j*3;for(let i=0;i<7;i++){let xx=rng()*w;R(x,xx,yy,8+rng()*24,1,j%3===0?'#69b1b76b':'#3b7a8a50');}}
   let lx=w*.84;poly(x,[[lx-10,ground-7],[lx-6,ground-78],[lx+6,ground-78],[lx+10,ground-7]],'#9aafb0');R(x,lx-7,ground-67,14,7,'#477786');R(x,lx-9,ground-81,18,10,'#ddd5b1');R(x,lx-12,ground-84,24,4,'#35465d');poly(x,[[lx-13,ground-84],[lx,ground-93],[lx+13,ground-84]],'#34485e');glow(x,lx,ground-76,37,p[3],.14);poly(x,[[lx-5,ground-78],[0,ground-95],[0,ground-50],[lx-5,ground-74]],'#b5eff012');
   for(let i=0;i<5;i++){let xx=rng()*w;R(x,xx,ground-21,2,22,'#5c7686');R(x,xx-3,ground-23,8,3,'#85a7ad');}
  } else if(biome==='glasshouse'){
   // A transparent observatory: opposing colored beams meet in glass prisms.
   // Purely scenic; the experiment, not these light beams, computes decisions.
   planet(x,w*.78,h*.21,Math.min(31,w*.085),'#dcb9d3',rng);
   const horizon=ground-14;
   for(let i=0;i<6;i++){
    const xx=(i-.2)*w/4.8,ww=w/5.7,hh=67+(i%3)*15,top=horizon-hh;
    poly(x,[[xx,horizon],[xx,top+18],[xx+ww/2,top-9],[xx+ww,top+18],[xx+ww,horizon]],'#5c92941b');
    line(x,[[xx,horizon],[xx,top+18],[xx+ww/2,top-9],[xx+ww,top+18],[xx+ww,horizon]],'#94ceca86');
    line(x,[[xx+ww/2,top-9],[xx+ww/2,horizon]],'#cae8ea55');
    line(x,[[xx,top+18],[xx+ww,top+18]],'#cae8ea55');
    line(x,[[xx,top+hh*.61],[xx+ww,top+hh*.61]],'#94ceca50');
    poly(x,[[xx+2,top+22],[xx+ww/2,top+1],[xx+ww/2-3,horizon],[xx+2,horizon]],'#c8e6dc08');
    const yy=top+hh*.65;
    cube(x,xx+ww/2,yy,Math.min(13,ww*.22),i%2?'#ebb9d8':'#a8f3de');
    glow(x,xx+ww/2,yy,25,i%2?'#ebb9d8':'#a8f3de',.14);
   }
   for(let i=0;i<3;i++){
    const yy=h*.24+i*13;
    line(x,[[0,yy],[w*.28,yy+25],[w*.53,yy-6],[w*.75,yy+28],[w,yy]],i%2?'#dfa2cd50':'#a7eade44');
    for(const xx of [w*.28,w*.53,w*.75]){circle(x,xx,yy+7,2,'#ccefe9');glow(x,xx,yy+7,10,'#a8f3de',.12);}
   }
   R(x,0,horizon+3,w,7,'#3e69714a');
   for(let i=0;i<12;i++)line(x,[[i*w/11,horizon+9],[i*w/11+18,ground]],'#bee2df26');
  } else if(biome==='memory'){
   for(let i=0;i<8;i++){let xx=i*w/7,yy=ground-18-rng()*80;R(x,xx,yy,34,42,'#24344e');R(x,xx+2,yy+3,30,3,p[3]);for(let j=0;j<4;j++){R(x,xx+5,yy+10+j*6,3,3,p[3]);R(x,xx+12,yy+11+j*6,14+rng()*7,1,'#6384aa');}line(x,[[xx+17,yy+42],[xx+17,ground]],'#506488',1);}
  } else if(['garden','ruins','underwater','vault'].includes(biome)){
   for(let i=0;i<9;i++){let xx=i*w/8;arch(x,xx,ground+8,23+rng()*20,35+rng()*90,p[4]);if(i%3===0)circle(x,xx+15,ground-58,4,p[3]);}
   for(let i=0;i<22;i++){let xx=rng()*w,yy=ground+rng()*25;let branch=[[xx,yy],[xx-5,yy-20],[xx+3,yy-32]];line(x,branch,p[3]+'60',2);circle(x,xx+3,yy-32,3,p[3]);circle(x,xx-8,yy-17,2,p[3]);}
   if(biome==='underwater'){for(let i=0;i<5;i++)poly(x,[[w*.65+i*32,-1],[w*.32+i*55,ground],[w*.44+i*55,ground],[w*.72+i*32,-1]],'#abe4e30a');}
  } else if(biome==='desert'){
   for(let k=0;k<4;k++){const pts=[[0,h]];for(let xx=0;xx<=w+10;xx+=8)pts.push([xx,ground-50+k*23+Math.sin(xx*.013+k*2.2)*15]);pts.push([w,h]);poly(x,pts,['#916963','#b28369','#c49474','#876674'][k]);}
   for(let i=0;i<5;i++){let xx=w*(.12+i*.19),yy=ground-5;line(x,[[xx,yy],[xx,yy-40]],'#57465c',2);line(x,[[xx,yy-35],[xx-16,yy-43]],'#ddbba6',2);line(x,[[xx,yy-35],[xx+13,yy-45]],'#ddbba6',2);line(x,[[xx,yy-35],[xx+1,yy-16]],'#ddbba6',2);poly(x,[[xx-20,yy],[xx-14,yy-11],[xx+6,yy-11],[xx,yy]],'#364c69');}
  } else if(biome==='glacier'){
   for(let i=0;i<7;i++){let xx=i*w/6,hh=40+rng()*65;poly(x,[[xx-40,ground+5],[xx,ground-hh],[xx+48,ground]],'#84b7c9');poly(x,[[xx-10,ground-hh+26],[xx,ground-hh],[xx+30,ground-hh+47]],'#c6e4e2');line(x,[[xx,ground-hh],[xx+5,ground-15],[xx+20,ground]],'#507f9c',1);}
  } else if(['dream','nebula','moon','orbital','cosmos'].includes(biome)){
   for(let i=0;i<7;i++){let xx=i*w/6,yy=h*(.48+rng()*.23);poly(x,[[xx-32,yy],[xx+33,yy],[xx+16,yy+17],[xx-12,yy+26]],p[4]);R(x,xx-34,yy-3,66,4,p[2]);R(x,xx-29,yy-5,58,2,p[3]+'50');if(i%2===0){arch(x,xx-10,yy-4,23,38,p[4]);glow(x,xx,yy-22,17,p[3],.17);}}
   if(biome==='orbital'){line(x,[[0,ground-5],[w,ground-5]],'#9985a8',2);for(let i=0;i<10;i++)R(x,i*w/9,ground-25,2,25,'#6d6287');}
  } else {
   for(let i=0;i<6;i++){let xx=i*w/5;R(x,xx,ground-100,9,103,'#302a40');R(x,xx+1,ground-100,2,103,'#9b766126');for(let j=0;j<3;j++){circle(x,xx+20,ground-50-j*19,7,'#563f4d');circle(x,xx+20,ground-50-j*19,3,'#261f33');}R(x,xx+4,ground-104,75,6,'#453243');}}
  // Lower platform, masonry, pipes and scattered details.
  const platformColor=biome==='glacier'?'#517b92':biome==='desert'?'#564152':'#111e31';R(x,0,ground,w,h-ground,platformColor);R(x,0,ground,w,3,p[3]+'88');R(x,0,ground+3,w,5,'#101423');
  for(let yy=ground+8;yy<h;yy+=13){for(let xx=(Math.floor(yy/13)%2)*-19;xx<w;xx+=38){R(x,xx,yy,36,11,['#182338','#1f2a3d','#253147'][Math.floor(rng()*3)]);R(x,xx+1,yy+1,34,1,'#b4c1cc10');if(rng()>.75)R(x,xx+12,yy+6,8,1,'#070d1c');}}
  for(let i=0;i<w/110;i++){let xx=i*110+20;R(x,xx,ground+15,43,21,'#090f21');R(x,xx+3,ground+18,37,2,p[4]);for(let a=0;a<4;a++)R(x,xx+5+a*9,ground+20,3,11,'#29374c');R(x,xx+41,ground+30,2,2,p[3]);}
  // Few foreground plants / cables so each scene feels inhabited.
  for(let i=0;i<8;i++){let xx=rng()*w;R(x,xx,ground-8,7,8,'#2b3144');for(let j=0;j<3;j++)line(x,[[xx+3,ground-8],[xx+(j-1)*6,ground-17-j*2]],p[4],2);}
  return c;
 }
 function lantern(c,x,y,color='#ffbc70') {R(c,x-1,y,2,51,'#24364b');R(c,x-6,y-3,12,3,'#293749');R(c,x-4,y,8,7,color);R(c,x-5,y-5,10,3,'#706064');glow(c,x,y+4,28,color,.18);R(c,x-4,y+51,8,3,'#384255');}
 function booth(c,x,y,id,col,t=0,done=false){
  const w=106,h=103;R(c,x-w/2-4,y-h-5,w+8,h+9,'#080e20');R(c,x-w/2,y-h,w,h,'#142337');R(c,x-w/2+3,y-h+3,w-6,h-6,'#0e1d2d');
  R(c,x-w/2,y-h,w,2,col);R(c,x-w/2,y-h,2,h,col+'80');R(c,x+w/2-2,y-h,2,h,col+'80');for(let yy=y-h+10;yy<y;yy+=8)R(c,x-w/2+4,yy,w-8,1,'#3e536315');
  glow(c,x,y-h*.5,64,col,.07);R(c,x-48,y-20,96,5,'#536071');R(c,x-43,y-15,4,15,'#425168');R(c,x+39,y-15,4,15,'#425168');
  if(id==='signal'){
   R(c,x-41,y-83,82,53,'#34485b');R(c,x-38,y-80,76,46,'#07192a');for(let j=0;j<4;j++)line(c,[[x-35,y-42-j*9],[x+35,y-42-j*9]],'#2c536055');let pts=[];for(let k=0;k<70;k++)pts.push([x-35+k,y-52-Math.sin(k*.24)*9-(k===44?19:0)]);line(c,pts,col,1);line(c,[[x+9,y-78],[x+9,y-33]],'#ff9b6d');R(c,x-31,y-17,21,6,'#243843');R(c,x-27,y-15,3,2,col);
  }else if(id==='agents'){
   const ns=[[x-24,y-67],[x+8,y-81],[x+25,y-55],[x-8,y-42]];line(c,[ns[0],ns[1],ns[2],ns[3],ns[0]],col+'80');line(c,[ns[0],ns[2]],col+'60');ns.forEach((p,i)=>{circle(c,...p,4,i===2?'#e3928e':col);circle(c,...p,2,'#18243c');});robot(c,x-28,y-20,.8,col,t);robot(c,x+18,y-20,.9,col,t+2);R(c,x-40,y-91,20,13,'#263e57');R(c,x-36,y-87,3,2,col);R(c,x-30,y-84,6,1,col);
  }else if(id==='depin'){
   for(let i=0;i<3;i++){let xx=x-31+i*31;R(c,xx-11,y-67,23,35,'#24374c');R(c,xx-9,y-65,19,29,'#0c1a2e');}poly(c,[[x-30,y-62],[x-39,y-49],[x-31,y-49],[x-34,y-37],[x-21,y-54],[x-29,y-54]],'#a4e8ac');cube(c,x,y-49,9,'#c2a0e4');R(c,x+24,y-59,16,20,'#86d8df');circle(c,x+31,y-54,3,'#244756');R(c,x+27,y-49,8,3,'#244756');line(c,[[x-32,y-74],[x+32,y-74]],col);for(let i=0;i<3;i++)line(c,[[x-32+i*32,y-75],[x-32+i*32,y-68]],col);
  }else if(id==='mbodi'){
   arm(c,[x-33,y-23],[x-28,y-53],[x-3,y-68],col,4);cube(c,x+19,y-36,11,col);for(let i=0;i<3;i++)circle(c,x-8+i*16,y-83-(i%2)*7,3,col);line(c,[[x-8,y-83],[x+8,y-90],[x+24,y-83]],col);R(c,x-42,y-26,21,5,'#647b7b');
  }else{
   arm(c,[x-35,y-23],[x-42,y-59],[x-21,y-79],'#ddd4e5',4);arm(c,[x+34,y-23],[x+39,y-58],[x+15,y-77],'#ddd4e5',4);let pts=[];for(let i=0;i<9;i++)pts.push([x-20+i*4.5,y-77+Math.sin(i/8*Math.PI)*17]);line(c,pts,col,2);for(let j=1;j<5;j++){let ps=pts.map((p,i)=>[p[0]+Math.sin(i)*j,p[1]+j*6]);line(c,ps,col+'80');}R(c,x+13,y-30,16,10,'#b28493');}
  if(done){text(c,'✦',x+37,y-h+13,'#b6ecc5',10);}R(c,x-14,y+2,28,3,col+'bb');
 }
 function cube(c,x,y,s,col){const a=[[x,y-s],[x+s,y-s/2],[x+s,y+s/2],[x,y+s],[x-s,y+s/2],[x-s,y-s/2],[x,y-s]];line(c,a,col);line(c,[[x-s,y-s/2],[x,y],[x+s,y-s/2]],col);line(c,[[x,y],[x,y+s]],col);}
 function arm(c,a,b,d,col='#b5dddd',width=5){line(c,[a,b,d],'#132735',width+3);line(c,[a,b,d],col,width);[a,b,d].forEach(p=>{circle(c,...p,width*.72,'#142d40');circle(c,...p,width*.37,col);});line(c,[[d[0]-4,d[1]+2],[d[0]-4,d[1]+8]],col,2);line(c,[[d[0]+4,d[1]+2],[d[0]+4,d[1]+8]],col,2);}
 function robot(c,x,y,s=1,col='#83d5d4',t=0){c.save();c.translate(x,y);c.scale(s,s);R(c,-8,-20,16,12,'#31435b');R(c,-6,-18,12,8,col);R(c,-4,-16,2,2,'#122236');R(c,2,-16,2,2,'#122236');R(c,-5,-7,10,6,'#667d8b');R(c,-8,-6,3,5,'#35465b');R(c,6,-6,3,5,'#35465b');R(c,-5,-1,3,2,'#7c8793');R(c,2,-1,3,2,'#7c8793');R(c,-1,-24,1,4,'#b6c5d4');R(c,-2,-25,3,2,col);c.restore();}
 const body=[
 '.....HHHHHH.....','....HHHHHHHH....','...HHHHHHHHHH...','...HHHHHHHHHH...','...HHSSSSSSHH...','...HSSSSKSSSH...','....SSSSKSSS....','....SSSSSSS.....','.....SSSS.......','....OOOOOO......','...OOLLOOOO.....','..OOOLOOOOBO....','..OOOLLOOOBOS...','..OOOLLOOOBOS...','..SSOLLOOOBOS...','..SSOOOOOOO.....','....OOOOOOO.....','....DDDDDDD.....','....DDD.DDD.....','....DDD.DDD.....','....DDD.DDD.....','....KKK.KKK.....','...KKKK.KKKK....'];
 function explorer(c,x,y,t,face=1,moving=false){c.save();c.translate(Math.round(x),Math.round(y));c.scale(face*1.45,1.45);const pal={H:'#142133',S:'#e9b887',K:'#101827',O:'#df8556',L:'#efb374',B:'#6d9b9b',D:'#354660'};body.forEach((row,j)=>[...row].forEach((v,i)=>{if(v!=='.'){let off=j>17&&moving?Math.round(Math.sin(t*11)*(i<8?2:-2)):0;R(c,i-8+off,j-23,1,1,pal[v]);}}));c.restore();}
 function dog(c,x,y,t,face=1,moving=false){c.save();c.translate(Math.round(x),Math.round(y));c.scale(face*1.2,1.2);R(c,-10,-9,14,6,'#b47551');R(c,-8,-8,8,3,'#dfbb84');R(c,1,-13,7,9,'#c68e58');poly(c,[[1,-12],[0,-17],[5,-13]],'#a47353');R(c,5,-10,5,3,'#e3c69a');R(c,9,-10,2,2,'#253145');R(c,5,-13,1,1,'#202734');let step=moving?Math.sin(t*12)*2:0;R(c,-8+step,-4,2,4,'#ddb986');R(c,1-step,-4,2,4,'#c69366');line(c,[[-10,-7],[-16,-11],[-16,-13]],'#b88864',2);c.restore();}
 function heart(c,x,y,s=1){poly(c,[[x,y+s],[x-s,y],[x-3*s,y],[x-4*s,y+s],[x-4*s,y+3*s],[x,y+7*s],[x+4*s,y+3*s],[x+4*s,y+s],[x+3*s,y],[x+s,y]],'#efa1bc');}
 function ambience(c,biome,w,h,t,reduced){if(reduced)return;let rng=rand(1289);for(let i=0;i<20;i++){let xx=(rng()*w+t*(i%2?1.7:-.8)+w*20)%w,yy=h*.15+((rng()*h*.62+Math.sin(t*.5+i)*4)% (h*.65));if(biome==='rain'){line(c,[[xx,yy],[xx-2,yy+5]],'#a2c5ff40');}else if(biome==='underwater'){c.strokeStyle='#7bd8ed35';c.beginPath();c.arc(xx,yy,1+i%3,0,7);c.stroke();}else{R(c,xx,yy,1,1,P[biome]?.[3]+'80'||'#edbcab80');}}}
 function thumbnail(canvas,wid){canvas.width=300;canvas.height=155;const c=canvas.getContext('2d');c.drawImage(makeBackground(wid.biome,300,155),0,0);booth(c,152,139,wid.id,wid.color,0,false);robot(c,60,130,1.1,wid.color);lantern(c,247,85,wid.color);}
 return {P,rand,hash,R,line,poly,circle,glow,text,planet,makeBackground,lantern,booth,cube,arm,robot,explorer,dog,heart,ambience,thumbnail};
})();
