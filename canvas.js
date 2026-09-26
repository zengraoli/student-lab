'use strict';
const $=id=>document.getElementById(id), stage=$('stage'), world=$('world');
const STORE='image-lab-canvas-v2';
let nodes=[], selected=[], camera={x:40,y:100,z:1}, tool='select', busy=false, space=false, presets=[], loraCatalog=[], edges=[], importAt=null, lastOp='generate', loraRows=[], group=new Set();
const SVG='http://www.w3.org/2000/svg';
const needsImage=['edit','merge','inpaint','remove_object','remove_background','upscale','collage','pose','pose_extract'];
const regional=['inpaint','remove_object'];
const names={generate:'文生图',edit:'参考编辑',inpaint:'选区重绘',remove_object:'移除物件',remove_background:'移除背景',merge:'多图融合',transparent:'透明素材',upscale:'插值放大',collage:'拼图',pose:'姿势迁移',pose_extract:'姿势骨架'};
const help={generate:'用文字生成新图。需要画布上的图片作参考，请点击“参考生成”。',edit:'选择一张参考图。已自动切到 2048 + VAE 分块：实测 1024 编辑容易过度锐化，甚至不执行指令。约 2 分钟。',inpaint:'选择一张图，用矩形或画笔画出要修改的区域，再描述新内容。输出保留原尺寸，选区外像素不变。',remove_object:'把物件连同阴影完整涂入选区，再在弹出框里写清楚要去掉什么、后面该补成什么（例如“去掉绿色垃圾桶，补成湖面和步道”）。只写“移除物件”时，模型有时会把物件重新画回来。补好的内容只贴回选区。',remove_background:'选择一张图，模型估计透明边缘，再保留原图颜色输出 PNG。复杂毛发、玻璃和轮廓边缘仍需检查。',merge:'Shift + 点击选择 2–4 张参考图，按编号（图1、图2）描述它们的关系。已自动切到 2048 + VAE 分块：实测 1024 融合会失败。约 2 分钟。',transparent:'从文字生成透明素材；棋盘格表示透明区域，下载 PNG 保留 alpha。',upscale:'选择一张图，使用 Lanczos 插值扩大尺寸，不是 AI 超分。',collage:'选择 2–4 张图片，按编号横向排版，不进行语义融合。',pose:'选中一张人物照片（或骨架图），按它的姿势生成新图。Z-Image 约 6 秒，可叠加 LoRA；Qwen 约 2 分钟，还可以再 Shift 选一张人物图保持长相。',pose_extract:'选中一张人物照片，提取 OpenPose 风格的姿势骨架（CPU，约 1–3 秒）。骨架图可以再拿来按姿势生成。'};
function status(text){$('status').textContent=text;}
function persist(){try{localStorage.setItem(STORE,JSON.stringify({nodes,camera,edges}));}catch(e){status('画布保存失败：'+e.message);}}
function transform(){world.style.transform=`translate(${camera.x}px,${camera.y}px) scale(${camera.z})`;$('zoomreset').textContent=Math.round(camera.z*100)+'%';stage.style.backgroundSize=(22*camera.z)+'px '+(22*camera.z)+'px';stage.style.backgroundPosition=`${camera.x}px ${camera.y}px`;}
function point(e){let r=stage.getBoundingClientRect();return{x:(e.clientX-r.left-camera.x)/camera.z,y:(e.clientY-r.top-camera.y)/camera.z};}
function chosen(){return selected.map(id=>nodes.find(n=>n.id===id)).filter(n=>n&&n.type!=='prompt');}
function changeTool(value){tool=value;stage.dataset.tool=value;document.querySelectorAll('[data-tool]').forEach(b=>b.classList.toggle('active',b.dataset.tool===value));document.querySelectorAll('.node canvas').forEach(c=>c.style.pointerEvents=['rect','brush'].includes(value)?'auto':'none');}
function loraAllowed(){const op=$('operation').value;return ['generate','pose'].includes(op)&&$('precision').value==='bf16'&&(op!=='pose'||$('model').value==='zimage');}
function loraOptions(){return loraCatalog.filter(x=>x.model===$('model').value);}
function loraTitle(r){const list=(r.loras&&r.loras.length)?r.loras:[r.lora&&{id:r.lora,scale:r.lora_scale},r.lora2&&{id:r.lora2,scale:r.lora2_scale}].filter(Boolean);return list.length?' · '+list.map(u=>(loraCatalog.find(x=>x.id===u.id)?.title||u.id)+' ×'+u.scale).join(' + '):'';}
function updateLora(){const opts=loraOptions(),allowed=loraAllowed();loraRows=loraRows.filter(r=>opts.some(o=>o.id===r.id));const box=$('lorarows');box.replaceChildren();loraRows.forEach((r,i)=>{const row=document.createElement('div');row.className='lorarow';const sel=document.createElement('select');sel.setAttribute('aria-label','LoRA '+(i+1));for(const o of opts){const op=new Option((o.custom?'★ ':'')+o.title,o.id);op.disabled=loraRows.some((x,j)=>j!==i&&x.id===o.id);sel.add(op);}sel.value=r.id;sel.disabled=!allowed;sel.onchange=()=>{r.id=sel.value;const e=opts.find(o=>o.id===r.id);r.scale=e?.default_scale??0.7;updateLora();};const sc=document.createElement('input');sc.type='number';sc.min=0;sc.max=1.5;sc.step=0.05;sc.value=r.scale;sc.disabled=!allowed;sc.title='强度（0–1.5）';sc.setAttribute('aria-label','LoRA '+(i+1)+' 强度');sc.oninput=()=>{r.scale=Number(sc.value);};const rm=document.createElement('button');rm.type='button';rm.textContent='×';rm.title='移除这个 LoRA';rm.onclick=()=>{loraRows.splice(i,1);updateLora();};row.append(sel,sc,rm);box.append(row);});$('addlora').disabled=!allowed||loraRows.length>=6||loraRows.length>=opts.length;const trig=loraRows.map(r=>opts.find(o=>o.id===r.id)?.trigger).filter(Boolean);$('loratrigger').hidden=!trig.length||!allowed;$('lorahint').textContent=!allowed?'LoRA 只在 BF16 文生图，以及 Z-Image 姿势迁移时可用。':loraRows.length?`已选 ${loraRows.length} 个 LoRA`+(trig.length?`，触发词：${trig.join('、')}`:'')+'。叠加多个时建议把每个强度调低一些。':(opts.length?'点“添加 LoRA”，最多叠加 6 个；★ 是自己训练的。':'这个模型还没有可用的 LoRA。');}
function addLoraRow(id){const opts=loraOptions().filter(o=>!loraRows.some(r=>r.id===o.id));const pick=id?opts.find(o=>o.id===id):opts[0];if(!pick)return false;loraRows.push({id:pick.id,scale:pick.default_scale??0.7});updateLora();return true;}
function collectLoras(){return loraAllowed()?loraRows.map(r=>({id:r.id,scale:Number(r.scale)})):[];}
function update(){const op=$('operation').value,cpu=['upscale','collage','pose_extract'].includes(op),zOk=['pose','pose_extract'];const big=['edit','merge'];if(op!==lastOp){if(big.includes(op)){$('width').value=2048;$('height').value=2048;$('vae_tiling').checked=true;}else if(big.includes(lastOp)){$('width').value=1024;$('height').value=1024;$('vae_tiling').checked=false;}lastOp=op;}$('form').classList.toggle('cpu',cpu);$('referencebox').hidden=!needsImage.includes(op);$('ophelp').textContent=help[op];$('factorbox').hidden=op!=='upscale';$('dimensions').hidden=regional.includes(op)||op==='remove_background'||zOk.includes(op);if(needsImage.includes(op)&&!cpu&&!zOk.includes(op)&&$('model').value!=='qwen'){$('model').value='qwen';$('steps').value=40;}$('model').disabled=needsImage.includes(op)&&op!=='pose';$('posebox').hidden=!(op==='pose'&&$('model').value==='zimage');$('precision').disabled=$('model').value==='zimage';if($('precision').disabled)$('precision').value='bf16';$('run').textContent=op==='pose_extract'?'提取骨架':cpu?'处理图片':regional.includes(op)?'生成选区内容':op==='remove_background'?'提取透明主体':op==='pose'?'按姿势生成':'开始生成';updateLora();}
function select(id,multiple=false){const isPrompt=x=>nodes.find(n=>n.id===x)?.type==='prompt';if(multiple&&(isPrompt(id)||selected.some(isPrompt)))multiple=false;if(multiple){if(selected.includes(id))selected=selected.filter(x=>x!==id);else if(selected.length<4)selected.push(id);else status('最多选择 4 张参考图');}else selected=[id];selection();}
function selection(){let list=chosen();document.querySelectorAll('.node').forEach(el=>{el.classList.toggle('selected',selected.includes(el.dataset.id));const badge=el.querySelector('.badge'),index=selected.indexOf(el.dataset.id);if(badge){badge.hidden=index<0;badge.textContent=index+1;}});$('refcount').textContent=list.length;$('refs').replaceChildren();for(const [i,n] of list.entries()){let b=document.createElement('button');b.type='button';let im=new Image();im.src=n.url;im.alt=n.title;let badge=document.createElement('b');badge.textContent=i+1;b.append(im,badge);b.onclick=()=>{selected=selected.filter(id=>id!==n.id);selection();};b.title='取消选择 '+n.title;$('refs').append(b);}$('selectiontools').hidden=!list.length;if(list[0]){$('download').href=list[0].url;$('download').download=list[0].asset;$('record').textContent=JSON.stringify(list[0].request||{来源:'上传图片',width:list[0].w,height:list[0].h},null,2);}else $('record').textContent='选一张图查看 prompt 与参数。';for(let id of ['regionedit','erase','removebg','enlarge','poseextract'])$(id).disabled=list.length!==1;$('posegen').disabled=!(list.length===1||list.length===2);}
function renderNode(n){if(n.type==='prompt')return renderPromptNode(n);let el=document.createElement('article');el.className='node';el.dataset.id=n.id;el.style.left=n.x+'px';el.style.top=n.y+'px';let head=document.createElement('div');head.className='node-head';head.textContent=n.title;let area=document.createElement('div');area.className='node-image';let img=new Image();img.src=n.url;img.alt=n.title;img.draggable=false;img.onerror=()=>{head.textContent=n.title+' · 图片不可用，请重新导入';};img.onload=()=>drawLinks();let mask=document.createElement('canvas');mask.width=n.w;mask.height=n.h;mask.setAttribute('aria-label','为 '+n.title+' 绘制选区');mask.dataset.painted='false';mask.style.pointerEvents=['rect','brush'].includes(tool)?'auto':'none';area.append(img,mask);let badge=document.createElement('span');badge.className='badge';badge.hidden=true;let port=document.createElement('span');port.className='port out';port.title='拖出连线：连到提示词节点；拖到空白处会新建提示词节点';port.onpointerdown=e=>{if(e.button===0)startLink(n,e);};el.append(head,area,badge,port);world.append(el);
 el.onpointerdown=e=>{if(e.button!==0||space||tool==='pan')return;if(e.target===mask)return;e.stopPropagation();if(group.size>1&&group.has(n.id)&&!e.shiftKey){dragNodes(e,el,groupList());return;}if(group.size)setGroup([]);select(n.id,e.shiftKey);if(e.shiftKey)return;dragNodes(e,el,[n]);};
 mask.onpointerdown=e=>{if(e.button!==0||space||tool==='pan')return;e.stopPropagation();$('regionpop').hidden=true;if(n.w>2048||n.h>2048){status('这张图大于 2048，请下载后重新导入，缩小到编辑尺寸再画选区。');return;}select(n.id);const ctx=mask.getContext('2d',{willReadFrequently:true}),loc=ev=>{let r=mask.getBoundingClientRect();return{x:(ev.clientX-r.left)*mask.width/r.width,y:(ev.clientY-r.top)*mask.height/r.height};},start=loc(e);let last=start;let box={x0:start.x,y0:start.y,x1:start.x,y1:start.y};const grow=(p,r=0)=>{box.x0=Math.min(box.x0,p.x-r);box.y0=Math.min(box.y0,p.y-r);box.x1=Math.max(box.x1,p.x+r);box.y1=Math.max(box.y1,p.y+r);};const saved=ctx.getImageData(0,0,mask.width,mask.height);ctx.fillStyle='rgba(25,106,201,0.48)';ctx.strokeStyle='rgba(25,106,201,0.48)';ctx.lineWidth=Number($('brush').value);ctx.lineCap='round';ctx.lineJoin='round';mask.setPointerCapture(e.pointerId);function stroke(p){ctx.beginPath();ctx.moveTo(last.x,last.y);ctx.lineTo(p.x,p.y);ctx.stroke();last=p;mask.dataset.painted='true';}if(tool==='brush'){ctx.beginPath();ctx.arc(start.x,start.y,ctx.lineWidth/2,0,Math.PI*2);ctx.fill();mask.dataset.painted='true';grow(start,ctx.lineWidth/2);}const move=ev=>{let p=loc(ev);if(tool==='rect'){ctx.putImageData(saved,0,0);ctx.fillRect(Math.min(start.x,p.x),Math.min(start.y,p.y),Math.abs(p.x-start.x),Math.abs(p.y-start.y));mask.dataset.painted='true';box={x0:Math.min(start.x,p.x),y0:Math.min(start.y,p.y),x1:Math.max(start.x,p.x),y1:Math.max(start.y,p.y)};}else{stroke(p);grow(p,ctx.lineWidth/2);}};const end=()=>{mask.removeEventListener('pointermove',move);mask.removeEventListener('pointerup',end);mask.removeEventListener('pointercancel',end);if(mask.dataset.painted==='true')showRegionPop(n,mask,box);status('选区已绘制：在弹出框里写这块区域要改成什么。');};mask.addEventListener('pointermove',move);mask.addEventListener('pointerup',end);mask.addEventListener('pointercancel',end);};
}
function addNode(asset,title,position,request=null){let n={id:crypto.randomUUID(),asset:asset.image_id||asset.id,url:asset.image||asset.url,w:asset.width,h:asset.height,title,x:position.x,y:position.y,request};nodes.push(n);renderNode(n);$('empty').hidden=true;select(n.id);persist();return n;}
function zoom(factor,cx=stage.clientWidth/2,cy=stage.clientHeight/2){const z=Math.max(.15,Math.min(3,camera.z*factor));camera.x=cx-(cx-camera.x)*z/camera.z;camera.y=cy-(cy-camera.y)*z/camera.z;camera.z=z;transform();persist();}
function fit(){if(!nodes.length){camera={x:40,y:100,z:1};transform();return;}let left=Math.min(...nodes.map(n=>n.x)),top=Math.min(...nodes.map(n=>n.y)),right=Math.max(...nodes.map(n=>n.x+320)),bottom=Math.max(...nodes.map(n=>n.y+30+316*n.h/n.w));camera.z=Math.max(.15,Math.min(1.5,(stage.clientWidth-70)/(right-left),(stage.clientHeight-190)/(bottom-top)));camera.x=(stage.clientWidth-(right-left)*camera.z)/2-left*camera.z;camera.y=85-top*camera.z;transform();persist();}
async function api(url,data){const response=await fetch(url,data===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});let result=await response.json();if(!response.ok)throw Error(typeof result.detail==='string'?result.detail:JSON.stringify(result.detail));return result;}
async function upload(files,position){if(busy){status('请等当前任务完成再导入。');return;}busy=true;$('run').disabled=true;let ids=[];try{for(let [i,file] of Array.from(files).entries()){if(!['image/png','image/jpeg','image/webp'].includes(file.type))throw Error('只支持 PNG、JPEG 和 WebP');if(file.size>16000000)throw Error('单张图片不能超过 16 MB');let data=await new Promise((resolve,reject)=>{let r=new FileReader();r.onload=()=>resolve(r.result);r.onerror=reject;r.readAsDataURL(file);});let asset=await api('/api/images',{data});ids.push(addNode(asset,file.name,{x:position.x+i*350,y:position.y}).id);}selected=ids.slice(0,4);selection();status(`已导入 ${ids.length} 张图。点击“参考生成”，或画选区进行局部编辑。`);fit();}catch(e){status(e.message);}finally{busy=false;$('run').disabled=false;}}
function maskData(n){const src=document.querySelector(`.node[data-id="${n.id}"] canvas`);if(src.dataset.painted!=='true')throw Error('请先用矩形或画笔绘制选区');let c=document.createElement('canvas');c.width=n.w;c.height=n.h;const ctx=c.getContext('2d'),data=src.getContext('2d').getImageData(0,0,n.w,n.h);let count=0;for(let i=0;i<data.data.length;i+=4){let v=data.data[i+3]>0?255:0;count+=v?1:0;data.data[i]=data.data[i+1]=data.data[i+2]=v;data.data[i+3]=255;}if(count<16)throw Error('选区太小，请覆盖完整的修改对象');ctx.putImageData(data,0,0);return c.toDataURL('image/png');}
async function run(event){if(event)event.preventDefault();if(busy)return;const op=$('operation').value,list=chosen();try{if(needsImage.includes(op)&&!list.length)throw Error('请在画布上选择参考图片');if(['edit','inpaint','remove_object','remove_background','upscale','pose_extract'].includes(op)&&list.length!==1)throw Error('此操作需要选择一张图');if(op==='pose'&&!(list.length===1||(list.length===2&&$('model').value==='qwen')))throw Error('姿势迁移：选一张姿势图；用 Qwen 时可以再 Shift 选一张人物图保持长相');if(op==='pose'&&!$('prompt').value.trim())throw Error('请先在“描述”里写新图的内容（人物、服装、场景）');if(['merge','collage'].includes(op)&&list.length<2)throw Error('请 Shift + 点击选择至少两张图');busy=true;$('run').disabled=true;let body={operation:op,model:$('model').value,precision:$('precision').value,prompt:$('prompt').value,images:needsImage.includes(op)?list.map(n=>n.asset):[],loras:collectLoras()};for(let k of ['width','height','steps','seed','factor'])body[k]=Number($(k).value);for(let k of ['kv_cache','vae_tiling'])body[k]=$(k).checked;if(op==='pose'){body.control_scale=Number($('control_scale').value);if(body.model==='qwen'){body.width=2048;body.height=2048;body.vae_tiling=true;}}if(regional.includes(op)){const mask=await api('/api/images',{data:maskData(list[0])});body.mask=mask.id;}const anchor=list[0];const ratio=['generate','transparent','edit','merge','pose'].includes(op)||!anchor?body.height/body.width:op==='collage'?anchor.h/anchor.w/list.length:anchor.h/anchor.w;let pos=freeSpot(anchor?{x:anchor.x+355,y:anchor.y}:{x:(stage.clientWidth/2-camera.x)/camera.z-160,y:(110-camera.y)/camera.z},34+316*ratio);let job=await api('/api/jobs',body);localStorage.setItem(STORE+'-job',JSON.stringify({id:job.id,position:pos}));await finish(job,pos);return true;}catch(e){status('未完成：'+e.message);return false;}finally{busy=false;$('run').disabled=false;}}
// 找一个不和已有节点重叠的位置：从 pos 开始，压到哪个节点就挪到它下面（dir='right' 时挪到右边）
function freeSpot(pos,h,dir='down'){pos={...pos};for(let i=0;i<60;i++){const hit=nodes.find(n=>{const b=nodeBox(n);return pos.x<b.x+b.w+20&&pos.x+340>b.x&&pos.y<b.y+b.h+20&&pos.y+h+20>b.y;});if(!hit)break;const b=nodeBox(hit);if(dir==='down')pos.y=b.y+b.h+40;else pos.x=b.x+b.w+35;}return pos;}
function fmtSec(s){s=Math.max(0,Math.round(s));return s<60?s+' 秒':Math.floor(s/60)+' 分 '+(s%60)+' 秒';}
// 估算进度：服务器按采样步数上报阶段（load / prepare / denoise / decode / save），前端换算成百分比和剩余时间
function progressInfo(job,track){
  const p=job.progress||{},now=job.now||Date.now()/1000,used=job.started?now-job.started:0;
  if(job.status==='queued')return{frac:.02,text:job.ahead?`排队中：前面还有 ${job.ahead} 个任务`:'排队中…'};
  if(!p.phase)return{frac:.03,text:'准备中… · 已用 '+fmtSec(used)};
  let frac,eta=null;
  if(p.phase==='load')frac=.02+.08*Math.min(1,(now-(p.at||now))/60);
  else if(p.phase==='prepare')frac=.1;
  else if(p.phase==='denoise'){
    if(track.t0===undefined){track.t0=p.at||now;track.s0=p.step;}
    else if(p.step>track.s0){const rate=((p.at||now)-track.t0)/(p.step-track.s0);eta=(p.total-p.step)*rate+(job.request?.vae_tiling?20:3)-(now-(p.at||now));}
    frac=.1+.8*p.step/p.total;
  }
  else if(p.phase==='decode')frac=.9+.08*Math.min(1,(now-(p.at||now))/40);
  else frac=.99;
  return{frac,text:`${p.message||'生成中'} · ${Math.round(frac*100)}% · 已用 ${fmtSec(used)}`+(eta!==null&&eta>1?` · 约剩 ${fmtSec(eta)}`:'')};
}
// 提示词节点里的进度条；节点不存在时返回空函数
function promptProgress(id,head=''){
  return info=>{const el=document.querySelector(`.node[data-id="${id}"]`);if(!el)return;const bar=el.querySelector('.prompt-progress'),st=el.querySelector('.prompt-status');
    if(!info){bar.hidden=true;return;}bar.hidden=false;bar.firstElementChild.style.width=Math.round(info.frac*100)+'%';st.textContent=(head?head+'\n':'')+info.text;};
}
async function finish(job,pos,onProgress){const track={};while(['queued','running'].includes(job.status)){const info=progressInfo(job,track);status(info.text);onProgress?.(info);await new Promise(r=>setTimeout(r,1500));job=await api('/api/jobs/'+job.id);}onProgress?.(null);localStorage.removeItem(STORE+'-job');if(job.status!=='succeeded')throw Error(job.error);const result=job.result;const node=addNode(result,(names[job.request.operation]||job.request.operation)+loraTitle(job.request)+' · '+result.width+'×'+result.height,pos,job.request);if(result.pose?.skeleton_image&&job.request.operation==='pose'&&result.pose.source!=='skeleton_input'){const sk=addNode({image_id:result.pose.skeleton_id,image:result.pose.skeleton_image,width:result.pose.size[0],height:result.pose.size[1]},'姿势骨架'+(result.pose.visible_keypoints?' · '+result.pose.visible_keypoints+' 个关键点':''),{x:pos.x,y:pos.y-60-316*result.pose.size[1]/result.pose.size[0]},{operation:'pose_extract',images:job.request.images.slice(0,1)});selected=[node.id];selection();}changeTool('select');fit();const m=result.metrics;status(`完成 · ${m.seconds.toFixed(2)} 秒`+(m.gpu_total_peak_gib?` · 显存峰值 ${m.gpu_total_peak_gib.toFixed(2)} GiB`:'')+(result.outside_mask_identical?'\n已验证选区外像素完全不变。':'')+(job.request.operation==='remove_background'?`\nAlpha 范围 ${result.alpha_extrema?.join('–')}；请在棋盘格上检查边缘。`:''));return node;}
function action(op,drawTool){$('operation').value=op;update();if(drawTool)changeTool(drawTool);if(op==='remove_object')$('prompt').value='移除选区中的物件及其阴影，补全周围背景。';if(op==='remove_background')$('prompt').value='';status(help[op]);}
$('precision').onchange=updateLora;$('addlora').onclick=()=>addLoraRow();$('loratrigger').onclick=()=>{for(const r of [...loraRows].reverse()){const item=loraCatalog.find(x=>x.id===r.id);if(item?.trigger&&!$('prompt').value.includes(item.trigger))$('prompt').value=item.trigger+', '+$('prompt').value;}};$('form').onsubmit=run;$('operation').onchange=()=>{update();if(regional.includes($('operation').value))changeTool('rect');};$('model').onchange=()=>{$('steps').value=$('model').value==='zimage'?9:40;update();};
$('upload').onclick=()=>$('files').click();$('files').onchange=()=>{upload($('files').files,importAt||{x:(60-camera.x)/camera.z,y:(100-camera.y)/camera.z});importAt=null;$('files').value='';};
stage.addEventListener('dragover',e=>{e.preventDefault();stage.classList.add('dragover');});stage.addEventListener('dragleave',()=>stage.classList.remove('dragover'));stage.addEventListener('drop',e=>{e.preventDefault();stage.classList.remove('dragover');upload(e.dataTransfer.files,point(e));});
stage.addEventListener('wheel',e=>{if(e.target.closest('button,input,a,textarea,select,#regionpop,#ctxmenu'))return;e.preventDefault();let r=stage.getBoundingClientRect();zoom(Math.exp(-e.deltaY*.0015),e.clientX-r.left,e.clientY-r.top);},{passive:false});
stage.addEventListener('pointerdown',e=>{if(e.button!==0||e.target.closest('#tools,#selectiontools,#canvasfooter,#regionpop,#ctxmenu'))return;if(e.target.closest('.node')&&!space&&tool!=='pan')return;if(e.shiftKey&&!space&&tool!=='pan'){marquee(e);return;}const initial={x:camera.x,y:camera.y},start={x:e.clientX,y:e.clientY};let moved=false;stage.setPointerCapture(e.pointerId);const move=ev=>{moved=moved||Math.abs(ev.clientX-start.x)+Math.abs(ev.clientY-start.y)>3;camera.x=initial.x+ev.clientX-start.x;camera.y=initial.y+ev.clientY-start.y;transform();};const end=ev=>{stage.removeEventListener('pointermove',move);stage.removeEventListener('pointerup',end);stage.removeEventListener('pointercancel',end);if(!moved&&ev.type==='pointerup'&&group.size){setGroup([]);status('已取消全选。');}persist();};stage.addEventListener('pointermove',move);stage.addEventListener('pointerup',end);stage.addEventListener('pointercancel',end);});
document.querySelectorAll('[data-tool]').forEach(b=>b.onclick=()=>changeTool(b.dataset.tool));$('fit').onclick=fit;$('zoomout').onclick=()=>zoom(.8);$('zoomin').onclick=()=>zoom(1.25);$('zoomreset').onclick=()=>zoom(1/camera.z);
$('reference').onclick=()=>action(selected.length>1?'merge':'edit');$('regionedit').onclick=()=>action('inpaint','rect');$('erase').onclick=()=>action('remove_object','brush');$('removebg').onclick=()=>action('remove_background');$('enlarge').onclick=()=>action('upscale');$('posegen').onclick=()=>{if($('model').value!=='qwen'&&chosen().length===2){$('model').value='qwen';$('steps').value=40;}action('pose');};$('poseextract').onclick=()=>{action('pose_extract');run();};
$('clearmask').onclick=()=>{for(let n of chosen()){let c=document.querySelector(`.node[data-id="${n.id}"] canvas`);c.getContext('2d').clearRect(0,0,c.width,c.height);c.dataset.painted='false';}status('选区已清除。');};
$('remove').onclick=()=>removeNodes([...selected]);
document.addEventListener('keydown',e=>{if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='a'&&!e.target.closest('input,textarea,select,[contenteditable]')){e.preventDefault();selectAll();return;}if(e.target.closest('input,textarea,select,button'))return;if(e.code==='Space'){space=true;e.preventDefault();}if(e.code==='Delete'&&group.size>1){e.preventDefault();const ids=[...group];if(confirm(`把选中的 ${ids.length} 个节点移出画布？服务器上的图片不会删除。`)){setGroup([]);removeNodes(ids);}return;}if(e.code==='Delete'&&selected.length){$('remove').click();e.preventDefault();}});document.addEventListener('keyup',e=>{if(e.code==='Space')space=false;});window.addEventListener('blur',()=>space=false);
$('loadpreset').onclick=()=>{let p=presets.find(p=>p.id===$('preset').value);if(!p){status('请先选择一个文档示例');return;}for(let k of ['prompt','width','height'])$(k).value=p[k];$('operation').value='generate';loraRows=[];$('seed').value=42;$('steps').value=$('model').value==='zimage'?9:40;$('precision').value='bf16';$('kv_cache').checked=true;$('vae_tiling').checked=false;update();status('已载入文档原始 prompt：seed 42，可改为 2026；Qwen 40 步 / Z 9 步。');};
$('reuseparams').onclick=()=>{let r=chosen()[0]?.request;if(!r){status('上传图片没有生成参数；请选择生成结果。');return;}for(let k of ['operation','model','precision','prompt','width','height','steps','seed','factor'])if(r[k]!==undefined)$(k).value=r[k];for(let k of ['kv_cache','vae_tiling'])$(k).checked=!!r[k];update();for(let k of ['width','height'])if(r[k]!==undefined)$(k).value=r[k];$('vae_tiling').checked=!!r.vae_tiling;if(r.control_scale!==undefined)$('control_scale').value=r.control_scale;loraRows=((r.loras&&r.loras.length)?r.loras:[r.lora&&{id:r.lora,scale:r.lora_scale},r.lora2&&{id:r.lora2,scale:r.lora2_scale}].filter(Boolean)).map(u=>({id:u.id,scale:u.scale}));updateLora();status('参数已复用；编辑类任务请重新选择所需参考图和选区。');};
(async()=>{try{let saved=JSON.parse(localStorage.getItem(STORE)||'null');if(saved&&Array.isArray(saved.nodes)){nodes=saved.nodes;camera=saved.camera||camera;edges=Array.isArray(saved.edges)?saved.edges:[];nodes.forEach(renderNode);refreshPromptInputs();drawLinks();$('empty').hidden=!!nodes.length;}transform();changeTool('select');update();loraCatalog=await api('/api/loras');updateLora();presets=await api('/api/presets');for(let p of presets){let option=new Option(p.title,p.id);$('preset').add(option);}let pending=JSON.parse(localStorage.getItem(STORE+'-job')||'null');if(pending){busy=true;$('run').disabled=true;const done=await finish(await api('/api/jobs/'+pending.id),pending.position,pending.from?promptProgress(pending.from):null);if(pending.from&&done&&nodes.some(x=>x.id===pending.from)){edges.push({id:crypto.randomUUID(),from:pending.from,to:done.id,kind:'result'});afterEdges();}}}catch(e){status(e.message);}finally{busy=false;$('run').disabled=false;}})();

// ---------- 节点流：提示词节点、连线与 @ 引用（参考 LibTV 的画布交互） ----------
function isPromptNode(n){return !!n&&n.type==='prompt';}
function inputsOf(id){return edges.filter(e=>e.to===id&&e.kind==='input').map(e=>nodes.find(n=>n.id===e.from)).filter(Boolean);}
function nodeBox(n){const el=document.querySelector(`.node[data-id="${n.id}"]`);const h=el&&el.offsetHeight>40?el.offsetHeight:(isPromptNode(n)?250:34+316*n.h/n.w);return{x:n.x,y:n.y,w:320,h};}
function curve(x1,y1,x2,y2){const dx=Math.max(40,Math.abs(x2-x1)/2);return `M${x1},${y1} C${x1+dx},${y1} ${x2-dx},${y2} ${x2},${y2}`;}
function drawLinks(){
  const svg=$('links');if(!svg)return;svg.replaceChildren();
  for(const e of edges){
    const a=nodes.find(n=>n.id===e.from),b=nodes.find(n=>n.id===e.to);if(!a||!b)continue;
    const A=nodeBox(a),B=nodeBox(b),d=curve(A.x+A.w,A.y+A.h/2,B.x,B.y+B.h/2);
    const g=document.createElementNS(SVG,'g');g.setAttribute('class','edge '+e.kind);
    const hit=document.createElementNS(SVG,'path');hit.setAttribute('d',d);hit.setAttribute('class','link-hit');
    const tip=document.createElementNS(SVG,'title');tip.textContent='点击断开这条连线';hit.append(tip);
    hit.addEventListener('pointerdown',ev=>{ev.stopPropagation();ev.preventDefault();edges=edges.filter(x=>x.id!==e.id);afterEdges();status('已断开连线。');});
    const line=document.createElementNS(SVG,'path');line.setAttribute('d',d);line.setAttribute('class','link');
    g.append(hit,line);svg.append(g);
  }
}
function afterEdges(){refreshPromptInputs();drawLinks();persist();}
function connect(from,to){
  const a=nodes.find(n=>n.id===from),b=nodes.find(n=>n.id===to);
  if(!a||!b||isPromptNode(a)||!isPromptNode(b)){status('连线规则：图片 → 提示词节点。');return false;}
  if(edges.some(e=>e.from===from&&e.to===to&&e.kind==='input')){status('这张图已经连到这个节点了。');return false;}
  const count=inputsOf(to).length;if(count>=4){status('一个提示词节点最多连 4 张参考图。');return false;}
  edges.push({id:crypto.randomUUID(),from,to,kind:'input'});afterEdges();
  status(`已连线：这张图在该节点里是“图${count+1}”，在提示词里输入 @ 即可引用。`);return true;
}
function startLink(n,e){
  e.stopPropagation();e.preventDefault();
  const port=e.currentTarget,A=nodeBox(n),x1=A.x+A.w,y1=A.y+A.h/2;
  const temp=document.createElementNS(SVG,'path');temp.setAttribute('class','temp');$('links').append(temp);
  port.setPointerCapture(e.pointerId);
  const move=ev=>{const p=point(ev);temp.setAttribute('d',curve(x1,y1,p.x,p.y));};
  const end=ev=>{
    port.removeEventListener('pointermove',move);port.removeEventListener('pointerup',end);port.removeEventListener('pointercancel',end);temp.remove();
    if(ev.type!=='pointerup')return;
    const target=document.elementFromPoint(ev.clientX,ev.clientY),el=target&&target.closest('.node');
    if(el){const t=nodes.find(x=>x.id===el.dataset.id);if(isPromptNode(t))connect(n.id,t.id);else if(t&&t.id!==n.id)status('图片只能连到提示词节点；拖到空白处可以新建一个。');return;}
    if(target&&stage.contains(target)&&!target.closest('#tools,#selectiontools,#canvasfooter,#regionpop,#ctxmenu')){const p=point(ev);const pn=addPromptNode({x:p.x,y:p.y-60});connect(n.id,pn.id);}
  };
  port.addEventListener('pointermove',move);port.addEventListener('pointerup',end);port.addEventListener('pointercancel',end);
}
function addPromptNode(p,text=''){
  const n={id:crypto.randomUUID(),type:'prompt',title:'提示词节点',text,model:$('model').value,x:Math.round(p.x),y:Math.round(p.y),w:320,h:250};
  nodes.push(n);renderNode(n);$('empty').hidden=true;select(n.id);persist();
  setTimeout(()=>document.querySelector(`.node[data-id="${n.id}"] textarea`)?.focus(),0);return n;
}
function renderPromptNode(n){
  const el=document.createElement('article');el.className='node prompt-node';el.dataset.id=n.id;el.style.left=n.x+'px';el.style.top=n.y+'px';
  el.innerHTML='<div class="node-head">提示词节点</div><div class="prompt-body"><div class="prompt-inputs"></div>'+
    '<textarea rows="4" placeholder="描述要生成的画面。输入 @ 引用连进来的图片，例如：让 @图1 的人物坐在 @图2 的湖边"></textarea>'+
    '<div class="mention-menu" hidden></div><div class="prompt-actions"><select aria-label="没有参考图时使用的模型"><option value="qwen">Qwen 2.1</option><option value="zimage">Z-Image-Turbo</option></select>'+
    '<button type="button" class="go">生成</button></div><div class="prompt-progress" role="progressbar" aria-label="生成进度" hidden><i></i></div><p class="prompt-status"></p></div>'+
    '<span class="port in" title="参考图从这里连进来"></span><span class="port out static"></span>';
  world.append(el);
  const ta=el.querySelector('textarea'),sel=el.querySelector('select'),menu=el.querySelector('.mention-menu');
  ta.value=n.text||'';sel.value=n.model||'qwen';
  el.onpointerdown=e=>{
    if(e.button!==0||space||tool==='pan')return;e.stopPropagation();
    if(!e.target.closest('.node-head')){if(!selected.includes(n.id))select(n.id);return;}
    if(group.size>1&&group.has(n.id)){dragNodes(e,el,groupList());return;}
    if(group.size)setGroup([]);select(n.id);dragNodes(e,el,[n]);
  };
  ta.addEventListener('input',()=>{n.text=ta.value;persist();mentionCheck(n,el);drawLinks();});
  ta.addEventListener('keydown',e=>mentionKeys(e,n,el));
  ta.addEventListener('blur',()=>setTimeout(()=>{menu.hidden=true;},150));
  sel.onchange=()=>{n.model=sel.value;persist();};
  el.querySelector('.go').onclick=()=>runPromptNode(n);
  refreshPromptInputs();
}
function refreshPromptInputs(){
  for(const n of nodes.filter(isPromptNode)){
    const el=document.querySelector(`.node[data-id="${n.id}"]`);if(!el)continue;
    const box=el.querySelector('.prompt-inputs'),sel=el.querySelector('select'),list=inputsOf(n.id);box.replaceChildren();
    if(!list.length)box.textContent='还没有参考图，直接文生图。把图片右侧的圆点拖到这里即可作为参考。';
    else list.forEach((img,i)=>{const chip=document.createElement('span');chip.className='chip';chip.title=img.title;const im=new Image();im.src=img.url;im.alt='';chip.append(im,document.createTextNode('图'+(i+1)));box.append(chip);});
    sel.disabled=!!list.length;sel.value=list.length?'qwen':(n.model||'qwen');
    sel.title=list.length?'有参考图时只能用 Qwen 2.1（Z-Image-Turbo 不支持编辑）':'';
  }
}
function mentionCheck(n,el){
  const ta=el.querySelector('textarea'),menu=el.querySelector('.mention-menu');
  if(!/@[^\s@]*$/.test(ta.value.slice(0,ta.selectionStart))){menu.hidden=true;return;}
  const list=inputsOf(n.id);menu.replaceChildren();
  if(!list.length){const p=document.createElement('p');p.className='hint';p.textContent='还没有连入图片：先把图片右侧的圆点拖到这个节点。';menu.append(p);}
  else list.forEach((img,i)=>{const b=document.createElement('button');b.type='button';b.dataset.index=i+1;const im=new Image();im.src=img.url;im.alt='';b.append(im,document.createTextNode(`图${i+1} · ${img.title}`));b.onpointerdown=ev=>{ev.preventDefault();ev.stopPropagation();insertMention(n,el,i+1);};menu.append(b);});
  menu.querySelector('button')?.classList.add('active');
  menu.style.top=(ta.offsetTop+ta.offsetHeight)+'px';menu.hidden=false;
}
function insertMention(n,el,k){
  const ta=el.querySelector('textarea'),pos=ta.selectionStart;
  const before=ta.value.slice(0,pos).replace(/@[^\s@]*$/,'@图'+k+' '),after=ta.value.slice(pos);
  ta.value=before+after;ta.selectionStart=ta.selectionEnd=before.length;n.text=ta.value;persist();
  el.querySelector('.mention-menu').hidden=true;ta.focus();
}
function mentionKeys(e,n,el){
  const menu=el.querySelector('.mention-menu');
  if(menu.hidden){if(e.key==='Enter'&&(e.ctrlKey||e.metaKey)){e.preventDefault();runPromptNode(n);}return;}
  const items=[...menu.querySelectorAll('button')];
  if(e.key==='Escape'){e.preventDefault();menu.hidden=true;return;}
  if(!items.length)return;
  let i=items.findIndex(b=>b.classList.contains('active'));
  if(e.key==='ArrowDown'||e.key==='ArrowUp'){e.preventDefault();items[i]?.classList.remove('active');i=(Math.max(i,0)+(e.key==='ArrowDown'?1:items.length-1))%items.length;items[i].classList.add('active');}
  else if(e.key==='Enter'||e.key==='Tab'){e.preventDefault();insertMention(n,el,Number(items[Math.max(0,i)].dataset.index));}
}
async function runPromptNode(n){
  const el=document.querySelector(`.node[data-id="${n.id}"]`),st=el.querySelector('.prompt-status'),go=el.querySelector('.go');
  if(busy){st.textContent='请等当前任务完成。';return;}
  try{
    const inputs=inputsOf(n.id),text=(n.text||'').trim();
    if(!text)throw Error('请先写提示词');
    const used=[];
    for(const m of text.matchAll(/@图(\d+)/g)){const k=Number(m[1]);if(k<1||k>inputs.length)throw Error(`@图${k} 不存在：这个节点只连了 ${inputs.length} 张图`);if(!used.includes(k))used.push(k);}
    // 只 @ 了部分图时，只把被引用的图作为参考，并按引用顺序重新编号成“图1、图2”
    const refs=used.length?used.map(k=>inputs[k-1]):inputs;
    const prompt=used.length?text.replace(/@图(\d+)/g,(m,k)=>'图'+(used.indexOf(Number(k))+1)):text;
    const op=refs.length===0?'generate':refs.length===1?'edit':'merge',model=refs.length?'qwen':(n.model||'qwen');
    const body={operation:op,model,precision:'bf16',prompt,images:refs.map(r=>r.asset),width:refs.length?2048:Number($('width').value),height:refs.length?2048:Number($('height').value),
      steps:model==='zimage'?9:($('model').value==='qwen'?Number($('steps').value):40),seed:Number($('seed').value),kv_cache:$('kv_cache').checked,vae_tiling:refs.length?true:$('vae_tiling').checked};
    if(op==='generate'&&$('model').value===model&&loraRows.length&&$('precision').value==='bf16')
      body.loras=loraRows.map(r=>({id:r.id,scale:Number(r.scale)}));
    busy=true;$('run').disabled=true;go.disabled=true;
    const head=`${names[op]} · ${model==='qwen'?'Qwen 2.1':'Z-Image-Turbo'}`+(refs.length?` · 参考 ${refs.length} 张图 · 2048 + VAE 分块，约 2 分钟`:'');
    st.textContent=head+' · 提交中…';
    let pos=freeSpot({x:n.x+380,y:n.y},34+316*body.height/body.width);
    const job=await api('/api/jobs',body);
    localStorage.setItem(STORE+'-job',JSON.stringify({id:job.id,position:pos,from:n.id}));
    const node=await finish(job,pos,promptProgress(n.id,head));
    edges.push({id:crypto.randomUUID(),from:n.id,to:node.id,kind:'result'});afterEdges();
    st.textContent=`完成：${names[op]}`+(refs.length?`，参考 ${refs.length} 张图`:'')+`，${node.w}×${node.h}`;
  }catch(e){el.querySelector('.prompt-progress').hidden=true;st.textContent='未完成：'+e.message;status('未完成：'+e.message);}
  finally{busy=false;$('run').disabled=false;go.disabled=false;}
}
// ---------- 全选、框选与整体拖动 ----------
function setGroup(ids){group=new Set(ids);document.querySelectorAll('.node').forEach(el=>el.classList.toggle('grouped',group.has(el.dataset.id)));}
function groupList(){return nodes.filter(n=>group.has(n.id));}
function selectAll(){if(!nodes.length){status('画布上还没有节点。');return;}setGroup(nodes.map(n=>n.id));status(`已全选 ${nodes.length} 个节点：拖动其中任意一个即可整体移动；Delete 移出画布，Esc 或点击空白处取消。`);}
function dragNodes(e,el,list){
  const start=point(e),init=list.map(n=>({n,x:n.x,y:n.y,el:document.querySelector(`.node[data-id="${n.id}"]`)}));el.setPointerCapture(e.pointerId);
  const move=ev=>{const p=point(ev),dx=p.x-start.x,dy=p.y-start.y;for(const it of init){it.n.x=it.x+dx;it.n.y=it.y+dy;if(it.el){it.el.style.left=it.n.x+'px';it.el.style.top=it.n.y+'px';}}drawLinks();};
  const end=()=>{el.removeEventListener('pointermove',move);el.removeEventListener('pointerup',end);el.removeEventListener('pointercancel',end);persist();};
  el.addEventListener('pointermove',move);el.addEventListener('pointerup',end);el.addEventListener('pointercancel',end);
}
// Shift + 在空白处拖动：框选
function marquee(e){
  const r=stage.getBoundingClientRect(),sx=e.clientX-r.left,sy=e.clientY-r.top,a=point(e),box=document.createElement('div');box.id='marquee';stage.append(box);
  stage.setPointerCapture(e.pointerId);
  const move=ev=>{const x=ev.clientX-r.left,y=ev.clientY-r.top;Object.assign(box.style,{left:Math.min(sx,x)+'px',top:Math.min(sy,y)+'px',width:Math.abs(x-sx)+'px',height:Math.abs(y-sy)+'px'});};
  const end=ev=>{
    stage.removeEventListener('pointermove',move);stage.removeEventListener('pointerup',end);stage.removeEventListener('pointercancel',end);box.remove();
    if(ev.type!=='pointerup')return;
    const b=point(ev),x0=Math.min(a.x,b.x),x1=Math.max(a.x,b.x),y0=Math.min(a.y,b.y),y1=Math.max(a.y,b.y);
    const hit=nodes.filter(n=>{const q=nodeBox(n);return q.x<x1&&q.x+q.w>x0&&q.y<y1&&q.y+q.h>y0;});
    setGroup(hit.map(n=>n.id));
    status(hit.length?`已框选 ${hit.length} 个节点：拖动其中任意一个即可整体移动；Delete 移出画布，Esc 或点击空白处取消。`:'框里没有节点。');
  };
  stage.addEventListener('pointermove',move);stage.addEventListener('pointerup',end);stage.addEventListener('pointercancel',end);
}
function removeNodes(ids){
  for(const id of ids){document.querySelector(`.node[data-id="${id}"]`)?.remove();group.delete(id);}
  nodes=nodes.filter(n=>!ids.includes(n.id));edges=edges.filter(e=>!ids.includes(e.from)&&!ids.includes(e.to));
  selected=selected.filter(id=>!ids.includes(id));selection();$('empty').hidden=!!nodes.length;
  refreshPromptInputs();drawLinks();persist();status('已移出画布；服务器原图未删除。');
}
// 右键菜单
function hideMenu(){$('ctxmenu').hidden=true;}
function showMenu(e,items){
  const menu=$('ctxmenu'),s=stage.getBoundingClientRect();menu.replaceChildren();
  for(const [label,fn] of items){const b=document.createElement('button');b.type='button';b.textContent=label;b.onclick=()=>{hideMenu();fn();};menu.append(b);}
  menu.hidden=false;
  menu.style.left=Math.max(8,Math.min(e.clientX-s.left,s.width-menu.offsetWidth-8))+'px';
  menu.style.top=Math.max(8,Math.min(e.clientY-s.top,s.height-menu.offsetHeight-8))+'px';
}
stage.addEventListener('contextmenu',e=>{
  if(e.target.closest('textarea,input,select,#tools,#selectiontools,#canvasfooter,#regionpop,#ctxmenu'))return;
  e.preventDefault();
  const el=e.target.closest('.node'),n=el&&nodes.find(x=>x.id===el.dataset.id),p=point(e);
  if(n&&!isPromptNode(n))showMenu(e,[['用这张图新建提示词节点',()=>{const pn=addPromptNode({x:n.x+380,y:n.y});connect(n.id,pn.id);}],['提取姿势骨架',()=>{selected=[n.id];selection();action('pose_extract');run();}],['按这张图的姿势生成…',()=>{selected=[n.id];selection();action('pose');$('prompt').focus();status('在左侧“描述”里写新图的人物、服装和场景，再点“按姿势生成”。');}],['移出画布',()=>removeNodes([n.id])]]);
  else if(n)showMenu(e,[['生成',()=>runPromptNode(n)],['断开所有参考图',()=>{edges=edges.filter(x=>!(x.to===n.id&&x.kind==='input'));afterEdges();}],['删除这个节点',()=>removeNodes([n.id])]]);
  else showMenu(e,[['新建提示词节点',()=>addPromptNode(p)],['在这里导入图片',()=>{importAt=p;$('files').click();}]]);
});
document.addEventListener('pointerdown',e=>{if(!e.target.closest('#ctxmenu'))hideMenu();},true);
document.addEventListener('keydown',e=>{if(e.key==='Escape'){hideMenu();$('regionpop').hidden=true;if(group.size){setGroup([]);status('已取消全选。');}}});
$('selectall').onclick=selectAll;
// 选区弹出框：画完选区直接写修改要求
function showRegionPop(n,mask,box){
  const pop=$('regionpop'),r=mask.getBoundingClientRect(),s=stage.getBoundingClientRect(),sx=r.width/mask.width,sy=r.height/mask.height;
  pop.dataset.node=n.id;pop.hidden=false;
  let left=r.left+box.x0*sx-s.left,top=r.top+box.y1*sy-s.top+10;
  if(top+pop.offsetHeight>s.height-54)top=r.top+box.y0*sy-s.top-pop.offsetHeight-10;
  pop.style.left=Math.max(10,Math.min(left,s.width-pop.offsetWidth-10))+'px';
  pop.style.top=Math.max(10,Math.min(top,s.height-pop.offsetHeight-54))+'px';
  $('regionprompt').focus();
}
async function regionRun(op){
  const pop=$('regionpop'),n=nodes.find(x=>x.id===pop.dataset.node);if(!n){pop.hidden=true;return;}
  const text=$('regionprompt').value.trim();
  if(op==='inpaint'&&!text){status('请先写这块区域要改成什么。');$('regionprompt').focus();return;}
  selected=[n.id];selection();$('operation').value=op;update();
  $('prompt').value=op==='remove_object'?(text||'移除选区中的物件及其阴影，补全周围背景。'):text;
  pop.hidden=true;
  if(await run()){const c=document.querySelector(`.node[data-id="${n.id}"] canvas`);if(c){c.getContext('2d').clearRect(0,0,c.width,c.height);c.dataset.painted='false';}$('regionprompt').value='';}
}
$('regiongo').onclick=()=>regionRun('inpaint');$('regionerase').onclick=()=>regionRun('remove_object');$('regioncancel').onclick=()=>{$('regionpop').hidden=true;};
$('regionprompt').addEventListener('keydown',e=>{if(e.key==='Enter'&&(e.ctrlKey||e.metaKey)){e.preventDefault();regionRun('inpaint');}});
$('regionpop').addEventListener('pointerdown',e=>e.stopPropagation());

// 给助手 / 训练面板用：把服务器结果放上画布、刷新 LoRA 列表、切换到指定 LoRA
window.lab={addResult(result,title,request){let pos=freeSpot({x:(stage.clientWidth/2-camera.x)/camera.z-160,y:(110-camera.y)/camera.z},34+316*result.height/result.width,'right');const node=addNode(result,title,pos,request);fit();return node;},async reloadLoras(){loraCatalog=await api('/api/loras');updateLora();return loraCatalog;},useLora(id){const item=loraCatalog.find(x=>x.id===id);if(!item)return false;$('operation').value='generate';$('model').value=item.model;$('steps').value=item.model==='zimage'?9:40;$('precision').value='bf16';update();loraRows=loraRows.filter(r=>r.id!==id);loraRows.push({id,scale:item.default_scale??0.8});updateLora();if(item.trigger&&!$('prompt').value.includes(item.trigger))$('prompt').value=item.trigger+', '+$('prompt').value;return true;},selectedAssets(){return chosen().map(n=>({id:n.asset,url:n.url,title:n.title,w:n.w,h:n.h}));},status,api};
