// 助手（Step 多模态模型 + Skills）与在线 LoRA 训练面板。画布操作通过 canvas.js 暴露的 window.lab 完成。
(() => {
const $ = id => document.getElementById(id);
const sleep = ms => new Promise(r => setTimeout(r, ms));
const store = {
  get(k, d) { try { const v = JSON.parse(localStorage.getItem(k)); return v ?? d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} },
};
async function req(method, url, data) {
  const init = { method };
  if (data !== undefined) { init.headers = { 'Content-Type': 'application/json' }; init.body = JSON.stringify(data); }
  const r = await fetch(url, init);
  let body = null;
  try { body = await r.json(); } catch {}
  if (!r.ok) throw Error(body && body.detail ? (typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)) : 'HTTP ' + r.status);
  return body;
}
function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') e.className = v;
    else if (k.startsWith('on')) e[k] = v;
    else if (v !== undefined && v !== null && v !== false) e.setAttribute(k, v === true ? '' : v);
  }
  for (const k of kids.flat()) if (k !== null && k !== undefined && k !== false) e.append(k instanceof Node ? k : String(k));
  return e;
}
const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
function md(text) {   // 只支持代码块、行内代码、粗体、标题，先转义再加标签
  const blocks = [];
  let s = esc(text).replace(/```[^\n]*\n?([\s\S]*?)```/g, (m, c) => { blocks.push(c); return `\u0000${blocks.length - 1}\u0000`; });
  s = s.replace(/`([^`\n]+)`/g, '<code>$1</code>').replace(/\*\*([^*\n]+)\*\*/g, '<b>$1</b>').replace(/^#{1,6}\s*(.+)$/gm, '<b>$1</b>').replace(/\n/g, '<br>');
  return s.replace(/\u0000(\d+)\u0000/g, (m, i) => `<pre>${blocks[i]}</pre>`);
}
const readDataURL = file => new Promise((resolve, reject) => { const r = new FileReader(); r.onload = () => resolve(r.result); r.onerror = reject; r.readAsDataURL(file); });
const fmtSec = s => s == null ? '?' : s < 90 ? Math.round(s) + ' 秒' : (s / 60).toFixed(1) + ' 分钟';

// ---------------------------------------------------------------- 标签页
function showTab(name) {
  document.querySelectorAll('#tabs button').forEach(b => { b.classList.toggle('active', b.dataset.tab === name); b.setAttribute('aria-selected', b.dataset.tab === name ? 'true' : 'false'); });
  for (const t of ['gen', 'agent', 'train']) $('tab-' + t).hidden = t !== name;
  document.querySelector('main').dataset.tab = name;
  store.set('lab-tab', name);
}
document.querySelectorAll('#tabs button').forEach(b => { b.onclick = () => showTab(b.dataset.tab); });

// ---------------------------------------------------------------- 模型设置与多模态检测
let agentCfg = null;
function badge(text, kind) { const b = $('agentbadge'); b.textContent = text; b.dataset.kind = kind; }
function showConfig(c) {
  agentCfg = c;
  $('ag_base').value = c.base_url || '';
  $('ag_model').value = c.model || '';
  $('ag_key').value = '';
  $('ag_key').placeholder = c.has_key ? `已保存（${c.api_key}），不修改就留空` : '粘贴 API Key';
  const p = c.probe, mm = !!(p && p.multimodal);
  if (!c.has_key) badge('未配置', 'warn');
  else if (!p) badge('未检测', 'warn');
  else if (!p.connected) badge('连接失败', 'bad');
  else if (mm) badge('可看图', 'ok');
  else badge('不能看图', 'bad');
  const hint = $('ag_probe');
  if (!c.has_key) { hint.textContent = '填写 API Key 后点“保存并检测”：会发一张测试图（红底白字 7），检查模型能不能看图。'; hint.className = 'hint'; }
  else if (!p) { hint.textContent = '还没有检测，点“保存并检测”。'; hint.className = 'hint'; }
  else {
    hint.textContent = `${p.model || c.model}：${p.message}` + (p.seconds ? `（${p.seconds} 秒）` : '') +
      (p.connected && !mm ? '\n请换成多模态模型（例如 step-5-preview）后重新保存检测；训练数据审核和“附带图片”都需要模型能看图。' : '');
    hint.className = 'hint ' + (mm ? 'ok' : 'bad');
  }
  if (!c.has_key || !mm) $('agentsettings').open = true;
  $('chat_attach').disabled = !mm;
  if (!mm) $('chat_attach').checked = false;
  $('chat_attach').parentElement.title = mm ? '' : '当前模型不能看图，不能附带图片';
}
$('ag_save').onclick = async () => {
  const body = { base_url: $('ag_base').value.trim(), model: $('ag_model').value.trim(), api_key: $('ag_key').value.trim() || null };
  if (!body.base_url || !body.model) { $('ag_probe').textContent = '请填写接口地址和模型名。'; return; }
  $('ag_save').disabled = true;
  badge('检测中…', 'warn');
  $('ag_probe').className = 'hint';
  $('ag_probe').textContent = `正在保存并检测 ${body.model}：发送测试图，确认它能不能看图…`;
  try { showConfig(await req('POST', '/api/agent/config', body)); }
  catch (e) { $('ag_probe').textContent = '保存失败：' + e.message; $('ag_probe').className = 'hint bad'; badge('出错', 'bad'); }
  finally { $('ag_save').disabled = false; }
};

// ---------------------------------------------------------------- Skills
let skills = [];
const picked = new Set(store.get('lab-skillpick', []));
function skillMsg(text, bad = false) { const m = $('skill_msg'); m.textContent = text; m.className = 'hint' + (bad ? ' bad' : ''); }
async function loadSkills() { skills = await req('GET', '/api/skills'); renderSkills(); }
function sourceLabel(src) {
  if (!src) return '未知来源';
  if (src.startsWith('github:')) return src.slice(7).replace(/@[^/]+/, '');
  if (src.startsWith('upload:')) return '上传：' + src.slice(7);
  return src;
}
function renderSkills() {
  $('skillcount').textContent = `${skills.filter(s => s.enabled).length}/${skills.length}`;
  const list = $('skilllist');
  list.replaceChildren();
  if (!skills.length) list.append(el('p', { class: 'hint' }, '还没有安装技能。'));
  for (const s of skills) {
    const box = el('input', { type: 'checkbox', 'aria-label': '启用 ' + s.name });
    box.checked = !!s.enabled;
    box.onchange = async () => {
      try { skills = await req('POST', `/api/skills/${encodeURIComponent(s.name)}/toggle`, { enabled: box.checked }); renderSkills(); }
      catch (e) { skillMsg(e.message, true); }
    };
    const files = Array.isArray(s.files) ? s.files.length : (s.files || 0);
    list.append(el('div', { class: 'skill' + (s.enabled ? '' : ' off') },
      el('label', { class: 'check', title: '启用后助手才能看到并使用这个技能' }, box, el('b', {}, s.name)),
      el('p', { class: 'desc', title: s.description || '' }, s.description || '（没有描述）'),
      el('div', { class: 'meta' },
        el('span', { title: s.source || '' }, `${sourceLabel(s.source)} · ${files} 个文件`),
        el('button', { type: 'button', onclick: () => viewSkill(s) }, '查看'),
        el('button', { type: 'button', onclick: async () => {
          if (!confirm(`删除技能「${s.name}」？`)) return;
          try { skills = await req('DELETE', `/api/skills/${encodeURIComponent(s.name)}`); picked.delete(s.name); renderSkills(); skillMsg('已删除 ' + s.name); }
          catch (e) { skillMsg(e.message, true); }
        } }, '删除'))));
  }
  renderPicker();
}
function renderPicker() {
  const box = $('chat_skillpick');
  box.replaceChildren();
  const enabled = skills.filter(s => s.enabled);
  for (const n of [...picked]) if (!enabled.some(s => s.name === n)) picked.delete(n);
  store.set('lab-skillpick', [...picked]);
  if (!enabled.length) return;
  box.append(el('span', { class: 'hint' }, '指定技能：'));
  for (const s of enabled) {
    const on = picked.has(s.name);
    box.append(el('button', { type: 'button', class: 'chip' + (on ? ' on' : ''), 'aria-pressed': on ? 'true' : 'false', title: (on ? '本次一定按这个技能工作。' : '不点也行：助手会按需自己读取已启用的技能。') + '\n' + (s.description || ''),
      onclick: () => { picked.has(s.name) ? picked.delete(s.name) : picked.add(s.name); renderPicker(); } }, s.name));
  }
}
async function viewSkill(s) {
  let d = $('skilldialog');
  if (!d) { d = el('dialog', { id: 'skilldialog', 'aria-label': '技能内容' }); document.body.append(d); d.addEventListener('click', e => { if (e.target === d) d.close(); }); }
  const pre = el('pre', {}, '加载中…');
  const show = async path => {
    pre.textContent = '加载中…';
    try { pre.textContent = (await req('GET', `/api/skills/${encodeURIComponent(s.name)}/file?path=${encodeURIComponent(path)}`)).content; }
    catch (e) { pre.textContent = '读取失败：' + e.message; }
  };
  const files = (Array.isArray(s.files) ? s.files : []).filter(f => /\.(md|txt|json|ya?ml|csv)$/i.test(f)).slice(0, 40);
  d.replaceChildren(
    el('div', { class: 'dlg-head' }, el('b', {}, s.name), el('button', { type: 'button', onclick: () => d.close() }, '关闭')),
    files.length > 1 ? el('div', { class: 'files' }, files.map(f => el('button', { type: 'button', onclick: () => show(f) }, f))) : null,
    pre);
  d.showModal();
  show('SKILL.md');
}
async function installSkill(source) {
  if (!source) { skillMsg('请填写 GitHub 地址或 owner/repo，例如 iamyoki/qwen-image-2.1-skill', true); return; }
  $('skill_install').disabled = true;
  skillMsg('正在从 GitHub 下载 ' + source + ' …');
  try {
    const r = await req('POST', '/api/skills/install', { source });
    $('skill_choices').replaceChildren();
    skillMsg(r.message);
    if (r.choices && r.choices.length) {
      for (const c of r.choices) {
        $('skill_choices').append(el('button', { type: 'button', title: c.description || c.path, onclick: () => installSkill(c.source) },
          el('b', {}, c.name), el('small', {}, c.description ? c.description.slice(0, 90) : c.path)));
      }
    } else {
      $('skill_src').value = '';
      await loadSkills();
    }
  } catch (e) { skillMsg('安装失败：' + e.message, true); }
  finally { $('skill_install').disabled = false; }
}
$('skill_install').onclick = () => installSkill($('skill_src').value.trim());
$('skill_src').addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); installSkill($('skill_src').value.trim()); } });
$('skill_upload_btn').onclick = () => $('skill_file').click();
$('skill_file').onchange = async () => {
  const f = $('skill_file').files[0];
  $('skill_file').value = '';
  if (!f) return;
  if (f.size > 8_000_000) { skillMsg('技能包不能超过 8 MB', true); return; }
  skillMsg('正在安装 ' + f.name + ' …');
  try { const r = await req('POST', '/api/skills/upload', { filename: f.name, data: await readDataURL(f) }); skillMsg(r.message); await loadSkills(); }
  catch (e) { skillMsg('安装失败：' + e.message, true); }
};

// ---------------------------------------------------------------- 对话
const TOOL = { generate_image: '出图', edit_image: '编辑图片', pose_transfer: '姿势迁移', wait_for_jobs: '等待结果', view_image: '检查图片',
  list_loras: '查看 LoRA 列表', load_skill: '读取技能', read_skill_file: '读取技能文件' };
const OPS = { generate: '文生图', edit: '参考编辑', merge: '多图融合', pose: '姿势迁移' };
let conv = store.get('lab-conv', null);
let chatItems = store.get('lab-chat', []);
let chatBusy = false;
const jobViews = new Map();
function saveChat() { if (chatItems.length > 150) chatItems = chatItems.slice(-150); store.set('lab-chat', chatItems); }
function scrollChat() { const log = $('chatlog'); log.scrollTop = log.scrollHeight; }
function renderItem(item) {
  let node;
  if (item.kind === 'user') {
    node = el('div', { class: 'msg user' }, item.text);
    if (item.skills && item.skills.length) node.append(el('div', { class: 'tags' }, '技能：' + item.skills.join('、')));
    if (item.images && item.images.length) node.append(el('div', { class: 'thumbs' }, item.images.map(u => el('img', { src: u, alt: '附带的图片' }))));
  } else if (item.kind === 'assistant') {
    node = el('div', { class: 'msg assistant' });
    node.innerHTML = md(item.text);
  } else if (item.kind === 'tool') {
    node = el('div', { class: 'msg tool' + (item.bad ? ' bad' : '') });
    if (item.detail) node.append(el('details', {}, el('summary', {}, '🔧 ' + item.text), el('div', { class: 'prompt' }, item.detail)));
    else node.append('🔧 ' + item.text);
  } else if (item.kind === 'skill') {
    node = el('div', { class: 'msg skill' }, '📘 ' + item.text);
  } else if (item.kind === 'job') {
    node = el('div', { class: 'msg job' });
    jobViews.set(item.job_id, node);
    paintJob(item);
  } else {
    node = el('div', { class: 'msg ' + (item.kind === 'error' ? 'error' : 'note') }, item.text);
  }
  $('chatlog').insertBefore(node, $('chattyping'));
  scrollChat();
}
function paintJob(item) {
  const node = jobViews.get(item.job_id);
  if (!node) return;
  const name = OPS[item.operation] || item.operation;
  const label = item.status === 'succeeded' ? `${name}完成 · ${item.size ? item.size.join('×') : ''}${item.seconds ? ' · ' + fmtSec(item.seconds) : ''} · 已放到画布上`
    : item.status === 'failed' ? `${name}失败：${item.error || ''}` : item.status === 'running' ? `${name}生成中…` : `${name}排队中…`;
  node.replaceChildren(item.image ? el('img', { src: item.image, alt: name + '结果' }) : el('span', { class: 'spinner', 'aria-hidden': 'true' }), el('span', {}, label));
  node.classList.toggle('bad', item.status === 'failed');
}
function addChat(item) { chatItems.push(item); saveChat(); renderItem(item); return item; }
function typing(on, text = '助手思考中…') { const t = $('chattyping'); t.hidden = !on; t.textContent = text; scrollChat(); }
async function watchJob(item) {
  while (true) {
    let job;
    try { job = await req('GET', '/api/jobs/' + item.job_id); }
    catch (e) { item.status = 'failed'; item.error = e.message; break; }
    if (item.status !== job.status) { item.status = job.status; paintJob(item); saveChat(); }
    if (job.status === 'succeeded') {
      const r = job.result;
      if (!item.placed) {
        window.lab.addResult(r, '助手 · ' + (OPS[job.request.operation] || job.request.operation) + ' · ' + r.width + '×' + r.height, job.request);
        item.placed = true;
      }
      Object.assign(item, { image: r.image, size: [r.width, r.height], seconds: r.metrics && r.metrics.seconds });
      break;
    }
    if (job.status === 'failed') { item.error = job.error; break; }
    await sleep(2000);
  }
  paintJob(item);
  saveChat();
}
function onEvent(ev) {
  if (ev.type === 'assistant') addChat({ kind: 'assistant', text: ev.text });
  else if (ev.type === 'tool') {
    if (ev.name === 'load_skill') return;   // 由 skill 事件显示
    const a = ev.args || {}, bits = [];
    if (a.model) bits.push(a.model === 'qwen' ? 'Qwen 2.1' : 'Z-Image');
    if (a.width && a.height) bits.push(`${a.width}×${a.height}`);
    if (a.loras && a.loras.length) bits.push('LoRA ' + a.loras.map(l => l.id + (l.scale != null ? ' ×' + l.scale : '')).join('、'));
    if (a.image_ids) bits.push(a.image_ids.length + ' 张参考图');
    if (a.job_ids) bits.push(a.job_ids.length + ' 个任务');
    if (ev.name === 'read_skill_file') bits.push(`${a.name}/${a.path}`);
    addChat({ kind: 'tool', text: (TOOL[ev.name] || ev.name) + (bits.length ? '（' + bits.join('，') + '）' : ''), detail: a.prompt || '' });
    typing(true, ev.name === 'wait_for_jobs' ? '等待出图…' : '助手处理中…');
  } else if (ev.type === 'tool_result') {
    if (ev.result && ev.result.error) addChat({ kind: 'tool', text: `${TOOL[ev.name] || ev.name}失败：${ev.result.error}`, bad: true });
  } else if (ev.type === 'skill') { if (!seenSkills.has(ev.name)) { seenSkills.add(ev.name); addChat({ kind: 'skill', text: '使用技能：' + ev.name }); } }
  else if (ev.type === 'job') { const item = addChat({ kind: 'job', job_id: ev.job_id, operation: ev.operation, status: 'queued' }); watchJob(item); }
  else if (ev.type === 'error') addChat({ kind: 'error', text: '出错了：' + ev.text });
}
let seenSkills = new Set();
async function follow(taskId, since) {
  seenSkills = new Set();
  chatBusy = true;
  $('chat_send').disabled = true;
  typing(true);
  let next = since;
  try {
    while (true) {
      let r;
      try { r = await req('GET', `/api/agent/tasks/${taskId}?since=${next}`); }
      catch (e) {
        if (/不存在/.test(e.message)) { addChat({ kind: 'note', text: '服务重启过，这一轮对话中断了，请重新发送。' }); break; }
        await sleep(2500); continue;
      }
      for (const ev of r.events) onEvent(ev);
      next = r.next;
      store.set('lab-chat-task', { id: taskId, next });
      if (r.events.some(e => e.type === 'end') || (r.status !== 'running' && !r.events.length)) break;
      await sleep(1000);
    }
  } finally {
    store.set('lab-chat-task', null);
    typing(false);
    chatBusy = false;
    $('chat_send').disabled = false;
  }
}
async function send() {
  const text = $('chat_input').value.trim();
  if (!text || chatBusy) return;
  if (!agentCfg || !agentCfg.has_key) { addChat({ kind: 'note', text: '请先在上面的“模型设置”里填写 API Key，并保存检测。' }); $('agentsettings').open = true; return; }
  const imgs = $('chat_attach').checked ? window.lab.selectedAssets().slice(0, 6) : [];
  const skillList = [...picked];
  addChat({ kind: 'user', text, images: imgs.map(i => i.url), skills: skillList });
  $('chat_input').value = '';
  try {
    const r = await req('POST', '/api/agent/chat', { conversation: conv, text, images: imgs.map(i => i.id), skills: skillList });
    conv = r.conversation;
    store.set('lab-conv', conv);
    await follow(r.task_id, 0);
  } catch (e) { addChat({ kind: 'error', text: '发送失败：' + e.message }); }
}
$('chat_send').onclick = send;
$('chat_input').addEventListener('keydown', e => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); send(); } });
$('chat_new').onclick = () => {
  if (chatBusy) return;
  conv = null; chatItems = []; jobViews.clear();
  store.set('lab-conv', null); saveChat();
  $('chatlog').querySelectorAll('.msg').forEach(n => n.remove());
  greet();
};
function greet() {
  if (!chatItems.length) renderItem({ kind: 'note', text: '我可以帮你写提示词并直接出图、编辑图片、按姿势生成，结果会自动放到画布上。选中画布上的图再发消息，我就能看到它们（图1、图2…）。Ctrl+Enter 发送。' });
}

// ---------------------------------------------------------------- 在线 LoRA 训练
let trImages = store.get('lab-train-images', []);
let trProblems = [];
let trPassed = false;   // 审核通过时问题图只作提醒（橙色），不通过时标红
let trDefaults = null;
function saveTrain() {
  store.set('lab-train-images', trImages);
  store.set('lab-train-form', { base: $('tr_base').value, kind: $('tr_kind').value, name: $('tr_name').value, trigger: $('tr_trigger').value });
}
function renderTrain() {
  $('tr_count').textContent = trImages.length;
  const box = $('tr_images');
  box.replaceChildren();
  trImages.forEach((im, i) => {
    const issue = trProblems.filter(p => p.image === i + 1).map(p => p.issue).join('；');
    box.append(el('div', { class: 'trimg' + (issue ? (trPassed ? ' warn' : ' bad') : ''), title: issue ? `图${i + 1}：${issue}` : (im.title || `图${i + 1}`) },
      el('img', { src: im.url, alt: '训练图 ' + (i + 1), loading: 'lazy' }), el('b', {}, String(i + 1)),
      el('button', { type: 'button', title: '移除这张', 'aria-label': '移除训练图 ' + (i + 1), onclick: () => { trImages.splice(i, 1); trProblems = []; renderTrain(); saveTrain(); } }, '×')));
  });
}
function applyDefaults() {
  const d = trDefaults && trDefaults.defaults[$('tr_base').value];
  if (!d) return;
  $('tr_steps').value = d.steps; $('tr_rank').value = d.rank; $('tr_lr').value = d.lr; $('tr_px').value = d.max_pixels;
}
function trBox() { return $('tr_result'); }
function trMsg(text, kind = '') { trBox().replaceChildren(el('p', { class: 'hint ' + kind }, text)); }
async function addTrainFiles(files) {
  let added = 0;
  for (const f of files) {
    if (!['image/png', 'image/jpeg', 'image/webp'].includes(f.type)) { trMsg(`${f.name}：只支持 PNG、JPEG、WebP`, 'bad'); continue; }
    if (f.size > 16_000_000) { trMsg(`${f.name}：单张不能超过 16 MB`, 'bad'); continue; }
    if (trImages.length >= 60) { trMsg('最多 60 张', 'bad'); break; }
    trMsg(`上传中 ${added + 1}/${files.length}…`);
    try {
      const a = await req('POST', '/api/images', { data: await readDataURL(f) });
      trImages.push({ id: a.id, url: a.url, w: a.width, h: a.height, title: f.name });
      added++;
      trProblems = [];
      renderTrain();
    } catch (e) { trMsg(`${f.name}：${e.message}`, 'bad'); }
  }
  saveTrain();
  if (added) trMsg(`已加入 ${added} 张，共 ${trImages.length} 张。`);
}
$('tr_add').onclick = () => $('tr_files').click();
$('tr_files').onchange = async () => { const files = [...$('tr_files').files]; $('tr_files').value = ''; await addTrainFiles(files); };
$('tr_fromcanvas').onclick = () => {
  const list = window.lab.selectedAssets();
  if (!list.length) { trMsg('先在画布上选中图片（Shift 多选，一次最多 4 张）。', 'bad'); return; }
  let added = 0;
  for (const a of list) if (!trImages.some(x => x.id === a.id) && trImages.length < 60) { trImages.push({ id: a.id, url: a.url, w: a.w, h: a.h, title: a.title }); added++; }
  trProblems = []; renderTrain(); saveTrain();
  trMsg(`从画布加入 ${added} 张，共 ${trImages.length} 张。`);
};
$('tr_clear').onclick = () => { if (trImages.length && !confirm('清空训练图片列表？（服务器上的图片不会删除）')) return; trImages = []; trProblems = []; renderTrain(); saveTrain(); trMsg(''); };
$('tr_base').onchange = () => { applyDefaults(); saveTrain(); };
for (const id of ['tr_kind', 'tr_name', 'tr_trigger']) $(id).addEventListener('change', saveTrain);
function verdictView(v) {
  const general = (v.problems || []).filter(p => !p.image), per = (v.problems || []).filter(p => p.image);
  return el('div', { class: 'verdict ' + (v.ok ? 'ok' : 'bad') },
    el('b', {}, v.ok ? `数据检查通过（${v.score ?? '-'} 分）` : `数据检查没通过（${v.score ?? '-'} 分）`),
    el('p', {}, v.summary || ''),
    general.length || per.length ? el('ul', {}, general.map(p => el('li', {}, p.issue)), per.map(p => el('li', {}, `图${p.image}：${p.issue}`))) : null,
    v.suggestions && v.suggestions.length ? el('div', {}, el('b', {}, '建议'), el('ul', {}, v.suggestions.map(s => el('li', {}, s)))) : null,
    v.captions && v.captions.length ? el('details', {}, el('summary', {}, `模型为每张图写的训练描述（${v.captions.length} 条）`), el('ol', { class: 'caps' }, v.captions.map(c => el('li', {}, c)))) : null);
}
function progressView() {
  const bar = el('i');
  const text = el('p', { class: 'hint' }, '准备中…');
  const view = el('div', { class: 'trprogress' }, el('div', { class: 'progress', role: 'progressbar', 'aria-label': '训练进度' }, bar), text);
  view.set = (frac, msg) => { bar.style.width = Math.round(Math.max(0, Math.min(1, frac)) * 100) + '%'; text.textContent = msg; };
  return view;
}
async function followTraining(id, box = trBox()) {
  $('tr_go').disabled = true;
  const view = progressView();
  box.append(view);
  try {
    while (true) {
      let job;
      try { job = await req('GET', '/api/jobs/' + id); }
      catch (e) {
        if (/不存在/.test(e.message)) { view.set(0, '找不到训练任务（服务可能重启过）。'); store.set('lab-train-job', null); return; }
        view.set(0, '查询训练状态失败，稍后重试：' + e.message); await sleep(5000); continue;
      }
      const used = job.started ? fmtSec(Date.now() / 1000 - job.started) : '';
      if (job.status === 'queued') view.set(0.01, '排队中：等前面的出图任务完成后开始训练。');
      else if (job.status === 'running') {
        const p = job.progress || {};
        const frac = p.total ? p.step / p.total : 0;
        view.set(p.phase === 'train' ? 0.12 + 0.88 * frac : 0.12 * frac, `${p.message || '加载模型…'} · 已用 ${used}（训练期间出图任务会排队）`);
      } else if (job.status === 'succeeded') {
        view.set(1, `训练完成，用时 ${fmtSec(job.result.seconds)}。`);
        store.set('lab-train-job', null);
        await trained(job, box);
        return;
      } else {
        view.set(0, '训练失败：' + (job.error || '未知错误'));
        view.classList.add('bad');
        store.set('lab-train-job', null);
        return;
      }
      await sleep(3000);
    }
  } finally { $('tr_go').disabled = false; }
}
async function trained(job, box) {
  const lora = job.result.lora;
  await refreshLoras();
  const use = el('button', { type: 'button', onclick: () => { window.lab.useLora(lora.id); showTab('gen'); window.lab.status(`已选中 LoRA「${lora.title}」并加入触发词 ${lora.trigger}，写好描述后点“开始生成”。`); } }, '去生成面板使用');
  const info = el('div', { class: 'verdict ok' },
    el('b', {}, `LoRA「${lora.title}」已加入列表（★ 标记）`),
    el('p', {}, `触发词 ${lora.trigger} · ${lora.train.images} 张图 · ${lora.train.steps} 步 · rank ${lora.train.rank}。正在用第一条训练描述生成预览图…`), use);
  box.append(info);
  const pv = job.result.preview_job;
  while (pv) {
    let p;
    try { p = await req('GET', '/api/jobs/' + pv); } catch { break; }
    if (p.status === 'succeeded') {
      window.lab.addResult(p.result, `训练预览 · ${lora.title} ×1.0`, p.request);
      info.querySelector('p').textContent = `触发词 ${lora.trigger} · ${lora.train.images} 张图 · ${lora.train.steps} 步 · rank ${lora.train.rank}。预览图已放到画布上（强度 1.0）。`;
      info.append(el('img', { class: 'preview', src: p.result.image, alt: '训练预览图' }));
      break;
    }
    if (p.status === 'failed') { info.append(el('p', { class: 'hint bad' }, '预览图生成失败：' + p.error)); break; }
    await sleep(2500);
  }
}
$('tr_go').onclick = async () => {
  const name = $('tr_name').value.trim(), trigger = $('tr_trigger').value.trim();
  if (!name) return trMsg('请填写 LoRA 名称。', 'bad');
  if (!/^[A-Za-z][A-Za-z0-9_\-]{1,39}$/.test(trigger)) return trMsg('触发词请用英文字母开头，只含字母、数字、下划线或连字符，例如 lin_yue。', 'bad');
  if (trImages.length < 8) return trMsg(`至少需要 8 张图（建议 15–30 张），现在只有 ${trImages.length} 张。`, 'bad');
  if (!agentCfg || !agentCfg.probe || !agentCfg.probe.multimodal) { showTab('agent'); $('agentsettings').open = true; return trMsg('数据审核需要能看图的模型：请先在“助手 → 模型设置”里配置并通过检测。', 'bad'); }
  saveTrain();
  $('tr_go').disabled = true;
  const box = trBox();
  box.replaceChildren(el('p', { class: 'hint' }, `正在让 ${agentCfg.model} 审核 ${trImages.length} 张训练图（主体是否一致、清晰度、多样性），约 20–90 秒…`));
  try {
    const v = await req('POST', '/api/train/check', { base: $('tr_base').value, kind: $('tr_kind').value, trigger, images: trImages.map(i => i.id) });
    trProblems = v.problems || [];
    trPassed = !!v.ok;
    renderTrain();
    box.replaceChildren(verdictView(v));
    if (!v.ok) { box.append(el('p', { class: 'hint bad' }, '请按上面的提示调整图片后重新检查。')); return; }
    const num = id => Number($(id).value) || undefined;
    const job = await req('POST', '/api/train/start', { check_id: v.check_id, name, steps: num('tr_steps'), rank: num('tr_rank'), lr: num('tr_lr'), max_pixels: num('tr_px') });
    box.append(el('p', { class: 'hint ok' }, '检查通过，已直接开始训练。可以切到别的标签页，进度会在这里继续更新。'));
    store.set('lab-train-job', job.id);
    await followTraining(job.id, box);
  } catch (e) { box.append(el('p', { class: 'hint bad' }, '未完成：' + e.message)); }
  finally { $('tr_go').disabled = false; }
};

// ---------------------------------------------------------------- 我的 LoRA
async function refreshLoras() {
  const list = await window.lab.reloadLoras();
  renderMyLoras(list);
  return list;
}
function renderMyLoras(list) {
  const mine = list.filter(x => x.custom);
  const box = $('mylora');
  box.replaceChildren();
  if (!mine.length) { box.append(el('p', { class: 'hint' }, '训练完成的 LoRA 会出现在这里，也会以 ★ 出现在生成面板的 LoRA 列表里。')); return; }
  for (const l of mine) {
    const t = l.train || {};
    const card = el('div', { class: 'mylora' },
      el('div', { class: 'head' }, el('b', {}, '★ ' + l.title), el('span', { class: 'tag' }, l.model === 'qwen' ? 'Qwen 2.1' : 'Z-Image')),
      el('p', { class: 'hint' }, `触发词 ${l.trigger || '无'} · 默认强度 ${l.default_scale}\n${t.images ?? '?'} 张图 · ${t.steps ?? '?'} 步 · rank ${t.rank ?? '?'} · 训练 ${fmtSec(t.seconds)}` + (l.notes ? `\n备注：${l.notes}` : '')));
    const acts = el('div', { class: 'acts' },
      el('button', { type: 'button', onclick: () => { window.lab.useLora(l.id); showTab('gen'); window.lab.status(`已选中「${l.title}」（强度 ${l.default_scale}），并加入触发词。`); } }, '使用'),
      el('button', { type: 'button', onclick: () => editLora(card, l) }, '修改参数'),
      el('button', { type: 'button', onclick: async () => {
        if (!confirm(`删除 LoRA「${l.title}」？文件会从服务器删除。`)) return;
        try { await req('DELETE', '/api/loras/' + encodeURIComponent(l.id)); await refreshLoras(); } catch (e) { alert(e.message); }
      } }, '删除'));
    card.append(acts);
    box.append(card);
  }
}
function editLora(card, l) {
  if (card.querySelector('form')) return;
  const f = el('form', {},
    el('label', {}, '名称'), el('input', { name: 'title', value: l.title, maxlength: 60 }),
    el('label', {}, '触发词（写进提示词才会生效）'), el('input', { name: 'trigger', value: l.trigger || '', maxlength: 80 }),
    el('label', {}, '默认强度（0–1.5，叠加多个 LoRA 时调低）'), el('input', { name: 'default_scale', type: 'number', min: 0, max: 1.5, step: 0.05, value: l.default_scale }),
    el('label', {}, '备注'), el('input', { name: 'notes', value: l.notes || '', maxlength: 400, placeholder: '例如：0.8 最像；配合 z_snapshot 更自然' }),
    el('div', { class: 'acts' }, el('button', { type: 'submit', class: 'save' }, '保存'), el('button', { type: 'button', onclick: () => f.remove() }, '取消')));
  const field = n => f.elements.namedItem(n);
  f.onsubmit = async e => {
    e.preventDefault();
    const body = { title: field('title').value.trim() || l.title, trigger: field('trigger').value.trim(), default_scale: Number(field('default_scale').value), notes: field('notes').value.trim() };
    try { await req('PATCH', '/api/loras/' + encodeURIComponent(l.id), body); await refreshLoras(); }
    catch (err) { alert('保存失败：' + err.message); }
  };
  card.append(f);
  field('title').focus();
}

// ---------------------------------------------------------------- 初始化
(async () => {
  showTab(store.get('lab-tab', 'gen'));
  const form = store.get('lab-train-form', null);
  if (form) { for (const k of ['base', 'kind', 'name', 'trigger']) if (form[k] != null) $('tr_' + k).value = form[k]; }
  renderTrain();
  for (const item of chatItems) renderItem(item);
  greet();
  try { showConfig(await req('GET', '/api/agent/config')); } catch (e) { badge('出错', 'bad'); $('ag_probe').textContent = e.message; }
  try { await loadSkills(); } catch (e) { skillMsg(e.message, true); }
  try { trDefaults = await req('GET', '/api/train/defaults'); applyDefaults(); } catch {}
  try { renderMyLoras(await req('GET', '/api/loras')); } catch {}
  for (const item of chatItems) if (item.kind === 'job' && !['succeeded', 'failed'].includes(item.status)) watchJob(item);
  const pending = store.get('lab-chat-task', null);
  if (pending && pending.id) follow(pending.id, pending.next || 0);
  const trainJob = store.get('lab-train-job', null);
  if (trainJob) { trBox().replaceChildren(el('p', { class: 'hint' }, '继续显示上次开始的训练：')); followTraining(trainJob); }
})();
})();
