"""图像实验台的智能助手：OpenAI 兼容的多模态大模型（默认阶跃星辰 step-5-preview）+ 工具调用 + Skills。

- 模型配置保存在 agent_config.json（权限 600，含 API Key，不随代码包分发）；前端只拿到打码后的 Key。
- 保存配置后自动做一次多模态探测：发一张红底白字“7”的测试图，答对颜色和数字才算支持图片输入。
- Skills 采用 skills.sh / Agent Skills 格式（目录 + SKILL.md，frontmatter 含 name、description）。
  它们只作为写提示词的说明交给模型阅读，实验台从不执行其中的脚本。
- 助手通过工具调用实验台已有的出图接口（文生图、编辑、融合、姿势迁移、LoRA），
  每次对话在后台线程里运行，前端轮询事件并把生成结果放到画布上。
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import shutil
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / 'agent_config.json'
SKILLS = ROOT / 'skills'
REGISTRY = SKILLS / 'registry.json'
DEFAULT_CONFIG = {'base_url': 'https://api.stepfun.com/step_plan/v1', 'model': 'step-5-preview', 'api_key': '',
                  'proxy': '', 'probe': None}
TEXT_EXT = {'.md', '.txt', '.json', '.yaml', '.yml', '.csv', '.tsv', '.xml', '.html', '.css', '.py', '.js', '.ts', '.sh', ''}
MAX_SKILL_BYTES = 8 * 1024 * 1024
_config_lock = threading.Lock()
_skill_lock = threading.Lock()


# ------------------------------------------------------------------ 配置与模型调用
def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG.exists():
        try:
            cfg.update(json.loads(CONFIG.read_text(encoding='utf-8')))
        except (OSError, ValueError):
            pass
    return cfg


def save_config(cfg: dict) -> None:
    with _config_lock:
        tmp = CONFIG.with_suffix('.tmp')
        tmp.write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding='utf-8')
        os.chmod(tmp, 0o600)
        tmp.replace(CONFIG)


def mask_key(key: str) -> str:
    return '' if not key else (key[:4] + '…' + key[-4:] if len(key) > 10 else '已设置')


def public_config(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    return {'base_url': cfg['base_url'], 'model': cfg['model'], 'api_key': mask_key(cfg.get('api_key', '')),
            'has_key': bool(cfg.get('api_key')), 'proxy': cfg.get('proxy', ''), 'probe': cfg.get('probe')}


def _opener(proxy: str):
    handlers = [urllib.request.ProxyHandler({'http': proxy, 'https': proxy} if proxy else {})]
    return urllib.request.build_opener(*handlers)


def chat_completion(cfg: dict, messages: list, tools: list | None = None, max_tokens: int = 4096, timeout: int = 240) -> dict:
    if not cfg.get('api_key'):
        raise RuntimeError('还没有配置 API Key：请在“助手”页的模型设置里填写')
    body = {'model': cfg['model'], 'messages': messages, 'max_tokens': max_tokens}
    if tools:
        body['tools'] = tools
    req = urllib.request.Request(cfg['base_url'].rstrip('/') + '/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + cfg['api_key']})
    try:
        with _opener(cfg.get('proxy', '')).open(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        detail = e.read().decode('utf-8', 'ignore')[:400]
        raise RuntimeError(f'模型接口返回 HTTP {e.code}：{detail}') from None
    except urllib.error.URLError as e:
        raise RuntimeError(f'连不上模型接口：{e.reason}') from None


def image_to_data_url(im: Image.Image, max_side: int = 768) -> str:
    im = im.convert('RGB')
    scale = min(1.0, max_side / max(im.size))
    if scale < 1:
        im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, 'JPEG', quality=88)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()


def extract_json(text: str):
    text = (text or '').strip()
    fence = re.search(r'```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```', text, re.S)
    if fence:
        text = fence.group(1)
    start = min([i for i in (text.find('{'), text.find('[')) if i >= 0], default=-1)
    if start < 0:
        raise ValueError('模型没有返回 JSON')
    depth, end, opener = 0, -1, text[start]
    closer = '}' if opener == '{' else ']'
    in_str = esc = False
    for i, ch in enumerate(text[start:], start):
        if in_str:
            esc = (ch == '\\' and not esc)
            if ch == '"' and not esc:
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                end = i
                break
    return json.loads(text[start:end + 1])


def probe(cfg: dict) -> dict:
    """探测连通性和图片理解能力：红底白色数字 7，答对才算多模态。"""
    im = Image.new('RGB', (256, 256), (215, 30, 30))
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default(size=170)
    except TypeError:
        font = ImageFont.load_default()
    draw.text((78, 30), '7', fill='white', font=font)
    messages = [{'role': 'user', 'content': [
        {'type': 'image_url', 'image_url': {'url': image_to_data_url(im, 256)}},
        {'type': 'text', 'text': '这张图的背景是什么颜色？图里写的数字是几？只用 JSON 回答：{"color":"...","digit":"..."}'}]}]
    started = time.time()
    result = {'checked_at': int(time.time()), 'model': cfg.get('model'), 'connected': False, 'multimodal': False}
    try:
        data = chat_completion(cfg, messages, max_tokens=300, timeout=90)
    except RuntimeError as e:
        text = str(e)
        result['message'] = text
        if 'HTTP 4' in text and re.search(r'image|vision|multimodal|图片|图像|content type', text, re.I):
            result.update(connected=True, message='接口可用，但这个模型不接受图片输入：' + text[:160])
        return result
    result['connected'] = True
    result['seconds'] = round(time.time() - started, 1)
    answer = (data.get('choices') or [{}])[0].get('message', {}).get('content') or ''
    result['answer'] = answer[:200]
    try:
        parsed = extract_json(answer)
        color, digit = str(parsed.get('color', '')).lower(), str(parsed.get('digit', ''))
    except (ValueError, AttributeError):
        color, digit = answer.lower(), answer
    result['multimodal'] = ('7' in digit) and ('red' in color or '红' in color)
    result['message'] = ('多模态可用：模型正确识别了测试图（红底、数字 7）。' if result['multimodal'] else
                         '模型能连通，但没能正确识别测试图，可能不支持图片输入。请换成多模态模型（例如 step-5-preview）。')
    return result


# ------------------------------------------------------------------ Skills
def _registry() -> list:
    try:
        return json.loads(REGISTRY.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return []


def _save_registry(items: list) -> None:
    SKILLS.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding='utf-8')


def list_skills() -> list:
    return [{k: s.get(k) for k in ('name', 'description', 'source', 'enabled', 'files', 'installed_at', 'bytes')} for s in _registry()]


def parse_frontmatter(text: str) -> dict:
    m = re.match(r'^﻿?---\s*\n(.*?)\n---\s*\n', text, re.S)
    if not m:
        return {}
    try:
        import yaml
        data = yaml.safe_load(m.group(1)) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        out = {}
        for line in m.group(1).splitlines():
            if ':' in line and not line.startswith(' '):
                k, v = line.split(':', 1)
                out[k.strip()] = v.strip().strip('"\'')
        return out


def _safe_name(name: str) -> str:
    name = re.sub(r'[^A-Za-z0-9._\-一-鿿]+', '-', str(name)).strip('-.')
    return name[:64] or 'skill'


def _collect(root: Path, subpath: str = '') -> list[Path]:
    base = (root / subpath).resolve() if subpath else root.resolve()
    if not str(base).startswith(str(root.resolve())) or not base.exists():
        raise ValueError(f'仓库里找不到路径：{subpath}')
    found = []
    for p in sorted(base.rglob('SKILL.md')):
        rel = p.relative_to(root)
        # 以 . 开头的隐藏目录里常是开发用技能或副本，默认跳过；明确指定子路径时除外
        if any(part.startswith('.') or part == 'node_modules' for part in rel.parts[:-1]) and not subpath:
            continue
        found.append(p.parent)
    return found


def _install_dirs(dirs: list[Path], source: str) -> list:
    installed = []
    with _skill_lock:
        items = _registry()
        for d in dirs:
            text = (d / 'SKILL.md').read_text(encoding='utf-8', errors='replace')
            meta = parse_frontmatter(text)
            name = _safe_name(meta.get('name') or d.name)
            desc = str(meta.get('description') or '').strip()
            total = sum(f.stat().st_size for f in d.rglob('*') if f.is_file())
            if total > MAX_SKILL_BYTES:
                raise ValueError(f'技能 {name} 太大（{total // 1024} KB），上限 8 MB')
            dest = SKILLS / name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(d, dest, ignore=shutil.ignore_patterns('.git', 'node_modules', '__pycache__'))
            files = sorted(str(f.relative_to(dest)).replace('\\', '/') for f in dest.rglob('*') if f.is_file())
            items = [s for s in items if s['name'] != name]
            items.append({'name': name, 'description': desc, 'source': source, 'enabled': True, 'files': files,
                          'bytes': total, 'installed_at': int(time.time())})
            installed.append(name)
        _save_registry(items)
    return installed


def _turbo_proxy() -> str:
    """AutoDL 学术加速（访问 GitHub raw / HuggingFace）；其他机器上不存在就返回空。"""
    try:
        m = re.search(r'https_proxy=(\S+)', Path('/etc/network_turbo').read_text())
        return m.group(1) if m else ''
    except OSError:
        return ''


def _download(url: str, timeout: int = 120) -> bytes:
    last = None
    for proxy in ('', _turbo_proxy()):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'image-lab-skills'})
            with _opener(proxy).open(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # 直连失败再走加速代理
            last = e
    raise RuntimeError(f'下载失败：{url}（{last}）')


def parse_skill_source(spec: str) -> dict:
    spec = spec.strip()
    m = re.match(r'^(?:https?://github\.com/)?([\w.\-]+)/([\w.\-]+?)(?:\.git)?(?:/(?:tree|blob)/([^/]+))?(?:/(.*?))?/?$', spec)
    if not m:
        raise ValueError('请填写 GitHub 地址或 owner/repo（可带子路径），例如 iamyoki/qwen-image-2.1-skill')
    owner, repo, branch, sub = m.groups()
    sub = (sub or '').strip('/')
    if sub.endswith('SKILL.md'):
        sub = sub[:-len('SKILL.md')].strip('/')
    return {'owner': owner, 'repo': repo, 'branch': branch, 'subpath': sub}


def install_from_github(spec: str) -> dict:
    src = parse_skill_source(spec)
    branch = src['branch']
    if not branch:
        try:
            branch = json.loads(_download(f"https://api.github.com/repos/{src['owner']}/{src['repo']}", 30)).get('default_branch') or 'main'
        except Exception:
            branch = 'main'
    data = _download(f"https://codeload.github.com/{src['owner']}/{src['repo']}/zip/refs/heads/{branch}")
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            z.extractall(tmp)
        roots = [p for p in Path(tmp).iterdir() if p.is_dir()]
        root = roots[0] if len(roots) == 1 else Path(tmp)
        dirs = _collect(root, src['subpath'])
        if not dirs:
            raise ValueError('这个位置下没有找到 SKILL.md')
        if len(dirs) > 3 and not src['subpath']:
            choices = []
            for d in dirs[:80]:
                rel = str(d.relative_to(root)).replace('\\', '/')
                meta = parse_frontmatter((d / 'SKILL.md').read_text(encoding='utf-8', errors='replace'))
                choices.append({'path': rel, 'name': str(meta.get('name') or d.name), 'description': str(meta.get('description') or '')[:240],
                                'source': f"{src['owner']}/{src['repo']}/tree/{branch}/{rel}"})
            return {'installed': [], 'choices': choices, 'message': f'仓库里有 {len(dirs)} 个技能，请选择要安装的一个（或填写 owner/repo/子路径）。'}
        source = f"github:{src['owner']}/{src['repo']}@{branch}" + (f"/{src['subpath']}" if src['subpath'] else '')
        names = _install_dirs(dirs, source)
    return {'installed': names, 'message': '已安装：' + '、'.join(names)}


def install_from_upload(filename: str, data: bytes) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        if filename.lower().endswith('.zip'):
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for info in z.infolist():
                    target = (root / info.filename).resolve()
                    if not str(target).startswith(str(root.resolve())):
                        raise ValueError('压缩包里有非法路径')
                z.extractall(root)
        elif filename.lower().endswith('.md'):
            (root / 'upload').mkdir()
            (root / 'upload' / 'SKILL.md').write_bytes(data)
        else:
            raise ValueError('请上传技能压缩包（.zip）或 SKILL.md')
        dirs = [d for d in _collect(root)]
        if not dirs:
            raise ValueError('没有找到 SKILL.md')
        names = _install_dirs(dirs, 'upload:' + filename)
    return {'installed': names, 'message': '已安装：' + '、'.join(names)}


def set_skill_enabled(name: str, enabled: bool) -> None:
    with _skill_lock:
        items = _registry()
        for s in items:
            if s['name'] == name:
                s['enabled'] = bool(enabled)
        _save_registry(items)


def delete_skill(name: str) -> None:
    with _skill_lock:
        items = [s for s in _registry() if s['name'] != name]
        shutil.rmtree(SKILLS / _safe_name(name), ignore_errors=True)
        _save_registry(items)


def read_skill_file(name: str, path: str = 'SKILL.md', limit: int = 60000) -> str:
    base = (SKILLS / _safe_name(name)).resolve()
    target = (base / path).resolve()
    if not str(target).startswith(str(base)) or not target.is_file():
        raise ValueError(f'技能 {name} 里没有文件 {path}')
    if target.suffix.lower() not in TEXT_EXT:
        raise ValueError('只能读取文本文件')
    text = target.read_text(encoding='utf-8', errors='replace')
    return text if len(text) <= limit else text[:limit] + '\n…（后面内容过长已截断）'


# ------------------------------------------------------------------ 训练数据检查
def check_dataset(images: list[Image.Image], base: str, trigger: str, kind: str, cfg: dict | None = None) -> dict:
    """让多模态模型判断训练图是否合格，并为每张图写英文 caption（以触发词开头）。"""
    cfg = cfg or load_config()
    kinds = {'person': '同一个人物（固定人脸和形象）', 'style': '同一种画风', 'object': '同一个物体或产品'}
    rubric = (f'你是 LoRA 训练数据审核员。用户要为 {"Qwen-Image-2.1" if base == "qwen" else "Z-Image-Turbo"} 训练一个 LoRA，'
              f'目标：{kinds.get(kind, kind)}，触发词：{trigger}。下面按顺序给出 {len(images)} 张训练图（图1…图{len(images)}）。\n'
              '请逐张检查并整体判断是否适合训练，标准：\n'
              '1. 主体一致：人物 LoRA 必须全部是同一个人（脸部特征一致），风格 LoRA 必须是同一种画风，物体 LoRA 必须是同一个物体；\n'
              '2. 质量：主体清晰、不严重模糊、不过暗过曝，没有大面积水印、边框或文字遮挡主体；\n'
              '3. 多样性：角度、表情、光线、背景、服装或构图有一定变化，不能几乎完全重复；\n'
              '4. 人物 LoRA 至少要有几张能看清脸的近景；图中如有多个人，主体要明确；\n'
              '5. 不含色情、未成年人或违法内容。\n'
              '只要存在会明显破坏训练效果的问题（例如混入另一个人、大部分图模糊）就判定不合格；个别小问题可以合格但要指出。\n'
              + ('主体一致性要严格判断：先写出图1人物的五官特征（脸型、眉眼、鼻子、嘴唇、发际线、痣或其他可辨认特征），'
                 '再把其余每张图和图1逐一对比。风格相近的 AI 生成年轻女性很容易被误认成同一个人，必须以五官细节为准，'
                 '不要只看发型、服装、妆容和画风。看不清脸的图填 "unsure"。\n' if kind == 'person' else
                 '主体一致性要严格判断：先写出图1的主体特征，再把其余每张图和图1逐一对比。\n')
              + f'同时为每张图写一句英文 caption，用于训练：以 “{trigger}” 开头，然后描述画面中除主体固有特征以外的内容'
              '（姿势、服装、背景、光线、构图、画风），20–40 个英文单词。\n'
              '只输出 JSON，不要其他文字：{"reference":"图1主体的特征（中文）","same_subject":[每张图是否与图1是同一主体：true/false/"unsure"，第1项为 true],'
              '"ok":true/false,"score":0-100,"summary":"一两句中文总体结论",'
              '"problems":[{"image":序号,"issue":"中文问题说明"}],"suggestions":["中文改进建议"],"captions":["每张图一条英文 caption，顺序与图片一致"]}')
    content = []
    for i, im in enumerate(images, 1):
        content.append({'type': 'text', 'text': f'图{i}'})
        content.append({'type': 'image_url', 'image_url': {'url': image_to_data_url(im, 512)}})
    content.append({'type': 'text', 'text': rubric})
    # step-5-preview 会先思考再输出，思考也算在 max_tokens 里；16 张图时思考 + JSON 约 6000 token，额度给足并在截断时重试一次
    verdict, answer, data = None, '', {}
    for _attempt in range(2):
        data = chat_completion(cfg, [{'role': 'user', 'content': content}], max_tokens=16000, timeout=420)
        choice = (data.get('choices') or [{}])[0]
        answer = choice.get('message', {}).get('content') or ''
        try:
            verdict = extract_json(answer)
            break
        except ValueError:
            continue
    if verdict is None:
        raise RuntimeError(f"模型没有按要求返回 JSON（finish_reason={choice.get('finish_reason')}）：" + answer[:200])
    captions = [str(c) for c in (verdict.get('captions') or [])]
    if len(captions) != len(images):
        captions = (captions + [f'{trigger}, a photo'] * len(images))[:len(images)]
    captions = [c if c.lower().startswith(trigger.lower()) else f'{trigger}, {c}' for c in captions]
    ok = bool(verdict.get('ok'))
    summary = verdict.get('summary', '')
    problems = [p for p in (verdict.get('problems') or []) if isinstance(p, dict)]
    same = verdict.get('same_subject')
    odd = []
    if isinstance(same, list) and len(same) == len(images):
        odd = [i for i, s in enumerate(same, 1) if s is False or str(s).lower() == 'false']
    if odd:   # 混入不同主体一律不合格：让用户移除后重新检查
        what = {'person': '人', 'style': '画风', 'object': '物体'}.get(kind, '主体')
        for i in odd:
            if not any(p.get('image') == i for p in problems):
                problems.append({'image': i, 'issue': f'和图1不是同一个{what}'})
        if ok:
            summary = (summary + ' ' if summary else '') + f'但图{"、图".join(map(str, odd))}和图1不是同一个{what}，请移除后重新检查。'
        ok = False
    problems.sort(key=lambda p: p.get('image') or 0)
    return {'ok': ok, 'score': verdict.get('score'), 'summary': summary, 'reference': verdict.get('reference', ''),
            'same_subject': same if isinstance(same, list) else None,
            'problems': problems, 'suggestions': verdict.get('suggestions') or [], 'captions': captions,
            'model': cfg.get('model'), 'usage': data.get('usage')}


# ------------------------------------------------------------------ 助手对话
TOOLS = [
    {'type': 'function', 'function': {'name': 'generate_image', 'description': '文生图。Z-Image-Turbo 9 步约 20 秒，适合快速出图；Qwen-Image-2.1 40 步约 45 秒，文字排版和复杂构图更好。',
     'parameters': {'type': 'object', 'properties': {
         'prompt': {'type': 'string', 'description': '完整提示词'},
         'model': {'type': 'string', 'enum': ['zimage', 'qwen'], 'description': '默认 zimage'},
         'width': {'type': 'integer', 'description': '宽，16 的倍数，256–2048，默认 1024'},
         'height': {'type': 'integer', 'description': '高，16 的倍数，256–2048，默认 1024'},
         'seed': {'type': 'integer'},
         'loras': {'type': 'array', 'description': '要叠加的 LoRA（必须与 model 对应），最多 4 个', 'items': {'type': 'object', 'properties': {'id': {'type': 'string'}, 'scale': {'type': 'number'}}, 'required': ['id']}}},
         'required': ['prompt']}}},
    {'type': 'function', 'function': {'name': 'edit_image', 'description': '用 Qwen-Image-2.1 编辑图片：1 张为参考编辑，2–4 张为多图融合（提示词里用“图1”“图2”指代，按 image_ids 顺序）。输出 2048，约 2 分钟。',
     'parameters': {'type': 'object', 'properties': {
         'image_ids': {'type': 'array', 'items': {'type': 'string'}, 'description': '画布图片 ID，1–4 个'},
         'prompt': {'type': 'string'}}, 'required': ['image_ids', 'prompt']}}},
    {'type': 'function', 'function': {'name': 'pose_transfer', 'description': '按参考图中人物的姿势生成新图：先自动提取骨架，再用 Z-Image-Turbo ControlNet（快，约 30 秒）或 Qwen-Image-2.1（可再给一张人物参考图保持长相，约 2 分钟）生成。',
     'parameters': {'type': 'object', 'properties': {
         'pose_image_id': {'type': 'string', 'description': '提供姿势的图片 ID（照片或骨架图都可以）'},
         'prompt': {'type': 'string', 'description': '新图的内容描述'},
         'model': {'type': 'string', 'enum': ['zimage', 'qwen']},
         'subject_image_id': {'type': 'string', 'description': '可选，仅 qwen：要保持长相的人物参考图'},
         'control_scale': {'type': 'number', 'description': '仅 zimage，姿势约束强度 0.5–1.0，默认 0.75'}},
         'required': ['pose_image_id', 'prompt']}}},
    {'type': 'function', 'function': {'name': 'wait_for_jobs', 'description': '等待已提交的任务完成（每个最多 5 分钟），返回结果图片 ID 和耗时。',
     'parameters': {'type': 'object', 'properties': {'job_ids': {'type': 'array', 'items': {'type': 'string'}}}, 'required': ['job_ids']}}},
    {'type': 'function', 'function': {'name': 'view_image', 'description': '查看一张图片（例如检查生成结果是否符合要求），图片会在下一轮对话中提供给你。',
     'parameters': {'type': 'object', 'properties': {'image_id': {'type': 'string'}}, 'required': ['image_id']}}},
    {'type': 'function', 'function': {'name': 'list_loras', 'description': '列出可用的 LoRA（包括用户自己训练的），含 ID、适用模型、触发词和推荐强度。',
     'parameters': {'type': 'object', 'properties': {}}}},
    {'type': 'function', 'function': {'name': 'load_skill', 'description': '读取一个已启用技能的完整说明（SKILL.md）和文件列表。需要按某个技能的规则工作时先调用。',
     'parameters': {'type': 'object', 'properties': {'name': {'type': 'string'}}, 'required': ['name']}}},
    {'type': 'function', 'function': {'name': 'read_skill_file', 'description': '读取技能目录里的参考文件（例如 references/t2i_rules.md）。',
     'parameters': {'type': 'object', 'properties': {'name': {'type': 'string'}, 'path': {'type': 'string'}}, 'required': ['name', 'path']}}},
]

SYSTEM = """你是“图像实验台”的创作助手，用中文回答，简洁、直接。
你可以调用工具在用户的服务器上出图，服务器有两个本地模型：
- Z-Image-Turbo：只能文生图，9 步约 20 秒，支持 LoRA 叠加和姿势控制（ControlNet），适合快速出图；
- Qwen-Image-2.1：文生图约 45 秒；也能编辑图片、多图融合（输出 2048，约 2 分钟），中英文排版更强。
工作规则：
1. 用户要出图时，先写好高质量提示词（主体、场景、构图、光线、风格都写具体），再调用工具；一次最多提交 4 个任务。
2. 提交后调用 wait_for_jobs 等结果；需要检查效果时用 view_image 看图，不满意可以改提示词再出一次（最多重试 2 次）。
3. 最后用一两句话告诉用户做了什么、结果在画布上的哪里（结果会自动出现在画布上），不要编造图片 ID 或结果。
4. 用户消息里的“画布图片”带有 ID，编辑、融合、姿势迁移时使用这些 ID。
5. 如果某个已启用技能与任务相关，先用 load_skill 读取它，并按它的规则写提示词；技能只是写作说明，不要尝试执行其中的脚本或命令。
6. LoRA 必须和模型对应，ID 从 list_loras 获取；有触发词的 LoRA 要把触发词写进提示词。"""


class Bridge:
    """由 app.py 注入：提交任务、等待任务、读取图片、列出 LoRA。"""
    submit = None       # (request dict) -> job id
    wait = None         # (job id, timeout) -> job dict
    image = None        # (image id) -> PIL.Image
    loras = None        # () -> list


bridge = Bridge()
TASKS: dict[str, dict] = {}
CONVERSATIONS: dict[str, list] = {}
_task_lock = threading.Lock()


def _event(task: dict, **ev):
    ev['t'] = round(time.time(), 2)
    with _task_lock:
        task['events'].append(ev)


def _catalog_text() -> str:
    enabled = [s for s in _registry() if s.get('enabled')]
    if not enabled:
        return '（当前没有启用的技能）'
    return '\n'.join(f"- {s['name']}：{s.get('description', '')[:400]}" for s in enabled)


def _run_tool(task: dict, name: str, args: dict, pending_images: list) -> dict:
    if name == 'generate_image':
        req = {'operation': 'generate', 'model': args.get('model') or 'zimage', 'prompt': args['prompt'],
               'width': int(args.get('width') or 1024), 'height': int(args.get('height') or 1024)}
        req['steps'] = 9 if req['model'] == 'zimage' else 40
        if args.get('seed') is not None:
            req['seed'] = int(args['seed'])
        if args.get('loras'):
            req['loras'] = [{'id': x['id'], 'scale': float(x.get('scale', 0.8))} for x in args['loras']][:4]
        job = bridge.submit(req)
        _event(task, type='job', job_id=job, operation='generate')
        return {'job_id': job, 'status': 'queued'}
    if name == 'edit_image':
        ids = list(args['image_ids'])[:4]
        req = {'operation': 'edit' if len(ids) == 1 else 'merge', 'model': 'qwen', 'prompt': args['prompt'], 'images': ids,
               'width': 2048, 'height': 2048, 'vae_tiling': True, 'steps': 40}
        job = bridge.submit(req)
        _event(task, type='job', job_id=job, operation=req['operation'])
        return {'job_id': job, 'status': 'queued'}
    if name == 'pose_transfer':
        model = args.get('model') or 'zimage'
        req = {'operation': 'pose', 'model': model, 'prompt': args['prompt'], 'images': [args['pose_image_id']]}
        if model == 'qwen' and args.get('subject_image_id'):
            req['images'].append(args['subject_image_id'])
        if model == 'zimage':
            req['steps'] = 9
            req['control_scale'] = float(args.get('control_scale') or 0.75)
        else:
            req.update(steps=40, width=2048, height=2048, vae_tiling=True)
        job = bridge.submit(req)
        _event(task, type='job', job_id=job, operation='pose')
        return {'job_id': job, 'status': 'queued'}
    if name == 'wait_for_jobs':
        out = []
        for jid in list(args['job_ids'])[:4]:
            job = bridge.wait(jid, 300)
            if job.get('status') == 'succeeded':
                r = job['result']
                out.append({'job_id': jid, 'status': 'succeeded', 'image_id': r['image_id'], 'size': [r['width'], r['height']],
                            'seconds': round(r.get('metrics', {}).get('seconds', 0), 1)})
            else:
                out.append({'job_id': jid, 'status': job.get('status'), 'error': job.get('error')})
        return {'results': out}
    if name == 'view_image':
        im = bridge.image(args['image_id'])
        pending_images.append((args['image_id'], im))
        return {'ok': True, 'note': '图片会在下一条消息中提供'}
    if name == 'list_loras':
        return {'loras': bridge.loras()}
    if name == 'load_skill':
        skill = next((s for s in _registry() if s['name'] == args['name']), None)
        if not skill:
            return {'error': '没有这个技能'}
        if not skill.get('enabled'):
            return {'error': '这个技能已被用户停用'}
        _event(task, type='skill', name=skill['name'])
        return {'name': skill['name'], 'content': read_skill_file(skill['name']), 'files': skill.get('files', [])}
    if name == 'read_skill_file':
        return {'content': read_skill_file(args['name'], args['path'], 30000)}
    return {'error': f'未知工具 {name}'}


def _run(task: dict, conv_id: str, user_text: str, image_ids: list[str], forced_skills: list[str]):
    cfg = load_config()
    try:
        history = CONVERSATIONS.setdefault(conv_id, [])
        system = SYSTEM + '\n\n已启用的技能（按需用 load_skill 读取）：\n' + _catalog_text()
        for name in forced_skills:
            try:
                system += f'\n\n用户要求本次使用技能「{name}」，完整说明如下：\n' + read_skill_file(name)
                _event(task, type='skill', name=name)
            except ValueError:
                pass
        content = []
        if image_ids:
            content.append({'type': 'text', 'text': '画布上选中的图片（按顺序）：' + '，'.join(f'图{i}={x}' for i, x in enumerate(image_ids, 1))})
            for x in image_ids[:6]:
                content.append({'type': 'image_url', 'image_url': {'url': image_to_data_url(bridge.image(x))}})
        content.append({'type': 'text', 'text': user_text})
        history.append({'role': 'user', 'content': content})
        for _round in range(10):
            data = chat_completion(cfg, [{'role': 'system', 'content': system}] + history, tools=TOOLS)
            msg = (data.get('choices') or [{}])[0].get('message', {})
            calls = msg.get('tool_calls') or []
            entry = {'role': 'assistant', 'content': msg.get('content') or ''}
            if calls:
                entry['tool_calls'] = calls
            history.append(entry)
            if msg.get('content'):
                _event(task, type='assistant', text=msg['content'])
            if not calls:
                break
            pending = []
            for call in calls:
                fn = call.get('function', {})
                try:
                    args = json.loads(fn.get('arguments') or '{}')
                except ValueError:
                    args = {}
                _event(task, type='tool', name=fn.get('name'), args=args)
                try:
                    result = _run_tool(task, fn.get('name'), args, pending)
                except Exception as e:  # 工具失败也要回传给模型，让它换办法
                    result = {'error': str(e)[:300]}
                _event(task, type='tool_result', name=fn.get('name'), result=result)
                history.append({'role': 'tool', 'tool_call_id': call.get('id'), 'content': json.dumps(result, ensure_ascii=False)})
            if pending:
                parts = [{'type': 'text', 'text': '这是你要求查看的图片：' + '，'.join(x for x, _ in pending)}]
                parts += [{'type': 'image_url', 'image_url': {'url': image_to_data_url(im)}} for _, im in pending]
                history.append({'role': 'user', 'content': parts})
        task['status'] = 'done'
    except Exception as e:
        task['status'] = 'failed'
        _event(task, type='error', text=str(e)[:500])
    finally:
        _event(task, type='end')
        # 控制对话长度：只保留最近 40 条，并去掉旧消息里的图片数据
        hist = CONVERSATIONS.get(conv_id, [])
        for m in hist[:-6]:
            if isinstance(m.get('content'), list):
                m['content'] = [p if p.get('type') != 'image_url' else {'type': 'text', 'text': '[图片已省略]'} for p in m['content']]
        # 从用户消息处截断，避免留下没有对应 tool_calls 的 tool 消息（接口会拒绝）
        cut = max(0, len(hist) - 40)
        while cut < len(hist) and hist[cut].get('role') != 'user':
            cut += 1
        CONVERSATIONS[conv_id] = hist[cut:]


def start_chat(conv_id: str | None, text: str, image_ids: list[str], forced_skills: list[str]) -> dict:
    conv_id = conv_id or uuid.uuid4().hex
    task = {'id': uuid.uuid4().hex, 'conversation': conv_id, 'status': 'running', 'events': [], 'created': time.time()}
    TASKS[task['id']] = task
    threading.Thread(target=_run, args=(task, conv_id, text, image_ids, forced_skills), daemon=True).start()
    return {'task_id': task['id'], 'conversation': conv_id}


def task_events(task_id: str, since: int = 0) -> dict:
    task = TASKS.get(task_id)
    if not task:
        raise KeyError(task_id)
    with _task_lock:
        events = task['events'][since:]
    return {'status': task['status'], 'events': events, 'next': since + len(events)}
