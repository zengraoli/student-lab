"""Single-GPU local teaching service. Run through SSH forwarding, not a public gateway."""
from __future__ import annotations
import base64, gc, hashlib, io, json, os, queue, re, shutil, threading, time, traceback, uuid
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator, ConfigDict
from PIL import Image, ImageOps, ImageChops, ImageFilter, UnidentifiedImageError
import agent

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
for folder in ('images', 'jobs', 'metrics'):
    (DATA / folder).mkdir(parents=True, exist_ok=True)
Image.MAX_IMAGE_PIXELS = 20_000_000
app = FastAPI(title='图像实验台', version='1.0')
app.mount('/images', StaticFiles(directory=DATA / 'images'), name='images')
# 前端文件每次都向服务器确认版本（未改动时只返回 304），避免更新后浏览器仍用旧页面
NO_CACHE = {'Cache-Control': 'no-cache'}
jobs: dict = {}
lock = threading.Lock()
work = queue.Queue(maxsize=24)
CUSTOM_LORAS = ROOT / 'loras_custom.json'
Z_CONTROLNET = Path(os.environ.get('ZIMAGE_CONTROLNET', '/root/autodl-tmp/addons/z-controlnet-union-2.1/Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors'))
LORAS: dict = {}
_custom_lock = threading.Lock()

def reload_loras():
    items = json.loads((ROOT/'loras.json').read_text(encoding='utf-8'))
    if CUSTOM_LORAS.exists():
        try:
            items += json.loads(CUSTOM_LORAS.read_text(encoding='utf-8'))
        except ValueError:
            pass
    LORAS.clear()
    LORAS.update({x['id']: x for x in items})

reload_loras()

class Upload(BaseModel):
    model_config = ConfigDict(extra='forbid')
    data: str = Field(max_length=22_000_000)

class LoraUse(BaseModel):
    model_config = ConfigDict(extra='forbid')
    id: str = Field(max_length=80)
    scale: float = Field(default=0.7, ge=0, le=1.5, allow_inf_nan=False)

class Request(BaseModel):
    model_config = ConfigDict(extra='forbid')
    operation: Literal['generate','edit','merge','transparent','upscale','collage','inpaint','remove_object','remove_background','pose','pose_extract'] = 'generate'
    model: Literal['qwen','zimage'] = 'qwen'
    precision: Literal['bf16','nf4'] = 'bf16'
    prompt: str = Field(default='', max_length=6000)
    images: list[str] = Field(default_factory=list, max_length=4)
    width: int = Field(default=1024, ge=256, le=2048, multiple_of=16)
    height: int = Field(default=1024, ge=256, le=2048, multiple_of=16)
    steps: int = Field(default=40, ge=1, le=60)
    seed: int = Field(default=42, ge=0, le=2147483647)
    kv_cache: bool = True
    vae_tiling: bool = False
    factor: int = Field(default=2, ge=2, le=4)
    reference_resolution: int = Field(default=1024, ge=512, le=1536, multiple_of=16)
    mask: str | None = None
    lora: str | None = None
    lora_scale: float = Field(default=0.7, ge=0, le=1.5, allow_inf_nan=False)
    lora2: str | None = None
    lora2_scale: float = Field(default=0.5, ge=0, le=1.5, allow_inf_nan=False)
    loras: list[LoraUse] = Field(default_factory=list, max_length=6)
    control_scale: float = Field(default=0.9, ge=0.2, le=1.5, allow_inf_nan=False)

    @model_validator(mode='after')
    def validate_operation(self):
        # 旧版请求用 lora / lora2 两个字段，这里统一转成 loras 列表
        if not self.loras:
            self.loras = [LoraUse(id=i, scale=s) for i, s in ((self.lora, self.lora_scale), (self.lora2, self.lora2_scale)) if i]
        ids = [u.id for u in self.loras]
        if len(set(ids)) != len(ids):
            raise ValueError('同一个 LoRA 不能重复添加')
        for use in self.loras:
            entry = LORAS.get(use.id)
            if not entry or entry['model'] != self.model:
                raise ValueError(f'LoRA {use.id} 不存在或与所选底模不匹配')
            if self.precision != 'bf16' or self.operation not in ('generate', 'pose'):
                raise ValueError('LoRA 目前只支持 BF16 文生图和姿势迁移')
            if self.operation == 'pose' and self.model != 'zimage':
                raise ValueError('姿势迁移叠加 LoRA 目前只支持 Z-Image-Turbo')
        count = len(self.images)
        if self.operation in ('edit','upscale','inpaint','remove_object','remove_background','pose_extract') and count != 1:
            raise ValueError('此操作需要一张参考图')
        if self.operation == 'pose' and not (count == 1 or (count == 2 and self.model == 'qwen')):
            raise ValueError('姿势迁移需要一张姿势参考图；用 Qwen 时可以再加一张人物参考图')
        if self.operation in ('merge','collage') and count < 2:
            raise ValueError('此操作需要至少两张参考图')
        if self.operation in ('generate','transparent') and count:
            raise ValueError('文生图操作不接收参考图，请选择编辑或多图融合')
        if self.model == 'zimage' and (self.operation not in ('generate','upscale','collage','pose','pose_extract') or self.precision != 'bf16'):
            raise ValueError('本示例 Z-Image-Turbo 仅支持 BF16 文生图和姿势迁移；编辑请选 Qwen')
        if self.operation not in ('upscale','collage','remove_object','remove_background','pose_extract') and not self.prompt.strip():
            raise ValueError('请输入提示词')
        if self.operation in ('inpaint','remove_object') and not self.mask:
            raise ValueError('请先绘制要修改的选区')
        if self.mask and self.operation not in ('inpaint','remove_object'):
            raise ValueError('此操作不使用选区')
        return self

def asset(name):
    if len(name) != 36 or not name.endswith('.png') or any(c not in '0123456789abcdef' for c in name[:-4]):
        raise HTTPException(400, '图片 ID 无效')
    path = DATA / 'images' / name
    if not path.is_file():
        raise HTTPException(404, '参考图不存在，请重新上传')
    return path

def save_job(job):
    temp = DATA / 'jobs' / (job['id'] + '.tmp')
    temp.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(temp.with_suffix('.json'))

@app.get('/')
def home():
    return FileResponse(ROOT / 'index.html', headers=NO_CACHE)

@app.get('/api/presets')
def presets():
    return json.loads((ROOT/'presets.json').read_text(encoding='utf-8'))

@app.get('/api/loras')
def loras():
    return [{k:v for k,v in entry.items() if k!='file'} for entry in LORAS.values()]

class LoraPatch(BaseModel):
    model_config = ConfigDict(extra='forbid')
    title: str | None = Field(default=None, max_length=60)
    trigger: str | None = Field(default=None, max_length=80)
    default_scale: float | None = Field(default=None, ge=0, le=1.5, allow_inf_nan=False)
    notes: str | None = Field(default=None, max_length=400)

def _custom_items():
    try:
        return json.loads(CUSTOM_LORAS.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return []

def _save_custom(items):
    CUSTOM_LORAS.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding='utf-8')
    reload_loras()

@app.patch('/api/loras/{lora_id}')
def edit_lora(lora_id: str, body: LoraPatch):
    with _custom_lock:
        items = _custom_items()
        entry = next((x for x in items if x['id'] == lora_id), None)
        if not entry:
            raise HTTPException(404, '只能修改自己训练的 LoRA')
        for k, v in body.model_dump(exclude_none=True).items():
            entry[k] = v.strip() if isinstance(v, str) else v
        _save_custom(items)
    return {k: v for k, v in entry.items() if k != 'file'}

@app.delete('/api/loras/{lora_id}')
def delete_lora(lora_id: str):
    with _custom_lock:
        items = _custom_items()
        entry = next((x for x in items if x['id'] == lora_id), None)
        if not entry:
            raise HTTPException(404, '只能删除自己训练的 LoRA')
        (ROOT / 'loras' / entry['file']).unlink(missing_ok=True)
        _save_custom([x for x in items if x['id'] != lora_id])
    return {'deleted': lora_id}

@app.get('/lora-comparison')
def lora_comparison():
    return FileResponse(ROOT/'lora_comparison.html')

@app.get('/api/lora-evaluation')
def lora_evaluation():
    records=[]
    for path in sorted((ROOT/'lora_evaluation').glob('*.json')):
        try:
            record=json.loads(path.read_text(encoding='utf-8'))
            if 'request' in record and 'status' in record:
                records.append({'example':path.stem,**record})
        except (OSError,ValueError):
            pass
    return records

@app.get('/canvas.js')
def canvas_js():
    return FileResponse(ROOT/'canvas.js', media_type='text/javascript', headers=NO_CACHE)

@app.get('/canvas.css')
def canvas_css():
    return FileResponse(ROOT/'canvas.css', media_type='text/css', headers=NO_CACHE)

@app.get('/panels.js')
def panels_js():
    return FileResponse(ROOT/'panels.js', media_type='text/javascript', headers=NO_CACHE)

@app.get('/api/health')
def health():
    cfg = agent.public_config()
    return {'status':'ok', 'queue':work.qsize(), 'comfyui_dependency':False,
            'models':['qwen','zimage'], 'image_editing':'Qwen reference-conditioned editing; no strength parameter',
            'upscale':'Pillow Lanczos interpolation, not learned super-resolution',
            'features': {'pose_controlnet': Z_CONTROLNET.is_file(), 'max_loras': 6, 'training': True,
                         'agent_model': cfg['model'], 'agent_ready': bool(cfg['has_key']),
                         'agent_multimodal': bool((cfg.get('probe') or {}).get('multimodal'))}}

@app.post('/api/images')
def upload(body: Upload):
    try:
        raw = base64.b64decode(body.data.split(',')[-1], validate=True)
        if len(raw) > 16_000_000:
            raise ValueError('文件不能超过 16 MB')
        with Image.open(io.BytesIO(raw)) as original:
            if original.width * original.height > 16_000_000:
                raise ValueError('图片不能超过 1600 万像素')
            img = ImageOps.exif_transpose(original).convert('RGBA')
            img.info.clear()
            img.thumbnail((2048,2048))
            name = uuid.uuid4().hex + '.png'
            img.save(DATA / 'images' / name)
        return {'id':name, 'url':'/images/'+name, 'width':img.width,'height':img.height}
    except (ValueError, OSError, UnidentifiedImageError, Image.DecompressionBombError) as e:
        raise HTTPException(400, str(e))

def enqueue(job):
    with lock:
        if work.full():
            raise HTTPException(429,'队列已满，请稍后重试')
        jobs[job['id']] = job
        save_job(job)
        work.put_nowait(job['id'])
    return job.copy()

@app.post('/api/jobs', status_code=202)
def submit(body: Request):
    for name in body.images:
        asset(name)
    if body.mask:
        with Image.open(asset(body.mask)) as m, Image.open(asset(body.images[0])) as source:
            if m.size != source.size:
                raise HTTPException(400, '遮罩尺寸必须与原图一致；大于 2048 的结果请重新上传后选择区域')
            if m.convert('L').point(lambda x: 255 if x > 127 else 0).getbbox() is None:
                raise HTTPException(400, '选区为空，请先绘制选区')
    if body.operation == 'pose' and body.model == 'zimage' and not Z_CONTROLNET.is_file():
        raise HTTPException(400, f'缺少 Z-Image ControlNet 权重：{Z_CONTROLNET}')
    return enqueue({'id':uuid.uuid4().hex,'status':'queued','created':time.time(),'request':body.model_dump()})

def wait_job(job_id, timeout=300):
    end = time.time() + timeout
    while time.time() < end:
        with lock:
            job = jobs.get(job_id)
            if not job:
                raise KeyError('任务不存在')
            if job['status'] in ('succeeded', 'failed'):
                return job.copy()
        time.sleep(1)
    return {'id': job_id, 'status': 'timeout'}

@app.get('/api/jobs/{job_id}')
def status(job_id: str):
    with lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, '任务不存在')
        out = job.copy()
        out['now'] = time.time()   # 服务器时间：前端算“已用时间”不依赖本机时钟
        if job['status'] == 'queued':
            # 前面还有几个任务（排队中的 + 正在跑的），给前端估算等待时间
            out['ahead'] = sum(1 for j in jobs.values() if j['status'] == 'running' or (j['status'] == 'queued' and j['created'] < job['created']))
        return out

def release_host_memory():
    """Keep the container under its memory limit (90 GiB on the AutoDL instance).

    Reading ~31 GiB of safetensors per model leaves the files in the page cache,
    which the container is charged for. With both models cached (~62 GiB) plus the
    offloaded pipeline in RAM (~24-36 GiB), loading the next model pushed the
    container over its limit and the process was killed without a traceback.
    Weights are already copied into process memory, so evicting the file cache is safe.
    """
    import ctypes
    for root in (os.environ.get('MODEL_ROOT', '/root/autodl-tmp/Qwen-Image-2.1'), os.environ.get('ZIMAGE_ROOT', '/root/autodl-tmp/Z-Image-Turbo'), str(Z_CONTROLNET.parent.parent)):
        for path in Path(root).glob('*/*.safetensors'):
            try:
                fd = os.open(path, os.O_RDONLY)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                finally:
                    os.close(fd)
            except OSError:
                pass
    try:
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except OSError:
        pass

class Engine:
    pipe = None
    key = None

    def unload(self):
        import torch
        self.pipe = None
        self.key = None
        gc.collect()
        torch.cuda.empty_cache()
        release_host_memory()

    @staticmethod
    def key_for(req):
        return (req.model, req.precision, tuple(u.id for u in req.loras), req.operation == 'pose' and req.model == 'zimage')

    def load(self, req):
        import torch
        key = self.key_for(req)
        control = key[3]
        if key == self.key:
            return 0.0
        self.unload()
        started = time.perf_counter()
        if control:
            # Z-Image-Turbo ControlNet Union 2.1（8 步版），骨架图作为 control image。
            from diffusers import ZImageControlNetPipeline, ZImageControlNetModel
            controlnet = ZImageControlNetModel.from_single_file(str(Z_CONTROLNET), config=str(Z_CONTROLNET.parent / 'config'), torch_dtype=torch.bfloat16)
            self.pipe = ZImageControlNetPipeline.from_pretrained(os.environ.get('ZIMAGE_ROOT','/root/autodl-tmp/Z-Image-Turbo'), controlnet=controlnet, torch_dtype=torch.bfloat16, local_files_only=True)
        elif req.model == 'qwen':
            from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel, BitsAndBytesConfig
            model = os.environ.get('MODEL_ROOT', '/root/autodl-tmp/Qwen-Image-2.1')
            extra = {}
            if req.precision == 'nf4':
                config = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_quant_type='nf4')
                extra['transformer'] = QwenImage21Transformer2DModel.from_pretrained(model,subfolder='transformer',torch_dtype=torch.bfloat16,quantization_config=config,local_files_only=True)
            self.pipe = QwenImage21Pipeline.from_pretrained(model,torch_dtype=torch.bfloat16,local_files_only=True,**extra)
        else:
            from diffusers import ZImagePipeline
            self.pipe = ZImagePipeline.from_pretrained(os.environ.get('ZIMAGE_ROOT','/root/autodl-tmp/Z-Image-Turbo'),torch_dtype=torch.bfloat16,local_files_only=True)
        # Base-model modules, captured before any adapter wraps them.
        modules = dict(self.pipe.transformer.named_modules())
        for index, use in enumerate(req.loras):
            adapter = f'lora{index}'
            entry = LORAS[use.id]
            path = ROOT/'loras'/entry['file']
            state = self.pipe.lora_state_dict(str(path), local_files_only=True)
            if req.model == 'qwen':
                from lora_compat import split_qwen21_gate_up
                state = split_qwen21_gate_up(state)
            else:
                from lora_compat import split_zimage_fused_qkv
                state = split_zimage_fused_qkv(state)
            # Reject unmatched tensors instead of silently accepting a partial adapter.
            for name, tensor in state.items():
                module, side = name.removeprefix('transformer.').split('.lora_',1)
                target = modules.get(module)
                if target is None or not hasattr(target,'weight'):
                    raise ValueError('LoRA 层与底模不匹配: '+name)
                axis = 1 if side.startswith('A.') else 0
                if tensor.shape[axis] != target.weight.shape[axis]:
                    raise ValueError('LoRA 权重尺寸不匹配: '+name)
            self.pipe.load_lora_weights(state, adapter_name=adapter)
            del state
        if control:
            # ControlNet 每一步都和 DiT 交替调用，放不进 CPU 卸载链，所以 DiT + ControlNet + VAE 常驻显卡（约 19 GiB）；
            # 文本编码器（约 8 GiB）留在内存，只在编码提示词时临时进显卡（encode_on_gpu）。
            # 整条管线都放进显卡时，加 LoRA 的峰值实测 31.5 GiB，太接近 32 GiB 上限。
            for name in ('transformer', 'controlnet', 'vae'):
                getattr(self.pipe, name).to('cuda')
        else:
            self.pipe.enable_model_cpu_offload()
        release_host_memory()
        self.key = key
        return time.perf_counter() - started

    def encode_on_gpu(self, prompt):
        import torch
        encoder = self.pipe.text_encoder
        encoder.to('cuda')
        try:
            with torch.no_grad():
                embeds, _ = self.pipe.encode_prompt(prompt=prompt, device=torch.device('cuda'), do_classifier_free_guidance=False)
        finally:
            encoder.to('cpu')
            torch.cuda.empty_cache()
        return embeds

    def run(self, req, job_id, progress=lambda **_: None):
        refs = [Image.open(asset(name)).convert('RGBA') for name in req.images]
        region = None
        original = None
        mask = None
        detail = {}
        if req.operation in ('inpaint','remove_object'):
            original = refs[0]
            mask = Image.open(asset(req.mask)).convert('L').point(lambda x: 255 if x > 127 else 0)
            box = mask.getbbox()
            pad = max(32, round(max(box[2]-box[0],box[3]-box[1])*.2))
            region = (max(0,box[0]-pad),max(0,box[1]-pad),min(original.width,box[2]+pad),min(original.height,box[3]+pad))
            # Only the masked pixels are ever pasted back, so locality never depends on the model.
            # inpaint: send the crop alone. A second black/white mask reference made Qwen paint large
            # white blocks into big selections (tested 2026-09-26). remove_object still needs the mask
            # to know which object to delete.
            refs = [original.crop(region)] if req.operation == 'inpaint' else [original.crop(region), mask.crop(region).convert('RGBA')]
        if req.operation in ('upscale','collage'):
            start = time.perf_counter()
            if req.operation == 'upscale':
                size = (refs[0].width*req.factor, refs[0].height*req.factor)
                if size[0]*size[1] > 20_000_000:
                    raise ValueError('放大后的像素不能超过 2000 万，请降低倍率')
                image = refs[0].resize(size, Image.Resampling.LANCZOS)
            else:
                height = min(1024, max(im.height for im in refs))
                parts = [im.resize((max(1, round(im.width*height/im.height)),height),Image.Resampling.LANCZOS) for im in refs]
                image = Image.new('RGBA',(sum(im.width for im in parts),height),(255,255,255,255))
                x=0
                for im in parts:
                    image.alpha_composite(im,(x,0)); x += im.width
            metrics = {'seconds':time.perf_counter()-start, 'load_seconds':0, 'backend':'Pillow CPU'}
        elif req.operation == 'pose_extract':
            import pose
            start = time.perf_counter()
            skeleton, info = pose.extract(refs[0])
            if not info['people']:
                raise ValueError('没有在图里检测到人物姿势，请换一张人物更完整、更清晰的图')
            image = skeleton.convert('RGBA')
            metrics = {'seconds':time.perf_counter()-start, 'load_seconds':0, 'backend':'rtmlib RTMW (CPU)'}
            detail = {'pose': info}
        else:
            import sys, torch
            sys.path.insert(0,str(ROOT.parent))
            from bench import Monitor
            skeleton = None
            if req.operation == 'pose':
                import pose
                if pose.looks_like_skeleton(refs[0]):
                    skeleton, info = refs[0].convert('RGB'), {'people': None, 'source': 'skeleton_input', 'size': list(refs[0].size)}
                else:
                    skeleton, info = pose.extract(refs[0])
                    if not info['people']:
                        raise ValueError('没有在参考图里检测到人物姿势，请换一张人物更完整、更清晰的图')
                skeleton_id = uuid.uuid4().hex + '.png'
                skeleton.save(DATA / 'images' / skeleton_id)
                detail['pose'] = dict(info, skeleton_id=skeleton_id, skeleton_image='/images/' + skeleton_id)
            if self.key != self.key_for(req):
                progress(phase='load', message='加载模型权重（首次运行或切换模型、LoRA 时需要，约 1 分钟）')
            load_seconds = self.load(req)
            if req.loras:
                self.pipe.set_adapters([f'lora{i}' for i in range(len(req.loras))], adapter_weights=[u.scale for u in req.loras])
                detail['loras'] = [dict({k: LORAS[u.id].get(k) for k in ('id','title','sha256','source','custom')}, scale=u.scale) for u in req.loras]
            if hasattr(self.pipe, 'vae') and not (req.operation == 'pose' and req.model == 'zimage'):
                self.pipe.vae.enable_tiling() if req.vae_tiling else self.pipe.vae.disable_tiling()
            prompt = req.prompt
            if req.operation == 'pose' and req.model == 'qwen':
                prompt = ('Image 1 is an OpenPose skeleton (colored stick figure on a black background). Generate a new photo: ' + req.prompt +
                          ' The main person must take exactly the same body pose as the skeleton in image 1 (head direction, torso, arms, hands, legs and feet).'
                          + (' The person should look like the person in image 2 (face, hairstyle and clothing).' if len(refs) > 1 else '')
                          + ' Do not draw the skeleton lines.')
                refs = [skeleton] + refs[1:2]
            if req.operation == 'transparent':
                prompt = 'This is an RGBA image with transparency. '+prompt+'. The image has alpha channel and the background is transparent.'
            if req.operation == 'remove_background':
                prompt = 'This is an RGBA image with transparency. Extract the main foreground subject from the reference image. Remove all background, including spaces between parts. Keep the entire subject, its identity, colors, details and composition. No replacement background, no shadow backdrop. The image has alpha channel and the background is transparent. '+req.prompt
            if region and req.operation == 'inpaint':
                prompt = req.prompt+' Keep the framing, camera angle, lighting and every part of the image not mentioned above unchanged.'
            elif region:
                instruction = 'Remove the object inside the white mask completely. Reconstruct the background naturally with matching perspective, texture and lighting. '+req.prompt
                prompt = 'Image 1 is the source crop. Image 2 is a black and white edit mask: WHITE is the editable region and BLACK must be preserved. '+instruction+' Preserve the crop framing and all content outside the white region. Output the edited image only, without drawing the mask.'
            args = dict(prompt=prompt,width=req.width,height=req.height,num_inference_steps=req.steps,generator=torch.Generator('cpu').manual_seed(req.seed))
            if req.operation == 'remove_background':
                args.update(width=max(256,min(2048,round(refs[0].width/16)*16)),height=max(256,min(2048,round(refs[0].height/16)*16)))
            if region:
                rw,rh=refs[0].size
                scale=min(1024/max(rw,rh), 2048/min(rw,rh))
                args.update(width=max(256,round(rw*scale/16)*16),height=max(256,round(rh*scale/16)*16))
            if skeleton is not None:
                # 输出比例跟随姿势参考图：Z 最长边 1024，Qwen 编辑最长边 2048
                side = 1024 if req.model == 'zimage' else 2048
                sw, sh = skeleton.size
                s = side / max(sw, sh)
                args.update(width=max(256, round(sw*s/16)*16), height=max(256, round(sh*s/16)*16))
                if req.model == 'zimage':
                    args.update(control_image=skeleton.resize((args['width'], args['height']), Image.Resampling.LANCZOS),
                                controlnet_conditioning_scale=req.control_scale)
                else:
                    self.pipe.vae.enable_tiling()
            if req.model == 'qwen':
                args.update(true_cfg_scale=1.0,use_kv_cache=req.kv_cache)
                if refs:
                    args['image'] = [im.convert('RGB') if req.operation == 'pose' else im for im in refs]
                    args['output_resolution'] = req.reference_resolution
            else:
                args['guidance_scale'] = 0.0
            total = req.steps
            def on_step(pipe, i, t, kwargs):
                done = i + 1
                progress(phase='decode' if done >= total else 'denoise', step=done, total=total,
                         message='解码图片' + ('（VAE 分块，2048 图约需几十秒）' if req.vae_tiling or refs else '') if done >= total else f'采样 {done}/{total} 步')
                return kwargs
            args['callback_on_step_end'] = on_step
            progress(phase='prepare', step=0, total=total, message='编码提示词和参考图')
            with Monitor('student_'+job_id) as monitor:
                if req.operation == 'pose' and req.model == 'zimage':
                    args['prompt_embeds'] = self.encode_on_gpu(args.pop('prompt'))
                image = self.pipe(**args).images[0]
                progress(phase='save', step=total, total=total, message='保存结果')
                if req.operation == 'remove_background':
                    alpha=image.convert('RGBA').getchannel('A').resize(refs[0].size,Image.Resampling.LANCZOS)
                    low,high=alpha.getextrema()
                    if low>32 or high<128:
                        raise ValueError('模型没有提取出有效透明主体，请更明确描述要保留的主体后重试')
                    image=refs[0].copy()
                    image.putalpha(ImageChops.multiply(refs[0].getchannel('A'),alpha))
                    detail={'foreground_rgb_preserved':True,'method':'qwen_alpha_on_original_rgb'}
                if region:
                    patch=image.resize(refs[0].size,Image.Resampling.LANCZOS)
                    # Generated RGBA may leave holes. Composite onto source context
                    # before applying the edit mask, never copy hidden RGB into it.
                    patch=Image.alpha_composite(original.crop(region),patch.convert('RGBA'))
                    local_mask=mask.crop(region)
                    feather=ImageChops.multiply(local_mask,local_mask.filter(ImageFilter.GaussianBlur(16)))
                    image=original.copy()
                    image.paste(patch,region[:2],feather)
                    outside=ImageChops.invert(mask)
                    difference=ImageChops.difference(image,original)
                    unchanged=all(ImageChops.multiply(channel,outside).getbbox() is None for channel in difference.split())
                    if not unchanged:
                        raise RuntimeError('选区外像素校验失败')
                    detail={'outside_mask_identical':True,'crop_box':region,'patch_size':[args['width'],args['height']],'method':'crop_reference_edit_then_masked_composite' if req.operation=='inpaint' else 'reference_and_mask_generation_then_masked_composite'}
                image.save(DATA / 'images' / (job_id+'.png'))
            metrics = dict(monitor.metrics,load_seconds=load_seconds,backend='Diffusers',gpu=torch.cuda.get_device_name(0))
        image.save(DATA / 'images' / (job_id+'.png'))
        return {'image':'/images/'+job_id+'.png','image_id':job_id+'.png','width':image.width,'height':image.height,'mode':image.mode,'alpha_extrema':image.getchannel('A').getextrema() if image.mode=='RGBA' else None,'metrics':metrics,**detail}

def run_training_job(engine, job):
    import trainer
    engine.unload()   # 训练独占显卡：先卸掉出图模型并清缓存
    spec = job['train']
    items = [(Image.open(asset(i)).convert('RGB'), c) for i, c in zip(spec['images'], spec['captions'])]
    last = [0.0]
    def progress(p):
        with lock:
            job['progress'] = p
            if time.time() - last[0] > 5:
                save_job(job); last[0] = time.time()
    out = trainer.run_training(job['id'], spec['base'], items, spec['steps'], spec['rank'], spec['lr'], spec['max_pixels'], progress)
    release_host_memory()
    lora_id = 'c_' + job['id'][:10]
    dest = ROOT / 'loras' / 'custom' / (lora_id + '.safetensors')
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out['lora_path'], dest)
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    entry = {'id': lora_id, 'model': spec['base'], 'title': spec['name'], 'file': 'custom/' + dest.name, 'sha256': digest,
             'bytes': dest.stat().st_size, 'source': '本机训练 ' + time.strftime('%Y-%m-%d %H:%M'), 'trigger': spec['trigger'],
             'default_scale': 1.0, 'custom': True, 'kind': spec['kind'],
             'train': {'images': len(items), 'steps': out['steps'], 'rank': spec['rank'], 'lr': spec['lr'], 'max_pixels': spec['max_pixels'],
                       'seconds': out['seconds'], 'timings': out['timings'], 'job': job['id'], 'check_score': spec.get('check_score')}}
    with _custom_lock:
        items_json = [x for x in _custom_items() if x['id'] != lora_id] + [entry]
        _save_custom(items_json)
    preview = enqueue({'id': uuid.uuid4().hex, 'status': 'queued', 'created': time.time(), 'request': Request(
        operation='generate', model=spec['base'], prompt=spec['captions'][0], steps=9 if spec['base'] == 'zimage' else 40,
        loras=[LoraUse(id=lora_id, scale=1.0)]).model_dump()})
    return {'lora': {k: v for k, v in entry.items() if k != 'file'}, 'preview_job': preview['id'], **out}

def worker():
    engine = Engine()
    while True:
        key = work.get()
        with lock:
            job = jobs[key]; job.update(status='running',started=time.time()); save_job(job)
        try:
            if job.get('kind') == 'train':
                result = run_training_job(engine, job)
            else:
                def progress(job=job, **p):
                    with lock:
                        job['progress'] = dict(p, at=time.time())
                result = engine.run(Request(**job['request']),key,progress)
            with lock:
                job.update(status='succeeded',result=result)
        except Exception as e:
            traceback.print_exc()
            engine.pipe = None; engine.key = None
            gc.collect()
            with lock:
                job.update(status='failed',error=f'{type(e).__name__}: {e}')
        finally:
            with lock:
                job['finished']=time.time(); save_job(job)
            work.task_done()

# ------------------------------------------------------------------ 在线 LoRA 训练
class TrainCheck(BaseModel):
    model_config = ConfigDict(extra='forbid')
    base: Literal['qwen', 'zimage'] = 'zimage'
    kind: Literal['person', 'style', 'object'] = 'person'
    trigger: str = Field(min_length=2, max_length=40)
    images: list[str] = Field(min_length=1, max_length=60)

class TrainStart(BaseModel):
    model_config = ConfigDict(extra='forbid')
    check_id: str
    name: str = Field(min_length=1, max_length=40)
    captions: list[str] | None = None
    steps: int | None = Field(default=None, ge=50, le=4000)
    rank: int | None = Field(default=None, ge=4, le=128)
    lr: float | None = Field(default=None, gt=0, le=1e-3)
    max_pixels: int | None = Field(default=None, ge=65536, le=1048576)

CHECKS: dict = {}

@app.get('/api/train/defaults')
def train_defaults():
    import trainer
    return {'defaults': trainer.DEFAULTS, 'min_images': 8, 'max_images': 60, 'recommended': '15–30 张'}

@app.post('/api/train/check')
def train_check(body: TrainCheck):
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_\-]{1,39}', body.trigger):
        raise HTTPException(400, '触发词请用英文字母开头，只含字母、数字、下划线或连字符，例如 lin_yue')
    images, hashes, problems = [], {}, []
    for i, name in enumerate(body.images, 1):
        with Image.open(asset(name)) as im:
            rgb = im.convert('RGB')
        if min(rgb.size) < 384:
            problems.append({'image': i, 'issue': f'分辨率太低（{rgb.width}×{rgb.height}），短边至少 384'})
        digest = hashlib.md5(rgb.resize((16, 16)).convert('L').tobytes()).hexdigest()
        if digest in hashes:
            problems.append({'image': i, 'issue': f'和图{hashes[digest]}重复'})
        hashes.setdefault(digest, i)
        images.append(rgb)
    if len(images) < 8:
        problems.insert(0, {'image': 0, 'issue': f'只有 {len(images)} 张图，至少需要 8 张（建议 15–30 张）'})
    if problems:
        verdict = {'ok': False, 'score': 0, 'summary': '基础检查没通过，没有提交给模型审核。', 'problems': problems,
                   'suggestions': ['补足图片数量、去掉重复或过小的图后再检查'], 'captions': []}
    else:
        try:
            verdict = agent.check_dataset(images, body.base, body.trigger, body.kind)
        except RuntimeError as e:
            raise HTTPException(502, str(e))
    check_id = uuid.uuid4().hex
    CHECKS[check_id] = {'request': body.model_dump(), 'verdict': verdict, 'created': time.time()}
    return {'check_id': check_id, **verdict}

@app.post('/api/train/start', status_code=202)
def train_start(body: TrainStart):
    import trainer
    check = CHECKS.get(body.check_id)
    if not check:
        raise HTTPException(404, '请先检查训练数据')
    if not check['verdict'].get('ok'):
        raise HTTPException(400, '数据检查没有通过，请按提示调整后重新检查')
    req = check['request']
    captions = body.captions or check['verdict']['captions']
    if len(captions) != len(req['images']):
        raise HTTPException(400, 'caption 数量和图片数量不一致')
    base = req['base']
    d = trainer.DEFAULTS[base]
    spec = {'base': base, 'kind': req['kind'], 'trigger': req['trigger'], 'name': body.name.strip(), 'images': req['images'],
            'captions': [c if c.lower().startswith(req['trigger'].lower()) else f"{req['trigger']}, {c}" for c in captions],
            'steps': body.steps or d['steps'], 'rank': body.rank or d['rank'], 'lr': body.lr or d['lr'],
            'max_pixels': body.max_pixels or d['max_pixels'], 'check_score': check['verdict'].get('score')}
    return enqueue({'id': uuid.uuid4().hex, 'status': 'queued', 'created': time.time(), 'kind': 'train', 'train': spec,
                    'request': {'operation': 'train', 'model': base}})

# ------------------------------------------------------------------ 智能助手与 Skills
class AgentConfigBody(BaseModel):
    model_config = ConfigDict(extra='forbid')
    base_url: str = Field(max_length=300)
    model: str = Field(max_length=120)
    api_key: str | None = Field(default=None, max_length=400)
    proxy: str = Field(default='', max_length=200)

class ChatBody(BaseModel):
    model_config = ConfigDict(extra='forbid')
    conversation: str | None = None
    text: str = Field(min_length=1, max_length=8000)
    images: list[str] = Field(default_factory=list, max_length=6)
    skills: list[str] = Field(default_factory=list, max_length=5)

class SkillInstall(BaseModel):
    model_config = ConfigDict(extra='forbid')
    source: str = Field(min_length=3, max_length=400)

class SkillUpload(BaseModel):
    model_config = ConfigDict(extra='forbid')
    filename: str = Field(max_length=200)
    data: str = Field(max_length=12_000_000)

class SkillToggle(BaseModel):
    model_config = ConfigDict(extra='forbid')
    enabled: bool

@app.get('/api/agent/config')
def agent_config():
    return agent.public_config()

@app.post('/api/agent/config')
def agent_config_save(body: AgentConfigBody):
    cfg = agent.load_config()
    cfg.update(base_url=body.base_url.strip().rstrip('/'), model=body.model.strip(), proxy=body.proxy.strip())
    if body.api_key and '…' not in body.api_key:
        cfg['api_key'] = body.api_key.strip()
    cfg['probe'] = agent.probe(cfg)   # 保存后立即探测多模态能力
    agent.save_config(cfg)
    return agent.public_config(cfg)

@app.post('/api/agent/probe')
def agent_probe():
    cfg = agent.load_config()
    cfg['probe'] = agent.probe(cfg)
    agent.save_config(cfg)
    return agent.public_config(cfg)

@app.post('/api/agent/chat', status_code=202)
def agent_chat(body: ChatBody):
    cfg = agent.load_config()
    if not cfg.get('api_key'):
        raise HTTPException(400, '请先在模型设置里填写 API Key')
    if body.images and not (cfg.get('probe') or {}).get('multimodal'):
        raise HTTPException(400, '当前模型没有通过多模态检测，不能发送图片；请换多模态模型后重新检测')
    for name in body.images:
        asset(name)
    return agent.start_chat(body.conversation, body.text, body.images, body.skills)

@app.get('/api/agent/tasks/{task_id}')
def agent_task(task_id: str, since: int = 0):
    try:
        return agent.task_events(task_id, since)
    except KeyError:
        raise HTTPException(404, '对话任务不存在')

@app.get('/api/skills')
def skills():
    return agent.list_skills()

@app.post('/api/skills/install')
def skills_install(body: SkillInstall):
    try:
        return agent.install_from_github(body.source)
    except (ValueError, RuntimeError, OSError) as e:
        raise HTTPException(400, str(e))

@app.post('/api/skills/upload')
def skills_upload(body: SkillUpload):
    try:
        return agent.install_from_upload(body.filename, base64.b64decode(body.data.split(',')[-1]))
    except (ValueError, RuntimeError, OSError) as e:
        raise HTTPException(400, str(e))

@app.post('/api/skills/{name}/toggle')
def skills_toggle(name: str, body: SkillToggle):
    agent.set_skill_enabled(name, body.enabled)
    return agent.list_skills()

@app.delete('/api/skills/{name}')
def skills_delete(name: str):
    agent.delete_skill(name)
    return agent.list_skills()

@app.get('/api/skills/{name}/file')
def skills_file(name: str, path: str = 'SKILL.md'):
    try:
        return {'name': name, 'path': path, 'content': agent.read_skill_file(name, path)}
    except ValueError as e:
        raise HTTPException(404, str(e))

def _bridge_submit(req):
    body = Request(**req)
    for name in body.images:
        asset(name)
    return enqueue({'id': uuid.uuid4().hex, 'status': 'queued', 'created': time.time(), 'request': body.model_dump(), 'source': 'agent'})['id']

agent.bridge.submit = _bridge_submit
agent.bridge.wait = wait_job
agent.bridge.image = lambda name: Image.open(asset(name)).convert('RGB')
agent.bridge.loras = lambda: [{'id': x['id'], 'model': x['model'], 'title': x['title'], 'trigger': x.get('trigger', ''),
                               'default_scale': x.get('default_scale', 0.7), 'custom': bool(x.get('custom'))} for x in LORAS.values()]

for path in (DATA / 'jobs').glob('*.json'):
    old = json.loads(path.read_text(encoding='utf-8'))
    if old['status'] in ('queued','running'):
        old.update(status='failed',error='服务重启中断了任务，请重新提交'); save_job(old)
    jobs[old['id']] = old
threading.Thread(target=worker,daemon=True).start()
