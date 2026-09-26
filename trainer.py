"""在线 LoRA 训练（Qwen-Image-2.1 / Z-Image-Turbo），底层是 DiffSynth-Studio。

流程：训练图 + caption 写成 DiffSynth 数据集 → 第 1 步缓存文本和图像编码（只加载文本编码器和 VAE）
→ 第 2 步只加载 DiT 训练 LoRA。两步都在子进程里跑，结束后显存和内存全部归还给实验台。

Z-Image-Turbo 是蒸馏模型，直接训练会破坏 8 步加速能力，所以训练时融合 ostris 的
zimage_turbo_training_adapter（DiffSynth 的 differential training 做法），推理时不加载它。
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parent
DIFFSYNTH = LAB / 'src' / 'DiffSynth-Studio-main'
QWEN_ROOT = Path(os.environ.get('MODEL_ROOT', '/root/autodl-tmp/Qwen-Image-2.1'))
Z_ROOT = Path(os.environ.get('ZIMAGE_ROOT', '/root/autodl-tmp/Z-Image-Turbo'))
Z_ADAPTER = Path(os.environ.get('ZIMAGE_TRAINING_ADAPTER', '/root/autodl-tmp/addons/zimage_turbo_training_adapter/zimage_turbo_training_adapter_v1.safetensors'))
RUNS = ROOT / 'data' / 'training'

DEFAULTS = {
    'qwen': {'steps': 600, 'rank': 16, 'lr': 1e-4, 'max_pixels': 327680},
    'zimage': {'steps': 600, 'rank': 32, 'lr': 1e-4, 'max_pixels': 589824},
}

RUNNER = r'''
import json, runpy, sys
cfg = json.loads(sys.argv[1])
if cfg.get("patch_qwen_flex"):
    # Torch 2.8 的 FlexAttention 反向传播在 RTX 50 系（SM120）上超出共享内存，改用 SDPA 两遍实现
    import diffsynth.models.qwen_image_21_dit as qwen_dit
    qwen_dit.FLEX_ATTN_AVAILABLE = False
sys.argv = [cfg["trainfile"]] + cfg["argv"]
runpy.run_path(cfg["trainfile"], run_name="__main__")
'''


def shards(folder: Path) -> list[str]:
    return sorted(str(p) for p in folder.glob('*.safetensors'))


def prepare(run: Path, items: list[tuple[Image.Image, str]]) -> Path:
    data = run / 'data'
    data.mkdir(parents=True, exist_ok=True)
    meta = []
    for i, (im, caption) in enumerate(items, 1):
        name = f'{i:03d}.png'
        rgb = im.convert('RGB')
        rgb.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        rgb.save(data / name)
        meta.append({'image': name, 'prompt': caption})
    (data / 'metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
    return data


def phases(base: str, run: Path, n_images: int, steps: int, rank: int, lr: float, max_pixels: int) -> list[tuple[str, dict]]:
    data, cache, out = run / 'data', run / 'cache', run / 'output'
    repeat = max(1, math.ceil(steps / n_images))
    total = repeat * n_images
    # --use_gradient_checkpointing 两个阶段都要传：缓存里会记下这个开关，训练阶段直接沿用缓存的值。
    # 只在训练阶段传的话实际没有开启，Z-Image 在约 850×670 时显存超过 32 GiB。
    common = ['--lora_base_model', 'dit', '--lora_rank', str(rank), '--dataset_num_workers', '0', '--max_pixels', str(max_pixels),
              '--use_gradient_checkpointing']
    if base == 'qwen':
        trainfile = DIFFSYNTH / 'examples' / 'qwen_image_21' / 'model_training' / 'train.py'
        common += ['--processor_path', str(QWEN_ROOT / 'processor'), '--lora_target_modules', '']
        cache_models = [shards(QWEN_ROOT / 'text_encoder'), str(QWEN_ROOT / 'vae' / 'diffusion_pytorch_model.safetensors')]
        train_models = [shards(QWEN_ROOT / 'transformer')]
        extra_train = []
    else:
        trainfile = DIFFSYNTH / 'examples' / 'z_image' / 'model_training' / 'train.py'
        common += ['--tokenizer_path', str(Z_ROOT / 'tokenizer'), '--lora_target_modules', 'to_q,to_k,to_v,to_out.0,w1,w2,w3']
        cache_models = [shards(Z_ROOT / 'text_encoder'), str(Z_ROOT / 'vae' / 'diffusion_pytorch_model.safetensors')]
        train_models = [shards(Z_ROOT / 'transformer')]
        if not Z_ADAPTER.is_file():
            raise FileNotFoundError(f'缺少 Z-Image 训练适配器：{Z_ADAPTER}')
        extra_train = ['--preset_lora_path', str(Z_ADAPTER), '--preset_lora_model', 'dit']
    cache_argv = ['--task', 'sft:data_process', '--model_paths', json.dumps(cache_models), '--dataset_base_path', str(data),
                  '--dataset_metadata_path', str(data / 'metadata.json'), '--dataset_repeat', '1', '--output_path', str(cache)] + common
    train_argv = ['--task', 'sft:train', '--model_paths', json.dumps(train_models), '--dataset_base_path', str(cache),
                  '--dataset_repeat', str(repeat), '--num_epochs', '1', '--learning_rate', str(lr), '--output_path', str(out),
                  '--remove_prefix_in_ckpt', 'pipe.dit.', '--save_steps', str(total)] + extra_train + common
    cfg = {'trainfile': str(trainfile), 'patch_qwen_flex': base == 'qwen'}
    return [('cache', {**cfg, 'argv': cache_argv, 'total': n_images}), ('train', {**cfg, 'argv': train_argv, 'total': total})]


_progress = re.compile(r'(\d+)/(\d+) \[')


def run_training(job_id: str, base: str, items: list[tuple[Image.Image, str]], steps: int, rank: int, lr: float,
                 max_pixels: int, on_progress) -> dict:
    run = RUNS / job_id
    run.mkdir(parents=True, exist_ok=True)
    prepare(run, items)
    plan = phases(base, run, len(items), steps, rank, lr, max_pixels)
    env = dict(os.environ, PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    env['PYTHONPATH'] = str(DIFFSYNTH) + os.pathsep + env.get('PYTHONPATH', '')
    log_path = run / 'train.log'
    started = time.time()
    timings = {}
    with open(log_path, 'w', encoding='utf-8') as log:
        for phase, cfg in plan:
            t = time.time()
            on_progress({'phase': phase, 'step': 0, 'total': cfg['total'], 'message': '缓存编码' if phase == 'cache' else '训练 LoRA'})
            proc = subprocess.Popen([sys.executable, '-c', RUNNER, json.dumps(cfg)], cwd=str(DIFFSYNTH), env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            last = 0.0
            buffer = ''
            while True:
                chunk = proc.stdout.read(256)
                if not chunk:
                    break
                log.write(chunk)
                buffer = (buffer + chunk)[-2000:]
                matches = _progress.findall(buffer)
                if matches and time.time() - last > 2:
                    step, total = map(int, matches[-1])
                    on_progress({'phase': phase, 'step': step, 'total': total,
                                 'message': ('缓存编码' if phase == 'cache' else '训练 LoRA') + f' {step}/{total}'})
                    last = time.time()
            code = proc.wait()
            log.flush()
            timings[phase] = round(time.time() - t, 1)
            if code != 0:
                tail = log_path.read_text(encoding='utf-8', errors='replace')[-1500:]
                raise RuntimeError(f'{phase} 阶段失败（退出码 {code}）：{tail[-600:]}')
    outputs = sorted((run / 'output').glob('*.safetensors'), key=lambda p: p.stat().st_mtime)
    if not outputs:
        raise RuntimeError('训练结束但没有找到 LoRA 文件')
    return {'lora_path': str(outputs[-1]), 'seconds': round(time.time() - started, 1), 'timings': timings,
            'steps': plan[1][1]['total'], 'log': str(log_path)}
