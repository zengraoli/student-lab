"""Reproducible Qwen Image 2.1 benchmarks; all metrics are measured, not estimated."""
import argparse, json, os, time, threading, traceback, platform
from pathlib import Path
os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS','4')
import psutil, torch, pynvml
from PIL import Image

GIB = 1024**3
LAB = Path(os.environ.get('LAB', '/root/autodl-tmp/Qwen-Image-2.1/lab'))
MODEL = os.environ.get('MODEL_ROOT', '/root/autodl-tmp/Qwen-Image-2.1')

class Monitor:
    def __init__(self, name):
        self.name=name; self.samples=[]; self.stop=threading.Event()
        pynvml.nvmlInit(); self.gpu=pynvml.nvmlDeviceGetHandleByIndex(0)
    def __enter__(self):
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        self.start=time.perf_counter(); self.thread=threading.Thread(target=self.poll,daemon=True); self.thread.start()
        return self
    def poll(self):
        proc=psutil.Process()
        while not self.stop.is_set():
            processes=[proc]+proc.children(recursive=True)
            rss=sum(p.memory_info().rss for p in processes if p.is_running())
            gpu=pynvml.nvmlDeviceGetMemoryInfo(self.gpu).used
            system=psutil.virtual_memory().used
            try: cgroup=int(Path('/sys/fs/cgroup/memory.current').read_text())
            except Exception: cgroup=0
            self.samples.append([time.perf_counter()-self.start,gpu,rss,system,cgroup,psutil.swap_memory().used])
            self.stop.wait(.05)
    def __exit__(self,*exc):
        torch.cuda.synchronize(); self.elapsed=time.perf_counter()-self.start
        self.stop.set(); self.thread.join()
        self.metrics={'seconds':round(self.elapsed,4),'gpu_total_peak_gib':max((r[1] for r in self.samples),default=0)/GIB,
          'process_tree_rss_peak_gib':max((r[2] for r in self.samples),default=0)/GIB,
          'host_used_peak_gib':max((r[3] for r in self.samples),default=0)/GIB,
          'container_memory_peak_sampled_gib':max((r[4] for r in self.samples),default=0)/GIB,
          'swap_peak_gib':max((r[5] for r in self.samples),default=0)/GIB,
          'torch_allocated_peak_gib':torch.cuda.max_memory_allocated()/GIB,
          'torch_reserved_peak_gib':torch.cuda.max_memory_reserved()/GIB,'sampling_interval_ms':50}
        import csv
        with (LAB/'results'/f'{self.name}.samples.csv').open('w') as f:
            w=csv.writer(f);w.writerow(['elapsed_s','gpu_used_bytes','process_tree_rss_bytes','host_used_bytes','container_bytes','swap_bytes']);w.writerows(self.samples)

def record(data):
    print(json.dumps(data,ensure_ascii=False),flush=True)
    with (LAB/'results'/'runs.jsonl').open('a') as f: f.write(json.dumps(data,ensure_ascii=False)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',default='offload',choices=['full','offload','sequential','bnb4','bnb8','group']);p.add_argument('--suite',required=True);p.add_argument('--tag',default='baseline');p.add_argument('--compile',action='store_true');p.add_argument('--vae-tiling',action='store_true');p.add_argument('--lora');p.add_argument('--lora-scale',type=float,default=1)
    a=p.parse_args()
    from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel, BitsAndBytesConfig
    extra={}
    with Monitor(a.tag+'_load') as load:
        if a.mode in ('bnb4','bnb8'):
            quant=BitsAndBytesConfig(load_in_4bit=a.mode=='bnb4',load_in_8bit=a.mode=='bnb8',bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_quant_type='nf4')
            extra['transformer']=QwenImage21Transformer2DModel.from_pretrained(MODEL,subfolder='transformer',torch_dtype=torch.bfloat16,quantization_config=quant,local_files_only=True)
        pipe=QwenImage21Pipeline.from_pretrained(MODEL,torch_dtype=torch.bfloat16,local_files_only=True,**extra)
        if a.lora: pipe.load_lora_weights(a.lora)
        if a.mode=='full': pipe.to('cuda')
        elif a.mode=='sequential': pipe.enable_sequential_cpu_offload()
        elif a.mode=='group':
            from diffusers.hooks import apply_group_offloading
            for component in [pipe.text_encoder,pipe.transformer,pipe.vae]:
                apply_group_offloading(component,onload_device=torch.device('cuda'),offload_device=torch.device('cpu'),offload_type='leaf_level',use_stream=False)
        else: pipe.enable_model_cpu_offload()
        if a.vae_tiling: pipe.vae.enable_tiling()
        if a.compile:
            from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21FlexAttnProcessor
            pipe.transformer.set_attn_processor(QwenImage21FlexAttnProcessor());pipe.transformer.compile()
    record({'event':'load','tag':a.tag,'mode':a.mode,**load.metrics,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0)})
    tasks=json.loads(Path(a.suite).read_text())
    for index,task in enumerate(tasks):
        name=a.tag+'_'+task['id']; output=LAB/'outputs'/f'{name}.png'
        kwargs=dict(prompt=task['prompt'],height=task.get('height',1024),width=task.get('width',1024),num_inference_steps=task.get('steps',40),true_cfg_scale=1.,use_kv_cache=True,generator=torch.Generator('cpu').manual_seed(task.get('seed',42)))
        if task.get('images'): kwargs['image']=[Image.open(LAB/x).convert('RGBA') for x in task['images']]
        if a.lora: kwargs['attention_kwargs']={'scale':a.lora_scale}
        try:
            with Monitor(name) as monitor:
                img=pipe(**kwargs).images[0];img.save(output)
            extrema=img.getextrema(); record({'event':'generation','tag':a.tag,'mode':a.mode,'index':index,'task':task,'output':str(output),'image_mode':img.mode,'extrema':extrema,'compile':a.compile,'vae_tiling':a.vae_tiling,'lora':a.lora,'lora_scale':a.lora_scale,**monitor.metrics})
        except Exception as e:
            record({'event':'error','tag':a.tag,'task':task,'error':str(e),'traceback':traceback.format_exc()}); raise
    print('BENCH_COMPLETE',flush=True)

if __name__=='__main__': main()
