"""Python HTTP examples, no ComfyUI client or workflow required."""
import argparse, base64, json, time
from pathlib import Path
from urllib.request import Request, urlopen

class Client:
    def __init__(self, url='http://127.0.0.1:6008'):
        self.url = url.rstrip('/')
    def call(self, path, data=None):
        body = None if data is None else json.dumps(data).encode()
        with urlopen(Request(self.url+path, data=body,headers={'Content-Type':'application/json'}), timeout=60) as r:
            return json.load(r)
    def upload(self, path):
        return self.call('/api/images', {'data':base64.b64encode(Path(path).read_bytes()).decode()})['id']
    def generate(self, timeout=1800, **kwargs):
        job = self.call('/api/jobs',kwargs)
        start=time.monotonic()
        while job['status'] in ('queued','running'):
            if time.monotonic()-start > timeout:
                raise TimeoutError(f'Task still on server: {job["id"]}; query /api/jobs/{job["id"]}')
            time.sleep(1)
            job=self.call('/api/jobs/'+job['id'])
        if job['status']!='succeeded':
            raise RuntimeError(job['error'])
        return job
    def download(self, job, path):
        with urlopen(self.url+job['result']['image'],timeout=60) as r:
            Path(path).write_bytes(r.read())

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--url',default='http://127.0.0.1:6008')
    p.add_argument('--operation',default='generate',choices=['generate','edit','merge','transparent','upscale','collage','inpaint','remove_object','remove_background'])
    p.add_argument('--model',default='qwen',choices=['qwen','zimage'])
    p.add_argument('--prompt',default='A watercolor mountain lake at dawn.')
    p.add_argument('--images',nargs='*',default=[])
    p.add_argument('--lora')
    p.add_argument('--lora-scale',type=float,default=.7)
    p.add_argument('--mask',help='Black/white PNG, white edits; same dimensions as source')
    p.add_argument('--output',default='result.png')
    p.add_argument('--steps',type=int)
    args=p.parse_args(); client=Client(args.url)
    job=client.generate(lora=args.lora,lora_scale=args.lora_scale,mask=client.upload(args.mask) if args.mask else None,operation=args.operation,model=args.model,prompt=args.prompt,images=[client.upload(f) for f in args.images],steps=args.steps or (9 if args.model=='zimage' else 40))
    client.download(job,args.output)
    print(json.dumps(job,ensure_ascii=False,indent=2))
