# Qwen 2.1 × Z-Image-Turbo 图像实验台

**v0.1（MVP）**：基于 Qwen-Image-2.1 和 Z-Image-Turbo 的无限画布图像实验台，支持文生图、参考编辑与多图融合、选区重绘与移除物件、抠图与透明素材、姿势迁移、多 LoRA 叠加、多模态助手和在线 LoRA 训练。

相关飞书开放文档：[Qwen-Image-2.1 与 Z-Image-Turbo：从零部署到实用入门](https://my.feishu.cn/docx/OgDdd6eWzoMACAxZPmhctW5wnJc?from=from_copylink)（部署步骤、每个功能的操作截图和实测结果）。

面向了解 Python / HTTP 的学生。一个 FastAPI 服务直接调用 Diffusers，网页与 Python 客户端调用同一组接口；无需 ComfyUI 进程、节点或工作流。

## 当前已部署环境

服务器：RTX 5090 D 32GB；Python 3.12；Torch 2.8.0+cu128；Transformers 5.17.0。
两份官方模型在 `/root/autodl-tmp/Qwen-Image-2.1` 与 `/root/autodl-tmp/Z-Image-Turbo`。
代码在 `/root/autodl-tmp/Qwen-Image-2.1/lab/student-lab`。ComfyUI 仍独立使用 6006。

```bash
# 在服务器执行；如果 6008 已有本服务，无需重复启动。
bash /root/autodl-tmp/Qwen-Image-2.1/lab/student-lab/start.sh
```

本机终端建立转发，终端保持开启。把 `<端口>`、`<主机>` 换成自己服务器的 SSH 端口和地址：

```bash
ssh -p <端口> -N -L 16008:127.0.0.1:6008 root@<主机>
```

浏览器访问 `http://127.0.0.1:16008`，接口文档在 `/docs`。服务仅监听服务器回环地址；示例没有用户账户管理，不应直接暴露为公共网站。

## Python 最短使用方法

客户端 `client.py` 只使用 Python 标准库，运行它的电脑不需要 GPU、Torch 或 ComfyUI。

```python
from client import Client
c = Client('http://127.0.0.1:16008')
# 文生图：Qwen 40 步；Z-Image-Turbo 9 步、服务端 guidance_scale=0。
job = c.generate(model='qwen', prompt='一座临湖的中国园林，清晨薄雾，摄影风格。', steps=40)
c.download(job, 'garden.png')

# 图生图 / 图像编辑：图作为条件，不是传统 strength 控制的去噪初始化。
ref = c.upload('garden.png')
edited = c.generate(operation='edit', images=[ref], prompt='保留亭子的结构和画面构图，将季节改成冬天，湖岸覆雪。')
c.download(edited, 'winter.png')

# 多图语义融合：让人物和场景成为同一幅画，输出会被重新生成。
person = c.upload('person.png')
merged = c.generate(operation='merge', images=[person, ref], prompt='图1的成年人物站在图2的湖边，保持人物面部特征和亭子的整体形状。')
c.download(merged, 'merged.png')

# 原生透明素材：下载 PNG 保留 alpha。
sticker = c.generate(operation='transparent', prompt='A cute blue whale-shaped teapot, isolated product sticker.')
c.download(sticker, 'sticker.png')

# 两种“变大”：普通插值不新增真实细节；高清改绘可能改变内容。
large = c.generate(operation='upscale', images=[ref], factor=2)
c.download(large, 'garden_2x.png')
redraw = c.generate(operation='edit', images=[ref], prompt='保留场景构图和建筑结构，重新绘制为细节清晰的高分辨率照片。', width=2048, height=2048, vae_tiling=True)
c.download(redraw, 'garden_redraw_2k.png')

# 拼图只是排版，不是融合。
sheet = c.generate(operation='collage', images=[person, ref])
c.download(sheet, 'sheet.png')
```

## 接口与任务状态

- `POST /api/images`：JSON `{"data":"base64 图片或 data URL"}` → 图片 ID。最大 16 MB / 1600 万像素，上传后最长边缩到 2048，保留 RGBA，移除 EXIF。
- `POST /api/jobs`：JSON 请求 → HTTP 202 与任务 ID。
- `GET /api/jobs/{id}`：`queued → running → succeeded / failed`，成功时提供 PNG URL 与时间/显存指标。运行中带 `progress`（`phase`：load / prepare / denoise / decode / save，`step`、`total`、`message`、`at`），排队时带 `ahead`（前面还有几个任务），`now` 是服务器时间，用来算已用时间。
- `GET /api/health`：模型列表与队列长度。
- `/images/{id}.png`：下载输出。

重启服务：`bash student-lab/restart.sh`（停掉占用 6008 的进程，后台重新启动，日志写到 `lab/logs/`）。

只启用一个 worker，GPU 串行处理；切换模型会卸载上一模型。重启时把中断的任务标记失败，避免伪装成仍在执行。模型加载时间单列，不混入生成时间；GPU 峰值是 50ms 采样的整卡占用，非理论模型尺寸。

## 局部修复

```bash
python repair_region.py original.png --box 288 176 704 496 --output repaired.png --prompt "Correct the two hands into a natural right-handed handshake. Preserve sleeves, lighting and the surrounding scene."
```

脚本裁出待修区域，调用 Qwen 编辑，再羽化贴回原图，验证框外像素完全相同。这保证了修改范围，不能保证手指一次就正确；仍需逐指检查手腕连接与比例。坐标必须针对自己的图片重新选择。

## 复测

`python verify_api.py` 验证错误输入处理；`python evaluate.py` 执行固定的双模型人物对照和 5090D 加速对照。脚本记录任务 ID，重跑时不会重复提交尚在执行的任务。失败任务保留，不自动刷种子挑图。

附件已带有完成的 `evaluation` 记录，默认命令会跳过它们。要重新实测，请指定新的输出目录，避免误把读取旧记录当成再次生成：

```bash
EVALUATION_DIR="$PWD/evaluation_retest_$(date +%Y%m%d_%H%M%S)" python evaluate.py
```

默认 `BF16 + model CPU offload + KV cache`。NF4 仅量化 DiT；文本编码器仍占显存。24 步是速度/质量取舍，不是无损加速。分块解码适用于显存压力大时，须检查接缝或异常色块。

## 已观察到的编辑边界

本应用为了快速比较默认输出 1024²，Qwen 官方 README 的默认尺寸为 2048²。两者不要混为同一实验条件。1024² 肖像换背景和动漫改绘出现明显过度锐化；改参考编码尺寸为 992/1056 未修复，DiffSynth 对照也复现。单独 VAE 编码/解码没有出现同等失真，尚未确定根因。场景改成傍晚与局部握手修复则有可用改善，不能由一个失败推断全部编辑无效。

2048² 不分块解码在当前 5090D 上曾出现 CUDA OOM；尝试 2K 时开启 `vae_tiling=True`。它改变的是解码内存策略，不保证修复所有图像问题。初始失败、后续诊断和实际请求均保留在 `evaluation`，没有把失败样本替换成成功样本。

## 模型与源码来源

- Qwen 官方：https://github.com/QwenLM/Qwen-Image-2.1
- Z-Image-Turbo：https://huggingface.co/Tongyi-MAI/Z-Image-Turbo
- 当前 Diffusers 源码：80c7ed262aeffbeb43ef13ae04baeb9b84515a69。
- 通用图像编辑概念：https://huggingface.co/docs/diffusers/using-diffusers/img2img 和 https://huggingface.co/docs/diffusers/using-diffusers/inpaint 。这些通用 Pipeline 的参数不能照搬给 QwenImage21Pipeline。

模型下载走 ModelScope（新机器才需要，当前服务器不要重复下载）：

```bash
cd /root/autodl-tmp
modelscope download --model Qwen/Qwen-Image-2.1 --local_dir ./Qwen-Image-2.1
modelscope download --model Tongyi-MAI/Z-Image-Turbo --local_dir ./Z-Image-Turbo
```

新环境重建与本机已测环境是不同的验证范围；本包不包含模型权重。依赖版本见 `environment.json`，服务端还需要同目录上一级的 `bench.py`（随复现包提供）用于采样监测。

## 换一台 Linux GPU 机器

以下是依据已测版本整理的安装步骤，未另外租空白机器做安装验收；当前服务器已经具备环境，不要重复安装。假设两份模型仍下载到上面的目录，将压缩包里的 `lab` 文件夹放到 `/root/autodl-tmp/Qwen-Image-2.1/lab`。所有依赖与缓存也放在数据盘。

```bash
export LAB=/root/autodl-tmp/Qwen-Image-2.1/lab
export MODEL_ROOT=/root/autodl-tmp/Qwen-Image-2.1
export ZIMAGE_ROOT=/root/autodl-tmp/Z-Image-Turbo
mkdir -p "$LAB"/{results,cache,tmp}
export PIP_CACHE_DIR="$LAB/cache/pip"
export HF_HOME="$LAB/cache/huggingface"
export MODELSCOPE_CACHE="$LAB/cache/modelscope"
export TMPDIR="$LAB/tmp"
python3.12 -m venv "$LAB/venv"
source "$LAB/venv/bin/activate"
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r "$LAB/student-lab/requirements-server.txt"
cd "$LAB/student-lab"
python -m uvicorn app:app --host 127.0.0.1 --port 6008 --workers 1
```

这里的 CUDA wheel 仍需主机驱动支持。LoRA 教程与 ComfyUI 工作流属于原实验环境的附加材料；新环境的最小 API 不依赖它们。人物 LoRA 的权重、训练图片和完整 DiffSynth 环境见旧报告附件，未在本包重复打包。

## ComfyUI 附加示例

当前服务器 6006 已安装所需节点。`lab/workflows/*.workflow.json` 可导入图形界面；对应 `.api.json` 用于 ComfyUI 的接口。工作流使用 `Qwen21OfficialLoader` 读取现有官方权重，不需再下载另一份模型。

若迁移 ComfyUI 环境，需要兼容 Qwen-Image-2.1 的版本（原实验源码基线 `0f74f7fb9f83a78bf46188fd4fd53e6bc44c1ae8`），把随包 `lab/comfy_original_loader.py` 放到其 `custom_nodes`，并把 `lab/comfy_inputs` 中两张原图放到其 `input`。在加载节点中填写自己的模型目录后重启 ComfyUI。这个工作流迁移步骤与上面的独立 API 分开；新 API 不需要安装任何这些节点。

## 无限画布与选区 API（本次新增）

导入图片后可拖动节点；滚轮缩放、空格拖动画布，Shift 点选最多四张参考图。点击“参考生成”会使用所选图，单图走 edit，多图走 merge。输出成为新节点，原图保留。画布位置保存在当前浏览器；选区不跨刷新保存。

选中一张图，点击“选区重绘”或“移除物件”，使用矩形/画笔覆盖需要修改的区域，再写修改要求。移除物件要覆盖物件及阴影。区域编辑为裁切上下文、Qwen 参考编辑、原图合成与选区内羽化；不是专用 inpainting 模型。程序逐通道验证选区外像素不变，但选区内可能改错、残留物件或有接缝。

移除背景使用 Qwen 估计 alpha，然后应用到原图 RGB，保留主体颜色。复杂发丝、玻璃、半透明材质和轮廓错位仍需人工检查。输出下载为 PNG。

```python
from client import Client
c = Client('http://127.0.0.1:16008')
source = c.upload('original.png')
mask = c.upload('mask.png')  # 同尺寸，白色修改、黑色保留
job = c.generate(operation='inpaint', images=[source], mask=mask,
    prompt='Change the blue ceramic whale teapot into a bright orange ceramic whale teapot. Preserve its shape and camera view.', steps=40, seed=42)
c.download(job, 'edited.png')
job = c.generate(operation='remove_object', images=[source], mask=mask,
    prompt='Remove the selected object and reconstruct the surrounding background.', steps=40)
c.download(job, 'removed.png')
job = c.generate(operation='remove_background', images=[source], steps=40)
c.download(job, 'transparent.png')
```

区域输出保持输入尺寸，内部裁切生成最长边 1024；去背景使用输入尺寸（模型要求尺寸对齐到 16，再将 alpha 还原），大图可开启 VAE 分块。参考图最长边 2048。超过该尺寸的放大结果请重新上传再编辑。

`presets.json` 与网页“从文档示例开始”包含第二章四组原始 prompt。`example_requests.json` 收集原始示例的 prompt、参数和输出名；其中 images 文件名需先上传替换为图片 ID。测试记录的原始 JSON 仍保留真实任务 ID。

新功能复测使用新目录，避免读取旧结果：
```bash
CANVAS_EVAL_DIR=canvas_evaluation_retest python test_canvas_features.py
```

第一轮结果保存在 canvas_evaluation，修复 alpha 合成后的复测保存在 canvas_evaluation_v2；不可把单次示例耗时当成重复性能中位数。

## 社区 LoRA（六款已下载文件）

网页选择底模后，在“LoRA 风格 / 人物”选择对应款式，设置强度，生成。当前只开放 BF16 文生图、单款 LoRA；量化与参考编辑组合未验收，接口会拒绝。角色设定款用“加入触发词”添加 `CharacterDesignIZT`。选择原始模型可回到无 LoRA；载入文档示例也会清除 LoRA，保证原始评测条件。

文件放在 `student-lab/loras/zimage-lora` 和 `student-lab/loras/qwen-image-lora`。`loras.json` 固定文件名、来源版本、SHA256 与底模；复现包不重复分发这些权重，按清单下载相同版本后放到对应位置。

```python
job = c.generate(model='zimage', lora='z_asian', lora_scale=0.7,
    prompt='Editorial photograph of one 25-year-old Chinese woman wearing a blue linen shirt, soft window daylight, neutral gray background.', steps=9, seed=42)
c.download(job, 'z_asian.png')
# 换成 Qwen：model='qwen', lora='q_asian', steps=40
```

可选 ID：Z 为 `z_asian`、`z_character`、`z_cinematic`；Qwen 为 `q_asian`、`q_nicegirls`、`q_lenovo`。强度范围 0–1.5，初测 0.7 是统一对照起点，并非作者最佳参数。接口不暗中修改 prompt。强度 0 仍加载适配器，真正无 LoRA 使用 `lora=None`。

后端使用当前 Diffusers 官方加载器转换 AI-Toolkit 键名，在挂载前检查每个权重层与底模尺寸。适配器未融合到基座；同适配器调强度无需重载，切换适配器或回到无 LoRA 会重新加载干净基座，额外加载时间单独记录。主测试采用1024×1024、BF16、Qwen 40 步/CFG 1、Z 9 步/guidance 0、seed 42 与2026。

`/lora-comparison` 为真实结果对照页，能查看每张图的原始 prompt 和参数。`evaluate_loras.py` 执行20张主对照及4张零强度/恢复测试。再次实测请设置新的 `LORA_EVAL_DIR`；默认读取已有记录以避免重复提交。两个种子用于初步视觉比较，不构成通用质量结论或稳定速度统计。此类审美/风格 LoRA 不等同于固定人物身份，也不保证手指和肢体修复。

### Qwen 2.1 社区权重兼容修正

这三份 AI-Toolkit 权重使用融合投影 `img_mlp.gate_up`，当前 Diffusers 使用独立 `gate_layer` 和 `proj`。只有去掉 `diffusion_model.` 前缀还不够，首轮加载因此被校验拒绝。`lora_compat.py` 按官方 AI-Toolkit 源码的 `[gate; up]` 行顺序复制 A、将 B 沿输出维切成两半，保持 `ΔW=B@A` 等价；不删除 MLP 权重，也不改动原始文件。

验证：`test_lora_compat.py` 对三款各32个融合层进行 float64 投影等价检查，最大绝对差均为0；转换后再逐层检查实际底模名称和尺寸。原失败请求保存在 `lora_evaluation/initial_failures`。数值转换正确并不替代出图质量验收。

官方实现参考：https://github.com/ostris/ai-toolkit/blob/main/extensions_built_in/diffusion_models/qwen_image_2/src/transformer.py
