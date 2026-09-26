"""AI-Toolkit Qwen 2.1 fused [gate; up] adapters -> Diffusers split SwiGLU.

Source layout documented in:
https://github.com/ostris/ai-toolkit/blob/main/extensions_built_in/diffusion_models/qwen_image_2/src/transformer.py
The original file is never changed. For delta W = B @ A, copy A and split B
on its output rows; this preserves the two projections and their rank/scale.
"""
def split_qwen21_gate_up(state):
    result=dict(state)
    for key in list(state):
        if not key.endswith('.img_mlp.gate_up.lora_A.weight'):
            continue
        bkey=key.replace('.lora_A.', '.lora_B.')
        a,b=state[key],state[bkey]
        if a.ndim!=2 or b.ndim!=2 or b.shape[0]%2 or a.shape[0]!=b.shape[1]:
            raise ValueError('Invalid fused Qwen 2.1 LoRA shape: '+key)
        for module,part in zip(('gate_layer','proj'),b.chunk(2,dim=0)):
            ak=key.replace('.gate_up.',f'.{module}.')
            bk=bkey.replace('.gate_up.',f'.{module}.')
            if ak in result or bk in result:
                raise ValueError('Ambiguous fused and split LoRA tensors: '+key)
            result[ak]=a.clone()
            result[bk]=part.contiguous()
        del result[key],result[bkey]
    return result

def split_zimage_fused_qkv(state):
    """ComfyUI-native Z-Image adapters -> Diffusers ZImageTransformer2DModel.

    ComfyUI keeps one fused attention.qkv projection (rows ordered q, k, v; Z-Image
    uses n_kv_heads == n_heads, so three equal thirds) and names the output
    projection attention.out. Diffusers has separate to_q/to_k/to_v and to_out.0.
    Same B @ A argument as above: copy A, split B into thirds.
    """
    result=dict(state)
    for key in list(state):
        if key.endswith('.attention.out.lora_A.weight') or key.endswith('.attention.out.lora_B.weight'):
            new=key.replace('.attention.out.','.attention.to_out.0.')
            if new in result:
                raise ValueError('Ambiguous fused and split LoRA tensors: '+key)
            result[new]=result.pop(key)
        if not key.endswith('.attention.qkv.lora_A.weight'):
            continue
        bkey=key.replace('.lora_A.', '.lora_B.')
        a,b=state[key],state[bkey]
        if a.ndim!=2 or b.ndim!=2 or b.shape[0]%3 or a.shape[0]!=b.shape[1]:
            raise ValueError('Invalid fused Z-Image qkv LoRA shape: '+key)
        for module,part in zip(('to_q','to_k','to_v'),b.chunk(3,dim=0)):
            ak=key.replace('.qkv.',f'.{module}.')
            bk=bkey.replace('.qkv.',f'.{module}.')
            if ak in result or bk in result:
                raise ValueError('Ambiguous fused and split LoRA tensors: '+key)
            result[ak]=a.clone()
            result[bk]=part.contiguous()
        del result[key],result[bkey]
    return result
