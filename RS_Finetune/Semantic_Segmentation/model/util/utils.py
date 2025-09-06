import os
import torch
# from mmcv.cnn.bricks.transformer import FFNMOE
from model.backbone.vit_moe import FFNMOE


def load_network(model, resume_path, rank=0, logger=None, 
                 decoder_datasets=('potsdam', 'OpenEarthMap', 'loveda')):
    """
    加载 checkpoint 到模型，并复制通用 semsegdecoder 权重到各 per-dataset decoders
    
    Args:
        model: 待加载的模型
        resume_path (str): checkpoint 路径
        rank (int): 当前进程 rank（仅 rank=0 打印日志）
        logger: 日志对象，可为 None
        decoder_datasets (tuple): 需要复制 decoder 权重的数据集
    """
    
    if not os.path.isfile(resume_path):
        if rank == 0 and logger:
            logger.warning(f"=> checkpoint not found at {resume_path}")
        return model

    if rank == 0 and logger:
        logger.info(f"=> loading checkpoint '{resume_path}'")

    checkpoint = torch.load(resume_path, map_location='cpu')
    ckpt_dict = checkpoint['model']

    # 去掉 "module." 前缀
    if list(ckpt_dict.keys())[0].startswith('module.'):
        ckpt_dict = {k[7:]: v for k, v in ckpt_dict.items()}

    # 过滤 shape 不匹配的参数
    model_dict = model.state_dict()
    filtered_ckpt_dict = {}
    for k, v in ckpt_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            filtered_ckpt_dict[k] = v
        else:
            if rank == 0 and logger:
                logger.warning(f"Skipping parameter: {k} with shape {v.shape} (does not match)")

    # 更新权重
    model_dict.update(filtered_ckpt_dict)
    model.load_state_dict(model_dict, strict=False)

    # 复制 semsegdecoder 权重
    if rank == 0 and logger:
        logger.info("=> copying semsegdecoder weights to per-dataset decoders...")

    base_decoder_sd = model.semsegdecoder.state_dict()
    for ds in decoder_datasets:
        decoder_module = getattr(model, f'semsegdecoder_{ds}', None)
        if decoder_module is not None:
            decoder_module.load_state_dict(base_decoder_sd, strict=False)

    return model, ckpt_dict





def load_moe_ffn_weights(model, pretrained_state_dict, part_features=128, verbose=True):

    for name, module in model.named_modules():
        if not isinstance(module, FFNMOE):
            continue

        old_fc2_weight_key = f'{name}.layers.1.weight'
        old_fc2_bias_key = f'{name}.layers.1.bias'

        if old_fc2_weight_key not in pretrained_state_dict:
            if verbose:
                print(f'[WARN] Pretrained key {old_fc2_weight_key} not found, skip {name}')
            continue

        fc2_weight = pretrained_state_dict[old_fc2_weight_key] 
        fc2_bias = pretrained_state_dict[old_fc2_bias_key]      

        shared_out_dim = module.embed_dims - module.part_features
        shared_weight = fc2_weight[:shared_out_dim, :]  
        shared_bias = fc2_bias[:shared_out_dim]        

        module.layers[-2].weight.data.copy_(shared_weight)
        module.layers[-2].bias.data.copy_(shared_bias)

        expert_weight = fc2_weight[shared_out_dim:, :] 
        expert_bias = fc2_bias[shared_out_dim:]         

        for i, expert in enumerate(module.experts):
            expert.weight.data.copy_(expert_weight)
            expert.bias.data.copy_(expert_bias)

        if verbose:
            print(f'[INFO] Loaded MoE FFN weights for {name}')
