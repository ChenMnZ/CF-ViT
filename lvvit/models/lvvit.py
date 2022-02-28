import torch
import torch.nn as nn

from timm.models.helpers import load_pretrained
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from timm.models.resnet import resnet26d, resnet50d, resnet101d
import numpy as np
import math

from .layers import *
from utils import batch_index_select,get_index



def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'LV_ViT_Tiny': _cfg(),
    'LV_ViT': _cfg(),
    'LV_ViT_Medium': _cfg(crop_pct=1.0),
    'LV_ViT_Large': _cfg(crop_pct=1.0),
}

def get_block(block_type, **kargs):
    if block_type=='mha':
        # multi-head attention block
        return MHABlock(**kargs)
    elif block_type=='ffn':
        # feed forward block
        return FFNBlock(**kargs)
    elif block_type=='tr':
        # transformer block
        return Block(**kargs)


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def get_dpr(drop_path_rate,depth,drop_path_decay='linear'):
    if drop_path_decay=='linear':
        # linear dpr decay
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
    elif drop_path_decay=='fix':
        # use fixed dpr
        dpr= [drop_path_rate]*depth
    else:
        # use predefined drop_path_rate list
        assert len(drop_path_rate)==depth
        dpr=drop_path_rate
    return dpr


class CF_LV_ViT(nn.Module):
    """ Vision Transformer with tricks
    Arguements:
        p_emb: different conv based position embedding (default: 4 layer conv)
        skip_lam: residual scalar for skip connection (default: 1.0)
        order: which order of layers will be used (default: None, will override depth if given)
        mix_token: use mix token augmentation for batch of tokens (default: False)
        return_dense: whether to return feature of all tokens with an additional aux_head (default: False)
    """
    def __init__(self, img_size_list = [112,224], patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., drop_path_decay='linear', hybrid_backbone=None, norm_layer=nn.LayerNorm, p_emb='4_2', head_dim = None,
                 skip_lam = 1.0,order=None, mix_token=False, return_dense=False):
        super().__init__()

        self.informative_selection = False
        self.alpha = 0.5
        self.beta_2 = 0.99
        # self.beta_2 = 0.8
        # self.target_index = [3,4,5,6,7,8,9,10,11,12,13,14,15]
        self.target_index = [3,4,5,6,7,8,9,10,11,12,13,14,15]
        self.patch_h = img_size_list[1]//patch_size
        self.patch_w = img_size_list[1]//patch_size
        self.img_size_list = img_size_list
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.output_dim = embed_dim if num_classes==0 else num_classes

        patch_embed_fn = MultiResoPatchEmbed4_2
        self.patch_embed = patch_embed_fn(img_size_list=img_size_list, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        self.num_patches_list = self.patch_embed.num_patches_list
        self.patches_h_list = [img_size//patch_size for img_size in img_size_list]


        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed_list = [nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim)) for num_patches in self.num_patches_list]
        self.pos_embed_list = nn.ParameterList(self.pos_embed_list)
        self.pos_drop = nn.Dropout(p=drop_rate)

        if order is None:
            dpr=get_dpr(drop_path_rate, depth, drop_path_decay)
            self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim, num_heads=num_heads, head_dim=head_dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, skip_lam=skip_lam)
                for i in range(depth)])
        else:
            # use given order to sequentially generate modules
            dpr=get_dpr(drop_path_rate, len(order), drop_path_decay)
            self.blocks = nn.ModuleList([
                get_block(order[i],
                    dim=embed_dim, num_heads=num_heads, head_dim=head_dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, skip_lam=skip_lam)
                for i in range(len(order))])

        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        self.return_dense=return_dense
        self.mix_token=mix_token

        if return_dense:
            self.aux_head=nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if mix_token:
            self.beta = 1.0
            assert return_dense, "always return all features when mixtoken is enabled"

        self.reuse_block = nn.Sequential(
                norm_layer(embed_dim),
                Mlp(in_features=embed_dim, hidden_features=int(mlp_ratio*embed_dim),out_features=embed_dim,act_layer=nn.GELU,drop=drop_rate)
            ) 

        for pos_embed in self.pos_embed_list:
            trunc_normal_(pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, GroupLinear):
            trunc_normal_(m.group_weight, std=.02)
            if isinstance(m, GroupLinear) and m.group_bias is not None:
                nn.init.constant_(m.group_bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed_list', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
    
    def forward_embeddings(self,x):
        x = self.patch_embed(x)
        return x

    def forward(self, xx):
        results = []
        # coarse stage
        self.first_stage_output = None
        x = self.forward_embeddings(xx[0])
        if self.mix_token and self.training:
            lam = np.random.beta(self.beta, self.beta)
            bbx1_0, bby1_0, bbx2_0, bby2_0 = rand_bbox(x.size(), lam)
            bbx1_1, bby1_1, bbx2_1, bby2_1 = bbx1_0*2, bby1_0*2, bbx2_0*2, bby2_0*2
            temp_x = x.clone()
            temp_x[:, :, bbx1_0:bbx2_0, bby1_0:bby2_0] = x.flip(0)[:, :, bbx1_0:bbx2_0, bby1_0:bby2_0]
            x = temp_x
        else:
            bbx1_1, bby1_1, bbx2_1, bby2_1 = 0,0,0,0
            bbx1_0, bby1_0, bbx2_0, bby2_0 = 0,0,0,0

        x = x.flatten(2).transpose(1, 2)
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed_list[0]
        embedding_coarse = x
        x = self.pos_drop(x)
        global_attention = 0
        for i, blk in enumerate(self.blocks):
            x,atten = blk(x)
            if i in self.target_index:
                global_attention = self.beta_2*global_attention + (1-self.beta_2)*atten
        x = self.norm(x)
        self.first_stage_output = x
        x_cls = self.head(x[:,0])
        if self.return_dense:
            x_aux = self.aux_head(x[:,1:])
            if not self.training:
                results.append((x_cls+0.25*x_aux.max(1)[0])/1.25)
            else:
                # recover the mixed part
                if self.mix_token and self.training:
                    x_aux = x_aux.reshape(x_aux.shape[0],self.patches_h_list[0], self.patches_h_list[0],x_aux.shape[-1])
                    temp_x = x_aux.clone()
                    temp_x[:, bbx1_0:bbx2_0, bby1_0:bby2_0, :] = x_aux.flip(0)[:, bbx1_0:bbx2_0, bby1_0:bby2_0, :]
                    x_aux = temp_x
                    x_aux = x_aux.reshape(x_aux.shape[0],self.num_patches_list[0],x_aux.shape[-1])
                results.append((x_cls, x_aux, (bbx1_0, bby1_0, bbx2_0, bby2_0)))
        else:
            results.append(x_cls)
        # fine stage
        x = self.forward_embeddings(xx[1])
        if self.mix_token and self.training:
            temp_x = x.clone()
            temp_x[:, :, bbx1_1:bbx2_1, bby1_1:bby2_1] = x.flip(0)[:, :, bbx1_1:bbx2_1, bby1_1:bby2_1]
            x = temp_x

        x = x.flatten(2).transpose(1, 2)
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        feature_temp = self.first_stage_output[:,1:,:]
        feature_temp = self.reuse_block(feature_temp)       
        B, new_HW, C = feature_temp.shape
        feature_temp = feature_temp.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        feature_temp = torch.nn.functional.interpolate(feature_temp, (self.patch_h, self.patch_w), mode='nearest').view(B, C, x.size(1) - 1).transpose(1, 2)
        feature_temp = torch.cat((torch.zeros(B, 1, self.embed_dim).cuda(), feature_temp), dim=1)
        x = x+feature_temp      # 直接shortcut
        x = x + self.pos_embed_list[1]
        embedding_fine = x
        x = self.pos_drop(x)

        if not self.informative_selection:
            for blk in self.blocks:
                x,_ = blk(x)
            x = self.norm(x)
            x_cls = self.head(x[:,0])

            if self.return_dense:
                x_aux = self.aux_head(x[:,1:])
                if not self.training:
                    results.append(x_cls+0.5*x_aux.max(1)[0])
                    # results.append(x_cls[0])
                else:
                    # recover the mixed part
                    if self.mix_token and self.training:
                        x_aux = x_aux.reshape(x_aux.shape[0],self.patches_h_list[1], self.patches_h_list[1],x_aux.shape[-1])
                        temp_x = x_aux.clone()
                        temp_x[:, bbx1_1:bbx2_1, bby1_1:bby2_1, :] = x_aux.flip(0)[:, bbx1_1:bbx2_1, bby1_1:bby2_1, :]
                        x_aux = temp_x
                        x_aux = x_aux.reshape(x_aux.shape[0],self.num_patches_list[1],x_aux.shape[-1])
                    results.append((x_cls, x_aux, (bbx1_1, bby1_1, bbx2_1, bby2_1)))
            else:
                results.append(x_cls)
        else:
            cls_attn = global_attention.mean(dim=1)[:,0,1:]
            import_token_num = math.ceil(self.alpha * self.patch_embed.num_patches_list[0])
            policy_index = torch.argsort(cls_attn, dim=1, descending=True)
            patch_unimportan_index = policy_index[:, import_token_num:]
            important_index = policy_index[:, :import_token_num]
            unimportan_tokens = batch_index_select(embedding_coarse, patch_unimportan_index+1)
            patch_important_index = get_index(important_index,image_size=self.img_size_list[1])
            cls_index = torch.zeros((B,1)).cuda().long()
            important_index = torch.cat((cls_index, patch_important_index+1), dim=1)
            important_tokens = batch_index_select(embedding_fine, important_index)
            x = torch.cat((important_tokens, unimportan_tokens), dim=1)
            for blk in self.blocks:
                x,_ = blk(x)
            x = self.norm(x)
            x_cls = self.head(x[:,0])

            if self.return_dense:
                x_aux = self.aux_head(x[:,1:])
                if not self.training:
                    results.append(x_cls+0.25*x_aux.max(1)[0])
                    # pdb.set_trace()
                    # results.append(0.5*x_aux.max(1)[0])

                else:
                    # recover the mixed part
                    if self.mix_token and self.training:
                        # 恢复成正方形,即不重要的区域复制成4份
                        # x_temp = torch.zeros(B,196,self.num_classes, dtype=torch.float16).cuda()
                        x_temp = torch.zeros(B,self.num_patches_list[1],self.num_classes, dtype=x_aux.dtype).cuda()

                        idx = patch_important_index
                        B, N, C = x_temp.size()
                        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
                        idx = idx + offset

                        # pdb.set_trace()
                        x_temp.view(B*N, C)[idx.reshape(-1)] =  x_aux[:,:(import_token_num*4)].reshape(-1,C)


                        idx = patch_unimportan_index
                        idx = get_index(idx,image_size=self.img_size_list[1])
                        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
                        idx = idx + offset
                        x_temp.view(B*N, C)[idx.reshape(-1)] =  x_aux[:,(import_token_num*4):].repeat(1,4,1).reshape(-1,C)

                        x_aux = x_temp.reshape(B,self.patches_h_list[1], self.patches_h_list[1],C)
                        temp_x = x_aux.clone()
                        temp_x[:, bbx1_1:bbx2_1, bby1_1:bby2_1, :] = x_aux.flip(0)[:, bbx1_1:bbx2_1, bby1_1:bby2_1, :]
                        x_aux = temp_x
                        x_aux = x_aux.reshape(x_aux.shape[0],self.num_patches_list[1],x_aux.shape[-1])
                    results.append((x_cls, x_aux,(bbx1_1, bby1_1, bbx2_1, bby2_1)))
            else:
                results.append(x_cls)
        # pdb.set_trace()
        return results

    def forward_early_exit(self, xx, threshold):
        # dynamic inference 
        self.first_stage_output = None
        x = self.forward_embeddings(xx[0])

        x = x.flatten(2).transpose(1, 2)
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed_list[0]
        embedding_coarse = x
        x = self.pos_drop(x)
        global_attention = 0
        for i, blk in enumerate(self.blocks):
            x,atten = blk(x)
            if i in self.target_index:
                global_attention = self.beta_2*global_attention + (1-self.beta_2)*atten
        x = self.norm(x)
        self.first_stage_output = x
        x_cls = self.head(x[:,0])
        x_aux = self.aux_head(x[:,1:])
        coarse_result = x_cls+0.5*x_aux.max(1)[0]
        logits_temp = F.softmax(coarse_result, 1)
        max_preds, _ = logits_temp.max(dim=1, keepdim=False)
        no_exit = max_preds < threshold
            
        x = xx[1][no_exit]

        x = self.forward_embeddings(x)

        x = x.flatten(2).transpose(1, 2)
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        feature_temp = self.first_stage_output[:,1:,:][no_exit]
        feature_temp = self.reuse_block(feature_temp)       
        B, new_HW, C = feature_temp.shape
        feature_temp = feature_temp.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        feature_temp = torch.nn.functional.interpolate(feature_temp, (self.patch_h, self.patch_w), mode='nearest').view(B, C, x.size(1) - 1).transpose(1, 2)
        feature_temp = torch.cat((torch.zeros(B, 1, self.embed_dim).cuda(), feature_temp), dim=1)
        x = x+feature_temp      # 直接shortcut
        x = x + self.pos_embed_list[1]
        embedding_fine = x
        x = self.pos_drop(x)

        
        cls_attn = global_attention[no_exit].mean(dim=1)[:,0,1:] # 不计算cls_token本身
        import_token_num = int(self.alpha * self.patch_embed.num_patches_list[0])
        policy_index = torch.argsort(cls_attn, dim=1, descending=True)
        patch_unimportan_index = policy_index[:, import_token_num:]
        important_index = policy_index[:, :import_token_num]
        unimportan_tokens = batch_index_select(embedding_coarse[no_exit], patch_unimportan_index+1)
        patch_important_index = get_index(important_index,image_size=self.img_size_list[1])
        cls_index = torch.zeros((B,1)).cuda().long()
        important_index = torch.cat((cls_index, patch_important_index+1), dim=1)
        important_tokens = batch_index_select(embedding_fine, important_index)
        x = torch.cat((important_tokens, unimportan_tokens), dim=1)
        for blk in self.blocks:
            x,_ = blk(x)
        x = self.norm(x)
        x_cls = self.head(x[:,0])

        x_aux = self.aux_head(x[:,1:])
        fine_result = x_cls+0.5*x_aux.max(1)[0]
        coarse_result[no_exit] = fine_result
        return coarse_result

@register_model
def cf_lvvit_small(pretrained=False, **kwargs):
    model = CF_LV_ViT(patch_size=16, embed_dim=384, depth=16, num_heads=6, mlp_ratio=3.,
        p_emb='4_2',skip_lam=2., return_dense=True,mix_token=True, **kwargs)
    model.default_cfg = default_cfgs['LV_ViT']
    return model


