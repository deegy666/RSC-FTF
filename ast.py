import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import os
import wget
import timm
from transformers import  Wav2Vec2Model
from copy import deepcopy
from timm.models.layers import to_2tuple,trunc_normal_


# override the timm package to relax the input shape constraint.
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        # patch_size = (32,8)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class ASTModel(nn.Module):
    """
    The AST model.
    :param label_dim: the label dimension, i.e., the number of total classes, it is 527 for AudioSet, 50 for ESC-50, and 35 for speechcommands v2-35
    :param fstride: the stride of patch spliting on the frequency dimension, for 16*16 patchs, fstride=16 means no overlap, fstride=10 means overlap of 6
    :param tstride: the stride of patch spliting on the time dimension, for 16*16 patchs, tstride=16 means no overlap, tstride=10 means overlap of 6
    :param input_fdim: the number of frequency bins of the input spectrogram
    :param input_tdim: the number of time frames of the input spectrogram
    :param imagenet_pretrain: if use ImageNet pretrained model
    :param audioset_pretrain: if use full AudioSet and ImageNet pretrained model
    :param model_size: the model size of AST, should be in [tiny224, small224, base224, base384], base224 and base 384 are same model, but are trained differently during ImageNet pretraining.
    """
    def __init__(self, label_dim=527, fstride=16, tstride=16, input_fdim=256, input_tdim=1024, imagenet_pretrain=True, audioset_pretrain=False, model_size='base384', verbose=True, mix_beta=None):
        super(ASTModel, self).__init__()
        assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5, the code might not be compatible with newer versions.'

        if verbose == True:
            print('---------------AST Model Summary---------------')
            print('ImageNet pretraining: {:s}, AudioSet pretraining: {:s}'.format(str(imagenet_pretrain),str(audioset_pretrain)))
        # override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        self.final_feat_dim = 768
        self.mix_beta = mix_beta
        self.wav2vec2_model = Wav2Vec2Model.from_pretrained("./Extract_Time-domain_Features/facebookwav2vec2-base-960h")
        self.conv = nn.Conv1d(in_channels=768, out_channels=768, kernel_size=230, stride=2, padding=0)
        self.scale = 768 ** -0.5
        self.attn_drop = nn.Dropout(0.1)
        self.proj = nn.Linear(768, 768)  # 把多个head进行Concat操作，然后通过Wo映射，这里用全连接层代替
        self.proj_drop = nn.Dropout(0.1)
        # if AudioSet pretraining is not used (but ImageNet pretraining may still apply)
        if audioset_pretrain == False:
            if model_size == 'tiny224':
                self.v = timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'small224':
                self.v = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'base224':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'base384':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=imagenet_pretrain)
            else:
                raise Exception('Model size must be one of tiny224, small224, base224, base384.')
            self.original_num_patches = self.v.patch_embed.num_patches
            self.oringal_hw = int(self.original_num_patches ** 0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))

            # automatcially get the intermediate shape
            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose == True:
                print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches))

            # the linear projection layer
            new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
            if imagenet_pretrain == True:
                new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
                new_proj.bias = self.v.patch_embed.proj.bias
            self.v.patch_embed.proj = new_proj

            # the positional embedding
            if imagenet_pretrain == True:
                # get the positional embedding from deit model, skip the first two tokens (cls token and distillation token), reshape it to original 2D shape (24*24).
                new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, self.original_num_patches, self.original_embedding_dim).transpose(1, 2).reshape(1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw)
                # cut (from middle) or interpolate the second dimension of the positional embedding
                if t_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[:, :, :, int(self.oringal_hw / 2) - int(t_dim / 2): int(self.oringal_hw / 2) - int(t_dim / 2) + t_dim]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(self.oringal_hw, t_dim), mode='bilinear')
                # cut (from middle) or interpolate the first dimension of the positional embedding
                if f_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[:, :, int(self.oringal_hw / 2) - int(f_dim / 2): int(self.oringal_hw / 2) - int(f_dim / 2) + f_dim, :]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
                # flatten the positional embedding
                new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1,2)
                # concatenate the above positional embedding with the cls token and distillation token of the deit model.
                self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
            else:
                # if not use imagenet pretrained model, just randomly initialize a learnable positional embedding
                # TODO can use sinusoidal positional embedding instead
                new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + 2, self.original_embedding_dim))
                self.v.pos_embed = new_pos_embed
                trunc_normal_(self.v.pos_embed, std=.02)

        # now load a model that is pretrained on both ImageNet and AudioSet
        elif audioset_pretrain == True:
            if audioset_pretrain == True and imagenet_pretrain == False:
                raise ValueError('currently model pretrained on only audioset is not supported, please set imagenet_pretrain = True to use audioset pretrained model.')
            if model_size != 'base384':
                raise ValueError('currently only has base384 AudioSet pretrained model.')
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            out_dir = './pretrained_models/'
            if not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)
            
            # sd =
            weights = torch.load(os.path.join(out_dir, 'audioset_16_16_0.4422.pth'), map_location=device)
            audio_model = ASTModel(label_dim=527, fstride=16, tstride=16, input_fdim=256, input_tdim=1024, imagenet_pretrain=False, audioset_pretrain=False, model_size='base384', verbose=False)
            audio_model = torch.nn.DataParallel(audio_model)

            current_model_dicts = audio_model.state_dict()

            for k1 in current_model_dicts.keys(): 
                for k2 in weights.keys():
                    if k1 == k2 and current_model_dicts[k1].size() == weights[k2].size():
                        current_model_dicts[k1] = weights[k2]
                    elif k1 == k2 and current_model_dicts[k1].size() != weights[k2].size():
                        print(f"mismatch: {k1}; {k2}")

            print(audio_model.load_state_dict(current_model_dicts, strict=False))

            # audio_model.load_state_dict(sd, strict=False)

            self.v = audio_model.module.v
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))

            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose == True:
                print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches))

            new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, 1024, 768).transpose(1, 2).reshape(1, 768, 16, 64)
            # if the input sequence length is larger than the original audioset (10s), then cut the positional embedding
            
            new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
            new_pos_embed = new_pos_embed.reshape(1, 768, num_patches).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))

    def get_shape(self, fstride, tstride, input_fdim=256, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    def square_patch(self, patch, hw_num_patch):
        h, w = hw_num_patch
        B, _, dim = patch.size()
        square = patch.reshape(B, h, w, dim)
        return square

    def flatten_patch(self, square):
        B, h, w, dim = square.shape
        patch = square.reshape(B, h * w, dim)
        return patch

    def patch_mix(self, image, target, time_domain=False, hw_num_patch=None):
        if self.mix_beta > 0:
            lam = np.random.beta(self.mix_beta, self.mix_beta)
        else:
            lam = 1

        batch_size, num_patch, dim = image.size()
        device = image.device

        index = torch.randperm(batch_size).to(device)

        if not time_domain:
            num_mask = int(num_patch * (1. - lam))
            mask = torch.randperm(num_patch)[:num_mask].to(device)

            image[:, mask, :] = image[index][:, mask, :]
            lam = 1 - (num_mask / num_patch)
        else:
            squared_1 = self.square_patch(image, hw_num_patch)
            squared_2 = self.square_patch(image[index], hw_num_patch)

            w_size = squared_1.size()[2]
            num_mask = int(w_size * (1. - lam))
            mask = torch.randperm(w_size)[:num_mask].to(device)

            squared_1[:, :, mask, :] = squared_2[:, :, mask, :]
            image = self.flatten_patch(squared_1)
            lam = 1 - (num_mask / w_size)
        
        y_a, y_b = target, target[index]
        return image, y_a, y_b, lam, index

    @autocast()
    def forward(self, x, audio,  y=None, patch_mix=False, time_domain=False):
        """
        :param x: the input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        :return: prediction
        """
        # x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        # h_patch, w_patch = int((x.size()[2] - 16) / 16) + 1, int((x.size()[3] - 16) / 16) + 1

        h_patch, w_patch = (16, 64)  # kernel_size = (16, 16)

        B = x.shape[0]
        x = self.v.patch_embed(x)

        if patch_mix:
            x, y_a, y_b, lam, index = self.patch_mix(x, y, time_domain=time_domain, hw_num_patch=[h_patch, w_patch])

        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for i, blk in enumerate(self.v.blocks):
            x = blk(x)
        x = self.v.norm(x) # [B, 1026, 768]
        x = x.transpose(1, 2)

        x = self.conv(x)
        x_1 = x
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool1d(x, kernel_size=1, stride=1) + x_1
        x = x.transpose(1, 2)   #[B, 399, 768]
        v = x
        #x = (x[:, 0] + x[:, 1]) / 2  #[B, 768]
        # x = self.mlp_head(x)
        audio = audio.squeeze(dim = 1) # [B, 1, 128000] -> [B, 128000]
        audio = self.wav2vec2_model(audio.cuda()).last_hidden_state # [B, 128000] -> [B, 399, 768]
        attn = (audio @ x.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)  # 通过softmax处理
        attn = self.attn_drop(attn)  # dropOut层
        x = (attn @ v).transpose(1, 2).reshape(B, 399, 768)
        x = self.proj(x)  # 通过全连接进行映射(相当于乘论文中的Wo)
        x = self.proj_drop(x)  # dropOut
        x = (x[:, 0] + x[:, 1]) / 2

        if not patch_mix:
            return x
        else:
            return x, y_a, y_b, lam, index