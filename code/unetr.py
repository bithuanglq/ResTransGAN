# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple, Union

import torch
import torch.nn as nn

from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.nets import ViT



class UNETR(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Tuple[int, int, int],
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "perceptron",
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = False,
        res_block: bool = True,
        dropout_rate: float = 0.0,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.

        Examples::

            # for single channel input 4-channel output with patch size of (96,96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=(96,96,96), feature_size=32, norm_name='batch')

            # for 4-channel input 3-channel output with patch size of (128,128,128), conv position embedding and instance norm
            >>> net = UNETR(in_channels=4, out_channels=3, img_size=(128,128,128), pos_embed='conv', norm_name='instance')

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise AssertionError("hidden size should be divisible by num_heads.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.num_layers = 12
        self.patch_size = (16, 16, 16)
        self.feat_size = (
            img_size[0] // self.patch_size[0],
            img_size[1] // self.patch_size[1],
            img_size[2] // self.patch_size[2],
        )
        self.hidden_size = hidden_size
        self.classification = False
        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            classification=self.classification,
            dropout_rate=dropout_rate,
        )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)  # type: ignore

    def proj_feat(self, x, hidden_size, feat_size):
        '''
          (b, w, h, d, c) --> (b, c, w, h, d)
        '''
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def load_from(self, weights):
        with torch.no_grad():
            res_weight = weights
            # copy weights from patch embedding
            for i in weights["state_dict"]:
                print(i)
            self.vit.patch_embedding.position_embeddings.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.position_embeddings_3d"]
            )
            self.vit.patch_embedding.cls_token.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.cls_token"]
            )
            self.vit.patch_embedding.patch_embeddings[1].weight.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.patch_embeddings.1.weight"]
            )
            self.vit.patch_embedding.patch_embeddings[1].bias.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.patch_embeddings.1.bias"]
            )

            # copy weights from  encoding blocks (default: num of blocks: 12)
            for bname, block in self.vit.blocks.named_children():
                print(block)
                block.loadFrom(weights, n_block=bname)
            # last norm layer of transformer
            self.vit.norm.weight.copy_(weights["state_dict"]["module.transformer.norm.weight"])
            self.vit.norm.bias.copy_(weights["state_dict"]["module.transformer.norm.bias"])

    def forward(self, x_in):
        x, hidden_states_out = self.vit(x_in)
        enc1 = self.encoder1(x_in)
        x2 = hidden_states_out[3]
        enc2 = self.encoder2(self.proj_feat(x2, self.hidden_size, self.feat_size))
        x3 = hidden_states_out[6]
        enc3 = self.encoder3(self.proj_feat(x3, self.hidden_size, self.feat_size))
        x4 = hidden_states_out[9]
        enc4 = self.encoder4(self.proj_feat(x4, self.hidden_size, self.feat_size))
        dec4 = self.proj_feat(x, self.hidden_size, self.feat_size)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)
        logits = self.out(out)
        return logits








'''
DataParallel(
  (module): UNETR(
    (vit): ViT(
      (patch_embedding): PatchEmbeddingBlock(
        (patch_embeddings): Sequential(
          (0): Rearrange('b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)', p1=16, p2=16, p3=16)
          (1): Linear(in_features=16384, out_features=768, bias=True)
        )
        (dropout): Dropout(p=0.0, inplace=False)
      )
      (blocks): ModuleList(
        (0): TransformerBlock(
          (mlp): MLPBlock(
            (linear1): Linear(in_features=768, out_features=3072, bias=True)
            (linear2): Linear(in_features=3072, out_features=768, bias=True)
            (fn): GELU()
            (drop1): Dropout(p=0.0, inplace=False)
            (drop2): Dropout(p=0.0, inplace=False)
          )
          (norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (attn): SABlock(
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
            (qkv): Linear(in_features=768, out_features=2304, bias=False)
            (input_rearrange): Rearrange('b h (qkv l d) -> qkv b l h d', qkv=3, l=12)
            (out_rearrange): Rearrange('b h l d -> b l (h d)')
            (drop_output): Dropout(p=0.0, inplace=False)
            (drop_weights): Dropout(p=0.0, inplace=False)
          )
          (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
        (1): TransformerBlock(
          (mlp): MLPBlock(
            (linear1): Linear(in_features=768, out_features=3072, bias=True)
            (linear2): Linear(in_features=3072, out_features=768, bias=True)
            (fn): GELU()
            (drop1): Dropout(p=0.0, inplace=False)
            (drop2): Dropout(p=0.0, inplace=False)
          )
          (norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (attn): SABlock(
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
            (qkv): Linear(in_features=768, out_features=2304, bias=False)
            (input_rearrange): Rearrange('b h (qkv l d) -> qkv b l h d', qkv=3, l=12)
            (out_rearrange): Rearrange('b h l d -> b l (h d)')
            (drop_output): Dropout(p=0.0, inplace=False)
            (drop_weights): Dropout(p=0.0, inplace=False)
          )
          (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
        (2): TransformerBlock(
          (mlp): MLPBlock(
            (linear1): Linear(in_features=768, out_features=3072, bias=True)
            (linear2): Linear(in_features=3072, out_features=768, bias=True)
            (fn): GELU()
            (drop1): Dropout(p=0.0, inplace=False)
            (drop2): Dropout(p=0.0, inplace=False)
          )
          (norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (attn): SABlock(
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
            (qkv): Linear(in_features=768, out_features=2304, bias=False)
            (input_rearrange): Rearrange('b h (qkv l d) -> qkv b l h d', qkv=3, l=12)
            (out_rearrange): Rearrange('b h l d -> b l (h d)')
            (drop_output): Dropout(p=0.0, inplace=False)
            (drop_weights): Dropout(p=0.0, inplace=False)
          )
          (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
        (3): TransformerBlock(
          (mlp): MLPBlock(
            (linear1): Linear(in_features=768, out_features=3072, bias=True)
            (linear2): Linear(in_features=3072, out_features=768, bias=True)
            (fn): GELU()
            (drop1): Dropout(p=0.0, inplace=False)
            (drop2): Dropout(p=0.0, inplace=False)
          )
          (norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (attn): SABlock(
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
            (qkv): Linear(in_features=768, out_features=2304, bias=False)
            (input_rearrange): Rearrange('b h (qkv l d) -> qkv b l h d', qkv=3, l=12)
            (out_rearrange): Rearrange('b h l d -> b l (h d)')
            (drop_output): Dropout(p=0.0, inplace=False)
            (drop_weights): Dropout(p=0.0, inplace=False)
          )
          (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
        (4): TransformerBlock(
          (mlp): MLPBlock(
            (linear1): Linear(in_features=768, out_features=3072, bias=True)
            (linear2): Linear(in_features=3072, out_features=768, bias=True)
            (fn): GELU()
            (drop1): Dropout(p=0.0, inplace=False)
            (drop2): Dropout(p=0.0, inplace=False)
          )
          (norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (attn): SABlock(
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
            (qkv): Linear(in_features=768, out_features=2304, bias=False)
            (input_rearrange): Rearrange('b h (qkv l d) -> qkv b l h d', qkv=3, l=12)
            (out_rearrange): Rearrange('b h l d -> b l (h d)')
            (drop_output): Dropout(p=0.0, inplace=False)
            (drop_weights): Dropout(p=0.0, inplace=False)
          )
          (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
        (5): TransformerBlock(
          (mlp): MLPBlock(
            (linear1): Linear(in_features=768, out_features=3072, bias=True)
            (linear2): Linear(in_features=3072, out_features=768, bias=True)
            (fn): GELU()
            (drop1): Dropout(p=0.0, inplace=False)
            (drop2): Dropout(p=0.0, inplace=False)
          )
          (norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (attn): SABlock(
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
            (qkv): Linear(in_features=768, out_features=2304, bias=False)
            (input_rearrange): Rearrange('b h (qkv l d) -> qkv b l h d', qkv=3, l=12)
            (out_rearrange): Rearrange('b h l d -> b l (h d)')
            (drop_output): Dropout(p=0.0, inplace=False)
            (drop_weights): Dropout(p=0.0, inplace=False)
          )
          (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
        (6): TransformerBlock(
          (mlp): MLPBlock(
            (linear1): Linear(in_features=768, out_features=3072, bias=True)
            (linear2): Linear(in_features=3072, out_features=768, bias=True)
            (fn): GELU()
            (drop1): Dropout(p=0.0, inplace=False)
            (drop2): Dropout(p=0.0, inplace=False)
          )
          (norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (attn): SABlock(
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
            (qkv): Linear(in_features=768, out_features=2304, bias=False)
            (input_rearrange): Rearrange('b h (qkv l d) -> qkv b l h d', qkv=3, l=12)
            (out_rearrange): Rearrange('b h l d -> b l (h d)')
            (drop_output): Dropout(p=0.0, inplace=False)
            (drop_weights): Dropout(p=0.0, inplace=False)
          )
          (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
        (7): TransformerBlock(
          (mlp): MLPBlock(
            (linear1): Linear(in_features=768, out_features=3072, bias=True)
            (linear2): Linear(in_features=3072, out_features=768, bias=True)
            (fn): GELU()
            (drop1): Dropout(p=0.0, inplace=False)
            (drop2): Dropout(p=0.0, inplace=False)
          )
          (norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (attn): SABlock(
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
            (qkv): Linear(in_features=768, out_features=2304, bias=False)
            (input_rearrange): Rearrange('b h (qkv l d) -> qkv b l h d', qkv=3, l=12)
            (out_rearrange): Rearrange('b h l d -> b l (h d)')
            (drop_output): Dropout(p=0.0, inplace=False)
            (drop_weights): Dropout(p=0.0, inplace=False)
          )
          (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
        (8): TransformerBlock(
          (mlp): MLPBlock(
            (linear1): Linear(in_features=768, out_features=3072, bias=True)
            (linear2): Linear(in_features=3072, out_features=768, bias=True)
            (fn): GELU()
            (drop1): Dropout(p=0.0, inplace=False)
            (drop2): Dropout(p=0.0, inplace=False)
          )
          (norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (attn): SABlock(
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
            (qkv): Linear(in_features=768, out_features=2304, bias=False)
            (input_rearrange): Rearrange('b h (qkv l d) -> qkv b l h d', qkv=3, l=12)
            (out_rearrange): Rearrange('b h l d -> b l (h d)')
            (drop_output): Dropout(p=0.0, inplace=False)
            (drop_weights): Dropout(p=0.0, inplace=False)
          )
          (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
        (9): TransformerBlock(
          (mlp): MLPBlock(
            (linear1): Linear(in_features=768, out_features=3072, bias=True)
            (linear2): Linear(in_features=3072, out_features=768, bias=True)
            (fn): GELU()
            (drop1): Dropout(p=0.0, inplace=False)
            (drop2): Dropout(p=0.0, inplace=False)
          )
          (norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (attn): SABlock(
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
            (qkv): Linear(in_features=768, out_features=2304, bias=False)
            (input_rearrange): Rearrange('b h (qkv l d) -> qkv b l h d', qkv=3, l=12)
            (out_rearrange): Rearrange('b h l d -> b l (h d)')
            (drop_output): Dropout(p=0.0, inplace=False)
            (drop_weights): Dropout(p=0.0, inplace=False)
          )
          (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
        (10): TransformerBlock(
          (mlp): MLPBlock(
            (linear1): Linear(in_features=768, out_features=3072, bias=True)
            (linear2): Linear(in_features=3072, out_features=768, bias=True)
            (fn): GELU()
            (drop1): Dropout(p=0.0, inplace=False)
            (drop2): Dropout(p=0.0, inplace=False)
          )
          (norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (attn): SABlock(
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
            (qkv): Linear(in_features=768, out_features=2304, bias=False)
            (input_rearrange): Rearrange('b h (qkv l d) -> qkv b l h d', qkv=3, l=12)
            (out_rearrange): Rearrange('b h l d -> b l (h d)')
            (drop_output): Dropout(p=0.0, inplace=False)
            (drop_weights): Dropout(p=0.0, inplace=False)
          )
          (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
        (11): TransformerBlock(
          (mlp): MLPBlock(
            (linear1): Linear(in_features=768, out_features=3072, bias=True)
            (linear2): Linear(in_features=3072, out_features=768, bias=True)
            (fn): GELU()
            (drop1): Dropout(p=0.0, inplace=False)
            (drop2): Dropout(p=0.0, inplace=False)
          )
          (norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (attn): SABlock(
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
            (qkv): Linear(in_features=768, out_features=2304, bias=False)
            (input_rearrange): Rearrange('b h (qkv l d) -> qkv b l h d', qkv=3, l=12)
            (out_rearrange): Rearrange('b h l d -> b l (h d)')
            (drop_output): Dropout(p=0.0, inplace=False)
            (drop_weights): Dropout(p=0.0, inplace=False)
          )
          (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
      )
      (norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    )
    (encoder1): UnetrBasicBlock(
      (layer): UnetResBlock(
        (conv1): Convolution(
          (conv): Conv3d(4, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        )
        (conv2): Convolution(
          (conv): Conv3d(16, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        )
        (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
        (norm1): InstanceNorm3d(16, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (norm2): InstanceNorm3d(16, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (conv3): Convolution(
          (conv): Conv3d(4, 16, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        )
        (norm3): InstanceNorm3d(16, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    )
    (encoder2): UnetrPrUpBlock(
      (transp_conv_init): Convolution(
        (conv): ConvTranspose3d(768, 32, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
      )
      (blocks): ModuleList(
        (0): Convolution(
          (conv): ConvTranspose3d(32, 32, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
        )
        (1): Convolution(
          (conv): ConvTranspose3d(32, 32, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
        )
      )
    )
    (encoder3): UnetrPrUpBlock(
      (transp_conv_init): Convolution(
        (conv): ConvTranspose3d(768, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
      )
      (blocks): ModuleList(
        (0): Convolution(
          (conv): ConvTranspose3d(64, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
        )
      )
    )
    (encoder4): UnetrPrUpBlock(
      (transp_conv_init): Convolution(
        (conv): ConvTranspose3d(768, 128, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
      )
      (blocks): ModuleList()
    )
    (decoder5): UnetrUpBlock(
      (transp_conv): Convolution(
        (conv): ConvTranspose3d(768, 128, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
      )
      (conv_block): UnetResBlock(
        (conv1): Convolution(
          (conv): Conv3d(256, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        )
        (conv2): Convolution(
          (conv): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        )
        (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
        (norm1): InstanceNorm3d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (norm2): InstanceNorm3d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (conv3): Convolution(
          (conv): Conv3d(256, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        )
        (norm3): InstanceNorm3d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    )
    (decoder4): UnetrUpBlock(
      (transp_conv): Convolution(
        (conv): ConvTranspose3d(128, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
      )
      (conv_block): UnetResBlock(
        (conv1): Convolution(
          (conv): Conv3d(128, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        )
        (conv2): Convolution(
          (conv): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        )
        (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
        (norm1): InstanceNorm3d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (norm2): InstanceNorm3d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (conv3): Convolution(
          (conv): Conv3d(128, 64, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        )
        (norm3): InstanceNorm3d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    )
    (decoder3): UnetrUpBlock(
      (transp_conv): Convolution(
        (conv): ConvTranspose3d(64, 32, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
      )
      (conv_block): UnetResBlock(
        (conv1): Convolution(
          (conv): Conv3d(64, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        )
        (conv2): Convolution(
          (conv): Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        )
        (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
        (norm1): InstanceNorm3d(32, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (norm2): InstanceNorm3d(32, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (conv3): Convolution(
          (conv): Conv3d(64, 32, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        )
        (norm3): InstanceNorm3d(32, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    )
    (decoder2): UnetrUpBlock(
      (transp_conv): Convolution(
        (conv): ConvTranspose3d(32, 16, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
      )
      (conv_block): UnetResBlock(
        (conv1): Convolution(
          (conv): Conv3d(32, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        )
        (conv2): Convolution(
          (conv): Conv3d(16, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        )
        (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
        (norm1): InstanceNorm3d(16, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (norm2): InstanceNorm3d(16, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (conv3): Convolution(
          (conv): Conv3d(32, 16, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        )
        (norm3): InstanceNorm3d(16, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    )
    (out): UnetOutBlock(
      (conv): Convolution(
        (conv): Conv3d(16, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1))
      )
    )
  )
)


'''
if __name__=='__main__':
    if True:
        from torchsummary import summary
        # from torchviz import make_dot
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = UNETR(in_channels=4, out_channels=1, img_size=(160,192,128), norm_name='instance').to(device)
        net = nn.DataParallel(net.to(device), device_ids=[0,1,2,3],output_device=0)
        print(net)
