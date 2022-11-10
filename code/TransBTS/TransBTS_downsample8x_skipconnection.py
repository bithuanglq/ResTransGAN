import torch
import torch.nn as nn
import os
# os.chdir('/home/hlq/Project/BraTS/code')
print('Current workspace:\t',os.getcwd())

if True:
    from TransBTS.Transformer import TransformerModel
    from TransBTS.PositionalEncoding import FixedPositionalEncoding,LearnedPositionalEncoding
    from TransBTS.Unet_skipconnection import Unet
else:
    from Transformer import TransformerModel
    from PositionalEncoding import FixedPositionalEncoding,LearnedPositionalEncoding
    from Unet_skipconnection import Unet
from typing import Tuple



class EnBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EnBlock, self).__init__()

        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        x3 = self.relu2(x2)

        return x2 + x3




class DeUp_Cat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeUp_Cat, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=2, stride=2)
        self.conv3 = nn.Conv3d(out_channels*2, out_channels, kernel_size=1)

    def forward(self, x, prev):
        '''
            x: (N, C, x, y, z)
        '''
        x1 = self.conv1(x)
        y = self.conv2(x1)
        # y = y + prev
        y = torch.cat((prev, y), dim=1)
        y = self.conv3(y)
        return y

class DeBlock(nn.Module):
    def __init__(self, in_channels):
        super(DeBlock, self).__init__()

        self.bn1 = nn.BatchNorm3d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(in_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        x1 = x1 + x

        return x1




from unet import Conv3x3, UnetUp

class My_TransBTS(nn.Module):
    def __init__(
        self, 
        in_channels:int,
        n_classes:int,
        embedding_dim=512,
        num_heads=8,        # multi-head self-attention
        num_layers=4,       # nums of transformer blocks
        dropout_rate=0.1,
        positional_encoding_type="learned",
    ):
        super(My_TransBTS, self).__init__()

        assert embedding_dim % num_heads == 0


        self.embedding_dim = embedding_dim


        filters = [64, 96, 128, 192, 256, 384, 512]
        self.filters = filters

        # downsampling
        self.encoder0 = nn.Sequential(
          Conv3x3(in_channels, filters[0], stride=1),
          Conv3x3(filters[0], filters[0])
        )
        for i in range(1, len(filters)):
          self.layer = nn.Sequential(
            Conv3x3(filters[i-1], filters[i], stride=2),
            Conv3x3(filters[i], filters[i])
          )
          setattr(self, f'encoder{i}', self.layer)



        # transformer
        self.seq_length = 3*3*3
        if positional_encoding_type == "learned":
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            )
        
        self.pe_dropout = nn.Dropout(dropout_rate)

        hidden_dim = 3*3*3
        self.transformer = TransformerModel(
            embedding_dim,
            num_layers,
            num_heads,
            hidden_dim,
            dropout_rate=dropout_rate,
            attn_dropout_rate=dropout_rate
        )

        self.pre_head_ln = nn.LayerNorm(embedding_dim)
        self.conv_x = nn.Conv3d(
            512,
            embedding_dim,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn = nn.BatchNorm3d(512)
        self.relu = nn.ReLU(inplace=True)

        # upsampling
        for i in reversed(range(len(filters) - 1)):
          self.layer = UnetUp(in_channels=filters[i+1], out_channels=filters[i])
          setattr(self, f'decoder{i}', self.layer)


        # for (160, 192, 160)
        self.decoder5 = nn.Sequential(
            nn.Upsample((5,6,5), mode='trilinear'),
            Conv3x3(512, 384),
            Conv3x3(384, 384)
        )

        # final conv (without any concat)
        self.final0 = nn.Conv3d(filters[0], n_classes, 1, 1, 0)
        self.final1 = nn.Conv3d(filters[1], n_classes, 1, 1, 0)
        self.final2 = nn.Conv3d(filters[2], n_classes, 1, 1, 0)
   

    def encode(self, x):
        e0 = self.encoder0(x)
        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        e6 = self.encoder6(e5)      # (1, 512, 3, 3, 3)

        x = self.bn(e6)
        x = self.relu(x)
        x = self.conv_x(x)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = x.view(x.size(0), -1, self.embedding_dim)

        x = self.position_encoding(x)
        x = self.pe_dropout(x)

        # apply transformer
        x, intmd_x = self.transformer(x)
        x = self.pre_head_ln(x)
        


        return e0, e1, e2, e3, e4, e6, x


    def decode(self, e0, e1, e2, e3, e4, e6, x, intmd_layers=[1, 2, 3, 4]):

        assert intmd_layers is not None, "pass the intermediate layers for MLA"

        x = self._reshape_output(x)
        # residual
        x = x + e6
        u5 = self.decoder5(x)
        u4 = self.decoder4(u5, e4)
        u3 = self.decoder3(u4, e3)
        u2 = self.decoder2(u3, e2)
        u1 = self.decoder1(u2, e1)
        u0 = self.decoder0(u1, e0)

        out0, out1, out2 = self.final0(u0), self.final1(u1), self.final2(u2)
        return out0, out1, out2


    def _reshape_output(self, x):
        x = x.view(
            x.size(0),
            3,
            3,
            3,
            self.embedding_dim,
        )
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        return x


    def forward(self, x):
        decoder_output = self.decode(*self.encode(x))
        return decoder_output 

#====================================
# ResTransGAN_without_deepsupervision
#====================================


class My_TransBTS_one_output(nn.Module):
    def __init__(
        self, 
        in_channels:int,
        n_classes:int,
        embedding_dim=512,
        num_heads=8,        # multi-head self-attention
        num_layers=4,       # nums of transformer blocks
        dropout_rate=0.1,
        positional_encoding_type="learned",
    ):
        super(My_TransBTS_one_output, self).__init__()

        assert embedding_dim % num_heads == 0


        self.embedding_dim = embedding_dim


        filters = [64, 96, 128, 192, 256, 384, 512]
        self.filters = filters

        # downsampling
        self.encoder0 = nn.Sequential(
          Conv3x3(in_channels, filters[0], stride=1),
          Conv3x3(filters[0], filters[0])
        )
        for i in range(1, len(filters)):
          self.layer = nn.Sequential(
            Conv3x3(filters[i-1], filters[i], stride=2),
            Conv3x3(filters[i], filters[i])
          )
          setattr(self, f'encoder{i}', self.layer)



        # transformer
        self.seq_length = 3*3*3
        if positional_encoding_type == "learned":
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            )
        
        self.pe_dropout = nn.Dropout(dropout_rate)

        hidden_dim = 3*3*3
        self.transformer = TransformerModel(
            embedding_dim,
            num_layers,
            num_heads,
            hidden_dim,
            dropout_rate=dropout_rate,
            attn_dropout_rate=dropout_rate
        )

        self.pre_head_ln = nn.LayerNorm(embedding_dim)
        self.conv_x = nn.Conv3d(
            512,
            embedding_dim,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn = nn.BatchNorm3d(512)
        self.relu = nn.ReLU(inplace=True)

        # upsampling
        for i in reversed(range(len(filters) - 1)):
          self.layer = UnetUp(in_channels=filters[i+1], out_channels=filters[i])
          setattr(self, f'decoder{i}', self.layer)


        # for (160, 192, 160)
        self.decoder5 = nn.Sequential(
            nn.Upsample((5,6,5), mode='trilinear'),
            Conv3x3(512, 384),
            Conv3x3(384, 384)
        )

        # final conv (without any concat)
        self.final0 = nn.Conv3d(filters[0], n_classes, 1, 1, 0)

   

    def encode(self, x):
        e0 = self.encoder0(x)
        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        e6 = self.encoder6(e5)      # (1, 512, 3, 3, 3)

        x = self.bn(e6)
        x = self.relu(x)
        x = self.conv_x(x)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = x.view(x.size(0), -1, self.embedding_dim)

        x = self.position_encoding(x)
        x = self.pe_dropout(x)

        # apply transformer
        x, intmd_x = self.transformer(x)
        x = self.pre_head_ln(x)
        


        return e0, e1, e2, e3, e4, e6, x



    def _reshape_output(self, x):
        x = x.view(
            x.size(0),
            3,
            3,
            3,
            self.embedding_dim,
        )
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        return x


    def decode(self, e0, e1, e2, e3, e4, e6, x, intmd_layers=[1, 2, 3, 4]):

        assert intmd_layers is not None, "pass the intermediate layers for MLA"

        x = self._reshape_output(x)
        # residual
        x = x + e6
        u5 = self.decoder5(x)
        u4 = self.decoder4(u5, e4)
        u3 = self.decoder3(u4, e3)
        u2 = self.decoder2(u3, e2)
        u1 = self.decoder1(u2, e1)
        u0 = self.decoder0(u1, e0)

        out = self.final0(u0)
        return out

    def forward(self, x):
        decoder_output = self.decode(*self.encode(x))
        return decoder_output 


#================================
# ResTransGAN_without_resnet
#================================

class My_TransBTS_withoout_resnet(nn.Module):
    def __init__(
        self, 
        in_channels:int,
        n_classes:int,
        embedding_dim=512,
        num_heads=8,        # multi-head self-attention
        num_layers=4,       # nums of transformer blocks
        dropout_rate=0.1,
        positional_encoding_type="learned",
    ):
        super(My_TransBTS_withoout_resnet, self).__init__()

        assert embedding_dim % num_heads == 0


        self.embedding_dim = embedding_dim


        filters = [64, 96, 128, 192, 256, 384, 512]
        self.filters = filters

        # downsampling
        self.encoder0 = nn.Sequential(
          Conv3x3(in_channels, filters[0], stride=1),
          Conv3x3(filters[0], filters[0])
        )
        for i in range(1, len(filters)):
          self.layer = nn.Sequential(
            Conv3x3(filters[i-1], filters[i], stride=2),
            Conv3x3(filters[i], filters[i])
          )
          setattr(self, f'encoder{i}', self.layer)



        # transformer
        self.seq_length = 3*3*3
        if positional_encoding_type == "learned":
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            )
        
        self.pe_dropout = nn.Dropout(dropout_rate)

        hidden_dim = 3*3*3
        self.transformer = TransformerModel(
            embedding_dim,
            num_layers,
            num_heads,
            hidden_dim,
            dropout_rate=dropout_rate,
            attn_dropout_rate=dropout_rate
        )

        self.pre_head_ln = nn.LayerNorm(embedding_dim)
        self.conv_x = nn.Conv3d(
            512,
            embedding_dim,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn = nn.BatchNorm3d(512)
        self.relu = nn.ReLU(inplace=True)

        # upsampling
        for i in reversed(range(len(filters) - 1)):
          self.layer = UnetUp(in_channels=filters[i+1], out_channels=filters[i])
          setattr(self, f'decoder{i}', self.layer)


        # for (160, 192, 160)
        self.decoder5 = nn.Sequential(
            nn.Upsample((5,6,5), mode='trilinear'),
            Conv3x3(512, 384),
            Conv3x3(384, 384)
        )

        # final conv (without any concat)
        self.final0 = nn.Conv3d(filters[0], n_classes, 1, 1, 0)
        self.final1 = nn.Conv3d(filters[1], n_classes, 1, 1, 0)
        self.final2 = nn.Conv3d(filters[2], n_classes, 1, 1, 0)
   

    def encode(self, x):
        e0 = self.encoder0(x)
        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        e6 = self.encoder6(e5)      # (1, 512, 3, 3, 3)

        x = self.bn(e6)
        x = self.relu(x)
        x = self.conv_x(x)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = x.view(x.size(0), -1, self.embedding_dim)

        x = self.position_encoding(x)
        x = self.pe_dropout(x)

        # apply transformer
        x, intmd_x = self.transformer(x)
        x = self.pre_head_ln(x)
        


        return e0, e1, e2, e3, e4, e6, x


    def decode(self, e0, e1, e2, e3, e4, e6, x, intmd_layers=[1, 2, 3, 4]):

        assert intmd_layers is not None, "pass the intermediate layers for MLA"

        x = self._reshape_output(x)
        # residual
        # x = x + e6
        u5 = self.decoder5(x)
        u4 = self.decoder4(u5, e4)
        u3 = self.decoder3(u4, e3)
        u2 = self.decoder2(u3, e2)
        u1 = self.decoder1(u2, e1)
        u0 = self.decoder0(u1, e0)

        out0, out1, out2 = self.final0(u0), self.final1(u1), self.final2(u2)
        return out0, out1, out2


    def _reshape_output(self, x):
        x = x.view(
            x.size(0),
            3,
            3,
            3,
            self.embedding_dim,
        )
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        return x


    def forward(self, x):
        decoder_output = self.decode(*self.encode(x))
        return decoder_output 



#================================
# ResTransGAN_without_trans
#================================

class My_TransBTS_without_trans(nn.Module):
    def __init__(
        self, 
        in_channels:int,
        n_classes:int,
        embedding_dim=512,
        num_heads=8,        # multi-head self-attention
        num_layers=4,       # nums of transformer blocks
        dropout_rate=0.1,
        positional_encoding_type="learned",
    ):
        super(My_TransBTS_without_trans, self).__init__()

        assert embedding_dim % num_heads == 0


        self.embedding_dim = embedding_dim


        filters = [64, 96, 128, 192, 256, 384, 512]
        self.filters = filters

        # downsampling
        self.encoder0 = nn.Sequential(
          Conv3x3(in_channels, filters[0], stride=1),
          Conv3x3(filters[0], filters[0])
        )
        for i in range(1, len(filters)):
          self.layer = nn.Sequential(
            Conv3x3(filters[i-1], filters[i], stride=2),
            Conv3x3(filters[i], filters[i])
          )
          setattr(self, f'encoder{i}', self.layer)



        # upsampling
        for i in reversed(range(len(filters) - 1)):
          self.layer = UnetUp(in_channels=filters[i+1], out_channels=filters[i])
          setattr(self, f'decoder{i}', self.layer)


        # for (160, 192, 160)
        self.decoder5 = nn.Sequential(
            nn.Upsample((5,6,5), mode='trilinear'),
            Conv3x3(512, 384),
            Conv3x3(384, 384)
        )

        # final conv (without any concat)
        self.final0 = nn.Conv3d(filters[0], n_classes, 1, 1, 0)
        self.final1 = nn.Conv3d(filters[1], n_classes, 1, 1, 0)
        self.final2 = nn.Conv3d(filters[2], n_classes, 1, 1, 0)
   

    def encode(self, x):
        e0 = self.encoder0(x)
        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        e6 = self.encoder6(e5)      # (1, 512, 3, 3, 3)


        return e0, e1, e2, e3, e4, e6


    def decode(self, e0, e1, e2, e3, e4, e6,  intmd_layers=[1, 2, 3, 4]):

        assert intmd_layers is not None, "pass the intermediate layers for MLA"

        x = e6
        u5 = self.decoder5(x)
        u4 = self.decoder4(u5, e4)
        u3 = self.decoder3(u4, e3)
        u2 = self.decoder2(u3, e2)
        u1 = self.decoder1(u2, e1)
        u0 = self.decoder0(u1, e0)

        out0, out1, out2 = self.final0(u0), self.final1(u1), self.final2(u2)
        return out0, out1, out2


    def _reshape_output(self, x):
        x = x.view(
            x.size(0),
            3,
            3,
            3,
            self.embedding_dim,
        )
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        return x


    def forward(self, x):
        decoder_output = self.decode(*self.encode(x))
        return decoder_output 




if __name__ == '__main__':
    if False:
        with torch.no_grad():
            import os
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            cuda0 = torch.device('cuda:0')
            x = torch.rand((4, 4, 160, 192, 160), device=cuda0)
            model = My_TransBTS(img_size=(160, 192, 160), batch_size=4, n_classes=3)
            model.cuda()
            y = model(x)
            print(y.shape)      # torch.Size([4, 3, 160, 192, 160])

    if False:
        cuda0 = torch.device('cuda:0')
        k = torch.rand((1, 8, 4096, 64), device=cuda0)
        print(k.transpose(-2,-1).shape)             # torch.Size([1, 8, 64, 4096])  transpose 用于交换两个指定维度
        print(k.softmax(dim=-1).shape)              # torch.Size([1, 8, 4096, 64])
   