# Original UNETR++ source code is copied from the official UNETR++ repository.
# https://github.com/Amshaker/unetr_plus_plus/blob/main/unetr_pp/network_architecture/tumor/unetr_pp_tumor.py
from torch import nn
from typing import Tuple, Union
import torch
from torch.nn.functional import interpolate
from monai.networks.blocks.dynunet_block import UnetOutBlock, UnetResBlock
from unetr_pp.model_components import UnetrPPEncoder, UnetrUpBlock


class UNETR_PP_AE(nn.Module):
    """
    UNETR++ based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        feature_size: int = 16,
        hidden_size: int = 256,
        num_heads: int = 4,
        pos_embed: str = "perceptron",
        norm_name: Union[Tuple, str] = "instance",
        dropout_rate: float = 0.0,
        depths=None,
        dims=None,
        conv_op=nn.Conv3d,
        do_ds=True,
        recon=True,
    ) -> None:
        super().__init__()
        self.recon = recon
        if depths is None:
            depths = [3, 3, 3, 3]
        self.do_ds = do_ds
        self.conv_op = conv_op
        self.num_classes = out_channels
        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.feat_size = (3, 3, 3,)
        self.hidden_size = hidden_size

        self.unetr_pp_encoder = UnetrPPEncoder(
            input_size=[24 * 24 * 24, 12 * 12 * 12, 6 * 6 * 6, 3 * 3 * 3],
            dims=dims, depths=depths, num_heads=num_heads
        )

        self.encoder1 = UnetResBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 16,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            # out_size=8*8*8,
            out_size=6*6*6,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            # out_size=16*16*16,
            out_size=12*12*12,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            # out_size=32*32*32,
            out_size=24*24*24,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=(4, 4, 4),
            norm_name=norm_name,
            # out_size=128*128*128,
            out_size=96*96*96,
            conv_decoder=True,
        )
        self.out1 = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)
        if self.do_ds:
            self.out2 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 2, out_channels=out_channels)
            self.out3 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 4, out_channels=out_channels)
        
        # Autoencoder part
        self.ae_decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 16,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            # out_size=8*8*8,
            out_size=6*6*6,
        )
        self.ae_decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            # out_size=16*16*16,
            out_size=12*12*12,
        )
        self.ae_decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            # out_size=32*32*32,
            out_size=24*24*24,
        )
        self.ae_decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=(4, 4, 4),
            norm_name=norm_name,
            # out_size=128*128*128,
            out_size=96*96*96,
            conv_decoder=True,
        )
        self.ae_out1 = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=in_channels)
        if self.do_ds:
            self.ae_out2 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 2, out_channels=in_channels)
            self.ae_out3 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 4, out_channels=in_channels)

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in):
        recon = self.recon
        
        x_output, hidden_states = self.unetr_pp_encoder(x_in)
        convBlock = self.encoder1(x_in)

        # Four encoders
        enc1 = hidden_states[0]
        enc2 = hidden_states[1]
        enc3 = hidden_states[2]
        enc4 = hidden_states[3]

        # Four decoders
        dec4 = self.proj_feat(enc4, self.hidden_size, self.feat_size)
        dec3 = self.decoder5(dec4, enc3)
        dec2 = self.decoder4(dec3, enc2)
        dec1 = self.decoder3(dec2, enc1)

        out = self.decoder2(dec1, convBlock)
        if self.training and self.do_ds:
            real_out = self.out1(out)
            logits = torch.stack([
                real_out,
                interpolate(self.out2(dec1), real_out.shape[2:]),
                interpolate(self.out3(dec2), real_out.shape[2:])],
            dim=1)
        else:
            logits = self.out1(out)
        if recon:
            ae_dec3 = self.ae_decoder5(dec4, enc3)
            ae_dec2 = self.ae_decoder4(ae_dec3, enc2)
            ae_dec1 = self.ae_decoder3(ae_dec2, enc1)
            ae_out = self.ae_decoder2(ae_dec1, convBlock)
            if self.training and self.do_ds:
                real_out = self.ae_out1(ae_out)
                ae_logits = torch.stack([
                    real_out, 
                    interpolate(self.ae_out2(ae_dec1), real_out.shape[2:]),
                    interpolate(self.ae_out3(ae_dec2), real_out.shape[2:])],
                dim=1)
            else:
                ae_logits = self.ae_out1(ae_out)
            return logits, ae_logits, ae_out
        else:
            return logits
