import os
import sys
import torch
from torch import nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from stage1 import RAE

def stage1_instance() -> RAE:
    model = RAE(
        encoder_cls='Dinov2withNorm',
        encoder_config_path='models/encoders/dinov2/wReg_base',
        encoder_input_size=224,
        encoder_params={'dinov2_path': 'models/encoders/dinov2/wReg_base', 'normalize': True},
        decoder_config_path='configs/decoder/ViTXL',
        pretrained_decoder_path='models/decoders/dinov2/wReg_base/ViTXL_n08/model.pt',
        noise_tau=0.8,
        reshape_to_2d=True,   
    )
    return model


def test_stage1_forward():
    model = stage1_instance()
    model.eval()
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        z = model.encode(x)
        out = model.decode(z)
    print("Stage 1 forward pass successful with output shape:", out.shape, "and latent shape:", z.shape)
    
if __name__ == "__main__":
    test_stage1_forward()
    