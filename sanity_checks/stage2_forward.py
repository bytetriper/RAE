import os
import sys
import torch
from torch import nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from stage1 import RAE
from stage2.model import STAGE2_ARCHS, DiTwDDTHead
from PIL import Image
from torchvision import transforms
def get_default_img():
    # Placeholder function to load a default image
    img_path = "assets/pixabay_cat.png"
    img = Image.open(img_path).resize((256, 256)).convert("RGB")
    img = transforms.ToTensor()(img).unsqueeze(0)  # Add batch dimension
    return img

def stage1_instance() -> RAE:
    model = RAE(
        encoder_cls='Dinov2withNorm',
        encoder_config_path='models/encoders/dinov2/wReg_base',
        encoder_input_size=224,
        encoder_params={'dinov2_path': 'models/encoders/dinov2/wReg_base', 'normalize': True},
        decoder_config_path='configs/decoder/ViTXL',
        pretrained_decoder_path='models/decoders/dinov2/wReg_base/ViTXL_n08/model.pt',
        noise_tau=0.,
        reshape_to_2d=True,   
        normalization_stat_path='models/stats/dinov2/wReg_base/imagenet1k/stat.pt',
    )
    return model

def stage2_instance() -> DiTwDDTHead:
    model = STAGE2_ARCHS['DDTXL'](
        token_dim=768,  # Assuming the latent token dimension from stage 1
        input_size=16,  # Assuming the latent size from stage 1 is 32x32 for 256x256 input
    )
    return model

def test_stage2_forward(on_cuda: bool = True):
    if on_cuda and torch.cuda.is_available():
        torch.cuda.set_device(0)
        print("Using GPU")
        device = torch.device("cuda")
    else:
        print("Using CPU")
        device = torch.device("cpu")
    stage1_model = stage1_instance().to(device)
    stage2_model = stage2_instance().to(device)
    stage1_model.eval()
    stage2_model.eval()
    x = get_default_img().to(device)  # Use the default image
    with torch.no_grad():
        z = stage1_model.encode(x)
        t = torch.linspace(0.05, 0.998, steps=100).to(device)  # Dummy timestep
        #y = torch.linspace(0, 999, steps=100).long().to(device)  # Dummy class labels
        #y is int
        y = torch.zeros(100, dtype=torch.long).to(device)  # Dummy class labels
        t = t.repeat(z.size(0))
        y = y.repeat(z.size(0))
        z = z.repeat_interleave(100, dim=0)  # Repeat latent for each timestep
        x_end = torch.randn_like(z)  # Random noise as starting point
        print("z shape:", z.shape, "t shape:", t.shape, "y shape:", y.shape, "x_end shape:", x_end.shape)
        zt = (1 - t.view(-1, 1, 1, 1)) * z + t.view(-1, 1, 1, 1) * x_end  # Add noise based on timestep
        out = stage2_model(zt, t, y)
        eps_pred = x_end - z
        mse_loss = ((out - eps_pred) ** 2).mean()
    print("Stage 2 forward pass successful with output shape:", out.shape, "and latent shape:", z.shape)
    print(f"Dummy MSE Loss: {mse_loss.item():.6f}")

if __name__ == "__main__":
    test_stage2_forward(on_cuda=True)