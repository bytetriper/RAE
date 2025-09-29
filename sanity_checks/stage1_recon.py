import os
import sys
import torch
from torch import nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from stage1 import RAE
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


def test_stage1_recon(on_cuda: bool = True):
    if on_cuda and torch.cuda.is_available():
        torch.cuda.set_device(0)
        print("Using GPU")
        device = torch.device("cuda")
    else:
        print("Using CPU")
        device = torch.device("cpu")
    model = stage1_instance().to(device)
    model.eval()
    x = get_default_img().to(device)  # Use the default image
    with torch.no_grad():
        z = model.encode(x)
        out = model.decode(z)
    print("Stage 1 forward pass successful with output shape:", out.shape, "and latent shape:", z.shape)
    # L1, L2 and PSNR
    l1_loss = (out.clamp(0, 1) - x.clamp(0, 1)).abs().mean()
    l2_loss = ((out.clamp(0, 1) - x.clamp(0, 1)) ** 2).mean()
    psnr = 10 * torch.log10(1 / l2_loss)
    print(f"L1 Loss: {l1_loss.item():.6f}, MSE Loss: {l2_loss.item():.6f}, PSNR: {psnr.item():.2f} dB")
    # save output image
    out_img = transforms.ToPILImage()(out.squeeze(0).clamp(0, 1))
    out_img.save("assets/pixabay_cat_recon.png")
if __name__ == "__main__":
    test_stage1_recon(on_cuda=True)
    