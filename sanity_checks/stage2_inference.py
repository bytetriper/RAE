import os
import sys
import torch
from torch import nn
from tqdm import tqdm

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

def load_stage2_weights(model: nn.Module, path: str):
    if os.path.isfile(path):
        state_dict = torch.load(path, map_location='cpu')
        model.load_state_dict(state_dict, strict=True)
        print(f"Loaded stage 2 model weights from {path}")
    else:
        raise FileNotFoundError(f"No checkpoint found at {path}")

def single_stage2_model_inference(on_cuda: bool = True):
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
    load_stage2_weights(stage2_model, 'models/DiTs/Dinov2/wReg_base/DDTXL/stage2_model.pt')  # Update with actual path
    x = get_default_img().to(device)  # Use the default image
    import numpy as np
    sampler_timesteps = np.linspace(1/1000, 1, 50)
    shift_ratio = 1/ 0.14433756729740643
    sampler_timesteps = sampler_timesteps * shift_ratio / (1 + (shift_ratio - 1) * sampler_timesteps)
    sampler_timesteps = sampler_timesteps.tolist()
    with torch.no_grad():
        z = stage1_model.encode(x)
        x_end = torch.randn_like(z).to(device)  # Random input
        zt = x_end
        y = torch.ones(z.size(0), dtype=torch.long).to(device)*250  # Dummy class labels
        tbar = tqdm(reversed(range(1, len(sampler_timesteps))), desc='Sampling', total = len(sampler_timesteps)-1)
        for i in tbar:
            u_t = torch.tensor(sampler_timesteps[i]).to(device).repeat(z.size(0))
            u_s = torch.tensor(sampler_timesteps[i - 1]).to(device).repeat(z.size(0))
            #print("u_t:", u_t, "u_s:", u_s)
            sigma_t, sigma_s = u_t, u_s
            v_pred = stage2_model(zt, u_t, y)
            delta_t = sigma_s - sigma_t
            delta_t = delta_t.view(-1, 1, 1, 1).to(device)
            zt = zt + delta_t * v_pred
        out = zt
    x_inference = stage1_model.decode(out)
    print("Stage 2 single model inference successful with output shape:", x_inference.shape, x_inference.min(), x_inference.max())
    # save output image
    out_img = transforms.ToPILImage()(x_inference.squeeze(0).clamp(0, 1))
    out_img.save("assets/stage2_inference.png")
    print("Saved inference image to assets/stage2_inference.png")
if __name__ == "__main__":
    single_stage2_model_inference(on_cuda=True)