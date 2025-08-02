import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

# ─── PIX2PIX UNET GENERATOR ────────────────────────────────────────
# Copied from Pix2Pix repo (abridged UNet for inference)
class UnetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, num_downs=8, ngf=64):
        super(UnetGenerator, self).__init__()
        from torch.nn import Conv2d, ConvTranspose2d, BatchNorm2d, ReLU, Tanh
        # Encoder
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, innermost=True)
        for _ in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, submodule=unet_block)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, submodule=unet_block)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, submodule=unet_block)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, submodule=unet_block)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True)

    def forward(self, input):
        return self.model(input)

class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False):
        super(UnetSkipConnectionBlock, self).__init__()
        from torch.nn import Conv2d, ConvTranspose2d, BatchNorm2d, ReLU, Tanh, LeakyReLU
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc
        downconv = Conv2d(input_nc, inner_nc, kernel_size=4,
                          stride=2, padding=1, bias=False)
        downrelu = LeakyReLU(0.2, True)
        downnorm = BatchNorm2d(inner_nc)
        uprelu = ReLU(True)
        upnorm = BatchNorm2d(outer_nc)

        if outermost:
            upconv = ConvTranspose2d(inner_nc * 2, outer_nc,
                                     kernel_size=4, stride=2,
                                     padding=1)
            down = [downconv]
            up = [uprelu, upconv, Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = ConvTranspose2d(inner_nc, outer_nc,
                                     kernel_size=4, stride=2,
                                     padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = ConvTranspose2d(inner_nc * 2, outer_nc,
                                     kernel_size=4, stride=2,
                                     padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

# ─── INFERENCE FUNCTION ─────────────────────────────────────────────
def run_inference(lq_image_pil, model_path="generator.pth", device="cpu"):
    device = torch.device(device)

    # 1. Transform input
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),                   # [0,1]
        transforms.Normalize((0.5, 0.5, 0.5),     # [-1,1]
                             (0.5, 0.5, 0.5))
    ])
    input_tensor = transform(lq_image_pil).unsqueeze(0).to(device)  # [1, 3, 512, 512]

    # 2. Load model
    netG = UnetGenerator()
    netG.load_state_dict(torch.load(model_path, map_location=device))
    netG.eval().to(device)

    # 3. Run inference
    with torch.no_grad():
        fake_tensor = netG(input_tensor)[0]  # [3, 512, 512]

    # 4. Postprocess
    fake_img = (fake_tensor.cpu().numpy().transpose(1, 2, 0) + 1) * 127.5  # [0,255]
    fake_img = np.clip(fake_img, 0, 255).astype(np.uint8)
    fake_pil = Image.fromarray(fake_img)

    return fake_pil
