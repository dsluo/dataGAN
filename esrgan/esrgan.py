import cv2
import numpy as np
import torch

from esrgan import architecture as arch


class ESRGAN:

    def __init__(self, model_path, device='cpu', upscale=4):
        self.device = torch.device(device)
        model = arch.RRDB_Net(3, 3, 64, 23, gc=32, upscale=upscale, norm_type=None, act_type='leakyrelu',
                              mode='CNA', res_scale=1, upsample_mode='upconv')
        model.load_state_dict(torch.load(model_path), strict=True)
        model.eval()
        for k, v in model.named_parameters():
            v.requires_grad = False
        self.model = model.to(device)

    def upscale(self, input, output):
        # read image and convert to 0-1 scale
        img = cv2.imread(input, cv2.IMREAD_COLOR)
        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)
        img_LR = img_LR.to(self.device)

        # neural network magic
        result = self.model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        result = np.transpose(result[[2, 1, 0], :, :], (1, 2, 0))
        result = (result * 255.0).round()
        cv2.imwrite(output, result)
