import argparse
import torch
import cv2

import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image

from models import SRCNN, SIMPLENET, VDSR
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, getpsnr, ssim


def reconstruct(image, model):
    image = np.array(image).astype(np.float32)
    ycbcr = convert_rgb_to_ycbcr(image)

    y = ycbcr[..., 0]
    y /= 255.
    y = torch.from_numpy(y).to(device)
    y = y.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        preds = model(y).clamp(0.0, 1.0)
    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    output = pil_image.fromarray(output)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--weights-file', type=str, default="outputs/SIMPLENET/best.pth")
    parser.add_argument('--image-file', type=str, default="test/test.png")
    parser.add_argument('--scale', type=int, default=3)

    parser.add_argument("--model",
                        type=str,
                        required=True)

    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if args.model == "SRCNN":
        model = SRCNN().to(device)
        args.weights_file = "outputs/SRCNN/best.pth"
    elif args.model == "SIMPLENET":
        model = SIMPLENET().to(device)
        args.weights_file = "outputs/SIMPLENET/best.pth"
    elif args.model == "VDSR":
        model = VDSR().to(device)
        args.weights_file = "outputs/VDSR/best.pth"
    else:
        raise ValueError("[ERROR] invalid model name given. Available models:\n\tSRCNN\n\tSIMPLENET\n\tVDSR")

    # model = SIMPLENET().to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()

    image = pil_image.open(args.image_file).convert('RGB')

    image_width = (image.width // args.scale) * args.scale
    image_height = (image.height // args.scale) * args.scale
    image = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    image = image.resize((image.width // args.scale, image.height // args.scale), resample=pil_image.BICUBIC)
    image.save(args.image_file.replace('.', '_down.'))
    image = image.resize((image.width * args.scale, image.height * args.scale), resample=pil_image.BICUBIC)
    # image = cv2.imread(args.image_file)
    # image = down_sample(image)

    image.save(args.image_file.replace('.', '_BICUBIC.'))
    # cv2.imwrite("test/test1.png",image)

    output = reconstruct(image, model)

    output.save(args.image_file.replace('.', '_RECONSTRUCT.'))

    pic_down = cv2.imread("test/test_down.png")
    pic0 = cv2.imread("test/test.png")
    pic1 = cv2.imread("test/test_BICUBIC.png")
    pic2 = cv2.imread("test/test_RECONSTRUCT.png")
    psnr1 = getpsnr(pic0, pic1)
    psnr2 = getpsnr(pic0, pic2)

    ssim1 = ssim(pic0, pic1)
    ssim2 = ssim(pic0, pic2)

    print('BICUBIC')
    print('psnr: {}'.format(psnr1))
    print('ssim: {}\n'.format(ssim1))
    print('RECONSTRUCT')
    print('psnr: {}'.format(psnr2))
    print('ssim: {}'.format(ssim2))
