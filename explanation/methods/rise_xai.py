import sys
import cv2
from matplotlib import pyplot as plt
sys.path.append('../../src')
from model.frame import FrameModel
from torchvision.transforms import v2
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#code from: https://github.com/yiskw713/RISE
class RISE(nn.Module):
    def __init__(self, model, n_masks=10000, p1=0.1, input_size=(224, 224), initial_mask_size=(7, 7), n_batch=128, mask_path=None):
        super().__init__()
        self.model = model
        self.n_masks = n_masks
        self.p1 = p1
        self.input_size = input_size
        self.initial_mask_size = initial_mask_size
        self.n_batch = n_batch

        if mask_path is not None:
            self.masks = self.load_masks(mask_path)
        else:
            self.masks = self.generate_masks()

    def generate_masks(self):
        # cell size in the upsampled mask
        Ch = np.ceil(self.input_size[0] / self.initial_mask_size[0])
        Cw = np.ceil(self.input_size[1] / self.initial_mask_size[1])

        resize_h = int((self.initial_mask_size[0] + 1) * Ch)
        resize_w = int((self.initial_mask_size[1] + 1) * Cw)

        masks = []

        for _ in range(self.n_masks):
            # generate binary mask
            binary_mask = torch.randn(
                1, 1, self.initial_mask_size[0], self.initial_mask_size[1])
            binary_mask = (binary_mask < self.p1).float()

            # upsampling mask
            mask = F.interpolate(
                binary_mask, (resize_h, resize_w), mode='bilinear', align_corners=False)

            # random cropping
            i = np.random.randint(0, Ch)
            j = np.random.randint(0, Cw)
            mask = mask[:, :, i:i+self.input_size[0], j:j+self.input_size[1]]

            masks.append(mask)

        masks = torch.cat(masks, dim=0)   # (N_masks, 1, H, W)

        return masks

    def load_masks(self, filepath):
        masks = torch.load(filepath)
        return masks

    def save_masks(self, filepath):
        torch.save(self.masks, filepath)

    def forward(self, x):
        # x: input image. (1, 3, H, W)
        device = x.device

        # keep probabilities of each class
        probs = []
        # shape (n_masks, 3, H, W)
        masked_x = torch.mul(self.masks, x.to('cpu').data)

        for i in range(0, self.n_masks, self.n_batch):
            input = masked_x[i:min(i + self.n_batch, self.n_masks)].to(device)
            out = self.model(input)
            probs.append(torch.softmax(out, dim=1).to('cpu').data)

        probs = torch.cat(probs)    # shape => (n_masks, n_classes)
        n_classes = probs.shape[1]

        # caluculate saliency map using probability scores as weights
        saliency = torch.matmul(
            probs.data.transpose(0, 1),
            self.masks.view(self.n_masks, -1)
        )
        saliency = saliency.view(
            (n_classes, self.input_size[0], self.input_size[1]))
        saliency = saliency / (self.n_masks * self.p1)

        # normalize
        m, _ = torch.min(saliency.view(n_classes, -1), dim=1)
        saliency -= m.view(n_classes, 1, 1)
        M, _ = torch.max(saliency.view(n_classes, -1), dim=1)
        saliency /= M.view(n_classes, 1, 1)
        return saliency.data

def visualize_explanation(img, cam):
    """
    Synthesize an image with CAM to make a result image.
    Args:
        img: (Tensor) shape => (1, 3, H, W)
        cam: (Tensor) shape => (1, 1, H', W')
    Return:
        synthesized image (Tensor): shape =>(1, 3, H, W)
    """

    _, _, H, W = img.shape
    cam = F.interpolate(cam, size=(H, W), mode='bilinear', align_corners=False)
    cam = 255 * cam.squeeze()
    heatmap = cv2.applyColorMap(np.uint8(cam), cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap.transpose(2, 0, 1))
    heatmap = heatmap.float() / 255
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b])

    result = heatmap + img.cpu()
    result = result.div(result.max())

    return result

def explain(inference_image,visualize_image,label,model,visualize=True):

    input_size = tuple(visualize_image.shape[2:])

    #Compute the explanation
    wrapped_model = RISE(model, n_masks=4000, input_size=input_size)
    with torch.no_grad():
        saliency = wrapped_model(inference_image.unsqueeze(0).to(model.device))

    #Get the saliency map
    saliency = saliency[label]

    #If selected visualize the result
    if(visualize):
        saliency_visualize = saliency.view(1, 1, *input_size)
        heatmap = visualize_explanation(visualize_image, saliency_visualize)
        hm = (heatmap.squeeze().numpy().transpose(1, 2, 0))
        plt.imshow(hm)
        plt.show()

    #Return the saliency map
    return saliency.numpy()


if __name__ == "__main__":
    #Load the model
    rs_size = 224
    model = FrameModel.load_from_checkpoint("../../model/checkpoint/ff_attribution.ckpt", map_location='cuda').eval()

    #Create the transforms for inference and visualization purposes
    interpolation = 3
    inference_transforms = v2.Compose([
        v2.ToImage(),
        v2.Resize(rs_size, interpolation=interpolation, antialias=False),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    visualize_transforms = v2.Compose([
        v2.ToImage(),
        v2.Resize(rs_size, interpolation=interpolation, antialias=False),
        v2.ToDtype(torch.float32, scale=True),
    ])

    #Open the image
    image = Image.open('test.jpg')
    #Apply the transformations
    inference_image = inference_transforms(image)
    visualize_image = visualize_transforms(image).unsqueeze(0)
    #Select the explanation label
    label = 0

    #Call the explanation method
    explain(inference_image,visualize_image,label,model)
