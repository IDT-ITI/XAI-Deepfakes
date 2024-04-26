import sys
sys.path.append('../../src')
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import v2
from model.frame import FrameModel
from sobol_attribution_method.torch_explainer import SobolAttributionMethod

#code from: https://github.com/fel-thomas/Sobol-Attribution-Method?tab=readme-ov-file
def explain(inference_image,visualize_image,label,model,visualize=True):

    #Function to visualize the explanation (taken from the method's documentation)
    def show(img, **kwargs):
        if torch.is_tensor(img):
            img = img.detach().cpu().numpy()
            img = np.moveaxis(img, 0, -1)

        img -= img.min()
        img /= img.max()
        plt.imshow(img, **kwargs)
        plt.axis('off')

    #Compute the explanation
    explainer = SobolAttributionMethod(model, grid_size=8, nb_design=32, batch_size=128)
    explanations = explainer(inference_image.unsqueeze(0).to(model.device), np.array([label]))

    #If selected visualize the result
    if(visualize):
        show(visualize_image)
        show(explanations[0], cmap='jet', alpha=0.5)
        plt.show()

    #Normalize and return the saliency map
    return (explanations[0] - explanations[0].min()) / (explanations[0].max() - explanations[0].min())

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
    inference_image=inference_transforms(image)
    visualize_image = visualize_transforms(image)
    #Select the explanation label
    label = 0

    #Call the explanation method
    explain(inference_image,visualize_image,label,model)