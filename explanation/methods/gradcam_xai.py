import torch
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.transforms import v2
import sys
sys.path.append('../../src')
from model.frame import FrameModel
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from matplotlib import pyplot as plt

def explain(inference_image,visualize_image,label,model,visualize=True):

    #Find the all of the convolutional layers to feed to gradcam to compute the explanation from
    target_layers=[]
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Conv2d):
            target_layers.append(layer)

    #Set as target the explanation label
    targets = [ClassifierOutputTarget(label)]

    #Compute the explanation
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    result=cam(input_tensor=inference_image.unsqueeze(0),targets=targets)

    #Get the saliency map
    saliance_map=result[0, :]

    #If selected visualize the result
    if(visualize):
        visualization = show_cam_on_image(visualize_image, saliance_map, use_rgb=True)
        plt.imshow(visualization)
        plt.show()

    #Return the saliency map
    return saliance_map


if __name__ == "__main__":
    # Load the model
    rs_size = 224
    model = FrameModel.load_from_checkpoint("../../model/checkpoint/ff_attribution.ckpt", map_location='cuda').eval()

    # Create the transforms for inference and visualization purposes
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
    visualize_image = visualize_transforms(image).permute(1, 2, 0).numpy()
    #Select the explanation label
    label = 0

    #Call the explanation method
    explain(inference_image, visualize_image, label, model)

