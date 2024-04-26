import torch
from torchvision.transforms import v2
import sys
sys.path.append('../src')
sys.path.append('./methods')
from model.frame import FrameModel
import numpy as np
from data.datasets import DeepfakeDataset
from methods.gradcam_xai import explain as GradCAM
from methods.rise_xai import explain as RISE
from methods.shap_xai import explain as SHAP
from methods.lime_xai import explain as LIME
from methods.sobol_xai import explain as SOBOL
from decimal import Decimal




#"GradCAM++" - "RISE" - "SHAP" - "LIME" - "SOBOL"
explanation_method="LIME"
#Select to explain the predicted label (overrides explain_real and select_explanation_label)
explain_predicted=True
#Select to explain the real label (overrides select_explanation_label)
explain_real=False
#Manually select label
select_explanation_label=-1
#"random" or int of the index of the example
dataset_example_index="random"




#Load the model
rs_size = 224
model = FrameModel.load_from_checkpoint("../model/checkpoint/ff_attribution.ckpt",map_location='cuda').eval()
task = "multiclass"

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

#Load the dataset
ds_path = "../data/csvs/ff_test.csv"

#Dataset with inference transformations
target_transforms = lambda x: torch.tensor(x, dtype=torch.float32)
ds = DeepfakeDataset(
    ds_path,
    "../data/xai_test_data.lmdb",
    transforms=inference_transforms,
    target_transforms=target_transforms,
    task=task
)
#Dataset with visualization transformations
ds_visualize = DeepfakeDataset(
    ds_path,
    "../data/xai_test_data.lmdb",
    transforms=visualize_transforms,
    target_transforms=target_transforms,
    task=task
)

#Set the example index
if (dataset_example_index=="random"):
    idx = np.random.randint(0, len(ds))
else:
    idx = dataset_example_index

#Get the images
inference_image, label_real = ds[idx]
visualize_image, _ = ds_visualize[idx]

#Compute the inference scores
with torch.no_grad():
    frame = inference_image.to(model.device)
    output = model(frame.unsqueeze(0))

output = output.cpu().reshape(-1, ).numpy()
real_label=int(label_real.reshape(-1, ).numpy()[0])

#Print the inference scores, the predicted and the ground truth label
print("Output scores: ",end='')
for o in output:
    o=Decimal(str(o))
    print(format(o, 'f') ,end=' ')
print("\nPredicted label: "+ str(np.argmax(output)))
print("Real label: "+ str(real_label))

#Print the label selected for explanation
if(explain_predicted):
    explanation_label_index = np.argmax(output)
    print("\nExplaining predicted label")
elif(explain_real):
    explanation_label_index = real_label
    print("\nExplaining real label")
else:
    explanation_label_index=select_explanation_label
    print("\nExplaining label "+str(select_explanation_label))

#Call the corresponding explanation method to calculate the explanation
if(explanation_method == "GradCAM++"):
    GradCAM(inference_image, visualize_image.permute(1, 2, 0).numpy(), explanation_label_index, model)
elif(explanation_method == "RISE"):
    RISE(inference_image, visualize_image.unsqueeze(0), explanation_label_index, model)
elif (explanation_method == "SHAP"):
    SHAP(inference_image, visualize_image.permute(1, 2, 0).numpy(), explanation_label_index, model)
elif (explanation_method == "LIME"):
    LIME(visualize_image.permute(1, 2, 0).numpy(), inference_transforms, explanation_label_index, model)
elif (explanation_method == "SOBOL" ):
    SOBOL(inference_image, visualize_image, explanation_label_index, model)
else:
    print("Incorrect explanation method")
    sys.exit(0)