import csv
import os
import sys
import warnings
import pandas as pd
import numpy as np
import torch
from torchvision.transforms import v2
sys.path.append('../src')
sys.path.append('./methods')
from model.frame import FrameModel
from data.datasets import DeepfakeDataset
from evaluation.compute_pipelines_comparison_metrics import computeExplanationMetrics
from evaluation.generate_ff_test_data import getMatchedFFPath




#"GradCAM++" - "RISE" - "SHAP" - "LIME" - "SOBOL" - "All"
evaluation_explanation_methods="All"




valid_methods=["GradCAM++", "RISE", "SHAP", "LIME", "SOBOL", "All"]
if(evaluation_explanation_methods not in valid_methods):
    print("Invalid explanation method(s) to evaluate")
    sys.exit(0)

#Load the model
rs_size = 224
task = "multiclass"
model = FrameModel.load_from_checkpoint("../model/checkpoint/ff_attribution.ckpt",map_location='cuda').eval()

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

#Create the deepfake and real pairs test examples and load the dataset
ds_path = getMatchedFFPath("../data/csvs/ff_test.csv")

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

if not os.path.exists('./results'):
    os.makedirs('./results')

#Call the corresponding function to compute them
computeExplanationMetrics(model, ds, ds_visualize, inference_transforms, evaluation_explanation_methods)


#Load the saved results
save_name="comparison_results_"+evaluation_explanation_methods
scores = np.load("./results/"+save_name+".npy")
scores = list(scores)

#Compute the mean values
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    scores=np.nanmean(scores,axis=0)

#Create a dataframe and save the results in a csv
if(evaluation_explanation_methods=="All"):
    index_values = ["Original", "GradCAM++", "RISE", "SHAP", "LIME", "SOBOL"]
else:
    index_values = ["Original", evaluation_explanation_methods]

column_values = ["Sufficiency Top 3 DF","Sufficiency Top 3 F2F","Sufficiency Top 3 FS","Sufficiency Top 3 NT",
                 "Sufficiency Top 3 Original DF","Sufficiency Top 3 Original F2F","Sufficiency Top 3 Original FS","Sufficiency Top 3 Original NT",
                 "Accuracy (Top 3) DF","Accuracy (Top 3) F2F","Accuracy (Top 3) FS","Accuracy (Top 3) NT",
                 "Accuracy Top 3 Original DF","Accuracy Top 3 Original F2F","Accuracy Top 3 Original FS","Accuracy Top 3 Original NT"]
df = pd.DataFrame(data=scores,index=index_values,columns=column_values)
csv_save_name="comparison_scores_"+evaluation_explanation_methods
df.round(3).to_csv("./results/"+csv_save_name+".csv",sep=',')
print(df.round(3).to_string())