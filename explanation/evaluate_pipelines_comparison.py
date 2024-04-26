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




#Set the names of the files that save the results of each example and the final produced csv file
save_name="comparison_results"
csv_save_name="ff_comparison_scores"




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
computeExplanationMetrics(model, ds, ds_visualize, inference_transforms, save_name)

#Open the matched pairs csv and load the saved results
file = open(ds_path, "r")
data = list(csv.reader(file, delimiter=","))
file.close()
data=[x[0].split('/')[1] for x in data[1::2]]

scores_all = np.load("./results/"+save_name+".npy")
scores_all = list(scores_all)

category=["Deepfakes","Face2Face","FaceSwap","NeuralTextures"]

#Create a nan array with the saved results placed in different indexes based on the category of each example
#This ensures that the mean scores will happen between examples of the same category
scores=[]
for i,s in enumerate(scores_all):
    idx=category.index(data[i])

    original=8*[np.nan]+idx*[np.nan]+[s[0][2]]+(7-idx)*[np.nan]
    grad=idx*[np.nan]+[s[1][0]]+(3-idx)*[np.nan]+idx*[np.nan]+[s[1][1]]+(3-idx)*[np.nan]+idx*[np.nan]+[s[1][2]]+(3-idx)*[np.nan]+idx*[np.nan]+[s[1][3]]+(3-idx)*[np.nan]
    rise=idx*[np.nan]+[s[2][0]]+(3-idx)*[np.nan]+idx*[np.nan]+[s[2][1]]+(3-idx)*[np.nan]+idx*[np.nan]+[s[2][2]]+(3-idx)*[np.nan]+idx*[np.nan]+[s[2][3]]+(3-idx)*[np.nan]
    shap=idx*[np.nan]+[s[3][0]]+(3-idx)*[np.nan]+idx*[np.nan]+[s[3][1]]+(3-idx)*[np.nan]+idx*[np.nan]+[s[3][2]]+(3-idx)*[np.nan]+idx*[np.nan]+[s[3][3]]+(3-idx)*[np.nan]
    lime=idx*[np.nan]+[s[4][0]]+(3-idx)*[np.nan]+idx*[np.nan]+[s[4][1]]+(3-idx)*[np.nan]+idx*[np.nan]+[s[4][2]]+(3-idx)*[np.nan]+idx*[np.nan]+[s[4][3]]+(3-idx)*[np.nan]
    sobol=idx*[np.nan]+[s[5][0]]+(3-idx)*[np.nan]+idx*[np.nan]+[s[5][1]]+(3-idx)*[np.nan]+idx*[np.nan]+[s[5][2]]+(3-idx)*[np.nan]+idx*[np.nan]+[s[5][3]]+(3-idx)*[np.nan]
    scores.append(np.array((original,grad,rise,shap,lime,sobol)))

#Compute the mean values
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    scores=np.nanmean(scores,axis=0)

#Create a dataframe and save the results in a csv
index_values = ["Original","GradCAM", "RISE", "SHAP", "LIME", "SOBOL"]
column_values = ["Sufficiency Top 3 DF","Sufficiency Top 3 F2F","Sufficiency Top 3 FS","Sufficiency Top 3 NT",
                 "Sufficiency Top 3 Original DF","Sufficiency Top 3 Original F2F","Sufficiency Top 3 Original FS","Sufficiency Top 3 Original NT",
                 "Accuracy (Top 3) DF","Accuracy (Top 3) F2F","Accuracy (Top 3) FS","Accuracy (Top 3) NT",
                 "Accuracy Top 3 Original DF","Accuracy Top 3 Original F2F","Accuracy Top 3 Original FS","Accuracy Top 3 Original NT"]
df = pd.DataFrame(data=scores,index=index_values,columns=column_values)
df.round(3).to_csv("./results/"+csv_save_name+".csv",sep=',')
print(df.round(3).to_string())