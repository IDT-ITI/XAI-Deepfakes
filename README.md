# Towards Quantitative Evaluation of Explainable AI Methods for Deepfake Detection

## PyTorch Implementation of CA-SUM [[Paper]()] [[DOI]()] [[Cite]()]
<div align="justify">

- From **"Towards Quantitative Evaluation of Explainable AI Methods for Deepfake Detection"**.
- Written by ... .
<!--
- This software can be used for training a deep learning architecture which estimates frames' importance by integrating a concentrated attention mechanism and utilizing information about the frames' uniqueness and diversity. The integrated mechanism is able to focus on non-overlapping blocks in the main diagonal of the attention matrix and make better estimates about the significance of different parts of the video by considering the uniqueness and diversity of the associated frames. Training is performed in an unsupervised manner without knowledge of any ground-truth data. Finally, after being trained on a collection of videos, the CA-SUM model is capable of producing summaries for unseen videos, according to a user-specified time-budget about the summary duration. --> 
</div>

## Dependencies
Developed, checked and verified on an `Ubuntu 20.04.6` PC with an `NVIDIA RTX 4090` GPU and an `i5-12600K` CPU. Required packages can be found inside the environment.yml file and can be installed through the Conda package and environment management system.

## Data
<div align="justify">

The database containing the frames of the original and the deepfake videos is available after asking for permission through the link: . The database needs to be placed into the /data folder for the code to work

Faceforensics++ GitHub: 

</div>

## Configurations
<div align="justify">
   
Arguments: 
|Parameter name | File | Description | Default Value | Options
| :--- | :--- | :--- | :---: | :---:
`explanation_method`|[`visualize.py`](explanation/visualize.py#L19:L21)| Explanation Method used to explain the image. | 'LIME' | 'GradCAM++', 'RISE', 'SHAP', 'LIME', 'SOBOL'

<!--
## Training
<div align="justify">

To train the model using one of the aforementioned datasets and for a number of randomly created splits of the dataset (where in each split 80% of the data is used for training and 20% for testing) use the corresponding JSON file that is included in the [data/splits](/data/splits) directory. This file contains the 5 randomly-generated splits that were utilized in our experiments.

For training the model using a single split, run:
```bash
for sigma in $(seq 0.5 0.1 0.9); do
    python model/main.py --split_index N --n_epochs E --batch_size B --video_type 'dataset_name' --reg_factor '$sigma'
done
```
where, `N` refers to the index of the used data split, `E` refers to the number of training epochs, `B` refers to the batch size, `dataset_name` refers to the name of the used dataset, and `$sigma` refers to the length regularization factor, a hyper-parameter of our method that relates to the length of the generated summary.

Alternatively, to train the model for all 5 splits, use the [`run_summe_splits.sh`](model/run_summe_splits.sh) and/or [`run_tvsum_splits.sh`](model/run_tvsum_splits.sh) script and do the following:
```shell-script
chmod +x model/run_summe_splits.sh    # Makes the script executable.
chmod +x model/run_tvsum_splits.sh    # Makes the script executable.
./model/run_summe_splits.sh           # Runs the script. 
./model/run_tvsum_splits.sh           # Runs the script.  
```
Please note that after each training epoch the algorithm performs an evaluation step, using the trained model to compute the importance scores for the frames of each video of the test set. These scores are then used by the provided [evaluation](evaluation) scripts to assess the overall performance of the model.

The progress of the training can be monitored via the TensorBoard platform and by:
- opening a command line (cmd) and running: `tensorboard --logdir=/path/to/log-directory --host=localhost`
- opening a browser and pasting the returned URL from cmd. </div>

## Model Selection and Evaluation 
<div align="justify">

The selection of a well-trained model is based on a two-step process. First, we keep one trained model per considered value for the length regularization factor sigma, by selecting the model (i.e., the epoch) that minimizes the training loss. Then, we choose the best-performing model (i.e., the sigma value) for a given data split through a mechanism that involves a fully-untrained model of the architecture and is based on transductive inference. More details about this assessment can be found in Section 4.2 of our work. To evaluate the trained models of the architecture and automatically select a well-trained one, define:
 - the [`dataset_path`](evaluation/compute_fscores.py#L25) in [`compute_fscores.py`](evaluation/compute_fscores.py),
 - the [`base_path`](evaluation/evaluate_factor.sh#L7) in [`evaluate_factor`](evaluation/evaluate_factor.sh),
 - the [`base_path`](evaluation/choose_best_model.py#L12) and [`annot_path`](evaluation/choose_best_model.py#L34) in [`choose_best_model`](evaluation/choose_best_model.py),

and run [`evaluate_exp.sh`](evaluation/evaluate_exp.sh) via
```bash
sh evaluation/evaluate_exp.sh '$exp_num' '$dataset' '$eval_method'
```
where, `$exp_num` is the number of the current evaluated experiment, `$dataset` refers to the dataset being used, and `$eval_method` describe the used approach for computing the overall F-Score after comparing the generated summary with all the available user summaries (i.e., 'max' for SumMe and 'avg' for TVSum).

For further details about the adopted structure of directories in our implementation, please check line [#7](evaluation/evaluate_factor.sh#L7) and line [#13](evaluation/evaluate_factor.sh#L13) of [`evaluate_factor.sh`](evaluation/evaluate_factor.sh). </div>

-->

## Trained model
<div align="justify">

We have released the [**`trained model`**]() used in our evaluation procedure.
The model needs to be placed inside the [`model/checkpoint`](model/checkpoint)  path for the code to work
</div>

## Citation
<div align="justify">
    
If you find our work, code or pretrained models, useful in your work, please cite the following publication:

E. Apostolidis, G. Balaouras, V. Mezaris, I. Patras, "<b>Summarizing Videos using Concentrated Attention and Considering the Uniqueness and Diversity of the Video Frames</b>", Proc. of the 2022 Int. Conf. on Multimedia Retrieval (ICMR ’22), June 2022, Newark, NJ, USA.
</div>

BibTeX:

```
@inproceedings{10.1145/3512527.3531404,
author = {Apostolidis, Evlampios and Balaouras, Georgios and Mezaris, Vasileios and Patras, Ioannis},
title = {Summarizing Videos Using Concentrated Attention and Considering the Uniqueness and Diversity of the Video Frames},
year = {2022},
isbn = {9781450392389},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3512527.3531404},
doi = {10.1145/3512527.3531404},
pages = {407-415},
numpages = {9},
keywords = {frame diversity, frame uniqueness, concentrated attention, unsupervised learning, video summarization},
location = {Newark, NJ, USA},
series = {ICMR '22}
}
```

## License
<div align="justify">

Copyright (c) 2022, Evlampios Apostolidis, Georgios Balaouras, Vasileios Mezaris, Ioannis Patras / CERTH-ITI. All rights reserved. This code is provided for academic, non-commercial use only. Redistribution and use in source and binary forms, with or without modification, are permitted for academic non-commercial use provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation provided with the distribution.

This software is provided by the authors "as is" and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall the authors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software, even if advised of the possibility of such damage.
</div>

## Acknowledgement
<div align="justify"> This work was supported by the EU Horizon 2020 programme under grant agreements H2020-832921 MIRROR and H2020-951911 AI4Media. </div>
