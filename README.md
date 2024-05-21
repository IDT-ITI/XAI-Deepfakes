# Towards Quantitative Evaluation of Explainable AI Methods for Deepfake Detection

## [[Paper](https://arxiv.org/pdf/2404.18649)] [[DOI](https://updatelink)] [[Cite](#citation)]
<div align="justify">

- From **"Towards Quantitative Evaluation of Explainable AI Methods for Deepfake Detection"**, Proc. ACM Int. Workshop on Multimedia AI against Disinformation (MAD’24) at the ACM Int. Conf. on Multimedia Retrieval (ICMR’24), Thailand, June 2024.
- Written by Konstantinos Tsigos, Evlampios Apostolidis, Spyridon Baxevanakis, Symeon Papadopoulos and Vasileios Mezaris.
- This software can be used to evaluate the performance of five explanation approaches from the literature (GradCAM++, RISE, SHAP, LIME, SOBOL), on explaining the output of a state-of-the-art model (based on Efficient-Net) for deepfake detection. Our evaluation framework assesses the ability of an explanation method to spot the regions of a fake image with the biggest influence on the decision of the deepfake detector, by examining the extent to which these regions can be modified through a set of adversarial attacks, in order to flip the detector's prediction or reduce its initial prediction.
</div>

## Dependencies
Developed, checked and verified on an `Ubuntu 20.04.6` PC with an `NVIDIA RTX 4090` GPU and an `i5-12600K` CPU.

Dependencies can be found inside the [environment.yml](/environment.yml) file, which can be used to set up a corresponding [conda](https://docs.conda.io/en/latest/) enviroment.

## Data
<div align="justify">

The data for evaluating the different explanation methods, as well as visualizing their results, consist of the manipulated and non-manipulated cropped and sampled frames of the test videos found in the test split of the [FaceForensics++](https://github.com/ondyari/FaceForensics) dataset.

The required database containing the afformentioned data can be created by following the steps below:

1. Download the [FaceForensics++](https://github.com/ondyari/FaceForensics#Access) dataset
2. Run the following script to preprocess the raw data:
```bash
python3 data/preprocess_ff.py prepro -r RAW_DATA_PATH -tr PREPROCESSED_DATA_PATH -d cuda:0 -mdcsv RAW_DATA_PATH/dataset_info.csv
```
where `RAW_DATA_PATH` is the path to the downloaded FF++ dataset and `PREPROCESSED_DATA_PATH` is the path to save the preprocessed data. The script will create a new file `faceforensics_frames.csv` containing the paths to the preprocessed frames.

3. Create a new LMDB database by running the following script:
```bash
python3 data/lmdb_storage.py add-csv -csv ./data/faceforensics_frames.csv -h -pc relative_path -d ./data/xai_test_data.lmdb -ms 21474836480 -v -b PREPROCESSED_DATA_PATH
```
where `faceforensics_frames.csv` is the file created in the previous step and `PREPROCESSED_DATA_PATH` is the path to the preprocessed data. The script will create a new LMDB database `xai_test_data.lmdb` containing the preprocessed frames. The `-ms` flag specifies the maximum size of the database in bytes, default is 20GB.

<!-- The LMDB database can then be placed inside the [data](/data) folder for the code to work properly. -->

</div>

## Trained model
<div align="justify">

### ff_attribution
The trained model employed, was trained for multiclass classification on the FaceForensics++ dataset. It outputs a probability for each of the 5 classes (0, 1, 2, 3, 4), corresponding to real, neural textures, face2face, deepfakes and faceswap.

#### Model characteristics
| Model | ff_attribution
| --- | --- |
| Task | multiclass |
| Arch. | efficientnetv2_b0 |
| Type | CNN |
| No. Params | 7.1M |
| No. Datasets | 1 |
| Input | (B, 3, 224, 224) |
| Output | (B, 5) |

#### Performance (FF++ test set)
| Metric | Value |
| --- | --- |
| MulticlassAccuracy | 0.9626 |
| MulticlassAUROC | 0.9970 |
| MulticlassF1Score | 0.9627 |
| MulticlassAveragePrecision | 0.9881 |

## Evaluation and visualization
<div align="justify">

To evaluate the explanation method(s) on our proposed approach, you run the [`evaluate.py`](explanation/evaluate.py) file.

To evaluate the explanation method(s) on our proposed approach, as well as the original methodology for performance comparison, done on a smaller subset of 600 images, you run the [`evaluate_pipelines_comparison.py`](explanation/evaluate_pipelines_comparison.py) file.

For visualizing the explanation mask returned by an explanation method for a specific image, you run the [`visualize.py`](explanation/visualize.py) file.

## Running parameters and evaluation results
<div align="justify">

|Parameter name | File | Description | Default Value | Options
| :--- | :--- | :--- | :---: | :---:
`explanation_method`|[`visualize.py`](explanation/visualize.py#L19:L20)| Explanation method to explain the image. | 'LIME' | 'GradCAM++', 'RISE', 'SHAP', 'LIME', 'SOBOL'
`dataset_example_index`|[`visualize.py`](explanation/visualize.py#L21:L22)| Index of the image in the database | 'random' | 'random', integer between [0,13837]
`evaluation_explanation_methods`|[`evaluate.py`](explanation/evaluate.py#L18:L19) [`evaluate_pipelines_comparison.py`](explanation/evaluate_pipelines_comparison.py#L19:L20)| Explanation method to evaluate | 'All' | 'All', 'GradCAM++', 'RISE', 'SHAP', 'LIME', 'SOBOL'

Evaluation results are printed onto the console and additionally saved into a csv format file, located at the `results` folder created at the [explanation](/explanation) path. In order to prevent the need of running the evaluation process from the beginning in case of a crash, the evaluation results of each subsequent image are accumulated into an npy format file, used as a checkpoint, also located at the `results` folder.

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

## Citation
<div align="justify">
    
If you find our work, code or pretrained models, useful in your work, please cite the following publication:

K. Tsigos, E. Apostolidis, S. Baxevanakis, S. Papadopoulos, V. Mezaris, "<b>Towards Quantitative Evaluation of Explainable AI Methods for Deepfake Detection</b>", Proc. ACM Int. Workshop on Multimedia AI against Disinformation (MAD’24) at the ACM Int. Conf. on Multimedia Retrieval (ICMR’24), Thailand, June 2024.
</div>

BibTeX:

```
@INPROCEEDINGS{Tsigos2024,
    author    = {Tsigos, Konstantinos, and Apostolidis, Evlampios and Baxevanakis, Spyridon and Papadopoulos, Symeon and Mezaris, Vasileios},
    title     = {Towards Quantitative Evaluation of Explainable AI Methods for Deepfake Detection},
    year      = {2024},
    publisher = {Association for Computing Machinery},
    address   = {New York, NY, USA},
    booktitle = {Proceedings of the 3rd ACM International Workshop on Multimedia AI against Disinformation},
    location  = {Phuket, Thailand},
    series    = {MAD '24}
}
```

## License
<div align="justify">
    
Copyright (c) 2024, Konstantinos Tsigos, Evlampios Apostolidis, Spyridon Baxevanakis, Symeon Papadopoulos, Vasileios Mezaris / CERTH-ITI. All rights reserved. This code is provided for academic, non-commercial use only. Redistribution and use in source and binary forms, with or without modification, are permitted for academic non-commercial use provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation provided with the distribution.

This software is provided by the authors "as is" and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall the authors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software, even if advised of the possibility of such damage.
</div>

## Acknowledgement
<div align="justify"> This work was supported by the EU Horizon Europe and Horizon 2020 programmes under grant agreements 101070190 AI4TRUST and 951911 AI4Media, respectively. </div>
