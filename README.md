# Towards Quantitative Evaluation of Explainable AI Methods for Deepfake Detection

## PyTorch Implementation [[Paper](https://arxiv.org/pdf/2404.18649)] [[DOI](https://doi.org/10.1145/3643491.3660292)] [[Cite](#citation)]
<div align="justify">

- From **"Towards Quantitative Evaluation of Explainable AI Methods for Deepfake Detection"**, Proc. ACM Int. Workshop on Multimedia AI against Disinformation (MAD’24) at the ACM Int. Conf. on Multimedia Retrieval (ICMR’24), Thailand, June 2024.
- Written by Konstantinos Tsigos, Evlampios Apostolidis, Spyridon Baxevanakis, Symeon Papadopoulos and Vasileios Mezaris.
- This software can be used to evaluate the performance of five explanation approaches from the literature (GradCAM++, RISE, SHAP, LIME, SOBOL), on explaining the output of a state-of-the-art model (based on Efficient-Net) for deepfake detection. Our evaluation framework assesses the ability of an explanation method to spot the regions of a fake image with the biggest influence on the decision of the deepfake detector, by examining the extent to which these regions can be modified through a set of adversarial attacks, in order to flip the detector's prediction or reduce its initial prediction.
</div>

## Dependencies
The code was developed, checked and verified on an `Ubuntu 20.04.6` PC with an `NVIDIA RTX 4090` GPU and an `i5-12600K` CPU. All dependencies can be found inside the [environment.yml](/environment.yml) file, which can be used to set up the necessary [conda](https://docs.conda.io/en/latest/) enviroment.

## Data
<div align="justify">

The data for re-producing our experiments are the videos from the test split of the [FaceForensics++](https://github.com/ondyari/FaceForensics) dataset. This dataset contains 1000 original videos and 4000 fake videos created using one of the following four classes of AI-based manipulation (1000 per class): "FaceSwap" (FS), "DeepFakes" (DF), "Face2Face" (F2F),  and "NeuralTextures" (NT). The videos of the FS class were created via a graphics-based approach that transfers the face region from a source to a target video. The videos of the DF class were produced using autoencoders to replace a face in a target sequence with a face in a source video or image collection. The videos of the F2F class were obtained by a facial reenactment system that transfers the expressions of a source to a target video while maintaining the identity of the target person. The videos of the NT class were generated by modifying the facial expressions corresponding to the mouth region, using a patch-based GAN-loss. The dataset is divided into training, validation, and test sets, comprised of 720, 140 and 140 videos, respectively.

To re-create the database file that we used in our experiments, please follow the steps below:

1. Download the entire [FaceForensics++](https://github.com/ondyari/FaceForensics#Access) dataset
2. Run the following script to preprocess the raw data:
```bash
python data/preprocess_ff.py prepro -r RAW_DATA_PATH -tr PREPROCESSED_DATA_PATH -d cuda:0 -mdcsv RAW_DATA_PATH/dataset_info.csv -orig
```
where `RAW_DATA_PATH` is the path to the downloaded FF++ dataset, `PREPROCESSED_DATA_PATH` is the path to save the preprocessed data, and `dataset_info.csv` is available in the [data/csvs](https://github.com/IDT-ITI/XAI-Deepfakes/blob/main/data/csvs) directory. The script will create a new file `faceforensics_frames.csv` containing the paths to the preprocessed frames.

3. Create a new LMDB database by running the following script:
```bash
python data/lmdb_storage.py add-csv -csv faceforensics_frames.csv -h -pc relative_path -d ./data/xai_test_data.lmdb -ms 21474836480 -v -b PREPROCESSED_DATA_PATH
```
where `faceforensics_frames.csv` is the file created in the previous step and `PREPROCESSED_DATA_PATH` is the path to the preprocessed data. The script will create a new LMDB database `xai_test_data.lmdb` containing the preprocessed frames. The `-ms` flag specifies the maximum size of the database in bytes, default is 20GB.
</div>

## Trained model
<div align="justify">

The employed model (called ff_attribution) was trained for multiclass classification on the FaceForensics++ dataset. It outputs a probability for each of the 5 classes (0, 1, 2, 3, 4), corresponding to "Real", "NeuralTextures", "Face2Face", "DeepFakes" and "FaceSwap", respectively.

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

To evaluate the explanation method(s) using our framework, run the [`evaluate.py`](explanation/evaluate.py) script.

To evaluate the explanation method(s) using both our framework and the one from [Gowrisankar et al. (2024)](https://arxiv.org/abs/2312.06627) (which was used as the basis for developing our framework), and compare the results (as we did in Tables 3 and 4 of our paper), run the [`evaluate_pipelines_comparison.py`](explanation/evaluate_pipelines_comparison.py) script.

To visualize the created explanation by an explanation method, for a specific image of the dataset, run the [`visualize.py`](explanation/visualize.py) script.

## Running parameters and evaluation results
<div align="justify">
The default parameters and the available options for running the above scripts, are listed below: 

|Parameter name | File | Description | Default Value | Options
| :--- | :--- | :--- | :---: | :---:
`evaluation_explanation_methods`|[`evaluate.py`](explanation/evaluate.py#L18:L19) [`evaluate_pipelines_comparison.py`](explanation/evaluate_pipelines_comparison.py#L19:L20)| Explanation method(s) to evaluate | 'All' | 'All', 'GradCAM++', 'RISE', 'SHAP', 'LIME', 'SOBOL'
`explanation_method`|[`visualize.py`](explanation/visualize.py#L19:L20)| Explanation method to explain the image. | 'LIME' | 'GradCAM++', 'RISE', 'SHAP', 'LIME', 'SOBOL'
`dataset_example_index`|[`visualize.py`](explanation/visualize.py#L21:L22)| Index of the image in the database | 'random' | 'random', integer between [0,13837]

The evaluation results are printed onto the console and saved into a CSV file which is stored within the `results` folder, created at the [explanation](/explanation) path. To eliminate the need to run the evaluation process from the beginning in case of an unexpected error, the evaluation results for each test image are accumulated into an NPY file, that is used as a checkpoint and is also located at the `results` folder.

## Citation
<div align="justify">
    
If you find our work, code or pretrained models, useful in your work, please cite the following publication:

K. Tsigos, E. Apostolidis, S. Baxevanakis, S. Papadopoulos, V. Mezaris, "<b>Towards Quantitative Evaluation of Explainable AI Methods for Deepfake Detection</b>", Proc. ACM Int. Workshop on Multimedia AI against Disinformation (MAD’24) at the ACM Int. Conf. on Multimedia Retrieval (ICMR’24), Thailand, June 2024.
</div>

The accepted version of this paper is available on ArXiv at: https://arxiv.org/abs/2404.18649

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
    
This code is provided for academic, non-commercial use only. Please also check for any restrictions applied in the code parts and datasets used here from other sources. For the materials not covered by any such restrictions, redistribution and use in source and binary forms, with or without modification, are permitted for academic non-commercial use provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation provided with the distribution.

This software is provided by the authors "as is" and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall the authors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software, even if advised of the possibility of such damage.
</div>

## Acknowledgement
<div align="justify"> This work was supported by the EU Horizon Europe and Horizon 2020 programmes under grant agreements 101070190 AI4TRUST and 951911 AI4Media, respectively. </div>
