import os
import sys

import torch
from methods.gradcam_xai import explain as GradCAM
from methods.rise_xai import explain as RISE
from methods.shap_xai import explain as SHAP
from methods.lime_xai import explain as LIME
from methods.sobol_xai import explain as SOBOL
from evaluation.adversarial_image_generation import AdversarialImageGeneration
from evaluation.metrics import ExplanationMetrics
import numpy as np
from skimage.segmentation import slic
from torchvision.transforms import v2

def inverseNormalization(img):
    invTrans = v2.Compose([v2.Normalize(mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                           v2.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])])
    return invTrans(img)

def computeDiscScores(i_img,evaluation,metrics,segments,ranked_segments,label,num_of_segments=3):

    #Create the segmentation mask by masking out the 3 top segments
    mask = evaluation.getSegmentsMask(segments, ranked_segments, num_of_segments)
    #Produce the adversarially attacked image
    a_img, _ = evaluation.generateAdversarialImage(mask)
    #Compute the comprehensiveness metric (disc plus) between the original and the adversarially attacked image for the label score
    disc_plus = metrics.disc(i_img, a_img, label)

    #Return the disc plus score and the adversarially attacked image
    return [disc_plus,a_img]

def computeGradCAMMetrics(model, i_img, v_img, i_r_img, v_r_img, best_label, true_label):

    #Create the ExplanationMetrics and AdversarialImageGeneration objects
    metrics = ExplanationMetrics(model)
    evaluation = AdversarialImageGeneration(i_img, model, 0.001, 40, 50, 16 / 255, 1 / 255)

    #Call the explanation method to produce the heatmaps for the two different pipelines
    heatmap = GradCAM(i_img, v_img.permute(1, 2, 0).numpy(), best_label, model, visualize=False)
    heatmap_original = GradCAM(i_r_img, v_r_img.permute(1, 2, 0).numpy(), 0, model, visualize=False)

    # In the rare occasions where the heatmap returned by GradCAM contains nan values
    if (np.any(np.isnan(heatmap))):
        #Recall the explanation method for our pipeline
        heatmap = GradCAM(i_img, v_img.permute(1, 2, 0).numpy(), best_label, model, visualize=False)

    if (np.any(np.isnan(heatmap_original))):
        # Recall the explanation method for the original pipeline
        heatmap_original = GradCAM(i_r_img, v_r_img.permute(1, 2, 0).numpy(), 0, model, visualize=False)

    #If the heatmaps don't contain nan values
    if (not np.any(np.isnan(heatmap)) and not np.any(np.isnan(heatmap_original))):
        #Segment the heatmap of our pipeline into segments and rank them based on the values of the heatmap for the indices of each one
        segments, ranked_segments = evaluation.getSegmentsfromHeatmap(heatmap)
        #Compute the comprehensiveness scores
        disc_plus, a_img = computeDiscScores(i_img, evaluation, metrics, segments, ranked_segments, best_label)
        #Compute the classifier's accuracy for the adversarially attacked image on the top 1-2-3 segments
        adv_label = np.argmax(model(a_img.unsqueeze(0).to(model.device)).cpu().detach().numpy())
        adv_acc = 1 if (adv_label == true_label) else 0

        #Do the same for the original pipeline
        segments, ranked_segments = evaluation.getSegmentsfromHeatmap(heatmap_original)
        disc_plus_original, a_img = computeDiscScores(i_img, evaluation, metrics, segments, ranked_segments, best_label)
        adv_label = np.argmax(model(a_img.unsqueeze(0).to(model.device)).cpu().detach().numpy())
        adv_acc_original = 1 if (adv_label == true_label) else 0

        torch.cuda.empty_cache()
        #Return the scores
        return [disc_plus, disc_plus_original, adv_acc, adv_acc_original]

    #If the heatmap still contains nan values
    else:
        torch.cuda.empty_cache()
        #Return nan values
        return [np.nan, np.nan, np.nan, np.nan]


def computeRISEMetrics(model, i_img, v_img, i_r_img, v_r_img, best_label, true_label):

    #Create the ExplanationMetrics and AdversarialImageGeneration objects
    metrics = ExplanationMetrics(model)
    evaluation = AdversarialImageGeneration(i_img, model, 0.001, 40, 50, 16 / 255, 1 / 255)

    #Call the explanation method to produce the heatmaps for the two different pipelines
    heatmap = RISE(i_img, v_img.unsqueeze(0), best_label, model, visualize=False)
    heatmap_original = RISE(i_r_img, v_r_img.unsqueeze(0), 0, model, visualize=False)

    #Segment the heatmap of our pipeline into segments and rank them based on the values of the heatmap for the indices of each one
    segments, ranked_segments = evaluation.getSegmentsfromHeatmap(heatmap)
    #Compute the comprehensiveness scores
    disc_plus, a_img = computeDiscScores(i_img, evaluation, metrics, segments, ranked_segments, best_label)
    #Compute the classifier's accuracy for the adversarially attacked image on the top 1-2-3 segments
    adv_label = np.argmax(model(a_img.unsqueeze(0).to(model.device)).cpu().detach().numpy())
    adv_acc = 1 if (adv_label == true_label) else 0

    #Do the same for the original pipeline
    segments, ranked_segments = evaluation.getSegmentsfromHeatmap(heatmap_original)
    disc_plus_original, a_img = computeDiscScores(i_img, evaluation, metrics, segments, ranked_segments, best_label)
    adv_label = np.argmax(model(a_img.unsqueeze(0).to(model.device)).cpu().detach().numpy())
    adv_acc_original = 1 if (adv_label == true_label) else 0

    torch.cuda.empty_cache()
    #Return the scores
    return [disc_plus, disc_plus_original, adv_acc, adv_acc_original]

def computeSHAPMetrics(model, i_img, v_img, i_r_img, v_r_img, best_label, true_label):

    #Create the ExplanationMetrics and AdversarialImageGeneration objects
    metrics = ExplanationMetrics(model)
    evaluation = AdversarialImageGeneration(i_img, model, 0.001, 40, 50, 16 / 255, 1 / 255)

    #Call the explanation method to produce the heatmaps for the two different pipelines
    heatmap = SHAP(i_img, v_img.permute(1, 2, 0).numpy(), best_label, model, visualize=False)
    heatmap_original = SHAP(i_r_img, v_r_img.permute(1, 2, 0).numpy(), 0, model, visualize=False)

    #Segment the heatmap of our pipeline into segments and rank them based on the values of the heatmap for the indices of each one
    segments, ranked_segments = evaluation.getSegmentsfromHeatmap(heatmap)
    #Compute the comprehensiveness scores
    disc_plus, a_img = computeDiscScores(i_img, evaluation, metrics, segments, ranked_segments, best_label)
    #Compute the classifier's accuracy for the adversarially attacked image on the top 1-2-3 segments
    adv_label = np.argmax(model(a_img.unsqueeze(0).to(model.device)).cpu().detach().numpy())
    adv_acc = 1 if (adv_label == true_label) else 0

    #Do the same for the original pipeline
    segments, ranked_segments = evaluation.getSegmentsfromHeatmap(heatmap_original)
    disc_plus_original, a_img = computeDiscScores(i_img, evaluation, metrics, segments, ranked_segments, best_label)
    adv_label = np.argmax(model(a_img.unsqueeze(0).to(model.device)).cpu().detach().numpy())
    adv_acc_original = 1 if (adv_label == true_label) else 0

    torch.cuda.empty_cache()
    #Return the scores
    return [disc_plus, disc_plus_original, adv_acc, adv_acc_original]

def computeLIMEMetrics(model, i_img, v_img, _, v_r_img, i_trans, best_label, true_label):
    #Custom segmentation function passed in LIME to segment the image
    def custom_seg(image):
        return slic(image, n_segments=50) - 1

    #Create the ExplanationMetrics and AdversarialImageGeneration objects
    metrics = ExplanationMetrics(model)
    evaluation = AdversarialImageGeneration(i_img, model, 0.001, 40, 50, 16 / 255, 1 / 255)

    #Call the explanation method to produce the segments and calculate the weights of the linear regressor that correspond to each one
    _, segments, exp = LIME(v_img.permute(1, 2, 0).numpy(), i_trans, best_label, model, custom_seg, visualize=False)
    #Rank the segments directly from the values of the weights
    ranked_segments = [x[0] for x in exp if x[1] > 0]
    #Compute the comprehensiveness scores
    disc_plus, a_img = computeDiscScores(i_img, evaluation, metrics, segments, ranked_segments, best_label)
    #Compute the classifier's accuracy for the adversarially attacked image on the top 1-2-3 segments
    adv_label = np.argmax(model(a_img.unsqueeze(0).to(model.device)).cpu().detach().numpy())
    adv_acc = 1 if (adv_label == true_label) else 0

    #Do the same for the original pipeline
    _, segments, exp = LIME(v_r_img.permute(1, 2, 0).numpy(), i_trans, 0, model, custom_seg, visualize=False)
    ranked_segments = [x[0] for x in exp if x[1] > 0]
    disc_plus_original, a_img = computeDiscScores(i_img, evaluation, metrics, segments, ranked_segments, best_label)
    adv_label = np.argmax(model(a_img.unsqueeze(0).to(model.device)).cpu().detach().numpy())
    adv_acc_original = 1 if (adv_label == true_label) else 0

    torch.cuda.empty_cache()
    #Return the scores
    return [disc_plus, disc_plus_original, adv_acc, adv_acc_original]

def computeSOBOLMetrics(model, i_img, v_img, i_r_img, v_r_img, best_label, true_label):

    #Create the ExplanationMetrics and AdversarialImageGeneration objects
    metrics = ExplanationMetrics(model)
    evaluation = AdversarialImageGeneration(i_img, model, 0.001, 40, 50, 16 / 255, 1 / 255)

    #Call the explanation method to produce the heatmaps for the two different pipelines
    heatmap = SOBOL(i_img, v_img, best_label, model, visualize=False)
    heatmap_original = SOBOL(i_r_img, v_r_img, 0, model, visualize=False)

    #Segment the heatmap of our pipeline into segments and rank them based on the values of the heatmap for the indices of each one
    segments, ranked_segments = evaluation.getSegmentsfromHeatmap(heatmap)
    #Compute the comprehensiveness scores
    disc_plus, a_img = computeDiscScores(i_img, evaluation, metrics, segments, ranked_segments, best_label)
    #Compute the classifier's accuracy for the adversarially attacked image on the top 1-2-3 segments
    adv_label = np.argmax(model(a_img.unsqueeze(0).to(model.device)).cpu().detach().numpy())
    adv_acc = 1 if (adv_label == true_label) else 0

    #Do the same for the original pipeline
    segments, ranked_segments = evaluation.getSegmentsfromHeatmap(heatmap_original)
    disc_plus_original, a_img = computeDiscScores(i_img, evaluation, metrics, segments, ranked_segments, best_label)
    adv_label = np.argmax(model(a_img.unsqueeze(0).to(model.device)).cpu().detach().numpy())
    adv_acc_original = 1 if (adv_label == true_label) else 0

    torch.cuda.empty_cache()
    #Return the scores
    return [disc_plus, disc_plus_original, adv_acc, adv_acc_original]

def computeExplanationMetrics(model,ds,ds_vis,i_trans,evaluation_explanation_methods):

    name="comparison_results_"+evaluation_explanation_methods
    scores=[]
    #Different deepfake categories
    category=["Deepfakes","Face2Face","FaceSwap","NeuralTextures"]

    start_index = 0
    #Load the saved results from the file and set the correct index
    if(os.path.isfile("./results/" + name + ".npy")):
        scores = np.load("./results/" + name + ".npy")
        scores = list(scores)
        start_index = 2*len(scores)

    #For the first 600 pairs of real and deepfake images of the dataset
    for idx in range(start_index,1200,2):

        #Print the index, get the inference and visualization frames and the ground truth label
        print((idx+2)//2)
        frame, label_real = ds[idx]
        true_label = int(label_real.reshape(-1, ).numpy()[0])
        visualize_frame, _ = ds_vis[idx]
        real_frame, _ = ds[idx+1]
        visualize_real_frame, _ = ds_vis[idx+1]

        #Set the index where the scores will be appended based on the category of the image
        pos=category.index(ds.df.loc[idx][0].split('/')[1])

        #Compute the predicted label
        labels = model(frame.unsqueeze(0).to(model.device)).cpu().detach().numpy()
        best_label = np.argmax(labels)

        #Compute the classifier's accuracy for the original image
        acc = 1 if (best_label == true_label) else 0

        if(evaluation_explanation_methods=="All"):
            #Call the different explanation methods and collect their evaluation scores
            scores_grad = computeGradCAMMetrics(model, frame, visualize_frame, real_frame, visualize_real_frame, best_label, true_label)
            scores_rise = computeRISEMetrics(model, frame, visualize_frame, real_frame, visualize_real_frame, best_label, true_label)
            scores_shap = computeSHAPMetrics(model, frame, visualize_frame, real_frame, visualize_real_frame, best_label, true_label)
            scores_lime = computeLIMEMetrics(model, frame, visualize_frame, _, visualize_real_frame, i_trans, best_label, true_label)
            scores_sobol = computeSOBOLMetrics(model, frame, visualize_frame, real_frame, visualize_real_frame, best_label, true_label)

            # Shift the scores based on the pos index, so the mean of the scores are computed for examples of the same category
            scores_original = 8*[np.nan]+pos*[np.nan]+[acc]+(7-pos)*[np.nan]
            scores_grad = pos*[np.nan]+[scores_grad[0]]+(3-pos)*[np.nan]+pos*[np.nan]+[scores_grad[1]]+(3-pos)*[np.nan]+pos*[np.nan]+[scores_grad[2]]+(3-pos)*[np.nan]+pos*[np.nan]+[scores_grad[3]]+(3-pos)*[np.nan]
            scores_rise = pos*[np.nan]+[scores_rise[0]]+(3-pos)*[np.nan]+pos*[np.nan]+[scores_rise[1]]+(3-pos)*[np.nan]+pos*[np.nan]+[scores_rise[2]]+(3-pos)*[np.nan]+pos*[np.nan]+[scores_rise[3]]+(3-pos)*[np.nan]
            scores_shap = pos*[np.nan]+[scores_shap[0]]+(3-pos)*[np.nan]+pos*[np.nan]+[scores_shap[1]]+(3-pos)*[np.nan]+pos*[np.nan]+[scores_shap[2]]+(3-pos)*[np.nan]+pos*[np.nan]+[scores_shap[3]]+(3-pos)*[np.nan]
            scores_lime = pos*[np.nan]+[scores_lime[0]]+(3-pos)*[np.nan]+pos*[np.nan]+[scores_lime[1]]+(3-pos)*[np.nan]+pos*[np.nan]+[scores_lime[2]]+(3-pos)*[np.nan]+pos*[np.nan]+[scores_lime[3]]+(3-pos)*[np.nan]
            scores_sobol = pos*[np.nan]+[scores_sobol[0]]+(3-pos)*[np.nan]+pos*[np.nan]+[scores_sobol[1]]+(3-pos)*[np.nan]+pos*[np.nan]+[scores_sobol[2]]+(3-pos)*[np.nan]+pos*[np.nan]+[scores_sobol[3]]+(3-pos)*[np.nan]

            # Accumulate the scores
            scores_img = np.array((scores_original, scores_grad, scores_rise, scores_shap, scores_lime, scores_sobol))
            scores.append(scores_img)

        elif(evaluation_explanation_methods=="GradCAM++"):
            scores_grad = computeGradCAMMetrics(model, frame, visualize_frame, real_frame, visualize_real_frame, best_label, true_label)

            scores_original = 8*[np.nan]+pos*[np.nan]+[acc]+(7-pos)*[np.nan]
            scores_grad = pos*[np.nan]+[scores_grad[0]]+(3-pos)*[np.nan]+pos*[np.nan]+[scores_grad[1]]+(3-pos)*[np.nan]+pos*[np.nan]+[scores_grad[2]]+(3-pos)*[np.nan]+pos*[np.nan]+[scores_grad[3]]+(3-pos)*[np.nan]

            scores_img = np.array((scores_original, scores_grad))
            scores.append(scores_img)

        elif (evaluation_explanation_methods == "RISE"):
            scores_rise = computeRISEMetrics(model, frame, visualize_frame, real_frame, visualize_real_frame, best_label, true_label)

            scores_original = 8*[np.nan]+pos*[np.nan]+[acc]+(7-pos)*[np.nan]
            scores_rise = pos*[np.nan]+[scores_rise[0]]+(3-pos)*[np.nan]+pos*[np.nan]+[scores_rise[1]]+(3-pos)*[np.nan]+pos*[np.nan]+[scores_rise[2]]+(3-pos)*[np.nan]+pos*[np.nan]+[scores_rise[3]]+(3-pos)*[np.nan]

            scores_img = np.array((scores_original, scores_rise))
            scores.append(scores_img)

        elif (evaluation_explanation_methods == "SHAP"):
            scores_shap = computeSHAPMetrics(model, frame, visualize_frame, real_frame, visualize_real_frame, best_label, true_label)

            scores_original = 8*[np.nan]+pos*[np.nan]+[acc]+(7-pos)*[np.nan]
            scores_shap = pos*[np.nan]+[scores_shap[0]]+(3-pos)*[np.nan]+pos*[np.nan]+[scores_shap[1]]+(3-pos)*[np.nan]+pos*[np.nan]+[scores_shap[2]]+(3-pos)*[np.nan]+pos*[np.nan]+[scores_shap[3]]+(3-pos)*[np.nan]

            scores_img = np.array((scores_original, scores_shap))
            scores.append(scores_img)

        elif (evaluation_explanation_methods == "LIME"):
            scores_lime = computeLIMEMetrics(model, frame, visualize_frame, _, visualize_real_frame, i_trans, best_label, true_label)

            scores_original = 8*[np.nan]+pos*[np.nan]+[acc]+(7-pos)*[np.nan]
            scores_lime = pos*[np.nan]+[scores_lime[0]]+(3-pos)*[np.nan]+pos*[np.nan]+[scores_lime[1]]+(3-pos)*[np.nan]+pos*[np.nan]+[scores_lime[2]]+(3-pos)*[np.nan]+pos*[np.nan]+[scores_lime[3]]+(3-pos)*[np.nan]

            scores_img = np.array((scores_original, scores_lime))
            scores.append(scores_img)

        elif (evaluation_explanation_methods == "SOBOL"):
            scores_sobol = computeSOBOLMetrics(model, frame, visualize_frame, real_frame, visualize_real_frame, best_label, true_label)

            scores_original = 8*[np.nan]+pos*[np.nan]+[acc]+(7-pos)*[np.nan]
            scores_sobol = pos*[np.nan]+[scores_sobol[0]]+(3-pos)*[np.nan]+pos*[np.nan]+[scores_sobol[1]]+(3-pos)*[np.nan]+pos*[np.nan]+[scores_sobol[2]]+(3-pos)*[np.nan]+pos*[np.nan]+[scores_sobol[3]]+(3-pos)*[np.nan]

            scores_img = np.array((scores_original, scores_sobol))
            scores.append(scores_img)

        else:
            print("Invalid explanation method(s) to evaluate")
            sys.exit(0)

        torch.cuda.empty_cache()
        #Save them on a file to avoid having to restart the whole procedure in case of the program crashing or something else going wrong
        np.save("./results/" + name + ".npy", scores)

    return