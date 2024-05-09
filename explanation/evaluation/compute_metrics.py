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
    disc_plus = []
    a_imgs=[]
    a_diff=None

    #For the range of the number of segments
    for i in range(num_of_segments):
        #Create the segmentation mask by masking out i segments sequentially
        mask = evaluation.getSegmentsMask(segments, ranked_segments, i+1)
        #In the case where all of the desired segments are masked
        if (i == num_of_segments - 1):
            #Produce a slight variation of the original image that has maximum score difference of 10/100 along the adversarially attacked
            a_img, a_diff = evaluation.generateAdversarialImage(mask, 10/100)
        else:
            #Produce the adversarially attacked image
            a_img, _ = evaluation.generateAdversarialImage(mask)

        #Compute the comprehensiveness metrics (disc plus) between the original and the adversarially attacked image for the label score
        result = metrics.disc(i_img, a_img, label)
        #Accumulate the adversarial images and the scores
        a_imgs.append(a_img)
        disc_plus.append(result)

    #Return the disc plus scores, the adversarially attacked images on top 1, top 1-2, top 1-2-3 and the slight variation
    return [disc_plus,a_imgs,a_diff]

def computeGradCAMMetrics(model, i_img, v_img, best_label, true_label):

    scores = []
    #Create the ExplanationMetrics and AdversarialImageGeneration objects
    metrics = ExplanationMetrics(model)
    evaluation = AdversarialImageGeneration(i_img, model, 0.001, 40, 50, 16 / 255, 1 / 255)

    #Call the explanation method to produce the heatmap for the model's predicted label for the image
    heatmap = GradCAM(i_img, v_img.permute(1, 2, 0).numpy(), best_label, model, visualize=False)
    #In the rare occasion where the heatmap returned by GradCAM contains nan values
    if (np.any(np.isnan(heatmap))):
        #Recall the explanation method
        heatmap = GradCAM(i_img, v_img.permute(1, 2, 0).numpy(), best_label, model, visualize=False)

    #If the heatmap doesn't contain nan values
    if (not np.any(np.isnan(heatmap))):
        #Segment the heatmap into segments and rank them based on the values of the heatmap for the indices of each one
        segments, ranked_segments = evaluation.getSegmentsfromHeatmap(heatmap)
        #Compute the comprehensiveness scores
        disc_plus, a_imgs, a_diff = computeDiscScores(i_img, evaluation, metrics, segments, ranked_segments, best_label)

        #If the image was not already classified as real by the model (meaning the adversarial attack occured and a_diff in not None)
        if(a_diff!=None):
            v_a_diff = inverseNormalization(a_diff)
            #Compute the heatmap for a slight variation of the original image
            heatmap_adv_diff = GradCAM(a_diff, v_a_diff.permute(1, 2, 0).numpy(), best_label, model, visualize=False)
            #Get the stability score of the explanation method by comparing the similarity of the heatmaps
            stability=metrics.stability(heatmap, heatmap_adv_diff)
        else:
            stability = np.nan

        #Compute the classifier's accuracy for the adversarially attacked image on the top 1, top 1-2 and top 1-2-3 segments
        adv_label_top_1 = np.argmax(model(a_imgs[0].unsqueeze(0).to(model.device)).cpu().detach().numpy())
        adv_acc_top_1=1 if(adv_label_top_1==true_label) else 0

        adv_label_top_2 = np.argmax(model(a_imgs[1].unsqueeze(0).to(model.device)).cpu().detach().numpy())
        adv_acc_top_2=1 if(adv_label_top_2==true_label) else 0

        adv_label_top_3 = np.argmax(model(a_imgs[2].unsqueeze(0).to(model.device)).cpu().detach().numpy())
        adv_acc_top_3=1 if(adv_label_top_3==true_label) else 0

        #Accumulate the scores
        scores.append(disc_plus[0]); scores.append(disc_plus[1]); scores.append(disc_plus[2])
        scores.append(stability)
        scores.append(adv_acc_top_1); scores.append(adv_acc_top_2); scores.append(adv_acc_top_3)

    #If the heatmap still contains nan values
    else:
        #Ignore the image and set nan values to the scores
        scores.append(np.nan); scores.append(np.nan); scores.append(np.nan)
        scores.append(np.nan)
        scores.append(np.nan); scores.append(np.nan); scores.append(np.nan)

    torch.cuda.empty_cache()
    #Return the scores
    return scores

def computeRISEMetrics(model, i_img, v_img, best_label, true_label):

    scores = []
    #Create the ExplanationMetrics and AdversarialImageGeneration objects
    metrics = ExplanationMetrics(model)
    evaluation = AdversarialImageGeneration(i_img, model, 0.001, 40, 50, 16 / 255, 1 / 255)

    #Call the explanation method to produce the heatmap for the model's predicted label for the image
    heatmap = RISE(i_img, v_img.unsqueeze(0), best_label, model, visualize=False)

    #Segment the heatmap into segments and rank them based on the values of the heatmap for the indices of each one
    segments, ranked_segments = evaluation.getSegmentsfromHeatmap(heatmap)
    #Compute the comprehensiveness scores
    disc_plus, a_imgs, a_diff = computeDiscScores(i_img, evaluation, metrics, segments, ranked_segments, best_label)

    #If the image was not already classified as real by the model (meaning the adversarial attack occured and a_diff in not None)
    if(a_diff!=None):
        v_a_diff = inverseNormalization(a_diff)
        #Compute the heatmap for a slight variation of the original image
        heatmap_adv_diff = RISE(a_diff, v_a_diff.unsqueeze(0), best_label, model, visualize=False)
        #Get the stability score of the explanation method by comparing the similarity of the heatmaps
        stability=metrics.stability(heatmap, heatmap_adv_diff)
    else:
        stability = np.nan

    #Compute the classifier's accuracy for the adversarially attacked image on the top 1, top 1-2 and top 1-2-3 segments
    adv_label_top_1 = np.argmax(model(a_imgs[0].unsqueeze(0).to(model.device)).cpu().detach().numpy())
    adv_acc_top_1 = 1 if (adv_label_top_1 == true_label) else 0

    adv_label_top_2 = np.argmax(model(a_imgs[1].unsqueeze(0).to(model.device)).cpu().detach().numpy())
    adv_acc_top_2 = 1 if (adv_label_top_2 == true_label) else 0

    adv_label_top_3 = np.argmax(model(a_imgs[2].unsqueeze(0).to(model.device)).cpu().detach().numpy())
    adv_acc_top_3 = 1 if (adv_label_top_3 == true_label) else 0

    #Accumulate the scores
    scores.append(disc_plus[0]); scores.append(disc_plus[1]); scores.append(disc_plus[2])
    scores.append(stability)
    scores.append(adv_acc_top_1); scores.append(adv_acc_top_2); scores.append(adv_acc_top_3)

    torch.cuda.empty_cache()
    #Return the scores
    return scores

def computeSHAPMetrics(model, i_img, v_img, best_label, true_label):

    scores = []
    #Create the ExplanationMetrics and AdversarialImageGeneration objects
    metrics = ExplanationMetrics(model)
    evaluation = AdversarialImageGeneration(i_img, model, 0.001, 40, 50, 16 / 255, 1 / 255)

    #Call the explanation method to produce the heatmap for the model's predicted label for the image
    heatmap = SHAP(i_img, v_img.permute(1, 2, 0).numpy(), best_label, model, visualize=False)

    #Segment the heatmap into segments and rank them based on the values of the heatmap for the indices of each one
    segments, ranked_segments = evaluation.getSegmentsfromHeatmap(heatmap)
    #Compute the comprehensiveness scores
    disc_plus, a_imgs, a_diff = computeDiscScores(i_img, evaluation, metrics, segments, ranked_segments, best_label)

    #If the image was not already classified as real by the model (meaning the adversarial attack occured and a_diff in not None)
    if(a_diff!=None):
        v_a_diff = inverseNormalization(a_diff)
        #Compute the heatmap for a slight variation of the original image
        heatmap_adv_diff = SHAP(a_diff, v_a_diff.permute(1, 2, 0).numpy(), best_label, model, visualize=False)
        #Get the stability score of the explanation method by comparing the similarity of the heatmaps
        stability=metrics.stability(heatmap, heatmap_adv_diff)
    else:
        stability = np.nan

    #Compute the classifier's accuracy for the adversarially attacked image on the top 1, top 1-2 and top 1-2-3 segments
    adv_label_top_1 = np.argmax(model(a_imgs[0].unsqueeze(0).to(model.device)).cpu().detach().numpy())
    adv_acc_top_1 = 1 if (adv_label_top_1 == true_label) else 0

    adv_label_top_2 = np.argmax(model(a_imgs[1].unsqueeze(0).to(model.device)).cpu().detach().numpy())
    adv_acc_top_2 = 1 if (adv_label_top_2 == true_label) else 0

    adv_label_top_3 = np.argmax(model(a_imgs[2].unsqueeze(0).to(model.device)).cpu().detach().numpy())
    adv_acc_top_3 = 1 if (adv_label_top_3 == true_label) else 0

    #Accumulate the scores
    scores.append(disc_plus[0]); scores.append(disc_plus[1]); scores.append(disc_plus[2])
    scores.append(stability)
    scores.append(adv_acc_top_1); scores.append(adv_acc_top_2); scores.append(adv_acc_top_3)

    torch.cuda.empty_cache()
    #Return the scores
    return scores

def computeLIMEMetrics(model, i_img, v_img, i_trans, best_label, true_label,num_of_segments=3):
    #Custom segmentation function passed in LIME to segment the image
    def custom_seg(image):
        return slic(image, n_segments=50) - 1

    scores = []
    #Create the ExplanationMetrics and AdversarialImageGeneration objects
    metrics = ExplanationMetrics(model)
    evaluation = AdversarialImageGeneration(i_img, model, 0.001, 40, 50, 16 / 255, 1 / 255)

    #Call the explanation method to produce the heatmap(binary mask) for the model's predicted label for the image
    heatmap, segments, exp = LIME(v_img.permute(1, 2, 0).numpy(), i_trans, best_label, model, custom_seg, visualize=False)

    #Extract the segments and rank them directly from LIME
    ranked_segments = [x[0] for x in exp if x[1] > 0]

    #If the number of segments returned by LIME are enough
    if(len(ranked_segments)>=num_of_segments):
        #Compute the comprehensiveness scores
        disc_plus, a_imgs, a_diff = computeDiscScores(i_img, evaluation, metrics, segments, ranked_segments, best_label)

        #If the image was not already classified as real by the model (meaning the adversarial attack occured and a_diff in not None)
        if (a_diff!=None):
            v_a_img_diff = inverseNormalization(a_diff)
            #Compute the heatmap for a slight variation of the original image
            heatmap_adv_diff, _, _ = LIME(v_a_img_diff.permute(1, 2, 0).numpy(), i_trans, best_label, model, custom_seg, visualize=False)
            #Get the stability score of the explanation method by comparing the similarity of the heatmaps
            stability = metrics.stability(heatmap, heatmap_adv_diff)
        else:
            stability = np.nan

        #Compute the classifier's accuracy for the adversarially attacked image on the top 1, top 1-2 and top 1-2-3 segments
        adv_label_top_1 = np.argmax(model(a_imgs[0].unsqueeze(0).to(model.device)).cpu().detach().numpy())
        adv_acc_top_1 = 1 if (adv_label_top_1 == true_label) else 0

        adv_label_top_2 = np.argmax(model(a_imgs[1].unsqueeze(0).to(model.device)).cpu().detach().numpy())
        adv_acc_top_2 = 1 if (adv_label_top_2 == true_label) else 0

        adv_label_top_3 = np.argmax(model(a_imgs[2].unsqueeze(0).to(model.device)).cpu().detach().numpy())
        adv_acc_top_3 = 1 if (adv_label_top_3 == true_label) else 0

        #Accumulate the scores
        scores.append(disc_plus[0]); scores.append(disc_plus[1]); scores.append(disc_plus[2])
        scores.append(stability)
        scores.append(adv_acc_top_1); scores.append(adv_acc_top_2); scores.append(adv_acc_top_3)

    else:
        #Ignore the image and set nan values to the scores
        scores.append(np.nan); scores.append(np.nan); scores.append(np.nan)
        scores.append(np.nan)
        scores.append(np.nan); scores.append(np.nan); scores.append(np.nan)

    torch.cuda.empty_cache()
    #Return the scores
    return scores

def computeSOBOLMetrics(model, i_img, v_img, best_label, true_label):

    scores = []
    #Create the ExplanationMetrics and AdversarialImageGeneration objects
    metrics = ExplanationMetrics(model)
    evaluation = AdversarialImageGeneration(i_img, model, 0.001, 40, 50, 16 / 255, 1 / 255)

    #Call the explanation method to produce the heatmap for the model's predicted label for the image
    heatmap = SOBOL(i_img, v_img, best_label, model, visualize=False)

    #Segment the heatmap into segments and rank them based on the values of the heatmap for the indices of each one
    segments, ranked_segments = evaluation.getSegmentsfromHeatmap(heatmap)
    #Compute the comprehensiveness scores
    disc_plus, a_imgs, a_diff = computeDiscScores(i_img, evaluation, metrics, segments, ranked_segments, best_label)

    #If the image was not already classified as real by the model (meaning the adversarial attack occured and a_diff in not None)
    if (a_diff!=None):
        v_a_img_diff = inverseNormalization(a_diff)
        #Compute the heatmap for a slight variation of the original image
        heatmap_adv_diff = SOBOL(a_diff, v_a_img_diff, best_label, model, visualize=False)
        #Get the stability score of the explanation method by comparing the similarity of the heatmaps
        stability = metrics.stability(heatmap, heatmap_adv_diff)
    else:
        stability = np.nan

    #Compute the classifier's accuracy for the adversarially attacked image on the top 1, top 1-2 and top 1-2-3 segments
    adv_label_top_1 = np.argmax(model(a_imgs[0].unsqueeze(0).to(model.device)).cpu().detach().numpy())
    adv_acc_top_1 = 1 if (adv_label_top_1 == true_label) else 0

    adv_label_top_2 = np.argmax(model(a_imgs[1].unsqueeze(0).to(model.device)).cpu().detach().numpy())
    adv_acc_top_2 = 1 if (adv_label_top_2 == true_label) else 0

    adv_label_top_3 = np.argmax(model(a_imgs[2].unsqueeze(0).to(model.device)).cpu().detach().numpy())
    adv_acc_top_3 = 1 if (adv_label_top_3 == true_label) else 0

    #Accumulate the scores
    scores.append(disc_plus[0]); scores.append(disc_plus[1]); scores.append(disc_plus[2])
    scores.append(stability)
    scores.append(adv_acc_top_1); scores.append(adv_acc_top_2); scores.append(adv_acc_top_3)

    torch.cuda.empty_cache()
    #Return the scores
    return scores

def computeExplanationMetrics(model,ds,ds_vis,i_trans,evaluation_explanation_methods):

    name="results_"+evaluation_explanation_methods
    scores=[]
    #Different deepfake categories
    category=["Deepfakes","Face2Face","FaceSwap","NeuralTextures"]

    start_index = 0
    #Load the saved results from the file and set the correct index
    if(os.path.isfile("./results/"+name+".npy")):
        scores = np.load("./results/"+name+".npy")
        scores = list(scores)
        start_index = len(scores)

    #For every image of the dataset
    for idx in range(start_index, len(ds.df)):

        #Print the index, get the inference and visualization frames and the ground truth label
        print(idx+1)
        frame, true_label = ds[idx]
        visualize_frame, _ = ds_vis[idx]

        #Set the index where the scores will be appended based on the category of the image
        pos=category.index(ds.df.loc[idx][0].split('/')[1])

        #Compute the predicted label
        labels = model(frame.unsqueeze(0).to(model.device)).cpu().detach().numpy()
        best_label = np.argmax(labels)

        #Compute the classifier's accuracy for the original image
        acc=1 if (best_label==true_label) else 0

        if(evaluation_explanation_methods=="All"):
            # Call the different explanation methods and collect their evaluation scores
            scores_grad=computeGradCAMMetrics(model,frame,visualize_frame,best_label,true_label)
            scores_rise=computeRISEMetrics(model,frame,visualize_frame,best_label,true_label)
            scores_shap=computeSHAPMetrics(model,frame,visualize_frame,best_label,true_label)
            scores_lime=computeLIMEMetrics(model,frame,visualize_frame,i_trans,best_label,true_label)
            scores_sobol=computeSOBOLMetrics(model,frame,visualize_frame,best_label,true_label)

            # Shift the scores based on the pos index, so the mean of the scores are computed for examples of the same category
            scores_original = 16 * [np.nan] + [acc] + (3 * pos) * [np.nan] + [acc] + (11 - (3 * pos)) * [np.nan]
            scores_grad = (4 * pos) * [np.nan] + scores_grad[:-3] + (12 - (4 * pos)) * [np.nan] + [np.nan] + (3 * pos) * [np.nan] + scores_grad[-3:] + (9 - (3 * pos)) * [np.nan]
            scores_rise = (4 * pos) * [np.nan] + scores_rise[:-3] + (12 - (4 * pos)) * [np.nan] + [np.nan] + (3 * pos) * [np.nan] + scores_rise[-3:] + (9 - (3 * pos)) * [np.nan]
            scores_shap = (4 * pos) * [np.nan] + scores_shap[:-3] + (12 - (4 * pos)) * [np.nan] + [np.nan] + (3 * pos) * [np.nan] + scores_shap[-3:] + (9 - (3 * pos)) * [np.nan]
            scores_lime = (4 * pos) * [np.nan] + scores_lime[:-3] + (12 - (4 * pos)) * [np.nan] + [np.nan] + (3 * pos) * [np.nan] + scores_lime[-3:] + (9 - (3 * pos)) * [np.nan]
            scores_sobol = (4 * pos) * [np.nan] + scores_sobol[:-3] + (12 - (4 * pos)) * [np.nan] + [np.nan] + (3 * pos) * [np.nan] + scores_sobol[-3:] + (9 - (3 * pos)) * [np.nan]

            # Accumulate the scores
            scores_img = np.array((scores_original, scores_grad, scores_rise, scores_shap, scores_lime, scores_sobol))
            scores.append(scores_img)

        elif(evaluation_explanation_methods=="GradCAM++"):
            scores_grad = computeGradCAMMetrics(model, frame, visualize_frame, best_label, true_label)

            scores_original = 16 * [np.nan] + [acc] + (3 * pos) * [np.nan] + [acc] + (11 - (3 * pos)) * [np.nan]
            scores_grad = (4 * pos) * [np.nan] + scores_grad[:-3] + (12 - (4 * pos)) * [np.nan] + [np.nan] + (3 * pos) * [np.nan] + scores_grad[-3:] + (9 - (3 * pos)) * [np.nan]

            scores_img = np.array((scores_original, scores_grad))
            scores.append(scores_img)

        elif (evaluation_explanation_methods == "RISE"):
            scores_rise = computeRISEMetrics(model, frame, visualize_frame, best_label, true_label)

            scores_original = 16 * [np.nan] + [acc] + (3 * pos) * [np.nan] + [acc] + (11 - (3 * pos)) * [np.nan]
            scores_rise = (4 * pos) * [np.nan] + scores_rise[:-3] + (12 - (4 * pos)) * [np.nan] + [np.nan] + (3 * pos) * [np.nan] + scores_rise[-3:] + (9 - (3 * pos)) * [np.nan]

            scores_img = np.array((scores_original, scores_rise))
            scores.append(scores_img)

        elif (evaluation_explanation_methods == "SHAP"):
            scores_shap = computeSHAPMetrics(model, frame, visualize_frame, best_label, true_label)

            scores_original = 16 * [np.nan] + [acc] + (3 * pos) * [np.nan] + [acc] + (11 - (3 * pos)) * [np.nan]
            scores_shap = (4 * pos) * [np.nan] + scores_shap[:-3] + (12 - (4 * pos)) * [np.nan] + [np.nan] + (3 * pos) * [np.nan] + scores_shap[-3:] + (9 - (3 * pos)) * [np.nan]

            scores_img = np.array((scores_original, scores_shap))
            scores.append(scores_img)

        elif (evaluation_explanation_methods == "LIME"):
            scores_lime = computeLIMEMetrics(model, frame, visualize_frame, i_trans, best_label, true_label)

            scores_original = 16 * [np.nan] + [acc] + (3 * pos) * [np.nan] + [acc] + (11 - (3 * pos)) * [np.nan]
            scores_lime = (4 * pos) * [np.nan] + scores_lime[:-3] + (12 - (4 * pos)) * [np.nan] + [np.nan] + (3 * pos) * [np.nan] + scores_lime[-3:] + (9 - (3 * pos)) * [np.nan]

            scores_img = np.array((scores_original, scores_lime))
            scores.append(scores_img)

        elif (evaluation_explanation_methods == "SOBOL"):
            scores_sobol = computeSOBOLMetrics(model, frame, visualize_frame, best_label, true_label)

            scores_original = 16 * [np.nan] + [acc] + (3 * pos) * [np.nan] + [acc] + (11 - (3 * pos)) * [np.nan]
            scores_sobol = (4 * pos) * [np.nan] + scores_sobol[:-3] + (12 - (4 * pos)) * [np.nan] + [np.nan] + (3 * pos) * [np.nan] + scores_sobol[-3:] + (9 - (3 * pos)) * [np.nan]

            scores_img = np.array((scores_original, scores_sobol))
            scores.append(scores_img)

        else:
            print("Invalid explanation method(s) to evaluate")
            sys.exit(0)

        torch.cuda.empty_cache()
        #Save them on a file to avoid having to restart the whole procedure in case of the program crashing or something else going wrong
        np.save("./results/"+name+".npy",scores)

    return