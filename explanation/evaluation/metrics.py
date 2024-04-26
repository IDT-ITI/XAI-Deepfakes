import numpy as np
from scipy.stats import pearsonr

class ExplanationMetrics():
    def __init__(self, model):
        self.model=model

    #Compute the comprehensiveness between original and adversarially attacked examples

    #x: Example with applied inference transformations
    #x_adv: Adversarially attacked example with applied inference transformations
    #label: The label to compute the metric
    #return: The difference between the scores of the original and the adversarially attacked example for label

    def disc(self, x, x_adv, label):
        score_original=self.model(x.unsqueeze(0).to(self.model.device)).cpu().detach().numpy()[0][label]
        score_adv=self.model(x_adv.unsqueeze(0).to(self.model.device)).cpu().detach().numpy()[0][label]
        return score_original-score_adv

    #Compute the stability of the output of an explanation method

    #image_explanation: Explanation saliency map of the original example
    #perturbed_image_explanation: Explanation saliency map of the perturbed/adversarially attacked example
    #return: The pearson correlation coefficient between the two explanation saliency maps

    def stability(self, image_explanation, perturbed_image_explanation):
        flat1 = image_explanation.flatten()
        flat2 = perturbed_image_explanation.flatten()
        return pearsonr(flat1,flat2).correlation