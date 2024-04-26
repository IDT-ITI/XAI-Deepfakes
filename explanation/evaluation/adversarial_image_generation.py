import numpy as np
from skimage.segmentation import slic
import torch

class AdversarialImageGeneration():
    def __init__(self, x, F, s, n, itr, e, a):
        self.image=x
        self.model=F
        self.search_variance=s
        self.num_of_samples=n
        self.image_dimensionality=x.shape[-1]
        self.max_iter=itr
        self.max_dist=e
        self.learn_rate=a

    def toNumpy(self,x):
        return x.permute(1, 2, 0).numpy()

    def toTorch(self,x):
        return torch.from_numpy(x).permute(2, 0, 1)

    #Extract the segments from the saliency map
    def getSegmentsfromHeatmap(self,heatmap):

        #Segment the image with the SLIC algorithm
        segments = slic(self.toNumpy(self.image), n_segments=50)-1

        #Compute the segment scores by calculating the mean value of the saliency map for the indices of each segment
        segment_scores=[]
        for seg in range(len(np.unique(segments))):
            seg_score=np.sum(heatmap[np.where(segments==seg)]) / np.unique(segments, return_counts=True)[1][seg]
            segment_scores.append(seg_score)
        ranked_segments=np.flip(np.argsort(segment_scores))

        #Return the segmentation and the sorted segments based on their segment scores
        return [segments,ranked_segments]

    #Create the binary mask of the top specified segments
    def getSegmentsMask(self,segments,ranked_segments,num_of_segments):
        mask=np.zeros((self.image_dimensionality,self.image_dimensionality))

        #Mask out the top number of segments sequentially, specified by num_of_segments
        for n in range(num_of_segments):
            mask[segments==ranked_segments[n]]=1

        #Return the binary mask
        return mask

    #Create an adversarially attacked image
    #mask: The binary mask
    #score_diff: The maximum difference between the scores of the original image and an adversarially attacked image img_diff
    #            This image is a slight variation of the original, used to compute the stability of the xai method
    #return: Adversarially attacked image and adversarially attacked image with maximum score difference score_diff
    def generateAdversarialImage(self,mask,score_diff=None):

        #Get the indices from the binary mask
        indices = np.where(mask == 1)

        #Get the score of the real label
        def getRealScore(scores):
            scores=scores.reshape(-1,)
            return scores[0]

        #Check if the score of the real label has the maximum value (image is real)
        def isRealImage(scores):
            scores = scores.reshape(-1,)
            return 1 if (np.argmax(scores) == 0) else 0

        #Clone the original image and set img_diff to None
        img = self.image.clone().detach()
        img_diff = None

        #If score_diff is not None
        if(score_diff!=None):
            #Compute the real label score of the original image
            initial_score=getRealScore(self.model(img.unsqueeze(0).to(self.model.device)).cpu().detach().numpy())

        #For the number of iterations
        for i in range(self.max_iter):

            #Check if the adversarially attacked image is considered as real by the model
            if (isRealImage(self.model(img.unsqueeze(0).to(self.model.device)).cpu().detach().numpy())):
                #If true then return it along with img_diff
                return [img,img_diff]

            #If score_diff is not None
            if(score_diff!=None):
                #Compute the score of the adversarially attacked image
                current_score=getRealScore(self.model(img.unsqueeze(0).to(self.model.device)).cpu().detach().numpy())
                #If the difference between the scores of the original image and the adversarially attacked image is less than score_diff
                if(current_score-initial_score<=score_diff):
                    #Clone the adversarially attacked image and set it to img_diff
                    img_diff=img.clone().detach()

            g=0
            #For the number of samples in each iteration
            for j in range(self.num_of_samples):

                #Create gaussian noise values with mean 0 and variance 1
                mean=np.zeros(self.image_dimensionality)
                cov_matrix=np.eye(self.image_dimensionality)
                uj=np.random.multivariate_normal(mean,cov_matrix,(self.image_dimensionality))
                uj=uj.reshape(self.image_dimensionality,self.image_dimensionality,-1)

                #Create a perturbation by adding the gaussian noise to the indices of the adversarially attacked image
                pert_img=self.toNumpy(img.clone().detach())
                pert_img[indices]=pert_img[indices]+(self.search_variance*uj[indices])
                pert_img=self.toTorch(pert_img).unsqueeze(0).to(self.model.device)
                #Add on g the real label score derived from the perturbation multiplied with the gaussian noise of the indices
                g=g+(getRealScore(self.model(pert_img).cpu().detach().numpy())*uj[indices])
                del pert_img

                #Create a perturbation by subtracting the gaussian noise from the indices of the adversarially attacked image
                pert_img=self.toNumpy(img.clone().detach())
                pert_img[indices]=pert_img[indices]-(self.search_variance*uj[indices])
                pert_img=self.toTorch(pert_img).unsqueeze(0).to(self.model.device)
                #Subtract from g the real label score derived from the perturbation multiplied with the gaussian noise of the indices
                g=g-(getRealScore(self.model(pert_img).cpu().detach().numpy())*uj[indices])
                del pert_img

            g=g/(2*self.num_of_samples*self.search_variance)

            img = self.toNumpy(img)
            #Update the adversarially attacked image by adding or subtracting the learning rate from the indices based on the sign of g
            img[indices]=img[indices] + np.clip((self.learn_rate*np.sign(g)), None, self.max_dist)
            img = self.toTorch(img)

        #Return the adversarially attacked images
        return [img,img_diff]

