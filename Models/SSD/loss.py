import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from torch.autograd import Variable
from box_utils import match, log_sum_exp,cxcy_to_gcxgcy,xy_to_cxcy,gcxgcy_to_cxcy
from box_utils import match_ious, bbox_overlaps_iou, bbox_overlaps_giou, bbox_overlaps_diou, bbox_overlaps_ciou, decode,find_jaccard_overlap,cxcy_to_xy


class FocalLoss(nn.Module):
    """
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num, 1)
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = alpha
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        print(self.gamma)
    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs,dim= 1)
        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class IouLoss(nn.Module):

    def __init__(self,pred_mode = "Corner",size_sum=True,variances=None,losstype='Giou'):
        super(IouLoss, self).__init__()
        self.size_sum = size_sum
        self.pred_mode = pred_mode
        self.variances = variances
        self.loss = losstype

    def forward(self, loc_p, loc_t,prior_data=None):
        num = loc_p.shape[0] 
        
        if self.pred_mode == "Center":
            decoded_boxes = decode(loc_p, prior_data, self.variances)
        elif self.pred_mode == "Corner":
            decoded_boxes = loc_p
        if self.loss == 'Iou':
            loss = torch.sum(1.0 - bbox_overlaps_iou(decoded_boxes, loc_t))
        else:
            if self.loss == 'Giou':
                loss = torch.sum(1.0 - bbox_overlaps_giou(decoded_boxes,loc_t))
            else:
                if self.loss == 'Diou':
                    loss = torch.sum(1.0 - bbox_overlaps_diou(decoded_boxes,loc_t))
                else:
                    loss = torch.sum(1.0 - bbox_overlaps_ciou(decoded_boxes, loc_t))            
     
        if self.size_sum:
            loss = loss
        else:
            loss = loss/num
        return loss

class MultiBoxLoss(nn.Module):
    """
    SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    """
    The MultiBox loss, a loss function for object detection.
    This is a combination of:
    (1) a localization loss for the predicted locations of the boxes, and
    (2) a confidence loss for the predicted class scores.
    """

    def __init__(self, priors_cxcy, losstype="l1",threshold=0.75, neg_pos_ratio=3, alpha=1.):
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(self.priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.losstype=losstype
        
        if self.losstype=="l1":
            self.loss=nn.SmoothL1Loss()
        elif self.losstype=="giou":
            self.loss = IouLoss("Corner",size_sum=False,losstype="Giou")
        elif self.losstype=="diou":
            self.loss=IouLoss("Corner",size_sum=False,losstype="Diou")
        elif self.losstype=="ciou":
            self.loss=IouLoss("Corner",size_sum=False,losstype="Ciou")
        elif self.losstype=="l1+diou":
            self.loss1=nn.SmoothL1Loss()
            self.loss2=IouLoss("Corner",size_sum=False,losstype="Diou")
        else:
            self.loss=nn.SmoothL1Loss()
        
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)
        # self.cross_entropy = nn.BCELoss()

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        """
        Forward propagation.
        :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
        :param boxes: true  object bounding boxes in boundary coordinates, a list of N tensors
        :param labels: true object labels, a list of N tensors
        :return: multibox loss, a scalar
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).cuda()  # (N, 8732, 4)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).cuda()  # (N, 8732)

        # For each image
        for i in range(batch_size):
            n_objects = boxes[i].size(0)

            overlap = find_jaccard_overlap(boxes[i],
                                           self.priors_xy)  # (n_objects, 8732)

            # For each prior(default box), find the object that has the maximum overlap
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)  # (8732)

            # We don't want a situation where an object is not represented in our positive (non-background) priors -
            # 1. An object might not be the best object for all priors, and is therefore not in object_for_each_prior.
            # 2. All priors with the object may be assigned as background based on the threshold (0.5).

            # To remedy this -
            # First, find the prior that has the maximum overlap for each object.
            _, prior_for_each_object = overlap.max(dim=1)  # (N_o)

            # Then, assign each object to the corresponding maximum-overlap-prior. (This fixes 1.)
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).cuda()

            # To ensure these priors qualify, artificially give them an overlap of greater than 0.5. (This fixes 2.)
            overlap_for_each_prior[prior_for_each_object] = 1.

            object_for_each_prior=list(object_for_each_prior)
            object_for_each_prior=[i.item() for i in object_for_each_prior]
            chose_prior=overlap_for_each_prior < self.threshold
            chose_prior=[i.item() for i in chose_prior]

            # Labels for each prior
            label_for_each_prior = labels[i][object_for_each_prior]  # (8732)

            # Set priors whose overlaps with objects are less than the threshold to be background (no object)
            label_for_each_prior[chose_prior] = 0  # (8732)

            # Store
            true_classes[i] = label_for_each_prior

            if self.losstype=="l1":
                 # Encode center-size object coordinates into the form we regressed predicted boxes to
                true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)
            else:
                # Since all the variants of Iou Loss regress on the box coordinates, we need to transform from offset to box_coordinates
                true_locs[i] = boxes[i][object_for_each_prior]  # (8732, 4)
                # Convert the predicted locs from offset to Corner Size coordinates
                predicted_locs[i]=cxcy_to_xy(gcxgcy_to_cxcy(predicted_locs[i],self.priors_cxcy))

        # Identify priors that are positive (object/non-background)
        positive_priors = true_classes != 0  # (N, 8732)

        # LOCALIZATION LOSS

        # Localization loss is computed only over positive (non-background) priors
        if self.losstype=="l1":
            loc_loss = self.loss(predicted_locs[positive_priors], true_locs[positive_priors]) # (), scalar
        else:
            # The model always predicts offsets, so we have to change the offset to center size coordinates
            loc_loss=self.loss(predicted_locs[positive_priors],true_locs[positive_priors])

        # Note: indexing with a torch.uint8 (byte) tensor flattens the tensor when indexing is across multiple dimensions (N & 8732)
        # So, if predicted_locs has the shape (N, 8732, 4), predicted_locs[positive_priors] will have (total positives, 4)

        # CONFIDENCE LOSS

        # Confidence loss is computed over positive priors and the most difficult (hardest) negative priors in each image
        # That is, FOR EACH IMAGE,
        # we will take the hardest (neg_pos_ratio * n_positives) negative priors, i.e where there is maximum loss
        # This is called Hard Negative Mining - it concentrates on hardest negatives in each image, and also minimizes pos/neg imbalance

        # Number of positive and hard-negative priors per image
        n_positives = positive_priors.sum(dim=1)  # (N)
        n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)

        # First, find the loss for all priors
        conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes), true_classes.view(-1))  # (N * 8732)
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)  # (N, 8732)

        # We already know which priors are positive
        conf_loss_pos = conf_loss_all[positive_priors]  # (sum(n_positives))

        # Next, find which priors are hard-negative
        # To do this, sort ONLY negative priors in each image in order of decreasing loss and take top n_hard_negatives
        conf_loss_neg = conf_loss_all.clone()  # (N, 8732)
        conf_loss_neg[positive_priors] = 0.  # (N, 8732), positive priors are ignored (never in top n_hard_negatives)
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)  # (N, 8732), sorted by decreasing hardness
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).cuda()  # (N, 8732)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, 8732)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negatives))

        # As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()  # (), scalar

        # TOTAL LOSS

        return conf_loss + self.alpha * loc_loss

class DecoderLoss(nn.Module):
    """ Loss function for Decoder"""
    def __init__(self,losstype="mse"):
        super(DecoderLoss,self).__init__()
        self.losstype="mse"
        if self.losstype=="mse":
            self.loss=nn.MSELoss()
    
    def forward(self,image_reconstructed,target_image,pred_loc,actual_loc):    
        # Now we need to match the template of the reconstructed image with target image.
        total_loss=[]
        batch_size=len(pred_loc)

        for batch in range(batch_size):
            loss_per_batch=0
            get_overlap=find_jaccard_overlap(torch.stack(list(pred_loc[batch].values())),actual_loc[batch])
            indices=get_overlap.max(dim=1)[1]
            for k in range(len(indices)):
                image=self.get_image(target_image[batch],actual_loc[batch][indices[k]])
                image=self.resize_image(target_image[batch])
                ls=self.loss(image_reconstructed[batch][k].to("cuda"),image.to("cuda"))
                loss_per_batch+=ls
            total_loss.append(loss_per_batch)
        
        loss=torch.mean(torch.stack(total_loss))
        return loss

        
    def get_image(self,image,box):
        box=torch.round(box*image.size(-1))
        box=[int(b) for b in box.tolist()]
        return image[box[1]:box[3],box[0]:box[2]]
    
    def resize_image(self,image,size=(100,100)):
        transforms_pil=transforms.ToPILImage()
        transforms_tensor=transforms.ToTensor()

        image=transforms_tensor(transforms_pil(image).resize(size,Image.NEAREST))
        return image
        
    def forward(self,image_reconstructed,image_true):
        loss=torch.mean(self.loss(image_reconstructed,image_true))
        return loss     
