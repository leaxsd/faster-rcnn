# %% ########################################## [md]
# # Faster R-CNN (PyTorch)
#
# `build Faster R-CNN from scratch`
# ---
# %% ##############################################
import torch
import torchvision
import torch.nn.functional as F
import numpy as np

# %% ########################################## [md]
# ## 1. Backend CNN
# `Create a backend CNN based on VGG16`
# %% ###############################################
# Pass dummy image trough VGG16
image = torch.zeros((1, 3, 800, 800)).float()
sub_sample = 16 # this is because output of VGG16 downsampling input 16 times

dummy_img = torch.zeros((1, 3, 800, 800)).float()

# download VGG16 pretrained
model = torchvision.models.vgg16(pretrained=True)
fe = list(model.features)

# trim VGG16
req_features = []
k = dummy_img.clone()
for i in fe:
    k = i(k)
    if k.size()[2] < 800//16:
        break
    req_features.append(i)
    out_channels = k.size()[1]

# this is the backend CNN
faster_rcnn_fe_extractor = torch.nn.Sequential(*req_features)

# Feature Map
out_map = faster_rcnn_fe_extractor(image) # [1,512, 50, 50]
print('backend CNN based on VGG16 : \n', '-'*20, '\n', faster_rcnn_fe_extractor[-5:])

# %% ########################################## [md]
#  ## 2. Anchor boxes
# `2.1 -Create anchors for each pixel on feature map`
# %% ###############################################

# 9 anchors to pixel
ratios = [0.5, 1, 2]
anchor_scales = [8, 16, 32]

fe_size = (800//16)
ctr_x = np.arange(16, (fe_size+1) * 16, 16)
ctr_y = np.arange(16, (fe_size+1) * 16, 16)

ctr = np.zeros((len(ctr_x) * len(ctr_y), 2), dtype=np.float32)

index = 0
for x in range(len(ctr_x)):
    for y in range(len(ctr_y)):
        ctr[index, 1] = ctr_x[x] - 8
        ctr[index, 0] = ctr_y[y] - 8
        index +=1

print('Center pixels in feature map', ctr.shape, '\n',  '-'*20, '\n', ctr[-5:])

# %% [md]
#  `2.2 - Generate anchor boxes for all pixels`
# %%

anchors = np.zeros(((fe_size * fe_size * 9), 4))
index = 0
for c in ctr:
  ctr_y, ctr_x = c
  for i in range(len(ratios)):
    for j in range(len(anchor_scales)):
        h = sub_sample * anchor_scales[j] * np.sqrt(ratios[i])
        w = sub_sample * anchor_scales[j] * np.sqrt(1./ ratios[i])
        anchors[index, 0] = ctr_y - h / 2.
        anchors[index, 1] = ctr_x - w / 2.
        anchors[index, 2] = ctr_y + h / 2.
        anchors[index, 3] = ctr_x + w / 2.
        index += 1

print('Generated Anchors :', anchors.shape, '\n', '-'*20, '\n', anchors[-5:])

# %% [md]
# `2.3 - Assign labels to bounding boxes`
#
# > (a) The anchor/anchors with the highest Intersection-over-Union(IoU) overlap with a ground-truth-box or \
# > (b) An anchor that has an IoU overlap higher than 0.7 with ground-truth box. \
# > (c) We assign a negative label to a non-positive anchor if its IoU ratio is lower than 0.3 for all ground-truth boxes. \
# > (d) Anchors that are neither positive nor negative do not contribute to the training objective.
# %%
# define example ground-truth
bbox = np.asarray([[20, 30, 400, 500], [300, 400, 500, 600]], dtype=np.float32) # [y1, x1, y2, x2] format
labels = np.asarray([6, 8], dtype=np.int8) # 0 represents background
# %% ###############################################
# valid anchors indexes
inside_index = np.where((anchors[:, 0] >= 0) &
                        (anchors[:, 1] >= 0) &
                        (anchors[:, 2] <= 800) &
                        (anchors[:, 3] <= 800)
                        )[0]

# create label and assign -1 to all
label = np.empty((len(inside_index), ), dtype=np.int32)
label.fill(-1)

# valid anchors
valid_anchors = anchors[inside_index]

# Calculate IOU between ground-truth and anchor bboxes

ious = np.empty((len(valid_anchors), 2), dtype=np.float32)
ious.fill(0)

for n1, i in enumerate(valid_anchors):
    ya1, xa1, ya2, xa2 = i
    anchor_area = (ya2 - ya1) * (xa2 -xa1)
    for n2, j in enumerate(bbox):
        yb1, xb1, yb2, xb2 = j
        box_area = (yb2 - yb1) * (xb2 -xb1)

        inter_x1 = max([xb1, xa1])
        inter_y1 = max([yb1, ya1])
        inter_x2 = min([xb2, xa2])
        inter_y2 = min([yb2, ya2])

        if (inter_x1 < inter_x2) and (inter_y1 < inter_y2):
            inter_area = (inter_y2 -inter_y1) * (inter_x2 - inter_x1)
            iou = inter_area / (anchor_area + box_area - inter_area)
        else:
            iou = 0.

        ious[n1, n2] = iou

gt_argmax_ious = ious.argmax(axis=0) # Tells which ground truth object has max iou with each anchor.
gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]
argmax_ious = ious.argmax(axis=1)
max_ious = ious[np.arange(len(inside_index)), argmax_ious] # Tells the max_iou with ground truth object with each anchor.
gt_argmax_ious = np.where(ious == gt_max_ious)[0] # Tells the anchors with the highest Intersection-over-Union (IoU) overlap with a ground-truth box.

# %%
# IOU positive and negative thresholds
pos_iou_threshold  = 0.7
neg_iou_threshold = 0.3

# %% [md]
#  `2.4 - Evaluate conditions (a), (b) and (c)`
# %%
# Assign negitive label (0) to all the anchor boxes which have max_iou less than negitive threshold [c]
label[max_ious < neg_iou_threshold] = 0

# Assign positive label (1) to all the anchor boxes which have highest IoU overlap with a ground-truth box [a]
label[gt_argmax_ious] = 1

# Assign positive label (1) to all the anchor boxes which have max_iou greater than positive threshold [b]
label[max_ious >= pos_iou_threshold] = 1

# %% [md]
# `2.5 - Assign locations to anchor boxes`
# %%
pos_ratio = 0.5
n_sample = 256

# %%
# positive samples
n_pos = pos_ratio * n_sample
pos_index = np.where(label == 1)[0]
if len(pos_index) > n_pos:
    disable_index = np.random.choice(pos_index, size=(len(pos_index) - n_pos), replace=False)
    label[disable_index] = -1

# negative sample
n_neg = n_sample - np.sum(label == 1)
neg_index = np.where(label == 0)[0]

if len(neg_index) > n_neg:
    disable_index = np.random.choice(neg_index, size=(len(neg_index) - n_neg), replace = False)
    label[disable_index] = -1

# %% [md]
# `2.6 - Parameterization`
#
# > $
# t_{x} = (x - x_{a})/w_{a} \\
# t_{y} = (y - y_{a})/h_{a} \\
# t_{w} = log(w/ w_a) \\
# t_{h} = log(h/ h_a) \\
# $
#
# > $x, y , w, h$ are the groud truth box center co-ordinates which has maxmimum iou with corresponding
# anchor, width and height. $x_a, y_a, h_a, w_a$ are anchor boxes center cooridinates, width and height.
# %%
# For each anchor box, find the ground-truth object which has max_iou as base
max_iou_bbox = bbox[argmax_ious]

# valid anchors height, width and center
height = valid_anchors[:, 2] - valid_anchors[:, 0]
width = valid_anchors[:, 3] - valid_anchors[:, 1]
ctr_y = valid_anchors[:, 0] + 0.5 * height
ctr_x = valid_anchors[:, 1] + 0.5 * width

# maximum iou bbox associate with each anchor height, width and center
base_height = max_iou_bbox[:, 2] - max_iou_bbox[:, 0]
base_width = max_iou_bbox[:, 3] - max_iou_bbox[:, 1]
base_ctr_y = max_iou_bbox[:, 0] + 0.5 * base_height
base_ctr_x = max_iou_bbox[:, 1] + 0.5 * base_width

# Use the above formulas to find the loc
eps = np.finfo(height.dtype).eps # minimum delta
height = np.maximum(height, eps)
width = np.maximum(width, eps)

dy = (base_ctr_y - ctr_y) / height
dx = (base_ctr_x - ctr_x) / width
dh = np.log(base_height / height)
dw = np.log(base_width / width)

# Anchors locs
anchor_locs = np.vstack((dy, dx, dh, dw)).transpose()

# %% [md]
# `2.7 - RPN targets`
# %%
# Anchors labels
anchor_labels = np.empty((len(anchors),), dtype=label.dtype)
anchor_labels.fill(-1)
anchor_labels[inside_index] = label

# Anchors locations
anchor_locations = np.empty((len(anchors),) + anchors.shape[1:], dtype=anchor_locs.dtype)
anchor_locations.fill(0)
anchor_locations[inside_index, :] = anchor_locs

print('anchor_locations [N,4]:', anchor_locations.shape, '\n', '-'*20, '\n', anchor_locations[np.where(anchor_labels >=0)[0]])
print('\nanchor_labels [N,]:', anchor_labels.shape, '\n', '-'*20, '\n', anchor_labels[np.where(anchor_labels >=0)[0]])

# %% ########################################## [md]
#  # 3. Region Proposal Network
# %% ###############################################
# define layers
mid_channels = 512
in_channels = 512 # depends on the output feature map
n_anchor = 9 # Number of anchors at each location
conv1 = torch.nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
reg_layer = torch.nn.Conv2d(mid_channels, n_anchor *4, 1, 1, 0)
cls_layer = torch.nn.Conv2d(mid_channels, n_anchor *2, 1, 1, 0)

# %%
# Layers initialization
# conv sliding layer
conv1.weight.data.normal_(0, 0.01)
conv1.bias.data.zero_()

# Regression layer
reg_layer.weight.data.normal_(0, 0.01)
reg_layer.bias.data.zero_()

# classification layer
cls_layer.weight.data.normal_(0, 0.01)
cls_layer.bias.data.zero_()

# %%
# RPN Prediction
x = conv1(out_map)
pred_anchor_locs = reg_layer(x)
pred_cls_scores = cls_layer(x)

# %%
# format pred_anchor_locs
pred_anchor_locs = pred_anchor_locs.permute(0, 2, 3, 1).contiguous().view(1, -1, 4)

# format pred_cls_scores
pred_cls_scores = pred_cls_scores.permute(0, 2, 3, 1).contiguous()

# objectness score *
objectness_score = pred_cls_scores.view(1, 50, 50, 9, 2)[:, :, :, :, 1].contiguous().view(1, -1)
print('objectness_score:', objectness_score.shape)

# pred_cls_scores *
pred_cls_scores  = pred_cls_scores.view(1, -1, 2)
print('pred_cls_scores:', pred_cls_scores.shape)

# %% ########################################## [md]
# `3.1 - RoI Network`
# %% ###############################################
nms_thresh = 0.7 # non-maxmimum supression threshold
n_train_pre_nms = 12000 # number of bboxes before nms during training
n_train_post_nms = 2000 # number of bboxes after nms during training
n_test_pre_nms = 6000 # number of bboxes before nms during testing
n_test_post_nms = 300 # number of bboxes after nms during testing
min_size = 16 # minimum height of the object required to create a proposal.

# %% [md]
# > Reverse parameterezation.
#
# > $
# x = (w_{a} * ctr\_x_{p}) + ctr\_x_{a} \\
# y = (h_{a} * ctr\_x_{p}) + ctr\_x_{a} \\
# h = np.exp(h_{p}) \times h_{a} \\
# w = np.exp(w_{p}) \times w_{a} \\
# $
# %%
# Convert anchors format from y1, x1, y2, x2 to ctr_x, ctr_y, h, w
anc_height = anchors[:, 2] - anchors[:, 0]
anc_width = anchors[:, 3] - anchors[:, 1]
anc_ctr_y = anchors[:, 0] + 0.5 * anc_height
anc_ctr_x = anchors[:, 1] + 0.5 * anc_width

# convert pred_achor_locs and abjectness_score to numpy
pred_anchor_locs_numpy = pred_anchor_locs[0].data.numpy()
objectness_score_numpy = objectness_score[0].data.numpy()

dy = pred_anchor_locs_numpy[:, 0::4]
dx = pred_anchor_locs_numpy[:, 1::4]
dh = pred_anchor_locs_numpy[:, 2::4]
dw = pred_anchor_locs_numpy[:, 3::4]

ctr_y = dy * anc_height[:, np.newaxis] + anc_ctr_y[:, np.newaxis]
ctr_x = dx * anc_width[:, np.newaxis] + anc_ctr_x[:, np.newaxis]
h = np.exp(dh) * anc_height[:, np.newaxis]
w = np.exp(dw) * anc_width[:, np.newaxis]

# convert [ctr_x, ctr_y, h, w] to [y1, x1, y2, x2] format
roi = np.zeros(pred_anchor_locs_numpy.shape, dtype=anchor_locs.dtype)
roi[:, 0::4] = ctr_y - 0.5 * h
roi[:, 1::4] = ctr_x - 0.5 * w
roi[:, 2::4] = ctr_y + 0.5 * h
roi[:, 3::4] = ctr_x + 0.5 * w


# clip the predicted boxes to the image
img_size = (800, 800) # Image size
roi[:, slice(0, 4, 2)] = np.clip(roi[:, slice(0, 4, 2)], 0, img_size[0])
roi[:, slice(1, 4, 2)] = np.clip(roi[:, slice(1, 4, 2)], 0, img_size[1])

# Remove predicted boxes with either height or width < threshold
hs = roi[:, 2] - roi[:, 0]
ws = roi[:, 3] - roi[:, 1]
keep = np.where((hs >= min_size) & (ws >= min_size))[0]
roi = roi[keep, :]
scores = objectness_score_numpy[keep]

# Sort all (proposal, score) pairs by score from highest to lowest
ordered_scores = scores.ravel().argsort()[::-1]

# Take top pre_nms_topN (e.g. 12000 while training and 300 while testing, use accordingly)
ordered_scores = ordered_scores[:n_train_pre_nms]
roi = roi[ordered_scores, :]
print('ROIs before nms:', roi.shape)

# %% [markdown]
# `3.2 - Non-Maximun supression`

# %%
y1 = roi[:, 0]
x1 = roi[:, 1]
y2 = roi[:, 2]
x2 = roi[:, 3]

areas = (x2 - x1 + 1) * (y2 - y1 + 1)
order = ordered_scores.argsort()[::-1]

keep = []

while order.size > 0:
    i = order[0]
    keep.append(i)
    xx1 = np.maximum(x1[i], x1[order[1:]])
    yy1 = np.maximum(y1[i], y1[order[1:]])
    xx2 = np.minimum(x2[i], x2[order[1:]])
    yy2 = np.minimum(y2[i], y2[order[1:]])

    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)
    inter = w * h
    ovr = inter / (areas[i] + areas[order[1:]] - inter)

    inds = np.where(ovr <= nms_thresh)[0]
    order = order[inds + 1]

# %%
# Final region proposals (Training/Test, use accordingly)
keep = keep[:n_train_post_nms]
roi = roi[keep]
print('ROI after nms:', roi.shape)

# %% ########################################## [md]
# `3.3 - Proposal targets`
# %% ###############################################
n_samples = 128 # Number of samples to sample from roi
pos_ratio = 0.25 # the number of positive examples out of the n_samples
pos_iou_thresh = 0.5 #  The minimum overlap of region proposal with any groundtruth
neg_iou_thresh_hi = 0.5 # The overlap value bounding required to consider a region proposal as negative
neg_iou_thresh_lo = 0.0 # The overlap value bounding required to consider a region proposal as background

# %%
# Find the iou of each ground truth object with the region proposals
ious = np.empty((len(roi), 2), dtype=np.float32)
ious.fill(0)
for num1, i in enumerate(roi):
    ya1, xa1, ya2, xa2 = i
    anchor_area = (ya2 - ya1) * (xa2 - xa1)
    for num2, j in enumerate(bbox):
        yb1, xb1, yb2, xb2 = j
        box_area = (yb2 - yb1) * (xb2 - xb1)

        inter_x1 = max([xb1, xa1])
        inter_y1 = max([yb1, ya1])
        inter_x2 = min([xb2, xa2])
        inter_y2 = min([yb2, ya2])

        if (inter_x1 < inter_x2) and (inter_y1 < inter_y2):
            iter_area = (inter_y2 - inter_y1) * (inter_x2 - inter_x1)
            iou = iter_area / (anchor_area + box_area - iter_area)
        else:
            iou = 0.
        ious[num1, num2] = iou

# %%
# Find out which ground truth has high IoU for each region proposal, Also find the maximum IoU
gt_assignment = ious.argmax(axis=1)
max_ious = ious.max(axis=1)

# Assign the labels to each proposal
gt_roi_label = labels[gt_assignment]

# Select the foreground rois as per the pos_iou_thesh
pos_index = np.where(max_ious >= pos_iou_thresh)[0]
pos_roi_per_this_image = int(n_samples * pos_ratio)
pos_roi_per_this_image = int(min(pos_roi_per_this_image, pos_index.size))
if pos_index.size > 0:
    pos_index = np.random.choice(pos_index, size=pos_roi_per_this_image, replace=False)

# Similarly we do for negitive (background) region proposals
neg_index = np.where((max_ious < neg_iou_thresh_hi) & (max_ious >= neg_iou_thresh_lo))[0]
neg_roi_per_this_image = n_samples - pos_roi_per_this_image
neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_index.size))
if neg_index.size > 0:
    neg_index = np.random.choice(neg_index, size=neg_roi_per_this_image, replace=False)

# gather positve samples index and negative samples
keep_index = np.append(pos_index, neg_index)
gt_roi_labels = gt_roi_label[keep_index]
gt_roi_labels[pos_roi_per_this_image:] = 0  # negative labels --> 0
sample_roi = roi[keep_index]

# Pick the ground truth objects for these sample_roi
bbox_for_sampled_roi = bbox[gt_assignment[keep_index]]

height = sample_roi[:, 2] - sample_roi[:, 0]
width = sample_roi[:, 3] - sample_roi[:, 1]
ctr_y = sample_roi[:, 0] + 0.5 * height
ctr_x = sample_roi[:, 1] + 0.5 * width
base_height = bbox_for_sampled_roi[:, 2] - bbox_for_sampled_roi[:, 0]
base_width = bbox_for_sampled_roi[:, 3] - bbox_for_sampled_roi[:, 1]
base_ctr_y = (bbox_for_sampled_roi[:, 0] + 0.5 * base_height)
base_ctr_x = (bbox_for_sampled_roi[:, 1] + 0.5 * base_width)

# Parameterize it
eps = np.finfo(height.dtype).eps
height = np.maximum(height, eps)
width = np.maximum(width, eps)
dy = (base_ctr_y - ctr_y) / height
dx = (base_ctr_x - ctr_x) / width
dh = np.log(base_height / height)
dw = np.log(base_width / width)
gt_roi_locs = np.vstack((dy, dx, dh, dw)).transpose()

print('sample_roi: ', sample_roi.shape, '\n', '-'*20, '\n', sample_roi[-5:], '\n')
print('gt_roi_locs: ', gt_roi_locs.shape, '\n', '-'*20, '\n', gt_roi_locs[-5:], '\n')
print('gt_roi_labels: ', gt_roi_labels.shape, '\n', '-'*20, '\n', gt_roi_labels)

# %% ########################################## [md]
# # 4. Fast R-CNN
# %% ###############################################
# Create ROI indices tensor PyTorch
rois = torch.from_numpy(sample_roi).float()
roi_indices = 0 * np.ones((len(rois),), dtype=np.int32) # on this example just one image
roi_indices = torch.from_numpy(roi_indices).float()
print(rois.shape, roi_indices.shape)

# Concatenate rois and roi_indices [N,5]
indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)
xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
indices_and_rois = xy_indices_and_rois.contiguous()
print(xy_indices_and_rois.shape)

# %%
# Define adaptive_max_pool
size = 7  # max pool 7x7
adaptive_max_pool = torch.nn.AdaptiveMaxPool2d(size)
output = []
rois = indices_and_rois.data.float()
rois[:, 1:].mul_(1 / 16.0)  # Subsampling ratio skipping the index
rois = rois.long()
num_rois = rois.size(0)
print(num_rois)

# %%
for i in range(num_rois):
    roi = rois[i]
    im_idx = roi[0]
    im = out_map.narrow(0, im_idx, 1)[..., roi[2]:(roi[4] + 1), roi[1]:(roi[3] + 1)]
    output.append(adaptive_max_pool(im))
output = torch.cat(output, 0)

# Reshape the tensor so that we can pass it through the feed forward layer.
k = output.view(output.size(0), -1)
print('adaptative_max_poll:' , output.shape)
# %%
# Define the classifier and regression networks
roi_head_classifier = torch.nn.Sequential(*[torch.nn.Linear(25088, 4096), torch.nn.Linear(4096, 4096)])
cls_loc = torch.nn.Linear(4096, 21 * 4)  # (VOC 20 classes + 1 background. Each will have 4 co-ordinates)
cls_loc.weight.data.normal_(0, 0.01)
cls_loc.bias.data.zero_()
score = torch.nn.Linear(4096, 21)  # (VOC 20 classes + 1 background)

k = roi_head_classifier(k)
roi_cls_loc = cls_loc(k)
roi_cls_score = score(k)

print('roi_cls_loc:', roi_cls_loc.shape, '\nroi_cls_score:', roi_cls_score.shape)

# %% ########################################## [md]
# # 5. RPN Loss
# %% ###############################################
# From RPN
rpn_loc = pred_anchor_locs[0]
rpn_score = pred_cls_scores[0]
gt_rpn_loc = torch.from_numpy(anchor_locations)
gt_rpn_score = torch.from_numpy(anchor_labels)

print('rpn_loc:', rpn_loc.shape)
print('rpn_score:', rpn_score.shape)
print('gt_rpn_loc:', gt_rpn_loc.shape)
print('gt_rpn_score:', gt_rpn_score.shape)

# %%
# Cross-entropy classification loss
rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_score.long(), ignore_index=-1)
print('rpn_cls_loss:', rpn_cls_loss)

# %%
# Smooth L1 loss regression loss
pos = gt_rpn_score > 0
mask = pos.unsqueeze(1).expand_as(rpn_loc)

# take positive labeled boxes
mask_loc_preds = rpn_loc[mask].view(-1, 4)
mask_loc_targets = gt_rpn_loc[mask].view(-1, 4)

# regression loss
x = torch.abs(mask_loc_targets - mask_loc_preds)
rpn_loc_loss = ((x < 1).float() * 0.5 * x**2) + ((x >= 1).float() * (x-0.5))
print('rpn_loc_loss', rpn_loc_loss.sum())

# %%
# RPN Loss
rpn_lambda = 10.
N_reg = (gt_rpn_score > 0).float().sum()
rpn_loc_loss = rpn_loc_loss.sum() / N_reg
rpn_loss = rpn_cls_loss + (rpn_lambda * rpn_loc_loss)
print('rpn_loss:', rpn_loss)

# %% ########################################## [md]
# # 6. Fast R-CNN Loss
# %% ###############################################
gt_roi_loc = torch.from_numpy(gt_roi_locs)
gt_roi_label = torch.from_numpy(np.float32(gt_roi_labels)).long()

print('roi_cls_loc:', roi_cls_loc.shape)
print('roi_cls_score:', roi_cls_score.shape)
print('gt_roi_loc:', gt_roi_loc.shape)
print('gt_roi_label:', gt_roi_label.shape)

# %%
# Classification loss
roi_cls_loss = F.cross_entropy(roi_cls_score, gt_roi_label, ignore_index=-1)
print('roi_cls_loss:', roi_cls_loss)

# %%
# Regression loss
n_sample = roi_cls_loc.shape[0]
roi_loc = roi_cls_loc.view(n_sample, -1, 4)

# take positive labeled boxes
roi_loc = roi_loc[torch.arange(0, n_sample).long(), gt_roi_label]
mask = gt_roi_label>0
mask_loc_preds = roi_loc[mask].view(-1, 4)
mask_loc_targets = gt_roi_loc[mask].view(-1, 4)

x = torch.abs(mask_loc_preds - mask_loc_targets)
roi_loc_loss = (x<1).float()*0.5*x**2 + (x>=1).float()*(x-0.5)
roi_loc_loss = roi_loc_loss
print('roi_loc_loss:', roi_loc_loss.sum())

# %% Total RoI Loss
roi_lambda = 10.
N_reg = (gt_roi_label>0).float().sum()
roi_loc_loss = roi_loc_loss.sum() / N_reg
roi_loss = roi_cls_loss + roi_lambda * roi_loc_loss
print('roi_loss:', roi_loss)

# %% ########################################## [md]
# # Total Loss
# %% ###############################################
total_loss = rpn_loss + roi_loss
print('total_loss:', total_loss)
