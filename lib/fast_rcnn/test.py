import argparse
import numpy as np
import cv2
import cPickle
import heapq
import os
import math
import tensorflow as tf
import matplotlib.pyplot as plt

from .config import cfg, get_output_dir

from ..utils.timer import Timer
from ..utils.cython_nms import nms, nms_new
from ..utils.blob import im_list_to_blob
from ..utils.boxes_grid import get_boxes_grid
from ..rpn_msr.generate_anchors import generate_anchors

from ..fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv


def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
        im (ndarray): a color image in BGR order
    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)

def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.
    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob
    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)

def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.
    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob
    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)
    scales = np.array(scales)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels

def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""
    if cfg.TEST.HAS_RPN:
        blobs = {'data' : None, 'rois' : None}
        blobs['data'], im_scale_factors = _get_image_blob(im)
    else:
        blobs = {'data' : None, 'rois' : None}
        blobs['data'], im_scale_factors = _get_image_blob(im)
        if cfg.IS_MULTISCALE:
            if cfg.IS_EXTRAPOLATING:
                blobs['rois'] = _get_rois_blob(rois, cfg.TEST.SCALES)
            else:
                blobs['rois'] = _get_rois_blob(rois, cfg.TEST.SCALES_BASE)
        else:
            blobs['rois'] = _get_rois_blob(rois, cfg.TEST.SCALES_BASE)

    return blobs, im_scale_factors

def _clip_boxes(boxes, im_shape):
    """Clip boxes to image boundaries."""
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
    return boxes


def _rescale_boxes(boxes, inds, scales):
    """Rescale boxes according to image rescaling."""

    for i in xrange(boxes.shape[0]):
        boxes[i,:] = boxes[i,:] / scales[int(inds[i])]

    return boxes


def im_detect(sess, net, im, boxes=None):
    """Detect object classes in an image given object proposals.
    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals
    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """

    blobs, im_scales = _get_blobs(im, boxes)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    im_blob = blobs['data']
    blobs['im_info'] = np.array(
        [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
        dtype=np.float32)

    # forward pass
    feed_dict={net.data: blobs['data'], net.im_info: blobs['im_info']}

    cls_score_P3, cls_prob_P3, bbox_pred_P3, \
    cls_score_P4, cls_prob_P4, bbox_pred_P4, \
    cls_score_P5, cls_prob_P5, bbox_pred_P5, \
    cls_score_P6, cls_prob_P6, bbox_pred_P6, \
    cls_score_P7, cls_prob_P7, bbox_pred_P7, \
        = sess.run([ \
                        net.get_output('cls_score/P3'), net.get_output('cls_prob/P3'), net.get_output('box_pred_reshape/P3'), \
                        net.get_output('cls_score/P4'), net.get_output('cls_prob/P4'), net.get_output('box_pred_reshape/P4'), \
                        net.get_output('cls_score/P5'), net.get_output('cls_prob/P5'), net.get_output('box_pred_reshape/P5'), \
                        net.get_output('cls_score/P6'), net.get_output('cls_prob/P6'), net.get_output('box_pred_reshape/P6'), \
                        net.get_output('cls_score/P7'), net.get_output('cls_prob/P7'), net.get_output('box_pred_reshape/P7'), \
                   ],\
                 feed_dict=feed_dict)

    # rois[P3~P7] = anchor_generator()
    num_anchor_ratio = 3  # 1:2, 1:1, 2:1
    num_anchor_scale = 3  # could be config
    _feat_strides = [8, 16, 32, 64, 128]
    anchor_sizes = [32, 64, 128, 256, 512]
    anchor_scales = np.array(anchor_sizes) / np.array(_feat_strides)

    sizes = np.power(2, [0.0, 1.0/3, 2.0/3])
    _anchors = [None, None, None, None, None]

    _anchors[0] = generate_anchors(base_size=_feat_strides[0], scales=anchor_scales[0]*sizes[:num_anchor_scale])
    _anchors[1] = generate_anchors(base_size=_feat_strides[1], scales=anchor_scales[1]*sizes[:num_anchor_scale])
    _anchors[2] = generate_anchors(base_size=_feat_strides[2], scales=anchor_scales[2]*sizes[:num_anchor_scale])
    _anchors[3] = generate_anchors(base_size=_feat_strides[3], scales=anchor_scales[3]*sizes[:num_anchor_scale])
    _anchors[4] = generate_anchors(base_size=_feat_strides[4], scales=anchor_scales[4]*sizes[:num_anchor_scale])

    _num_anchors = [anchor.shape[0] for anchor in _anchors]

    # Algorithm:
    #
    # for each (H, W) location i
    #   generate 3 anchor boxes centered on cell i
    #   apply predicted bbox deltas at cell i to each of the 3 anchors
    # measure GT overlap

    pred_boxes_P = []

    cls_score_P = [cls_score_P3, cls_score_P4, cls_score_P5, cls_score_P6, cls_score_P7]
    cls_prob_P = [cls_prob_P3, cls_prob_P4, cls_prob_P5, cls_prob_P6, cls_prob_P7]
    bbox_pred_P = [bbox_pred_P3, bbox_pred_P4, bbox_pred_P5, bbox_pred_P6, bbox_pred_P7]

    for idx, elem in enumerate(zip(cls_score_P, bbox_pred_P)):

        cls_score, box_deltas = elem

        assert cls_score.shape[0] == 1, \
            'Only single item batches are supported'

        # map of shape (..., H, W)
        height, width = cls_score.shape[1:3]

        # 1. Generate proposals from bbox deltas and shifted anchors
        shift_x = np.arange(0, width) * _feat_strides[idx]
        shift_y = np.arange(0, height) * _feat_strides[idx]
        shift_x, shift_y = np.meshgrid(shift_x, shift_y) # in W H order
        # K is H x W
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = _num_anchors[idx]
        K = shifts.shape[0]
        all_anchors = (_anchors[idx].reshape((1, A, 4)) +
                       shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
        all_anchors = all_anchors.reshape((K * A, 4))
        #total_anchors = int(K * A)

        all_anchors = all_anchors / im_scales[0]

        '''
        all_anchors_list.append(all_anchors)
        total_anchors_sum += total_anchors
        '''

        '''
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = _clip_boxes(pred_boxes, im.shape)
        '''
        pred_boxes = bbox_transform_inv(all_anchors, box_deltas)
        pred_boxes = _clip_boxes(pred_boxes, im.shape)

        pred_boxes_P.append(pred_boxes)

    '''
    assert len(im_scales) == 1, "Only single-image batch implemented"
    boxes = rois[:, 1:5] / im_scales[0]
    '''


    '''
    if cfg.TEST.SVM:
        # use the raw scores before softmax under the assumption they
        # were trained as linear SVMs
        scores = cls_score
    else:
        # use softmax estimated probabilities
        scores = cls_prob

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = _clip_boxes(pred_boxes, im.shape)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]
        pred_boxes = pred_boxes[inv_index, :]
    '''

    #return scores, pred_boxes
    return cls_prob_P, pred_boxes_P

def vis_detections(im, class_name, dets, thresh=0.8):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt 
    #im = im[:, :, (2, 1, 0)]
    for i in xrange(np.minimum(10, dets.shape[0])):
        bbox = dets[i, :4] 
        score = dets[i, -1] 
        if score > thresh:
            #plt.cla()
            #plt.imshow(im)
            plt.gca().add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='g', linewidth=3)
                )
            plt.gca().text(bbox[0], bbox[1] - 2,
                 '{:s} {:.3f}'.format(class_name, score),
                 bbox=dict(facecolor='blue', alpha=0.5),
                 fontsize=14, color='white')

            plt.title('{}  {:.3f}'.format(class_name, score))
    #plt.show()

def apply_nms(all_boxes, thresh):
    """Apply non-maximum suppression to all predicted boxes output by the
    test_net method.
    """
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(num_classes)]
    for cls_ind in xrange(num_classes):
        for im_ind in xrange(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue

            x1 = dets[:, 0]
            y1 = dets[:, 1]
            x2 = dets[:, 2]
            y2 = dets[:, 3]
            scores = dets[:, 4]
            inds = np.where((x2 > x1) & (y2 > y1) & (scores > cfg.TEST.DET_THRESHOLD))[0]
            dets = dets[inds,:]
            if dets == []:
                continue

            keep = nms(dets, thresh)
            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_boxes


def test_net(sess, net, imdb, weights_filename , max_per_image=1000, thresh=0.05, vis=False):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    all_boxes_P = [[[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]
                 for _ in xrange(5)] #  P3 ~ P7

    output_dir = get_output_dir(imdb, weights_filename)
    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    if not cfg.TEST.HAS_RPN:
        roidb = imdb.roidb

    det_file = os.path.join(output_dir, 'detections.pkl')
    # if os.path.exists(det_file):
    #     with open(det_file, 'rb') as f:
    #         all_boxes = cPickle.load(f)

    for i in xrange(num_images):
        # filter out any ground truth boxes
        if cfg.TEST.HAS_RPN:
            box_proposals = None
        else:
            # The roidb may contain ground-truth rois (for example, if the roidb
            # comes from the training or val split). We only want to evaluate
            # detection on the *non*-ground-truth rois. We select those the rois
            # that have the gt_classes field set to 0, which means there's no
            # ground truth.
            box_proposals = roidb[i]['boxes'][roidb[i]['gt_classes'] == 0]

        im = cv2.imread(imdb.image_path_at(i))
        _t['im_detect'].tic()
        scores, boxes = im_detect(sess, net, im, box_proposals)
        detect_time = _t['im_detect'].toc(average=False)

        _t['misc'].tic()
        if vis:
            image = im[:, :, (2, 1, 0)] 
            plt.cla()
            plt.imshow(image)

        print "scores[0].shape:"
        print scores[0].shape
        print "boxes[0].shape:"
        print boxes[0].shape

        for k in xrange(5):     # P3 ~ P7
            # skip j = 0, because it's the background class
            for j in xrange(1, imdb.num_classes):
                inds = np.where(scores[k][:, j] > thresh)[0]
                #print "scores[k][inds, j].shape:"
                #print scores[k][inds, j].shape
                cls_scores = scores[k][inds, j]
                #cls_scores = np.squeeze(scores[k][inds, j])
                #print "np.squeeze(scores[k][inds, j]).shape:"
                #print np.squeeze(scores[k][inds, j]).shape
                #cls_boxes = boxes[k][inds, j*4:(j+1)*4]
                #print "boxes[k][inds, :].shape:"
                #print boxes[k][inds, :].shape
                cls_boxes = boxes[k][inds, :]
                #cls_boxes = np.squeeze(boxes[k][inds, :])
                #print "np.squeeze(boxes[k][inds, :]).shape:"
                #print np.squeeze(boxes[k][inds, :]).shape
                cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                    .astype(np.float32, copy=False)
                try:
                    keep = nms(cls_dets, cfg.TEST.NMS)
                except:
                    print "cls_dets.shape:"
                    print cls_dets.shape
                    print "cls_dets:"
                    print cls_dets
                cls_dets = cls_dets[keep, :]
                if vis:
                    vis_detections(image, imdb.classes[j], cls_dets)
                all_boxes_P[k][j][i] = cls_dets
            if vis:
               plt.show()
            '''
            # Limit to max_per_image detections *over all classes*
            if max_per_image > 0:
                image_scores = np.hstack([all_boxes_P[k][j][i][:, -1]
                                          for j in xrange(1, imdb.num_classes)])
                if len(image_scores) > max_per_image:
                    image_thresh = np.sort(image_scores)[-max_per_image]
                    for j in xrange(1, imdb.num_classes):
                        keep = np.where(all_boxes_P[k][j][i][:, -1] >= image_thresh)[0]
                        all_boxes_P[k][j][i] = all_boxes_P[k][j][i][keep, :]
            '''

            for j in xrange(1, imdb.num_classes):
                all_boxes[j][i].append(all_boxes_P[k][j][i])

        for j in xrange(1, imdb.num_classes):
            all_boxes[j][i] = np.concatenate(all_boxes[j][i], axis=0)

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in xrange(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        # merge Ps operation

        nms_time = _t['misc'].toc(average=False)

        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, detect_time, nms_time)


    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    print 'Evaluating detections'
    imdb.evaluate_detections(all_boxes, output_dir)

