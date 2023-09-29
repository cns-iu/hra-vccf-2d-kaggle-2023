import numpy as np, zlib, base64, cv2, pandas as pd
from pycocotools import mask as coco_mask

from .models.research.object_detection.utils.object_detection_evaluation import OpenImagesInstanceSegmentationChallengeEvaluator
from .models.research.object_detection.metrics.oid_challenge_evaluation_utils import (
    build_groundtruth_dictionary,
    build_predictions_dictionary,
)

OID_CLASS_LABEL_MAP = {'blood_vessel': 1}
OID_CATEGORIES = [{"id": 1, "name": "blood_vessel"}]

def encode_binary_mask(mask: np.ndarray):
    """Converts a binary mask into OID challenge encoding ascii text."""
    
    # check input mask --
    if mask.dtype != bool:
        raise ValueError(
            "encode_binary_mask expects a binary mask, received dtype == %s" %
            mask.dtype)
    
    mask = np.squeeze(mask)
    if len(mask.shape) != 2:
        raise ValueError(
            "encode_binary_mask expects a 2d mask, received shape == %s" %
            mask.shape)
    
    # convert input mask to expected COCO API input --
    mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
    mask_to_encode = mask_to_encode.astype(np.uint8)
    mask_to_encode = np.asfortranarray(mask_to_encode)
    
    # RLE encode mask --
    encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]
    
    # compress and base64 encoding --
    binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
    base64_str = base64.b64encode(binary_str)
    return base64_str

def coordinates_to_masks(coordinates, shape):
    masks = []
    for coord in coordinates:
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(coord)], 1)
        masks.append(mask)
    return masks

def polygonann2objdetect(ann, img_h = 512, img_w = 512):
    """
    Args:
        ann: annotation dictionary {type: --, coordinates: [--,--,...]}
        img_h: image height
        img_w: image width
    Returns:
        XMin, XMax, YMin, YMax, ImageWidth, ImageHeight, Mask
        
        XMin: minimum pixel location (x) where object appear
        XMax: maximum pixel location (x) where object appear
        YMin: minimum pixel location (y) where object appear
        YMax: maximum pixel location (y) where object appear
        ImageWidth: image width
        ImageHeight: image height
        Mask: rle + encoded mask
        
    objdetect dataframe should have: ImageID, LabelName, XMin, XMax, YMin, YMax, IsGroupOf, Confidence, ImageWidth, ImageHeight, Mask.
    
    """
    # get coordinates
    coordinates = ann['coordinates']
    # make binary mask
    mask_img = coordinates_to_masks(coordinates, (img_h, img_w))[0]
    # get coordinate info
    ys, xs = np.where(mask_img)
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    xmin = x1 / img_w
    xmax = x2 / img_w
    ymin = y1 / img_h
    ymax = y2 / img_h
    # get rle + encoded mask
    mask_img = mask_img.astype(bool)
    encoded_mask = encode_binary_mask(mask_img).decode()

    return xmin, xmax, ymin, ymax, img_h, img_w, encoded_mask

def get_oid_dict_gt(df):
    df.rename(columns = {'Confidence': 'ConfidenceImageLabel'}, inplace = True)
    out_dicts = {}
    
    for image_id, rows in df.groupby('ImageID'):
        out_dict = build_groundtruth_dictionary(
            rows, OID_CLASS_LABEL_MAP
        )
        out_dicts[image_id] = out_dict
    return out_dicts

def make_gt_dict_from_dataframe(df_solution):
    list_rows = []
    for idx, row in df_solution.iterrows():
        hash_id = row['id']
        
        img_h, img_w = row['height'], row['width']
        is_group_of = 0
        confidence = 1
        label_name = 'blood_vessel'
        # list_encoded_mask = [j for j in [i.strip() for i in row['prediction_string'].split('0 0 0 0 0 0')] if j != '']
        list_encoded_mask = [i for i in row['prediction_string'].split() if i != '0']
        rows = []
        for encoded_mask in list_encoded_mask:
            # convert encoded_mask to binary mask
            binary_str_recovered = base64.b64decode(encoded_mask)
            encoded_mask_recovered = zlib.decompress(binary_str_recovered)
            binary_mask = coco_mask.decode({'size': [512,512], 'counts': encoded_mask_recovered})
            # get xmin, xmax, ymin, ymax from binary mask
            xs, ys = np.where(binary_mask)
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            rows.append([hash_id, label_name, xmin, xmax, ymin, ymax, is_group_of, confidence, img_w, img_h, encoded_mask])
        list_rows.extend(rows)
    df_gt_obj = pd.DataFrame(list_rows, columns = ['ImageID', 'LabelName', 'XMin', 'XMax', 'YMin', 'YMax', 'IsGroupOf', 'Confidence', 'ImageWidth', 'ImageHeight', 'Mask'])
    return df_gt_obj

def subm_to_pred_df(subm_df):
    ids = []
    label = []
    image_width = []
    image_height = []
    score = []
    mask = []
    for index, row in subm_df.iterrows():
        pred_strs = row["prediction_string"].split(" ")
        labels_str = pred_strs[0::3]
        scores = pred_strs[1::3]
        img_masks = pred_strs[2::3]
        assert len(scores) == len(img_masks)
        for i in range(len(scores)):
            ids.append(row["id"])
            # FIX - 1 dummy class - only consider blood_vessel
            # label.append(SUBM_TO_CLASS_LABEL[labels_str[i]])
            label.append('blood_vessel')
            image_width.append(row["width"])
            image_height.append(row["height"])
            score.append(scores[i])
            mask.append(img_masks[i])
    pred_df = pd.DataFrame(
        {
            "ImageID": ids,
            "LabelName": label,
            "ImageWidth": image_width,
            "ImageHeight": image_height,
            "Score": score,
            "Mask": mask,
        }
    )
    return pred_df

def get_oid_dict_pred(df):
    df.rename(columns = {'Confidence': 'ConfidenceImageLabel'}, inplace = True)
    out_dicts = {}
    
    for image_id, rows in (df.groupby('ImageID')):
        out_dict = build_predictions_dictionary(
            rows, OID_CLASS_LABEL_MAP
        )
        out_dicts[image_id] = out_dict
    return out_dicts

def get_oid_metrics(gt_dicts, pred_dicts, iou_thresholds=[0.6]):
    print("Calculating metrics for different iou_thresholds")
    output = {}
    list_image_id = list(gt_dicts.keys())
    for iou_threshold in iou_thresholds:
        challenge_evaluator = OpenImagesInstanceSegmentationChallengeEvaluator(
            OID_CATEGORIES, matching_iou_threshold=iou_threshold
        )
        print("Iou_threshold: ", iou_threshold)
        for image_id in list_image_id:
            challenge_evaluator.add_single_ground_truth_image_info(
                image_id, gt_dicts[image_id]
            )
            challenge_evaluator.add_single_detected_image_info(
                image_id, pred_dicts[image_id]
            )

        metrics = challenge_evaluator.evaluate()[f'OpenImagesInstanceSegmentationChallenge_Precision/mAP@{iou_threshold}IOU']
        output[iou_threshold] = metrics
    return output

def get_oid_metrics_per_patch(gt_dicts, pred_dicts, iou_thresholds=[0.6]):
    print("Calculating metrics for different iou_thresholds")
    output = {}
    list_image_id = list(gt_dicts.keys())
    for iou_threshold in iou_thresholds:
        challenge_evaluator = OpenImagesInstanceSegmentationChallengeEvaluator(
            OID_CATEGORIES, matching_iou_threshold=iou_threshold
        )
        print("Iou_threshold: ", iou_threshold)
        list_metrics = []
        for image_id in list_image_id:
            challenge_evaluator.add_single_ground_truth_image_info(
                image_id, gt_dicts[image_id]
            )
            challenge_evaluator.add_single_detected_image_info(
                image_id, pred_dicts[image_id]
            )
            metrics = challenge_evaluator.evaluate()[f'OpenImagesInstanceSegmentationChallenge_Precision/mAP@{iou_threshold}IOU']
            list_metrics.append(metrics)
        output[iou_threshold] = list_metrics
    return output

def non_max_suppression(boxes, scores, threshold):
    """
    Perform non-max suppression on a set of bounding boxes and corresponding scores.

    :param boxes: a list of bounding boxes in the format [xmin, ymin, xmax, ymax]
    :param scores: a list of corresponding scores
    :param threshold: the IoU (intersection-over-union) threshold for merging bounding boxes
    :return: a list of indices of the boxes to keep after non-max suppression
    """
    # Sort the boxes by score in descending order
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        for j in order:
            # Calculate the IoU between the two boxes
            intersection = max(0, min(boxes[i][2], boxes[j][2]) - max(boxes[i][0], boxes[j][0])) * \
                           max(0, min(boxes[i][3], boxes[j][3]) - max(boxes[i][1], boxes[j][1]))
            union = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]) + \
                    (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1]) - intersection
            iou = intersection / union

            # Remove boxes with IoU greater than the threshold
            if iou > threshold:
                order.remove(j)
    return keep

def non_max_suppression_mask(masks, scores, threshold):
    """
    Perform non-max suppression on a set of bounding boxes and corresponding scores.

    :param masks: 3D numpy matrix (n, img_h, img_w) for masks
    :param scores: a list of corresponding scores
    :param threshold: the IoU (intersection-over-union) threshold for merging bounding boxes
    :return: a list of indices of the boxes to keep after non-max suppression
    """
    # Sort the boxes by score in descending order
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        for j in order:
            # Calculate the IoU between the two boxes
            intersection = (masks[i] * masks[j]).sum()
            union = (masks[i] + masks[j]).sum()
            iou = intersection / union
            # Remove boxes with IoU greater than the threshold
            if iou > threshold:
                order.remove(j)
    return keep

def iou_matrix_mask(gt_masks, pred_masks):
    intersection = (pred_masks[:,np.newaxis] * gt_masks[np.newaxis]).sum(axis = (2,3))
    union = (pred_masks[:,np.newaxis] + gt_masks[np.newaxis]).sum(axis = (2,3)) - intersection
    iou = intersection / union
    return iou