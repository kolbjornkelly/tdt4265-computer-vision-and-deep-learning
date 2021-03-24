import numpy as np
import matplotlib.pyplot as plt
from tools import read_predicted_boxes, read_ground_truth_boxes


def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.

    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

        returns:
            float: value of the intersection of union for the two boxes.
    """

    # Find coordinates of intersection
    ymax = np.minimum(prediction_box[3], gt_box[3])
    xmax = np.minimum(prediction_box[2], gt_box[2])
    ymin = np.maximum(prediction_box[1], gt_box[1])
    xmin = np.maximum(prediction_box[0], gt_box[0])

    # Compute area of intersection
    intersec_width = np.maximum((xmax - xmin), 0)
    intersec_height = np.maximum((ymax - ymin), 0)
    intersection = intersec_width*intersec_height

    # Compute area of pred- and gt-boxes
    pred_width = prediction_box[2] - prediction_box[0]
    pred_height = prediction_box[3] - prediction_box[1]
    pred_area = pred_width * pred_height

    gt_width = gt_box[2] - gt_box[0]
    gt_height = gt_box[3] - gt_box[1]
    gt_area = gt_width * gt_height

    # Compute union
    union = pred_area + gt_area - intersection

    # Compute IoU
    iou = intersection / union
    assert iou >= 0 and iou <= 1
    return iou


def calculate_precision(num_tp, num_fp, num_fn):
    """ Calculates the precision for the given parameters.
        Returns 1 if num_tp + num_fp = 0

    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of precision
    """

    if (num_tp + num_fp > 0):
        return num_tp / (num_tp + num_fp)
    else:
        return 1


def calculate_recall(num_tp, num_fp, num_fn):
    """ Calculates the recall for the given parameters.
        Returns 0 if num_tp + num_fn = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of recall
    """
    if (num_tp + num_fn > 0):
        return num_tp / (num_tp + num_fn)
    else:
        return 0


def get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold):
    """Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.

        Remember: Matching of bounding boxes should be done with decreasing IoU order!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    """

    pred_matches = []
    gt_matches = []

    # Find all possible matches with a IoU >= iou_threshold

    for gt in gt_boxes:
        max_iou = 0
        gt_match = None
        for pred in prediction_boxes:
            iou = calculate_iou(pred, gt)
            if (iou > max_iou) and (iou >= iou_threshold):
                max_iou = iou
                gt_match = gt
        if (max_iou >= iou_threshold):
            pred_matches.append(pred)
            gt_matches.append(gt_match)

    return np.array(pred_matches), np.array(gt_matches)


def calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes,
       calculates true positives, false positives and false negatives
       for a single image.
       NB: prediction_boxes and gt_boxes are not matched!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        dict: containing true positives, false positives, true negatives, false negatives
            {"true_pos": int, "false_pos": int, false_neg": int}
    """

    # Find matches
    matched_preds, matched_gts = get_all_box_matches(
        prediction_boxes, gt_boxes, iou_threshold)

    # Compute data
    out_data = {
        "true_pos": matched_preds.shape[0],
        "false_pos": prediction_boxes.shape[0] - matched_preds.shape[0],
        "false_neg": gt_boxes.shape[0] - matched_preds.shape[0]
    }

    return out_data


def calculate_precision_recall_all_images(
        all_prediction_boxes, all_gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates recall and precision over all images
       for a single image.
       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        tuple: (precision, recall). Both float.
    """

    tps = 0
    fps = 0
    fns = 0

    for pred, gt in zip(all_prediction_boxes, all_gt_boxes):
        data = calculate_individual_image_result(pred, gt, iou_threshold)
        tps += data["true_pos"]
        fps += data["false_pos"]
        fns += data["false_neg"]

    precision = calculate_precision(tps, fps, fns)
    recall = calculate_recall(tps, fps, fns)

    return (precision, recall)


def get_precision_recall_curve(
        all_prediction_boxes, all_gt_boxes, confidence_scores, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates the recall-precision curve over all images.
       for a single image.

       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        scores: (list of np.array of floats): each element in the list
            is a np.array containting the confidence score for each of the
            predicted bounding box. Shape: [number of predicted boxes]

            E.g: score[0][1] is the confidence score for a predicted bounding box 1 in image 0.
    Returns:
        precisions, recalls: two np.ndarray with same shape.
    """
    # Instead of going over every possible confidence score threshold to compute the PR
    # curve, we will use an approximation
    confidence_thresholds = np.linspace(0, 1, 500)

    precisions = []
    recalls = []

    # Loop over every threshold
    for threshold in confidence_thresholds:
        preds = []
        # Loop over every image
        for image, pred_boxes in enumerate(all_prediction_boxes):
            conf_preds = []
            # Loop over every prediction for the current image
            for box_num, pred_box in enumerate(pred_boxes):
                # Add confident predictions
                if confidence_scores[image][box_num] >= threshold:
                    conf_preds.append(pred_box)

            preds.append(np.array(conf_preds))

        # Compute average precision and recall
        precision, recall = calculate_precision_recall_all_images(
            preds, all_gt_boxes, iou_threshold)
        # Append to final arrays
        precisions.append(precision)
        recalls.append(recall)
    return np.array(precisions), np.array(recalls)


def plot_precision_recall_curve(precisions, recalls):
    """Plots the precision recall curve.
        Save the figure to precision_recall_curve.png:
        'plt.savefig("precision_recall_curve.png")'

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        None
    """
    plt.figure(figsize=(20, 20))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.8, 1.0])
    plt.ylim([0.8, 1.0])
    plt.savefig("precision_recall_curve.png")


def calculate_mean_average_precision(precisions, recalls):
    """ Given a precision recall curve, calculates the mean average
        precision.

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        float: mean average precision
    """
    # Calculate the mean average precision given these recall levels.
    recall_levels = np.linspace(0, 1.0, 11)

    precision = 0
    for r in recall_levels:
        max_prec = 0
        for p, r_hat in zip(precisions, recalls):
            if (p > max_prec) and (r_hat >= r):
                max_prec = p
        precision += max_prec

    average_precision = precision / 11
    return average_precision


def mean_average_precision(ground_truth_boxes, predicted_boxes):
    """ Calculates the mean average precision over the given dataset
        with IoU threshold of 0.5

    Args:
        ground_truth_boxes: (dict)
        {
            "img_id1": (np.array of float). Shape [number of GT boxes, 4]
        }
        predicted_boxes: (dict)
        {
            "img_id1": {
                "boxes": (np.array of float). Shape: [number of pred boxes, 4],
                "scores": (np.array of float). Shape: [number of pred boxes]
            }
        }
    """
    # DO NOT EDIT THIS CODE
    all_gt_boxes = []
    all_prediction_boxes = []
    confidence_scores = []

    for image_id in ground_truth_boxes.keys():
        pred_boxes = predicted_boxes[image_id]["boxes"]
        scores = predicted_boxes[image_id]["scores"]

        all_gt_boxes.append(ground_truth_boxes[image_id])
        all_prediction_boxes.append(pred_boxes)
        confidence_scores.append(scores)

    precisions, recalls = get_precision_recall_curve(
        all_prediction_boxes, all_gt_boxes, confidence_scores, 0.5)
    plot_precision_recall_curve(precisions, recalls)
    mean_average_precision = calculate_mean_average_precision(
        precisions, recalls)
    print("Mean average precision: {:.4f}".format(mean_average_precision))


if __name__ == "__main__":
    ground_truth_boxes = read_ground_truth_boxes()
    predicted_boxes = read_predicted_boxes()
    mean_average_precision(ground_truth_boxes, predicted_boxes)
