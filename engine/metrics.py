import numpy as np


def intersect_and_union(pred_label, label, num_classes, ignore_index):

    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect, _ = np.histogram(intersect, bins=np.arange(num_classes + 1))
    area_pred_label, _ = np.histogram(pred_label, bins=np.arange(num_classes + 1))
    area_label, _ = np.histogram(label, bins=np.arange(num_classes + 1))
    area_union = area_pred_label + area_label - area_intersect

    return area_intersect, area_union


def total_intersect_and_union(results, gt_seg_maps, num_classes, ignore_index):
    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs
    total_area_intersect = np.zeros((num_classes, ), dtype=float)
    total_area_union = np.zeros((num_classes, ), dtype=float)
    for i in range(num_imgs):
        area_intersect, area_union = intersect_and_union(results[i], gt_seg_maps[i], num_classes,ignore_index)
        total_area_intersect += area_intersect
        total_area_union += area_union
    return total_area_intersect, total_area_union


def mean_iou(results, gt_seg_maps, num_classes, ignore_index):

    total_area_intersect, total_area_union = total_intersect_and_union(results, gt_seg_maps, num_classes, ignore_index)
    
    return total_area_intersect / total_area_union


def fast_hist(a, b, n, ignore=None):
	'''
	a and b are predict and mask respectively
	n is the number of classes
	'''
	k = (a >= 0) & (a < n) & (a!=ignore)
	return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def fast_hist_ignore_2(a, b, n, ignore1, ignore2):
	'''
	a and b are predict and mask respectively
	n is the number of classes
	'''
	k = (a >= 0) & (a < n) & (a!=ignore1) & (a!=ignore2)
	return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
	epsilon = 1e-5
	return (np.diag(hist) + epsilon) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)


if __name__ == "__main__":
    from tqdm import tqdm
    total_hist = 0
    for i in tqdm(range(625)):
        prediction = np.random.randint(0, 25, (16, 1280, 720))
        label = np.random.randint(0, 25, (16, 1280, 720))
        # iou = mean_iou(
        #     prediction,
        #     label,
        #     25,
        #     -1
        # )
        # print(iou)
        total_hist += fast_hist(prediction, label, 25)
    iou = per_class_iu(total_hist)
    print(iou)