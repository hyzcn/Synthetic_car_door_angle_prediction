import numpy as np

def eveluate_iou(label, pred, n_class, epsilon=1e-12):
    assert label.shape == pred.shape, \
        'label and pred shape mismatchL {} vs {}'.format(
            label.shape, pred.shape
        )
    
    ious = np.zeros(n_class)
    tps = np.zeros(n_class)
    fns = np.zeros(n_class)
    fps = np.zeros(n_class)

    for cls_id in range(n_class):
        tp = np.sum(pred[label == cls_id] == cls_id)
        fp = np.sum(label[pred == cls_id] != cls_id)
        fn = np.sum(pred[label == cls_id] != cls_id)

        ious[cls_id] = tp/(tp+fn+fp+epsilon)
        tps[cls_id] = tp
        fps[cls_id] = fp
        fns[cls_id] = fn

    return ious, tps, fps, fns

