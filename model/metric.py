import numpy as np
import cv2

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

def vis_seg(pred, n_class):
    color_dict = [[255,10,245], [127,255,0], [0,0,0], 
                [128,0,0], [0,128,0], [128,128,0],
                [0,0,128], [128,0,128], [0,128,128]]
    img = np.zeros([pred.shape[0], pred.shape[1], 3], np.uint8)
    for i in range(n_class):
        img[pred == i] = color_dict[i]
    
    cv2.imwrite("./a.png", img)
    print("saved")

def eveluate_acc(label, pred, n_class, epsilon=1e-12):
    assert label.shape == pred.shape, \
        'label and pred shape mismatchL {} vs {}'.format(
            label.shape, pred.shape
        )

    tps = np.zeros(n_class)
    tns = np.zeros(n_class)
    fns = np.zeros(n_class)
    fps = np.zeros(n_class)

    for cls_id in range(n_class):
        tp = np.sum(pred[label == cls_id] == cls_id)
        tn = np.sum(label[pred != cls_id] != cls_id)
        fp = np.sum(label[pred == cls_id] != cls_id)
        fn = np.sum(pred[label == cls_id] != cls_id)

        tps[cls_id] = tp
        tns[cls_id] = tn
        fps[cls_id] = fp
        fns[cls_id] = fn

    return tps, tns, fps, fns



