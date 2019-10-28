import numpy as np

cls = np.array([[0.1, 0.2, 0.3],
                [0.1, 0.2, 0.4],
                [0.1, 0.3, 0.9],
                [0.2, 0.3, 0.1],
                [0.1, 0.8, 0.5]])

scores, classes = cls.max(dim=1)

anchor_nms_idx = []

for c in classes.unique():
    idx = np.where(classes == c)[0].tolist()
    anchor_nms_idx .append(nms(bbox[idx], self.nms_thd))
