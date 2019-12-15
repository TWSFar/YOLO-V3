
# Append to pycocotools JSON dictionary
'''
if save_json:
# [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
image_id = int(Path(paths[si]).stem.split('_')[-1])
box = pred[:, :4].clone()  # xyxy
scale_coords(imgs[si].shape[1:], box, shapes[si])  # to original shape
box = xyxy2xywh(box)  # xywh
box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
for di, d in enumerate(pred):
jdict.append({
'image_id': image_id,
'category_id': coco91class[int(d[6])],
'bbox': [float3(x) for x in box[di]],
'score': float(d[4])
})
'''