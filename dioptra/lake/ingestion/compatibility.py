import numpy as np
from skimage.draw import polygon

def process_bbox(bbox, datapoint):
    if bbox.get('coco_polygon'):
        image_height = datapoint['metadata']['height']
        image_width = datapoint['metadata']['width']
        poly = np.array(bbox['coco_polygon'], dtype=np.int32)
        # Coco coordinate sequence is [x, y, x, y, x, y, ...]
        columns = [poly[i] for i in range(len(poly)) if i % 2 == 0]
        rows = [poly[i] for i in range(len(poly)) if i % 2 == 1]
        rr, cc = polygon(rows, columns)
        img = np.zeros((image_height + 1, image_width + 1), dtype=np.uint8)
        img[rr, cc] = 1
        bbox['segmentation_mask'] = img.tolist()

    if bbox.get('segmentation_mask'):
        bbox['segmentation_mask'] = np.array(bbox['segmentation_mask']).astype(np.uint8).tolist()
        # top is the first row with a non-zero value
        bbox['top'] = bbox.get('top', np.where(np.any(bbox['segmentation_mask'], axis=1))[0][0])
        # left is the first column with a non-zero value
        bbox['left'] = bbox.get('left', np.where(np.any(bbox['segmentation_mask'], axis=0))[0][0])
        # height is the last row with a non-zero value minus the first row with a non-zero value
        bbox['height'] = bbox.get('height', np.where(np.any(bbox['segmentation_mask'], axis=1))[0][-1] - bbox['top'])
        # width is the last column with a non-zero value minus the first column with a non-zero value
        bbox['width'] = bbox.get('width', np.where(np.any(bbox['segmentation_mask'], axis=0))[0][-1] - bbox['left'])
    
    return bbox

def process_prediction(prediction, datapoint):
    if prediction.get('bboxes'):
        bboxes = prediction['bboxes']
        for i, bbox in enumerate(bboxes):
            prediction['bboxes'][i] = process_bbox(bbox, datapoint)

    return prediction

def process_groundtruth(groundtruth, datapoint):
    if groundtruth.get('bboxes'):
        bboxes = groundtruth['bboxes']
        for i, bbox in enumerate(bboxes):
            groundtruth['bboxes'][i] = process_bbox(bbox, datapoint)

    return groundtruth
