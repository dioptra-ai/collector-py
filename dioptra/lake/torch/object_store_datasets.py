import os
import pickle
from functools import partial
from multiprocessing import Pool
import hashlib

import tqdm
from torch.utils.data import Dataset
import smart_open
from PIL import Image
import numpy as np
import yaml

from dioptra.lake.utils import _decode_to_np_array

class ImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        """
        A wrapper class on top of torch datasets.
        The goal is to allow either a data streaming from an object store or to cache the data locally and load it from there

        Parameters:
            dataframe: the datagframe to use as a data source
            transform: the transform to be applied when calling __getitem__

        """
        self.cache_dir = os.path.join(
            os.environ.get('DIOPTRA_CACHE_DIR',
            os.path.join(os.path.expanduser('~'), '.dioptra')))
        self.dataframe = dataframe
        self.transform = transform
        self.use_caching = True

        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def __getitem__(self, index, is_prefetch=False):
        """
        Main method to get the dataset items.
        In prefetch mode, it will pull the images from an object store, skip `transform` but won't return them
        If the dataset is already prefetched, the results will be loaded from the cache dir

        Parameters:
            index: index of the datapoint to be retreived
            is_prefetch: whether in prefetch mode or not

        """
        row = self.dataframe.iloc[index].copy()

        # Resolving the image field
        if 'metadata' in row and row['type'] =='IMAGE' and 'uri' in row['metadata']:
            if self.use_caching:
                row['image'] = self._handle_cache(
                    row['metadata']['uri'], self._download_img, is_prefetch)
            else:
                row['image'] = self._download_img(row['metadata']['uri'])

        # Resolving the groundtruths field
        if 'groundtruths' in row:
            for groundtruth in row['groundtruths']:
                if groundtruth['task_type'] == 'CLASSIFICATION':
                    row['class_name'] = groundtruth['class_name']
                    break
                if groundtruth['task_type'] == 'SEGMENTATION':
                    if self.use_caching:
                        row['segmentation_class_mask'] = self._handle_cache(
                            groundtruth['encoded_segmentation_class_mask'],
                            _decode_to_np_array,
                            is_prefetch)
                    else:
                        row['segmentation_class_mask'] = _decode_to_np_array(
                            groundtruth['encoded_segmentation_class_mask'])

                    if row['segmentation_class_mask'] is not None:
                        row['segmentation_class_mask'] = row['segmentation_class_mask'].astype(np.int32)
                    break
                if groundtruth['task_type'] == 'INSTANCE_SEGMENTATION':
                    if 'bboxes' in groundtruth and groundtruth['bboxes'] is not None:
                        row['polygons'] = [
                            {
                                **({'class_name': bbox['class_name']} if bbox is not None and 'class_name' in bbox and bbox['class_name'] is not None else {}),
                                **({'coco_polygon': bbox['coco_polygon']} if  bbox is not None and 'coco_polygon' in bbox and bbox['coco_polygon'] is not None else {})
                            } for bbox in groundtruth['bboxes']]
                    break
                if groundtruth['task_type'] == 'LANE_DETECTION':
                    if 'lanes' in groundtruth and groundtruth['lanes'] is not None:
                        row['lanes'] = [
                            {
                                **({'class_name': lane['class_name']} if lane is not None and 'class_name' in lane and lane['class_name'] is not None else {}),
                                **({'coco_polygon': lane['coco_polygon']} if lane is not None and 'coco_polygon' in lane and lane['coco_polygon'] is not None else {})
                            } for lane in groundtruth['lanes']]
                    break

        if self.transform is not None and not is_prefetch:
            return self.transform(row)

        return row

    def _handle_cache(self, field, field_transform, is_prefetch):
        field_hash = hashlib.md5(field.encode()).hexdigest()
        cached_object_path = os.path.join(self.cache_dir, field_hash)

        if is_prefetch and os.path.exists(cached_object_path):
            return

        if self.use_caching and os.path.exists(cached_object_path):
            try:
                with open(cached_object_path, 'rb') as file:
                    return pickle.load(file)
            except EOFError:
                print(f'Error while loading {field}, will reprocess it')

        object_transformed = field_transform(field)
        with open(cached_object_path, 'wb') as file:
            pickle.dump(object_transformed, file)
        return object_transformed

    def _download_img(self, image_path):
        return Image.open(smart_open.open(image_path, 'rb'))

    def __len__(self):
        """
        Length of the dataset

        """
        return len(self.dataframe)

    def prefetch_images(self, num_workers=1):
        """
        Run the multi processed prefetch on the dataset
        Parameters:
            num_workers: number of processors to be used

        """
        if not self.use_caching:
            raise RuntimeError('Turn use_caching to True to be able to prefetch images')

        with Pool(num_workers) as my_pool:
            list(tqdm.tqdm(
                my_pool.imap(partial(self.__getitem__, is_prefetch=True), range(self.__len__())),
                total=self.__len__(),
                desc='Prefetching your images ...',
                ncols=100
            ))

    def _export_record(self, index, path, format, class_names):
        processed_row = self.__getitem__(index)

        if format == 'yolov7':
            if 'image' not in processed_row or \
                    'polygons' not in processed_row or \
                    'metadata' not in processed_row or \
                    'uri' not in processed_row['metadata'] or \
                    'width' not in processed_row['metadata'] or \
                    'height' not in processed_row['metadata']:
                print('Skipping datapoint as it does not have the required fields: image, polygons, metadata.uri, metadata.width, metadata.height')
                return

            datapoint_hash = hashlib.md5(processed_row['metadata']['uri'].encode()).hexdigest()
            img_name = os.path.basename(processed_row['metadata']['uri'])
            img_name, img_ext = os.path.splitext(os.path.basename(processed_row['metadata']['uri']))
            datapoint_name = f'{datapoint_hash}_{img_name}'
            img_width = processed_row['metadata']['width']
            img_height = processed_row['metadata']['height']
            img_path = os.path.join(path, 'images' , datapoint_name + img_ext)
            processed_row['image'].save(img_path)

            with open(os.path.join(path, 'labels' , datapoint_name + '.txt'), 'w') as file:
                file.write('')

            for polygon in processed_row['polygons']:
                if 'class_name' not in polygon or 'coco_polygon' not in polygon:
                    continue
                if polygon['class_name'] not in class_names:
                    continue
                class_index = class_names.index(polygon['class_name'])
                normalized_polygon = [
                    str(value / img_width) if index % 2 else str(value / img_height) for index, value in enumerate(polygon['coco_polygon'])]
                with open(os.path.join(path, 'labels' , datapoint_name + '.txt'), 'a') as file:
                    file.write(f'{class_index} {" ".join(normalized_polygon)}\n')


    def export(self, path, format, class_names, num_workers=1):
        """
        Export the dataset to a given format
        Parameters:
            path: path to the file to be exported
            format: format of the file to be exported
            class_names: list of class names to be used in the export
            num_workers: number of processors to be used

        """
        if format != 'yolov7':
            raise NotImplementedError(f'Exporting to {format} is not implemented yet')
        if os.path.exists(path):
            raise RuntimeError(f'Dir {path} already exists')

        if format == 'yolov7':
            os.makedirs(path)
            os.makedirs(os.path.join(path, 'images'))
            os.makedirs(os.path.join(path, 'labels'))

            self.use_caching = False
            self.transform = None

            with Pool(num_workers) as my_pool:
                list(tqdm.tqdm(
                    my_pool.imap(partial(self._export_record, path=path, format=format, class_names=class_names), range(self.__len__())),
                    total=self.__len__(),
                    desc='Exporting your dataset ...',
                    ncols=100
                ))

            with open(os.path.join(path, 'data.yaml'), 'w') as file:
                yaml.dump({'names': class_names}, file)
