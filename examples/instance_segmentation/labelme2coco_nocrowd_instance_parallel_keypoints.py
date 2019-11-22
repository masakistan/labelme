#!/usr/bin/env python

from collections import Counter

import argparse
import datetime
import glob
import json
import os
import os.path as osp
import sys

import numpy as np
import PIL.Image

from torch.utils.data import DataLoader, Dataset
import labelme

from keypoints import skeletons, keypoints
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

try:
    import pycocotools.mask
except ImportError:
    print('Please install pycocotools:\n\n    pip install pycocotools\n')
    sys.exit(1)


def collate_images(batch):
    #ret = [img for img in batch]
    #print(batch)
    return batch[0]

expected_name_kps = 1 + 50
expected_occupation_kps = 1 + 50 + 50
expected_veteran_kps = 1 + 50 + 50

class FormsDataset(Dataset):
    def __init__(self, img_dir, out_dir, class_name_to_id):
        self.out_dir = out_dir
        self.class_name_to_id = class_name_to_id
        self.out_ann_file = osp.join(out_dir, 'annotations.json')
        self.label_files = glob.glob(osp.join(img_dir, '*.json'))
        #print(self.label_files)

    def __len__(self):
        return len(self.label_files)

    def __getitem__(self, image_id):
        label_file = osp.join(self.label_files[image_id])
        img, out_img_file, annotations = process(image_id, label_file, self.out_dir, self.class_name_to_id)
        return image_id, img, out_img_file, annotations
 

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('input_dir', help='input annotated directory')
    parser.add_argument('output_dir', help='output dataset directory')
    parser.add_argument('--labels', help='labels file', required=True)
    args = parser.parse_args()

    if osp.exists(args.output_dir):
        print('Output directory already exists:', args.output_dir)
        sys.exit(1)
    os.makedirs(args.output_dir)
    os.makedirs(osp.join(args.output_dir, 'JPEGImages'))
    print('Creating dataset:', args.output_dir)

    now = datetime.datetime.now()

    data = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=now.year,
            contributor=None,
            date_created=now.strftime('%Y-%m-%d %H:%M:%S.%f'),
        ),
        licenses=[dict(
            url=None,
            id=0,
            name=None,
        )],
        images=[
            # license, url, file_name, height, width, date_captured, id
        ],
        type='instances',
        annotations=[
            # segmentation, area, iscrowd, image_id, bbox, category_id, id
        ],
        categories=[
            # supercategory, id, name
        ],
    )

    class_name_to_id = {}
    class_id = 0
    for i, line in enumerate(open(args.labels).readlines()):
        class_name = line.strip()
        if i == 0:
            assert class_name == '__ignore__'
            continue
        elif i == 1:
            assert class_name == '_background_'
        class_name_to_id[class_name] = i
        d = dict(
            supercategory=None,
            id=i,
            name=class_name,
        )

        if class_name in skeletons:
            #print(class_name, len(skeletons[class_name]), len(keypoints[class_name]))
            d['skeleton'] = skeletons[class_name]
            d['keypoints'] = keypoints[class_name]
        data['categories'].append(d)
        
    annot_count = 0
    out_ann_file = osp.join(args.output_dir, 'annotations.json')

    img_dataset = FormsDataset(args.input_dir, args.output_dir, class_name_to_id)
    img_dataloader = DataLoader(
        img_dataset,
        batch_size = 1,
        shuffle = False,
        num_workers = 0,
        collate_fn = collate_images
    )


    for image_id, img, out_img_file, annotations in img_dataloader:
        #print(image_id, out_img_file)
        data['images'].append(dict(
            license=0,
            url=None,
            file_name=osp.relpath(out_img_file, osp.dirname(out_ann_file)),
            height=img.shape[0],
            width=img.shape[1],
            date_captured=None,
            id=image_id,
        ))

        for annotation in annotations:
            annotation['id'] = annot_count
            annot_count += 1

        data['annotations'].extend(annotations)
        
    with open(out_ann_file, 'w') as f:
        json.dump(data, f)

        
    #label_files = glob.glob(osp.join(args.input_dir, '*.json'))
    #for image_id, label_file in enumerate(label_files):

def process(image_id, label_file, output_dir, class_name_to_id):
    #print('Generating dataset from:', label_file)
    with open(label_file) as f:
        label_data = json.load(f)

    base = osp.splitext(osp.basename(label_file))[0]
    out_img_file = osp.join(
        output_dir, 'JPEGImages', base + '.jpg'
    )

    # NOTE: copy the image file
    img_file = osp.join(
        osp.dirname(label_file), label_data['imagePath']
    )
    img = np.asarray(PIL.Image.open(img_file))

    # NOTE: gather polygons
    polygons = []

    fields = {
        'name_col_field': [],
        'occupation_col_industry_field': [],
        'occupation_col_occupation_field': [],
        'veteran_col_war_or_expedition': [],
        'veteran_col_yes_or_no': [],
    }
    
    veteran_col = []
    name_col = []
    occupation_col = []

    tkps = Counter()
    
    #masks = {}
    for shape in label_data['shapes']:
        points = shape['points']
        label = shape['label']
        shape_type = shape.get('shape_type', None)
        mask = labelme.utils.shape_to_mask(
            img.shape[:2], points, shape_type
        )

        mask = np.asfortranarray(mask.astype(np.uint8))

        if shape['shape_type'] == 'polygon':
            coords = [coord for coords in points for coord in coords]
        elif shape['shape_type'] == 'rectangle':
            x1, y1 = points[0]
            x2, y2 = points[1]
            coords = [x1, y1, x1, y2, x2, y2, x2, y1]
        else:
            raise Exception('unknown shape type')
        ys = [coords[i] for i in range(1, len(coords), 2)]
        y_center = int((min(ys) + max(ys)) // 2)
        xs = [coords[i] for i in range(0, len(coords), 2)]
        x_center = int((min(xs) + max(xs)) // 2)
        #print(label, y_center)

        #coords = [
        #    [coords[0], coords[1]],
        #    [coords[2], coords[3]],
        #    [coords[4], coords[5]],
        #    [coords[6], coords[7]],
        #]
        coords = list(map(int, coords))
        polygon = (x_center, y_center, 2, label, mask, coords)

        if label in fields:
            fields[label].append(polygon)
            tkps[label] += 1
        elif 'header' in label:
            if 'occupation' in label:
                tkps['occupation_col'] += 1
                occupation_col.append(polygon)
            elif 'name' in label:
                tkps['name_col'] += 1
                name_col.append(polygon)
            elif 'veteran' in label:
                tkps['veteran_col'] += 1
                veteran_col.append(polygon)
        else:
            polygons.append((label, mask, coords))

    for key, vals in fields.items():
        #print("before")
        #print(list([(i[0], i[1]) for i in vals]))
        vals.sort(key = lambda x: x[0])
        #print("after")
        #print(list([(i[0], i[1]) for i in vals]))


    print('before', tkps)
    label = 'name_col_field'
    tkps['name_col'] += tkps[label]
    if len(fields[label]) < 50:
        #print('padding name', 50 - len(fields[label]))
        pad = list([(0, 0, 0, None, None, None) for i in range(50 - len(fields[label]))])
        fields[label] = fields[label] + pad

    label = 'veteran_col_yes_or_no'
    tkps['veteran_col'] += tkps[label]
    if len(fields[label]) < 50:
        pad = list([(0, 0, 0, None, None, None) for i in range(50 - len(fields[label]))])
        fields[label] = fields[label] + pad

    label = 'veteran_col_war_or_expedition'
    tkps['veteran_col'] += tkps[label]
    if len(fields[label]) < 50:
        pad = list([(0, 0, 0, None, None, None) for i in range(50 - len(fields[label]))])
        fields[label] = fields[label] + pad

    label = 'occupation_col_industry_field'
    tkps['occupation_col'] += tkps[label]
    if len(fields[label]) < 50:
        pad = list([(0, 0, 0, None, None, None) for i in range(50 - len(fields[label]))])
        fields[label] = fields[label] + pad

    label = 'occupation_col_occupation_field'
    tkps['occupation_col'] += tkps[label]
    if len(fields[label]) < 50:
        pad = list([(0, 0, 0, None, None, None) for i in range(50 - len(fields[label]))])
        fields[label] = fields[label] + pad

    
    name_col += fields['name_col_field']
    veteran_col += fields['veteran_col_yes_or_no'] + fields['veteran_col_war_or_expedition']
    occupation_col += fields['occupation_col_occupation_field'] + fields['occupation_col_industry_field']
    #print(len(polygons))
    #print(list([(i[0], i[1], i[2]) for i in name_col]))

    #print(tkps)

    print(tkps)
        
    annotations = []
    for label, mask, polygon in polygons:
        print(label)
        #if label != 'name_col':
        #    continue

        nkps = tkps[label]
        #print('total kps', nkps)
        if label == 'name_col':
            kps = name_col
            exp = expected_name_kps
        elif label == 'occupation_col':
            kps = occupation_col
            exp = expected_occupation_kps
        elif label == 'veteran_col':
            kps = veteran_col
            exp = expected_veteran_kps
            
        fkps = []
        fkps_bbox = []
        #print(kps)
        for kp in kps:
            #print(kp)
            x, y, v, _, _, coords = kp
            fkps.append(x)
            fkps.append(y)
            fkps.append(v)

            # NOTE: convert coord to box
            if coords is None:
                #fkps_bbox.append([[0, 0]] * 4)
                for i in range(8):
                    fkps.append(0)
            else:
                assert len(coords) == 8, 'coords for field is incorrect {}'.format(coords)
                for i in coords:
                    fkps.append(i)
                #fkps_bbox.append(coords)
        #print(label, polygon)
        cls_name = label.split('-')[0]
        if cls_name not in class_name_to_id:
            continue
        
        cls_id = class_name_to_id[cls_name]
        segmentation = pycocotools.mask.encode(mask)
        segmentation['counts'] = segmentation['counts'].decode()
        area = float(pycocotools.mask.area(segmentation))
        bbox = list(pycocotools.mask.toBbox(segmentation))
        annotations.append(dict(
            segmentation = [polygon],
            num_keypoints = nkps,
            keypoints = fkps,
            #keypoint_bboxes = fkps_bbox,
            #segmentation = segmentation,
            area=area,
            iscrowd = 0,
            #iscrowd = 1,
            image_id=image_id,
            category_id=cls_id,
            id=-1,
            bbox=bbox,
        ))
        #annot_count += 1

    #PIL.Image.fromarray(img).save(out_img_file)
       
    return img, out_img_file, annotations


if __name__ == '__main__':
    main()
