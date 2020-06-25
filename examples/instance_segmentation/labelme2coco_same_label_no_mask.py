#!/usr/bin/env python

import multiprocessing

import argparse
import collections
import datetime
import glob
import json
import os
import os.path as osp
import sys
import uuid

import numpy as np
import PIL.Image

import labelme

try:
    import pycocotools.mask
except ImportError:
    print('Please install pycocotools:\n\n    pip install pycocotools\n')
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('input_dir', help='input annotated directory')
    parser.add_argument('output_dir', help='output dataset directory')
    #parser.add_argument('--labels', help='labels file', required=True)
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
    labels = [
            '__ignore__',
            'cell'
            ]
    class_id = 0
    #for i, line in enumerate(open(args.labels).readlines()):
    for i, line in enumerate(labels):
        class_name = line.strip()
        if i == 0:
            assert class_name == '__ignore__'
            continue
        class_name_to_id[class_name] = class_id
        data['categories'].append(dict(
            supercategory=None,
            id=i,
            name=class_name,
        ))

    out_ann_file = osp.join(args.output_dir, 'annotations.json')
    print('writing to {}'.format(out_ann_file))
    label_files = glob.glob(osp.join(args.input_dir, '*.json'))
    work = []
    for image_id, label_file in enumerate(label_files):
        work.append((args.output_dir, image_id, label_file, out_ann_file, class_name_to_id))

        if len(work) == 1:
            break


    with multiprocessing.Pool(processes=32) as pool:
        results = pool.starmap(process_image, work)

    annot_count = 0
    for result in results:
        for img in result['images']:
            data['images'].append(img)
        for annot in result['annotations']:
            annot['id'] = annot_count
            data['annotations'].append(annot)
            annot_count += 1

    with open(out_ann_file, 'w') as f:
        json.dump(data, f)


def process_image(output_dir, image_id, label_file, out_ann_file, class_name_to_id):
    print('Generating dataset from:', label_file)
    data = {
            'images': [],
            'annotations': []
            }

    with open(label_file) as f:
        label_data = json.load(f)

    base = osp.splitext(osp.basename(label_file))[0]
    out_img_file = osp.join(
        output_dir, 'JPEGImages', base + '.jpg'
    )

    img_file = osp.join(
        osp.dirname(label_file), label_data['imagePath']
    )
    img = np.asarray(PIL.Image.open(img_file))
    PIL.Image.fromarray(img).save(out_img_file)
    data['images'].append(dict(
        license=0,
        url=None,
        file_name=osp.relpath(out_img_file, osp.dirname(out_ann_file)),
        height=img.shape[0],
        width=img.shape[1],
        date_captured=None,
        id=image_id,
    ))

    '''
    masks = {}                                     # for area
    segmentations = collections.defaultdict(list)  # for segmentation
    for shape in label_data['shapes']:
        points = shape['points']
        label = shape['label']
        group_id = shape.get('group_id')
        shape_type = shape.get('shape_type')

        continue

        mask = labelme.utils.shape_to_mask(
            img.shape[:2], points, shape_type
        )

        if group_id is None:
            group_id = uuid.uuid1()

        instance = (label, group_id)

        if instance in masks:
            masks[instance] = masks[instance] | mask
        else:
            masks[instance] = mask

        points = np.asarray(points).flatten().tolist()
        segmentations[instance].append(points)
    segmentations = dict(segmentations)
    '''

    for shape in label_data['shapes']:
        points = shape['points']
        label = shape['label']
        group_id = shape.get('group_id')
        shape_type = shape.get('shape_type')

        #print(points)

        min1, min2 = min([x[0] for x in points]), min([x[1] for x in points])
        max1, max2 = max([x[0] for x in points]), max([x[1] for x in points])

        bbox = [min1, min2, max1 - min1, max2 - min2]
        #bbox = list(map(int, bbox))
        area = max1 * max2
        #print(bbox)

        #cls_name, group_id = instance
        cls_name = 'cell'
        if cls_name not in class_name_to_id:
            continue
        cls_id = class_name_to_id[cls_name]

        #mask = np.asfortranarray(mask.astype(np.uint8))
        #mask = pycocotools.mask.encode(mask)
        #area = float(pycocotools.mask.area(mask))
        #bbox = pycocotools.mask.toBbox(mask).flatten().tolist()

        data['annotations'].append(dict(
            id=None,
            image_id=image_id,
            category_id=cls_id,
            #segmentation=segmentations[instance],
            area=area,
            bbox=bbox,
            iscrowd=0,
        ))
        #annot_count += 1
    return data


if __name__ == '__main__':
    main()
