#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
import json
from os.path import isfile

import _init_paths
from fast_rcnn.config import cfg, cfg_from_file
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import exifread

CLASSES = ('__background__',
           'mandarine')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}


def save_detections2(im, class_name, dets, imgName, outputDir, thresh=0.3):
    """Draw detected bounding boxes."""
    import matplotlib.pyplot as plt
    inds = np.where(dets[:, -1] >= thresh)[0]

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.3),
                fontsize=10, color='white')

    ax.set_title(('{} {} detections with '
                  'p({} | box) >= {:.1f}').format(len(inds), class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()

    (ignore, filename) = os.path.split(imgName)
    outfile = os.path.join(outputDir, filename)
    if not os.path.isdir(outputDir):
        os.makedirs(outputDir)
    print("Saving test image with boxes in {}".format(outfile))
    plt.savefig(outfile)
    plt.close()


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def process_image(net, im_file, outfolder):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.5
    NMS_THRESH = 0.3
    all_dets = {}
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        all_dets[cls] = dets
        save_detections2(im, cls, dets, im_file, outfolder, thresh=CONF_THRESH)

    return all_dets

def dms2dd(degrees, minutes, seconds, direction):
    '''
    See http://en.proft.me/2015/09/20/converting-latitude-and-longitude-decimal-values-p/
    Convert degrees gps to decimal
    '''
    dd = float(degrees) + float(minutes) / 60 + float(seconds) / (60 * 60);
    if direction == 'S' or direction == 'W':
        dd *= -1
    return dd;

def rtf(ratio):
    '''
    Ratio to float function
    '''
    if ratio.den == 0:
        return ratio.num
    else:
        return ratio.num / float(ratio.den)

def lat(tags):
    degrees_and_ref = map(rtf, tags['GPS GPSLatitude'].values) + [str(tags['GPS GPSLatitudeRef'].values)]
    return dms2dd(*degrees_and_ref)

def lon(tags):
    degrees_and_ref = map(rtf, tags['GPS GPSLongitude'].values) + [str(tags['GPS GPSLongitudeRef'].values)]
    return dms2dd(*degrees_and_ref)

def exiftags(file):
    # Open image file for reading (binary mode)
    with open(file, 'rb') as f:
        # Return Exif tags
        return exifread.process_file(f)

def counts_to_geojson(img_folder, counts_per_image_dict):
    def feature(lat, lon, properties_dict):
        return {
            "type": "Feature",
            "geometry":{
                "type":"Point",
                "coordinates":[lat, lon]
            },
            "properties": properties_dict
        }

    ret = {
        "type":"FeatureCollection",
        "features":[]
    }
    for image, count in counts_per_image.items():
        imgpath = os.path.join(img_folder, image)
        tags = exiftags(imgpath)
        ret["features"].append(feature(
            lat(tags), lon(tags), count,
            {"image_name": image,
             "count": count}
        ))

    return json.dumps(ret)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--imgfolder', dest='imgfolder',
                        help='Folder with images to use for inferece',
                        default=None, type=str)
    parser.add_argument('--outfolder', dest='outfolder',
                        help='Folder to use for storing resultant infereced images',
                        default=None, type=str)
    parser.add_argument('--imgsetfile', dest='imgsetfile',
                        help='Use a image set file (test.txt or so) to filter for files in imgfolder.',
                        default=None, type=str)



    args = parser.parse_args()

    return args

def ensure_file_exists(file):
    if not os.path.isfile(file):
        raise IOError("File {:s} not found.".format(file))

def ensure_dir_exists(dir):
    if not os.path.isdir(dir):
        raise IOError("Folder {:s} not found.".format(dir))


if __name__ == '__main__':
    args = parse_args()

    prototxt = "models/mandarines/VGG16/faster_rcnn_end2end/test.prototxt"
    cfgfile = "experiments/cfgs/faster_rcnn_end2end.yml"

    caffemodel = args.caffemodel

    for file in [prototxt, cfgfile, caffemodel]: ensure_file_exists(file)
    for dir in [args.imgfolder]: ensure_dir_exists(dir)

    cfg_from_file(cfgfile)

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print('\n\nLoaded network {:s}'.format(caffemodel))

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)


    image_names = []
    if not args.imgsetfile == None:
        assert os.path.exists(args.imgsetfile), \
                'Image set file does not exist: {}'.format(args.imgsetfile)
        with open(args.imgsetfile) as f:
            im_names = [x.strip() + ".jpg" for x in f.readlines()]
    else:
        im_names = [f for f in os.listdir(args.imgfolder)
                    if isfile(os.path.join(args.imgfolder, f)) and f.lower().endswith(".jpg")]

    counts_per_image = {}
    for im_name in im_names:
        print('Processing {} from {} and saving into {}'.format(im_name, args.imgfolder, args.outfolder))
        dets = process_image(net, os.path.join(args.imgfolder, im_name), args.outfolder)
        counts_per_image[im_name] = len(dets['mandarine'])

    target_geojson = os.path.join(args.outfolder, "counts.geojson")
    print('Writing '.format(target_geojson))
    with open(target_geojson, "w") as text_file:
        text_file.write(counts_to_geojson(args.imgfolder, counts_per_image))