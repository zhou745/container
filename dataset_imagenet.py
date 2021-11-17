import os.path as osp
import json
import requests
import time
import numpy as np
import io
from PIL import Image
import logging
import torch
import random

logger = logging.getLogger('global')

from dataset_base import BaseDataset
from datasets import build_transform


class ImageNetDataset(BaseDataset):
    """
    ImageNet Dataset.

    Arguments:
        - root_dir (:obj:`str`): root directory of dataset
        - meta_file (:obj:`str`): name of meta file
        - transform (list of ``Transform`` objects): list of transforms
        - read_type (:obj:`str`): read type from the original meta_file
        - evaluator (:obj:`Evaluator`): evaluate to get metrics
        - image_reader_type (:obj:`str`): reader type 'pil' or 'ks'
        - server_cfg (list): server configurations

    Metafile example::
        "n01440764/n01440764_10026.JPEG 0\n"
    """

    def __init__(self,
                 is_train,
                 args):

        if args.load_type == "ori":
            super(ImageNetDataset, self).__init__(read_from="ori",conf_path=args.conf_path)
        elif args.load_type == "petrel":
            super(ImageNetDataset, self).__init__(read_from="petrel",conf_path=args.conf_path)
        elif args.load_type == "mc":
            super(ImageNetDataset, self).__init__(read_from="mc",conf_path=args.conf_path)
        else:
            raise RuntimeError("unknown load type")

        split = "train" if is_train else "val"
        self.data_root = osp.join(args.data_path,split)
        if args.meta_file == "None":
            self.meta_file = osp.join(args.data_path, "meta/" + split + ".txt")
        else:
            self.meta_file = args.meta_file
        self.load_type = args.load_type
        self.image_transform = build_transform(is_train, args)

        # read in the meta files
        with open(self.meta_file) as f:
            lines = f.readlines()

        self.num = len(lines)
        self.metas = []
        for line in lines:
            filename, label = line.rstrip().split()
            self.metas.append({'filename': filename, 'label': label})

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        curr_meta = self.metas[idx]
        filename = osp.join(self.data_root, curr_meta['filename'])
        label = int(curr_meta['label'])

        if self.read_from == "ori":
            image = Image.open(filename)
            image = image.convert('RGB')
        else:
            img_bytes = self.read_file(filename)
            image = self.image_reader(img_bytes, filename)


        image = self.image_transform(image)

        # dict_out = {
        #     "images": image,
        #     "label": torch.tensor(label, dtype=torch.long),
        # }
        # dict_nouse = {
        #     "image_id": filename
        # }

        return image, torch.tensor(label, dtype=torch.long)

    def image_reader(self, img_bytes, filepath):
        buff = io.BytesIO(img_bytes)
        try:
            with Image.open(buff) as img:
                img = img.convert('RGB')
        except IOError:
            logger.info('Failed in loading {}'.format(filepath))
        return img
