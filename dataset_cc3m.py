import torch
from dataset_base import BaseDataset
from datasets import build_transform

from typing import Callable, Dict, List,Tuple
import os,glob,json
from collections import defaultdict
import io
import logging
from PIL import Image
import random
import numpy as np
import json

logger = logging.getLogger('global')


ImageID = int
Captions = List[str]



class CC300Dataset_cls(BaseDataset):

    def __init__(
        self,
        is_train,
        args
    ):

        if args.load_type == "ori":
            super(CC300Dataset_cls,self).__init__(read_from="ori",conf_path=args.conf_path)
        elif args.load_type == "petrel":
            super(CC300Dataset_cls,self).__init__(read_from="petrel",conf_path=args.conf_path)
        elif args.load_type == "mc":
            super(CC300Dataset_cls,self).__init__(read_from="mc",conf_path=args.conf_path)
        else:
            raise RuntimeError("unknown load type")

        #coco tokenizer and clip tokenizer

        #meta file load
        self.data_root = args.data_path
        self.meta_data = []
        with open(args.meta_file,"r") as f:
            lines = f.readlines()
            for line in lines:
                self.meta_data.append(json.loads(line))

        self.image_transform = build_transform(is_train, args)


    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:

        #read img
        filename, label = self.meta_data[idx]['filename'],self.meta_data[idx]['label']
        #image shoud be pil to use transform
        # t0 = time.time()
        if self.read_from == "ori":
            image = Image.open(filename)
            image = image.convert('RGB')
        else:
            try:
                filename_read = os.path.join(self.data_root,filename)
                img_bytes = self.read_file(filename_read)
                image = self.image_reader(img_bytes, filename_read)
            except Exception as e:
                print("dataset error can not find file {name}".format(name=filename), flush=True)
                idx = random.randint(0, self.__len__() - 1)
                return(self.__getitem__(idx))

        # Transform image and convert image from HWC to CHW format.
        image = self.image_transform(image)

        # dict_out = {
        #     "images": image,
        #     "label": torch.tensor(label, dtype=torch.long),
        #     }
        # dict_nouse = {
        #     "image_id": idx
        #     }
        return image,torch.tensor(label, dtype=torch.long)

    def image_reader(self,img_bytes, filepath):
        buff = io.BytesIO(img_bytes)
        try:
            with Image.open(buff) as img:
                img = img.convert('RGB')
        except IOError:
            logger.info('Failed in loading {}'.format(filepath))
        return img