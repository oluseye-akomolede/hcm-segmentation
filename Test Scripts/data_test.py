import os
from argparse import ArgumentParser, Namespace

import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

import pytorch_lightning as pl
from pytorch_lightning.utilities import argparse
from pytorch_lightning.core import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from datasets import load_dataset, IterableDataset, DownloadConfig, Dataset, Image
from torch.utils.data.distributed import DistributedSampler

import torch.distributed as dist
from PIL import Image, PngImagePlugin
import io

from huggingface_hub import login, logout
from tqdm import tqdm

workers = 0
batch_size = 180
dataset_name = "imagenet-1k"
PngImagePlugin.MAX_TEXT_CHUNK = 20000000  # Example: 20MB

class Basic_Dataloader():
    def __init__(self):
        self.train_dataset = None
        self.train_loader = None
        self.local_dataset = True
        return None
    @staticmethod
    def convert_bytes_to_image(image_bytes):
        try:
            image = Image.open(io.BytesIO(image_bytes))
            return image
        except Exception as e:
            print(f"Error converting bytes to image: {e}")
            return None
        
    @staticmethod
    def train_transforms(examples):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        nn_transforms =  transforms.Compose(
            [
                transforms.RandomResizedCrop(224), 
                transforms.RandomHorizontalFlip(), 
                transforms.ToTensor(), 
                normalize
            ]

        )

        try:
            examples["pixel_values"] = [nn_transforms(
                Basic_Dataloader.convert_bytes_to_image(examples["image"]["bytes"])
                .convert("RGB")
                )]
            return examples
        except Exception as e:
            print(e)
            return None

    @staticmethod
    def collate_fn(batch):
        if batch:
            return {
                'pixel_values': torch.stack([x['pixel_values'][0] for x in batch]),
                'label': torch.tensor([x['label'] for x in batch])
            }
        else:
            return None

    def train_dataloader(self):
        train_dataset = None

        d_cfg = DownloadConfig(
        resume_download = True,
        max_retries= 500
        )

        login(
            token=os.environ['HF_TOKEN']
        )

        train_dataset = load_dataset(
            dataset_name, 
            split='train', 
            streaming=True, 
            token=os.environ['HF_TOKEN'], 
            trust_remote_code=True,
            download_config=d_cfg,
            )
        
        logout()

        print(train_dataset)
            
        #train_dataset = split_dataset_by_node(train_dataset, rank=self.rank,world_size=self.world_size)
        train_dataset = train_dataset.map(Basic_Dataloader.train_transforms)
        train_dataset = train_dataset.with_format("torch")  
        try:
            self.train_dataset = train_dataset
            self.train_loader = torch.utils.data.DataLoader(
                dataset=self.train_dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=workers,
                collate_fn=Basic_Dataloader.collate_fn,      
                #pin_memory=True,
                drop_last=True,      
            )
        except Exception as exc:
            raise exc

        return self.train_loader

bdl = Basic_Dataloader()
data_loader = bdl.train_dataloader()
for epoch in range(2):
    print(data_loader)
    data_iterator = tqdm(
        data_loader,
        leave=True,
        unit="batch",
        postfix={
            "step": "training",
            "epo": 0,
            "lss": "%.8f" % 0.0,
            "acc1": "%.4f" % 0.0,
            "acc5": "%.4f" % 0.0,
        },
        disable=False,
        )

    for batch_idx, batch in enumerate(data_iterator):
        batch_b = batch
        if batch is not None:
            print(batch_idx)
            batch = None
        else:
            print("encountered None batch!!")
        #print(f"batch: {batch}")
        #print(f"batch_idx: {batch_idx}")
        #break
        