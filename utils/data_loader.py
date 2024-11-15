import torch
from torch.utils.data import DataLoader
from utils.Augmentation import get_augmentation
from datasets.video import Video_dataset

def train_data_loader(config, logger):
    transform_train = get_augmentation(True, config)
    # logger.info('train transforms: {}'.format(transform_train.transforms))

    train_data = Video_dataset(
        config.data.train_root, config.data.train_list,
        config.data.label_list, num_segments=config.data.num_segments,
        modality=config.data.modality,
        image_tmpl=config.data.image_tmpl, random_shift=config.data.random_shift,
        transform=transform_train, dense_sample=config.data.dense)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)                       
    train_loader = DataLoader(train_data,
        batch_size=config.data.batch_size, num_workers=config.data.workers,
        sampler=train_sampler, drop_last=True, pin_memory=True, persistent_workers=True)
    
    return train_loader, train_data

def val_data_loader(config, logger):
    transform_val = get_augmentation(False, config)
    # logger.info('val transforms: {}'.format(transform_val.transforms))

    val_data = Video_dataset(
        config.data.val_root, config.data.val_list, config.data.label_list,
        random_shift=False, num_segments=config.data.num_segments,
        modality=config.data.modality,
        image_tmpl=config.data.image_tmpl,
        transform=transform_val, dense_sample=config.data.dense)        
    
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data, shuffle=False)
    val_loader = DataLoader(val_data,
        batch_size=config.data.batch_size,num_workers=config.data.workers,
        sampler=val_sampler, drop_last=False, pin_memory=True, persistent_workers=True)
    
    return val_loader