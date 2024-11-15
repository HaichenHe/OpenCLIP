import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
import time
from utils.utils import init_distributed_mode, AverageMeter, reduce_tensor, accuracy
from slowfast.utils.misc import log_model_info
import slowfast.utils.logging as logging

from slowfast.models import clip
import numpy as np
from utils.logger import setup_logger
import yaml
from dotmap import DotMap
from datasets.video import Video_dataset
from datasets.transforms import GroupScale, GroupCenterCrop, Stack, ToTorchFormatTensor, GroupNormalize, GroupOverSample, GroupFullResSample
from slowfast.models.text_prompt import text_prompt
from slowfast.models import build_model

# logger = logging.get_logger(__name__)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='global config file')
    parser.add_argument('--log_time', default='001')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local-rank", type=int,
                        help='local rank for DistributedDataParallel')
    parser.add_argument(
        "--precision",
        choices=["amp", "fp16", "fp32"],
        default="amp",
        help="Floating point precition."
    )
    parser.add_argument('--test_crops', type=int, default=1)   
    parser.add_argument('--test_clips', type=int, default=3) 
    parser.add_argument('--dense', default=False, action="store_true",
                    help='use dense sample for test as in Non-local I3D')
    args = parser.parse_args()
    return args

def update_dict(dict):
    new_dict = {}
    for k, v in dict.items():
        new_dict[k.replace('module.', '')] = v
    return new_dict

def validate(val_loader, classes, device, model, config, n_class, test_crops, test_clips):
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()
    proc_start_time = time.time()

    sim_logits = []     # 
    labels = []     # 
    i_features = []

    with torch.no_grad():
        text_inputs = classes.to(device)
        cls_emb = model.module.encode_text(text_inputs)
        for i, (image, class_id) in enumerate(val_loader):
            batch_size = class_id.numel()
            num_crop = test_crops

            num_crop *= test_clips  # 4 clips for testing when using dense sample

            class_id = class_id.to(device)
            n_seg = config.data.num_segments
            image = image.view((-1, n_seg, 3) + image.size()[-2:])
            b, t, c, h, w = image.size()
            image_input = image.to(device).view(-1, c, h, w) 
            image_features = model.module.encode_image(image_input).view(b, t, -1) #b*num_crop, t,d
            cnt_time = time.time() - proc_start_time

            vid_emb = image_features.reshape(batch_size, num_crop*t, -1).mean(dim=1, keepdim=False)  # b,d
            vid_emb = vid_emb / vid_emb.norm(dim=-1, keepdim=True)
            cls_emb = cls_emb / cls_emb.norm(dim=-1, keepdim=True)
            similarity = vid_emb @ cls_emb.t() 
            similarity = model.module.model.logit_scale.exp() * similarity


            ########## gathering 
            i_features.append(concat_all_gather(vid_emb))
            sim_logits.append(concat_all_gather(similarity))
            labels.append(concat_all_gather(class_id))
            ##########


            prec = accuracy(similarity, class_id, topk=(1, 5))
            prec1 = reduce_tensor(prec[0])
            prec5 = reduce_tensor(prec[1])

            top1.update(prec1.item(), class_id.size(0))
            top5.update(prec5.item(), class_id.size(0))

            if i % config.logging.print_freq == 0 and dist.get_rank() == 0:
                runtime = float(cnt_time) / (i+1) / (batch_size * dist.get_world_size())
                print(
                    ('Test: [{0}/{1}], average {runtime:.4f} sec/video \t'
                     'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                     'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), runtime=runtime, top1=top1, top5=top5)))

    if dist.get_rank() == 0:
        ## half-classes evaluation
        sim, la = sim_logits[0], labels[0]
        vid_feat = i_features[0]
        for i in range(1, len(sim_logits)): 
            sim = torch.cat((sim, sim_logits[i]), 0)
            la = torch.cat((la, labels[i]), 0)
            vid_feat = torch.cat((vid_feat, i_features[i]), 0)

        text_feat = cls_emb/ cls_emb.norm(dim=-1, keepdim=True)
        acc_split, acc_split_top5 = multi_split_test(vid_feat.cpu(), text_feat.cpu(), la.cpu())
        accuracy_split, accuracy_split_std = np.mean(acc_split), np.std(acc_split)
        accuracy_split_top5, accuracy_split_top5_std = np.mean(acc_split_top5), np.std(acc_split_top5)
        print('-----Half-classes Evaluation-----')
        print('Top1: mean {:.03f}%, std {:.03f}%'.format(accuracy_split, accuracy_split_std))
        print('Top5: mean {:.03f}%, std {:.03f}%'.format(accuracy_split_top5, accuracy_split_top5_std))  
    return top1.avg

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output.cpu()

def compute_accuracy(vis_emb, text_emb, label):
    n_class = len(text_emb)
    n_samples = len(vis_emb)
    similarity=(100.0 * vis_emb @ text_emb.T)
    similarity=similarity.view(n_samples, -1, n_class).softmax(dim = -1)
    similarity=similarity.mean(dim = 1, keepdim = False)  # b 101
    prec=accuracy(similarity, label, topk = (1, 5))
    return prec[0], prec[1]
 
def multi_split_test(vis_embs, text_embs, true_label):
    # vis_embs: [10000, 768]
    # text_embs: [101, 768]
    # true_label: [10000,]
    full_acc1, full_acc5 = compute_accuracy(vis_embs, text_embs, true_label)
    print('-----Full-classes Evaluation------')
    print('Overall Top1 {:.03f}% Top5 {:.03f}%'.format(full_acc1.item(), full_acc5.item()))
 
    # Calculate accuracy per split
    # Only when the model has been trained on a different dataset
    true_label = true_label.numpy()
    accuracy_split, accuracy_split_top5 = np.zeros(10), np.zeros(10)
    for split in range(len(accuracy_split)):
        np.random.seed(split)
        sel_classes = np.random.permutation(len(text_embs))[:len(text_embs) // 2]  # [50, ]
        sel = [l in sel_classes for l in true_label]  # len = 10000
        subclasses = np.unique(true_label[sel])  # [50, ]
        # label_map = {}
        # for i in range(len(subclasses)):
        #     label_map[subclasses[i]] = i
        # new_label = np.array([label_map[l] for l in true_label[sel]])
        # new_label = torch.from_numpy(new_label)
        # label mapping: [4900, ], new label
        tl = np.array([int(np.where(l == subclasses)[0]) for l in true_label[sel]])
        tl = torch.from_numpy(tl)
        acc, acc5 = compute_accuracy(vis_embs[sel], text_embs[subclasses], tl)
        accuracy_split[split] = acc
        accuracy_split_top5[split] = acc5
    
    return accuracy_split, accuracy_split_top5

def main(args):
    init_distributed_mode(args)

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = DotMap(config)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        cudnn.benchmark = True
    
    model = build_model(config)

    logger_dir = os.path.join('exps', config['data']['dataset'], config['network']['arch'], args.log_time)
    logger = setup_logger(output=logger_dir,
                          distributed_rank=dist.get_rank(),
                          name=f'OpenCLIP')
    log_model_info(model, config,logger, use_train_input=False)

    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]
    scale_size = 256 if config.data.input_size == 224 else config.data.input_size

    # crop size
    input_size = config.data.input_size

    # control the spatial crop
    if args.test_crops == 1: # one crop
        cropping = torchvision.transforms.Compose([
            GroupScale(scale_size),
            GroupCenterCrop(input_size),
        ])
    elif args.test_crops == 3:  # do not flip, so only 3 crops (left right center)
        cropping = torchvision.transforms.Compose([
            GroupFullResSample(
                crop_size=input_size,
                scale_size=scale_size,
                flip=False)
        ])
    elif args.test_crops == 5:  # do not flip, so only 5 crops
        cropping = torchvision.transforms.Compose([
            GroupOverSample(
                crop_size=input_size,
                scale_size=scale_size,
                flip=False)
        ])
    elif args.test_crops == 10:
        cropping = torchvision.transforms.Compose([
            GroupOverSample(
                crop_size=input_size,
                scale_size=scale_size,
            )
        ])
    else:
        raise ValueError("Only 1, 3, 5, 10 crops are supported while we got {}".format(args.test_crops))


    val_data = Video_dataset(
        config.data.val_root, config.data.val_list, config.data.label_list,
        random_shift=False, num_segments=config.data.num_segments,
        modality=config.data.modality,
        image_tmpl=config.data.image_tmpl,
        test_mode=True,
        transform=torchvision.transforms.Compose([
            cropping,
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            GroupNormalize(input_mean, input_std),
        ]),
        dense_sample=args.dense,
        test_clips=args.test_clips)

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
    val_loader = DataLoader(val_data,
        batch_size=config.data.batch_size, num_workers=config.data.workers,
        sampler=val_sampler, pin_memory=True, drop_last=False)



    classes = text_prompt(val_data)
    n_class = classes.size(0)

    prec1 = validate(
        val_loader, classes, device,
        model,  config, n_class, args.test_crops, args.test_clips)
    return



if __name__ == '__main__':
    args = get_parser()
    main(args)

