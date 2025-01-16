import os
import argparse
import datetime
import json
import time
import numpy as np
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

import timm
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from engine import train_one_epoch, val_one_epoch
from llama import Tokenizer
from llama_vqa import LLaMA_VQA
from dataloader import load_data

os.environ["CUDA_VISIBLE_DEVICES"]="2"

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=2, type=int, help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--accum_iter', default=1, type=int, help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--llama_model_path', default='/mnt/data/LLama/pretrained/llama/', type=str, help='path of llama model')
    parser.add_argument('--model', default='7B', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--adapter_layer', type=int, default=32, metavar='LENGTH', help='the number of adapter layer')
    parser.add_argument('--adapter_len', type=int, default=10, metavar='LENGTH', help='the adapter length')
    parser.add_argument('--max_seq_len', type=int, default=128, metavar='LENGTH', help='the maximum sequence length')
    parser.add_argument('--max_feats', type=int, default=10, metavar='LENGTH', help='the maximum feature length')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.14, help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=9e-2, metavar='LR', help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR', help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=2, metavar='N', help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--dataset', default='nextqa', type=str, help='dataset')
    parser.add_argument('--output_dir', default='./checkpoint/nextqa', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    parser.add_argument('--vaq', default=False, help='vaq loss') #'store_true'
    parser.add_argument('--qav', default=False, help='qav loss') #'store_true'
    parser.add_argument('--bias', type=float, default=3.5, help='attention bias')
    parser.add_argument('--tau', type=float, default=100., help='tau')
    parser.add_argument('--sub', default=False, help='subtitles for VLEP and TVQA')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    tokenizer = Tokenizer(model_path=f'{args.llama_model_path}./tokenizer.model')


    model = LLaMA_VQA(args)
    model.to(device)

    model_without_ddp = model

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)
    
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()
    best_acc = 0.

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    qn_types = ['TP', 'CW', 'DC', 'TC', 'DL', 'DO', 'TN', 'CH']
    processed_type = []
    acc_matrix = np.zeros((len(qn_types), len(qn_types)))

    for type in qn_types:
        print('training type: ' + type)
        processed_type.append(type)

        for epoch in range(args.start_epoch, args.epochs):

            data_loader_train = load_data(args, tokenizer, split='train', type=type)

            if args.distributed:
                data_loader_train.sampler.set_epoch(epoch)
                data_loader_val.sampler.set_epoch(epoch)

            train_stats = train_one_epoch(model, data_loader_train, optimizer, epoch, loss_scaler, args=args)

            average_acc = 0
            for i in range(len(processed_type)):
                model.eval()
                type = processed_type[i]
                data_loader_val = load_data(args, tokenizer, split='val', type=type)
                val_stats = val_one_epoch(model_without_ddp, data_loader_val, optimizer, epoch, args=args)
                acc_matrix[i, len(processed_type) -1] = val_stats['acc']

                average_acc += val_stats['acc']

            average_acc = average_acc / (len(processed_type))
            result_acc = "\taverage_acc: {:.4f}".format(average_acc)
            print(result_acc)
            
            if len(processed_type) - 1 >0:
                forgetting = np.mean((np.max(acc_matrix, axis=1) - acc_matrix[:, len(processed_type)-1])[:len(processed_type)-1])
                
                result_str = "\tForgetting: {:.4f}".format(forgetting)

                print(result_str)

            model_name = f'{type}_checkpoint_best'
            misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch=epoch, name=model_name)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch, **{f'val_{k}': v for k, v in val_stats.items()}}

            if args.output_dir and misc.is_main_process():
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
