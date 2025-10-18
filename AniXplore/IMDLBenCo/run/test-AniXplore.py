import os
import json
import time
import types
import inspect
import argparse
import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

import IMDLBenCo.training_scripts.utils.misc as misc

from IMDLBenCo.registry import MODELS, POSTFUNCS
from IMDLBenCo.datasets import AnimeDatasetNoReal
from IMDLBenCo.transforms import get_albu_transforms
from IMDLBenCo.evaluation import PixelF1, ImageF1, PixelIOU, ImageAccuracy, PixelAccuracy, ImageAUC, PixelAUC

from IMDLBenCo.training_scripts.tester import test_one_epoch

def get_args_parser():
    parser = argparse.ArgumentParser('IMDLBench testing launch!', add_help=True)

    parser.add_argument('--epoch', default=0, type=int)
    parser.add_argument('--if_test_ImageF1', action='store_true')
    parser.add_argument('--if_test_ImageAccuracy', action='store_true')
    parser.add_argument('--if_test_PixelAccuracy', action='store_true')
    parser.add_argument('--if_test_ImageAUC', action='store_true')
    parser.add_argument('--if_test_PixelAUC', action='store_true')
    parser.add_argument('--if_test_PixelF1', action='store_true')
    parser.add_argument('--if_test_PixelIOU', action='store_true')


    parser.add_argument('--raw_img_data_root', type=str)
    parser.add_argument('--edited_img_data_root', type=str)
    parser.add_argument('--model', default=None, type=str,
                        help='The name of applied model', required=True)
    
    # ----Dataset parameters----
    parser.add_argument('--image_size', default=512, type=int,
                        help='image size of the images in datasets')
    
    parser.add_argument('--if_padding', action='store_true',
                        help='padding all images to same resolution.')
    
    parser.add_argument('--if_resizing', action='store_true', 
                        help='resize all images to same resolution.')
    # If edge mask activated
    parser.add_argument('--edge_mask_width', default=None, type=int,
                        help='Edge broaden size (in pixels) for edge maks generator.')
    parser.add_argument('--test_data_json', default='/root/Dataset/CASIA1.0', type=str,
                        help='test dataset json, should be a json file contains many datasets. Details are in readme.md')
    # ------------------------------------
    # Testing parameters
    parser.add_argument('--checkpoint_path', default = '/root/workspace/IML-ViT/output_dir', type=str, help='path to the dir where saving checkpoints')
    parser.add_argument('--test_batch_size', default=2, type=int,
                        help="batch size for testing")
    parser.add_argument('--no_model_eval', action='store_true', 
                        help='Do not use model.eval() during testing.')

    # Log parameters-----------
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    # -----------------------
    
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    args, remaining_args = parser.parse_known_args()


    model_class = MODELS.get(args.model)
    model_parser = misc.create_argparser(model_class)
    model_args = model_parser.parse_args(remaining_args)

    return args, model_args

def main(args, model_args):
    print("\nINTO test-anime-no-real !!!!!!!!!!!!\n")
    # init parameters for distributed training
    misc.init_distributed_mode(args)
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("=====args:=====")
    print("{}".format(args).replace(', ', ',\n'))
    print("=====Model args:=====")
    print("{}".format(model_args).replace(', ', ',\n'))
    device = torch.device(args.device)
    
    test_transform = get_albu_transforms('test')

    with open(args.test_data_json, "r") as f:
        test_dataset_json = json.load(f)
    
    
    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
    else:
        global_rank = 0
    
    # ========define the model directly==========
    # model = IML_ViT(
    #     vit_pretrain_path = model_args.vit_pretrain_path,
    #     predict_head_norm= model_args.predict_head_norm,
    #     edge_lambda = model_args.edge_lambda
    # )
    
    # --------------- or -------------------------
    # Init model with registry
    model = MODELS.get(args.model)
    
    # Filt usefull args
    if isinstance(model,(types.FunctionType, types.MethodType)):
        model_init_params = inspect.signature(model).parameters
    else:
        model_init_params = inspect.signature(model.__init__).parameters
        
    combined_args = {k: v for k, v in vars(args).items() if k in model_init_params}
    combined_args.update({k: v for k, v in vars(model_args).items() if k in model_init_params})
    model = model(**combined_args)
    # ============================================
    evaluator_list = [
        PixelF1(threshold=0.5, mode="origin"),
    ]
    if args.if_test_ImageF1:
        evaluator_list.append(ImageF1())
        print(f"add test ImageF1")
    if args.if_test_ImageAccuracy:
        evaluator_list.append(ImageAccuracy())
        print(f"add test ImageAccuracy")
    if args.if_test_PixelAccuracy:
        evaluator_list.append(PixelAccuracy())
        print(f"add test PixelAccuracy")
    if args.if_test_ImageAUC:
        evaluator_list.append(ImageAUC())
        print(f"add test ImageAUC")
    if args.if_test_PixelAUC:
        evaluator_list.append(PixelAUC())
        print(f"add test PixelAUC")
    if args.if_test_PixelF1:
        evaluator_list.append(PixelF1())
        print(f"add test PixelF1")
    if args.if_test_PixelIOU:
        evaluator_list.append(PixelIOU())
        print(f"add test PixelIOU")



    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module
    
    start_time = time.time()
    # get post function (if have)
    post_function_name = f"{args.model}_post_func".lower()
    print(f"Post function check: {post_function_name}")
    print(POSTFUNCS)
    if POSTFUNCS.has(post_function_name):
        post_function = POSTFUNCS.get(post_function_name)
    else:
        post_function = None
    
    # Start go through each datasets:
    for dataset_name, dataset_path in test_dataset_json.items():
        args.full_log_dir = os.path.join(args.log_dir, dataset_name)

        if global_rank == 0 and args.full_log_dir is not None:
            os.makedirs(args.full_log_dir, exist_ok=True)
            log_writer = SummaryWriter(log_dir=args.full_log_dir)
        else:
            log_writer = None
        
        # ---- dataset with crop augmentation ----
        dataset_test = AnimeDatasetNoReal(
            dataset_path,
            is_padding=args.if_padding,
            is_resizing=args.if_resizing,
            output_size=(args.image_size, args.image_size),
            common_transforms=test_transform,
            edge_width=args.edge_mask_width,
            post_funcs=post_function,
            raw_img_data_root=args.raw_img_data_root,
            edited_img_data_root=args.edited_img_data_root
        )
        
        # ------------------------------------
        print(dataset_test)
        print("len(dataset_test)", len(dataset_test))
        
        # Sampler
        if args.distributed:
            sampler_test = torch.utils.data.DistributedSampler(
                dataset_test, 
                num_replicas=num_tasks, 
                rank=global_rank, 
                shuffle=False,
                drop_last=False
            )
            print("Sampler_test = %s" % str(sampler_test))
        else:
            sampler_test = torch.utils.data.RandomSampler(dataset_test)

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, 
            sampler=sampler_test,
            batch_size=args.test_batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )

        print(f"Start testing on {dataset_name}! ")


        ckpt_path = args.checkpoint_path
        print(f"🔄 Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location='cuda')
        print(type(ckpt))
        print(ckpt.keys())

        if hasattr(model, 'module'):
            model.module.load_state_dict(ckpt['model'], strict=True)
        else:
            model.load_state_dict(ckpt['model'], strict=True)

        test_stats = test_one_epoch(
            model=model,
            data_loader=data_loader_test,
            evaluator_list=evaluator_list,
            device=device,
            epoch=args.epoch,
            log_writer=log_writer,
            args=args
        )
        log_stats = {
            **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': args.epoch}
    
        if args.full_log_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.full_log_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
        print(f"log_stats = {log_stats}")


        local_time = time.time() - start_time
        local_time_str = str(datetime.timedelta(seconds=int(local_time)))
        print(f'Testing on dataset {dataset_name} takes {local_time_str}')
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Total testing time {}'.format(total_time_str))
    exit(0)    
        


if __name__ == '__main__':
    args, model_args = get_args_parser()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, model_args)
