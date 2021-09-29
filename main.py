# -*- coding: utf-8 -*-
import os
import time
import argparse
import numpy as np

import torch
import torch.optim as optim
import torch.cuda.amp as amp
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import libs.trainer as trainer
from libs.model.model import ROG
from settings import plan_experiment
from libs.autoattack import AutoAttack
from libs.dataloader import dataloader, helpers
from libs.utilities import losses, utils, test, test_pgd

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

tasks = {
    '1': 'Task01_BrainTumour',
    '2': 'Task02_Heart',
    '3': 'Task03_Liver',
    '4': 'Task04_Hippocampus',
    '5': 'Task05_Prostate',
    '6': 'Task06_Lung',
    '7': 'Task07_Pancreas',
    '8': 'Task08_HepaticVessel',
    '9': 'Task09_Spleen',
    '10': 'Task10_Colon',
    '11': 'Task11_KiTS'
}


def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '1234' + port

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def main(rank, world_size, args):
    print(f"Running on rank {rank}.")
    setup(rank, world_size, args.port)

    training = args.test

    if args.ft:
        args.resume = True

    info, model_params = plan_experiment(
        tasks[args.task], args.batch, args.patience, args.fold, rank)

    # PATHS AND DIRS
    args.save_path = os.path.join(
        info['output_folder'], args.name, f'fold_{args.fold}')
    images_path = os.path.join(args.save_path, 'volumes')
    if args.adv:
        adv_path = os.path.join(args.save_path, 'autoattack')
        os.makedirs(adv_path, exist_ok=True)
        clean_path = os.path.join(adv_path, 'clean')
        os.makedirs(clean_path, exist_ok=True)
    load_path = args.save_path  # If we're resuming the training of a model
    if args.pretrained is not None:
        load_path = os.path.join(
            'Results', tasks[args.task], args.pretrained, f'fold_{args.fold}')

    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(images_path, exist_ok=True)

    # SEEDS
    np.random.seed(info['seed'])
    torch.manual_seed(info['seed'])

    cudnn.deterministic = False  # Normally is False
    cudnn.benchmark = args.benchmark  # Normaly is True

    # CREATE THE NETWORK ARCHITECTURE
    model = ROG(model_params).to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    if rank == 0:
        f = open(os.path.join(args.save_path, 'architecture.txt'), 'w')
        print(model, file=f)
    scaler = amp.GradScaler()

    if training or args.ft:
        # Initialize optimizer
        optimizer = optim.Adam(
            ddp_model.parameters(), lr=args.lr,
            weight_decay=1e-5, amsgrad=True)
        annealing = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, verbose=True, patience=info['patience'], factor=0.5)
        # Save experiment description
        if rank == 0:
            name_d = 'description_train.txt'
            name_a = 'args_train.txt'
            if not training:
                name_d = 'description_test.txt'
                name_a = 'args_test.txt'

            with open(os.path.join(args.save_path, name_d), 'w') as f:
                for key in info:
                    print(key, ': ', info[key], file=f)
                for key in model_params:
                    print(key, ': ', model_params[key], file=f)
                print(
                    'Number of parameters:',
                    sum([p.data.nelement() for p in model.parameters()]),
                    file=f)

                with open(os.path.join(args.save_path, name_a), 'w') as f:
                    for arg in vars(args):
                        print(arg, ':', getattr(args, arg), file=f)

    # CHECKPOINT
    epoch = 0
    best_dice = 0
    if args.resume:
        name = 'checkpoint.pth.tar'
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint = torch.load(
            os.path.join(load_path, name),
            map_location=map_location)
        # Only for training. Must be loaded before loading the model
        if not args.ft:
            np.random.set_state(checkpoint['rng'][0])
            torch.set_rng_state(checkpoint['rng'][1])

        if rank == 0:
            print('Loading model epoch {}.'.format(checkpoint['epoch']))

        ddp_model.load_state_dict(
            checkpoint['state_dict'], strict=(not args.ft))
        # if ft, we do not need the previous optimizer
        if not args.ft:
            best_dice = checkpoint['best_dice']
            epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            annealing.load_state_dict(checkpoint['scheduler'])
        args.load_model = 'best_dice'

    criterion = losses.segmentation_loss(alpha=1)
    metrics = utils.Evaluator(info['classes'])

    # DATASETS
    train_dataset = dataloader.Medical_data(
        True, info['train_file'], info['root'], info['p_size'])
    val_dataset = dataloader.Medical_data(
        True, info['val_file'], info['root'], info['val_size'], val=True)
    test_dataset = dataloader.Medical_data(
        False, info['test_file'], info['root'], info['val_size'])

    # SAMPLERS
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, drop_last=True)
    train_collate = helpers.collate(info['in_size'])
    val_collate = helpers.collate_val(list(map(int, info['val_size'])))

    # DATALOADERS
    train_loader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=info['batch'],
        num_workers=8, collate_fn=train_collate)
    val_loader = DataLoader(
        val_dataset, sampler=None, batch_size=info['test_batch'],
        num_workers=8, collate_fn=val_collate)
    test_loader = DataLoader(
        test_dataset, sampler=None, shuffle=False, batch_size=1, num_workers=0)

    if args.adv:
        adv_loader = dataloader.Medical_data(
            False, info['val_file'], info['root'], info['val_size'],
            adv_path, val=True, pgd=True)

    # TRAIN THE MODEL
    is_best = False
    torch.cuda.empty_cache()

    def moving_average(cum_loss, new_loss, n=5):
        if cum_loss is None:
            cum_loss = new_loss
        cum_loss = np.append(cum_loss, new_loss)
        if len(cum_loss) > n:
            cum_loss = cum_loss[1:]
        return cum_loss.mean()

    if training:
        accumulated_val_loss = None
        out_file = open(os.path.join(args.save_path, 'progress.csv'), 'a+')
        noise_data = torch.zeros(
            [info['batch'], model_params['modalities']] + info['in_size'],
            device=rank)
        for epoch in range(epoch + 1, args.epochs + 1):
            lr = utils.get_lr(optimizer)
            if rank == 0:
                print('--------- Starting Epoch {} --> {} ---------'.format(
                    epoch, time.strftime("%H:%M:%S")))
                print('Current learning rate:', lr)

            train_sampler.set_epoch(epoch)
            train_loss, noise_data = trainer.train(
                args, info, ddp_model, train_loader, noise_data, optimizer,
                criterion, scaler, rank)
            val_loss, dice = trainer.val(
                args, ddp_model, val_loader, criterion, metrics, rank)

            accumulated_val_loss = moving_average(
                accumulated_val_loss, val_loss)
            # if epoch % 2 == 0:
            annealing.step(accumulated_val_loss)

            mean = sum(dice) / len(dice)
            is_best = best_dice < mean
            best_dice = max(best_dice, mean)

            # Save ckeckpoint (every 100 epochs, best model and last)
            if rank == 0:
                state = {
                    'epoch': epoch,
                    'state_dict': ddp_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': annealing.state_dict(),
                    'rng': [np.random.get_state(),
                            torch.get_rng_state()],
                    'loss': [train_loss, val_loss],
                    'lr': lr,
                    'dice': dice,
                    'best_dice': best_dice}
                checkpoint = epoch % 100 == 0
                utils.save_epoch(
                    state, mean, args.save_path, out_file,
                    checkpoint=checkpoint, is_best=is_best)

            if lr <= (args.lr / (2 ** 4)):
                print('Stopping training: learning rate is too small')
                break
        out_file.close()

    # Loading the best model for testing
    dist.barrier()
    torch.cuda.empty_cache()
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    name = args.load_model + '.pth.tar'
    checkpoint = torch.load(
        os.path.join(args.save_path, name), map_location=map_location)
    torch.set_rng_state(checkpoint['rng'][1])  # TODO: Is this still necessary?
    ddp_model.load_state_dict(checkpoint['state_dict'])
    if rank == 0:
        print('Testing epoch with best dice ({}: dice {})'.format(
            checkpoint['epoch'], checkpoint['dice']))

    # TEST THE MODEL
    if args.adv:
        adversary = AutoAttack(
            ddp_model.forward, adv_loader, dice_thresh=0.5, device=rank,
            n_target_classes=info['classes']-1, eps=args.eps,
            n_iter=args.adv_iters, im_path=adv_path,
            log_path=os.path.join(args.save_path, 'log_autoattack'),
            model_name=args.load_model)
        # WORKING: APGD-CE, PGD, square, fab
        # TODO: Check apgd
        attck = ['apgd-ce', 'pgd', 'fab', 'square']
        if info['classes'] > 2:
            # We can do this attack only if the task is not binary
            attck += ['apgd-dlr', 'pgd', 'fab', 'square']
        adversary.attacks_to_run = attck
        # _ = adversary.run_standard_evaluation(bs=info['test_batch'])
        _ = adversary.run_standard_evaluation_individual(bs=info['test_batch'])
    else:
        # EVALUATE THE MODEL
        trainer.test(
            info, ddp_model, test_loader, images_path,
            info['test_file'], rank, world_size)
        dist.barrier()
        # CALCULATE THE FINAL METRICS
        if rank == 0:
            test.test(
                images_path, info['root'], info['test_file'], info['classes'])
    cleanup()


if __name__ == '__main__':
    # SET THE PARAMETERS
    parser = argparse.ArgumentParser()
    # EXPERIMENT DETAILS
    parser.add_argument('--task', type=str, default='4',
                        help='Task to train/evaluate (default: 4)')
    parser.add_argument('--name', type=str, default='ROG',
                        help='Name of the current experiment (default: ROG)')
    parser.add_argument('--AT', action='store_true', default=False,
                        help='Train a model with Free AT')
    parser.add_argument('--fold', type=str, default=0,
                        help='Which fold to run. Value from 0 to 4')

    parser.add_argument('--test', action='store_false', default=True,
                        help='Evaluate a model')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Continue training a model')
    parser.add_argument('--ft', action='store_true', default=False,
                        help='Fine-tune a model (will not load the optimizer)')
    parser.add_argument('--load_model', type=str, default='best_dice',
                        help='Weights to load (default: best_dice)')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Name of the folder with the pretrained model')

    # TRAINING HYPERPARAMETERS
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate (default: 1e-3)')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Maximum number of epochs (default: 1000)')
    parser.add_argument('--patience', type=int, default=50,
                        help='Patience of the scheduler (default: 50)')
    parser.add_argument('--batch', type=int, default=2,
                        help='Batch size (default: 2)')

    # ADVERSARIAL TRAINING AND TESTING
    parser.add_argument('--eps', type=float, default=8.,
                        help='Epsilon for the adv. attack (default: 8/255)')
    parser.add_argument('--adv_iters', type=int, default=5,
                        help='Number of iterations for AutoAttack')
    parser.add_argument('--adv', action='store_true', default=False,
                        help='Evaluate a model\'s robustness')

    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU(s) to use (default: 0)')
    parser.add_argument('--port', type=str, default='5')
    parser.add_argument('--benchmark', action='store_false', default=True,
                        help='Deactivate CUDNN benchmark')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['MKL_THREADING_LAYER'] = 'GNU'

    world_size = torch.cuda.device_count()
    if world_size > 1:
        mp.spawn(main, args=(world_size, args,), nprocs=world_size, join=True)
    else:
        # To allow breakpoints
        main(0, 1, args)
