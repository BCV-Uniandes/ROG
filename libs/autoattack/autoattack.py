import os
import json
import time
import nibabel as nib

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adv_utils import Logger
import libs.dataloader.helpers as helpers
from libs.utilities import test_pgd

# Attacks
from .autopgd import APGDAttack
from .fab import FABAttack
from .square import SquareAttack


class AutoAttack():
    def __init__(self, model, loader, dice_thresh, n_target_classes, im_path,
                 eps=.3, seed=None, verbose=True, attacks_to_run=[],
                 device='cuda', log_path=None, n_iter=20, model_name=None,
                 visualizations=False):
        self.model = model
        self.epsilon = eps / 255.
        self.seed = seed
        self.verbose = verbose
        self.attacks_to_run = attacks_to_run
        self.device = device
        self.logger = Logger(log_path)
        self.n_iter = n_iter
        self.visualizations = visualizations
        # Dice loss
        self.dice_thresh = dice_thresh
        self.dice = Dice(eps=1e-5)
        # Dataloader
        self.loader = loader
        self.model_name = model_name
        self.im_path = im_path
        self.classes = n_target_classes

        self.apgd = APGDAttack(
            self.model, dice_thresh, n_restarts=5, n_iter=n_iter,
            verbose=False, eps=self.epsilon, eot_iter=1,
            rho=.75, seed=self.seed, device=self.device)

        self.fab = FABAttack(
            self.model, dice_thresh, n_target_classes=n_target_classes,
            n_restarts=5, n_iter=n_iter, eps=self.epsilon, seed=self.seed,
            verbose=False, device=self.device)

        self.square = SquareAttack(
            self.model, dice_thresh, p_init=0.8, n_queries=self.n_iter,
            eps=self.epsilon, n_restarts=1, seed=self.seed,
            verbose=False, device=self.device, resc_schedule=False)

    def get_seed(self):
        return time.time() if self.seed is None else self.seed

    def check_calculated(self, idx, suffix):
        name = self.loader.filenames.iloc[idx][0][11:]
        im_name = os.path.join(
            self.im_path, self.loader.folder, name[:-7] + suffix + name[-7:])
        im_exists = os.path.isfile(im_name)
        if not im_exists:
            return False

        gt_name = os.path.join(
            self.im_path, 'clean', name[:-7] + '_gt' + name[-7:])
        gt_exists = os.path.isfile(gt_name)
        assert self.loader.folder == 'clean' and gt_exists, 'GT does not exist'
        return True

    @torch.no_grad()
    def run_standard_evaluation(self, bs=3):
        if self.verbose:
            print('Including {}'.format(', '.join(self.attacks_to_run)))

        # calculate the initial performance
        os.makedirs(os.path.join(self.im_path, 'clean'), exist_ok=True)
        results_file = os.path.join(self.im_path, 'clean.txt')

        if os.path.isfile(results_file):
            # Load the results if they have already been calculated
            with open(results_file) as outf:
                results = json.load(outf)
            clean_dice = torch.tensor(
                list(results['Individual Dice'].values()))
            mean = results['Mean Dice']
            clean_class = torch.tensor(
                list(results['Individual per class'].values()))
        else:
            for b_idx in range(len(self.loader)):
                self.loader.folder = '/clean'

                # Skip if we already evaluated this image
                exists = self.check_calculated(b_idx, '_pred')
                if exists:
                    continue

                x, y, p_name, affine = self.loader.update_pgd(b_idx)
                output = self.model(x.to(self.device))

                # Save predictions, input patch and ground truth
                output = torch.argmax(F.softmax(output, 1), dim=1).cpu()[0]
                x = x.cpu().permute(0, 2, 3, 4, 1)[0]
                im_name = p_name[:-7] + '_pred' + p_name[-7:]
                helpers.save_image(
                    output, os.path.join(self.im_path, 'clean', im_name),
                    affine)

                im_name = p_name[:-7] + '_gt' + p_name[-7:]
                helpers.save_image(
                    y.cpu(), os.path.join(self.im_path, 'clean', im_name),
                    affine)

                if self.visualizations:
                    im_name = p_name[:-7] + '_input' + p_name[-7:]
                    helpers.save_image(
                        x, os.path.join(self.im_path, 'clean', im_name),
                        affine)

            # Save the results
            clean_path = os.path.join(self.im_path, 'clean')
            dice = test_pgd.test(
                clean_path, clean_path, '_pred', self.loader.csv_file,
                self.classes + 1)

            clean_dice = {
                row[0]: sum(row[1:(self.classes + 1)]) / self.classes
                for row in dice}
            mean = clean_dice.pop('mean')
            clean_class = {row[0]: row[1:(self.classes + 1)] for row in dice}
            with open(results_file, 'w') as outf:
                json.dump(
                    {'Mean Dice': mean,
                     'Individual Dice': clean_dice,
                     'Individual per class': clean_class},
                    outf
                )
            clean_dice = torch.tensor(list(clean_dice.values()))
            clean_class = torch.tensor(list(clean_class.values()))

        self.dice_thresh = mean * self.dice_thresh
        robust_flags = clean_dice > self.dice_thresh
        robust_acc = torch.sum(robust_flags).item() / len(self.loader)

        if self.verbose:
            self.logger.log('====== {} ======'.format(self.model_name))
            txt = 'Eps: {}, Dice ths: {},  Initial acc: {:.2%}'
            self.logger.log(
                txt.format(self.epsilon, self.dice_thresh, robust_acc))
            self.logger.log('Dice scores: {}, mean: {}'.format(
                clean_dice, mean))
            self.logger.log('Dice by class: {}, mean: {}'.format(
                clean_class[:-1], clean_class[-1]))

        new_dice = clean_dice.clone().flatten().float()
        for attack in self.attacks_to_run:
            self.loader.folder = '/' + attack + str(self.n_iter)
            robust_flags = new_dice > self.dice_thresh
            robust_acc = torch.sum(robust_flags).item() / len(self.loader)
            txt = '\n Eps: {}, Dice ths: {}, Iters: {}'
            self.logger.log(
                txt.format(self.epsilon, self.dice_thresh, self.n_iter))
            num_robust = torch.sum(robust_flags).item()
            if num_robust == 0:
                break

            robust_lin_idcs = torch.nonzero(robust_flags, as_tuple=False)
            if num_robust > 1:
                robust_lin_idcs.squeeze_()

            x, y, names, batch_datapoint_idcs = [], [], [], []
            for b_idx in robust_lin_idcs:
                exists = self.check_calculated(int(b_idx), '_adv')
                if exists:
                    continue
                x_, y_, p_name, affine = self.loader.update_pgd(int(b_idx))
                x.append(x_)
                y.append(y_)
                names.append(p_name)
                batch_datapoint_idcs.append(int(b_idx))

                if not (len(x) == bs or (b_idx == robust_lin_idcs[-1]
                                         and len(x) > 0)):
                    continue

                # Only the correct examples
                x = torch.cat(x, dim=0).to(self.device)
                y = torch.cat(y, dim=0).long().to(self.device)
                batch_datapoint_idcs = torch.tensor(
                    batch_datapoint_idcs, device=self.device)

                # run attack
                if attack == 'apgd-ce':
                    self.apgd.loss = 'ce'
                    self.apgd.dice_thresh = self.dice_thresh
                    self.apgd.seed = self.get_seed()
                    _, adv_curr = self.apgd.perturb(x, y)

                elif attack == 'apgd-dlr':
                    self.apgd.loss = 'dlr'
                    self.apgd.dice_thresh = self.dice_thresh
                    self.apgd.seed = self.get_seed()
                    _, adv_curr = self.apgd.perturb(x, y)

                elif attack == 'fab':
                    self.fab.n_iter = 5
                    self.fab.seed = self.get_seed()
                    self.fab.dice_thresh = self.dice_thresh
                    adv_curr = self.fab.perturb(x, y)

                elif attack == 'square':
                    self.square.seed = self.get_seed()
                    self.square.dice_thresh = self.dice_thresh
                    adv_curr = self.square.perturb(x, y)

                elif attack == 'pgd':
                    epsilon = torch.tensor(
                        self.epsilon, device=self.device).unsqueeze(0).expand(
                        x.shape[1]).view(x.shape[1], 1, 1, 1)
                    delta = attack_pgd(
                        self.model, x, y, epsilon, self.device, epsilon / 3.,
                        iters=self.n_iter)
                    adv_curr = x + delta

                else:
                    raise ValueError('Attack not supported')

                output = self.model(adv_curr)
                dice_score = self.dice(output, y)
                false_batch = ~(dice_score > self.dice_thresh).to(self.device)
                # print(dice_score, (x == adv_curr).all())
                non_robust_lin_idcs = batch_datapoint_idcs[false_batch]
                robust_flags[non_robust_lin_idcs] = False
                new_dice[batch_datapoint_idcs] = dice_score.cpu()

                if self.verbose:
                    num_non_robust_batch = torch.sum(false_batch)
                    txt = '{} - {} out of {} successfully perturbed'
                    self.logger.log(txt.format(
                        attack, num_non_robust_batch, len(output)))

                adv_curr = adv_curr.cpu().permute(0, 2, 3, 4, 1)
                x = x.cpu().permute(0, 2, 3, 4, 1)
                output = torch.argmax(F.softmax(output, 1), dim=1).cpu()
                path_save = os.path.join(
                    self.im_path, attack + str(self.epsilon))
                os.makedirs(path_save, exist_ok=True)

                for idx in range(len(batch_datapoint_idcs)):
                    im_name = names[idx][:-7] + '_pred' + names[idx][-7:]
                    helpers.save_image(
                        output[idx], os.path.join(path_save, im_name), affine)
                    if self.visualizations:
                        im_name = names[idx][:-7] + '_delta' + names[idx][-7:]
                        helpers.save_image(
                            x[idx] - adv_curr[idx].cpu(),
                            os.path.join(path_save, im_name),
                            affine
                        )

                x, y, names, batch_datapoint_idcs = [], [], [], []

            clean_path = os.path.join(self.im_path, 'clean')
            dice_final = test_pgd.test(
                path_save, clean_path, '_pred', self.loader.csv_file,
                self.classes + 1)
            rob_dice = {
                row[0]: sum(row[1:(self.classes + 1)]) / self.classes
                for row in dice_final}
            rob_mean = rob_dice.pop('mean')
            rob_class = {
                row[0]: row[1:(self.classes + 1)] for row in dice_final}

            if self.verbose:
                self.logger.log(
                    'Mean scores: {}, \nClass scores: {}'.format(
                        rob_mean, rob_class['mean']
                    )
                )

        return rob_dice

    def run_standard_evaluation_individual(self, bs=3):
        if self.verbose:
            print('Including {}'.format(', '.join(self.attacks_to_run)))

        l_attacks = self.attacks_to_run
        adv = {}
        verbose_indiv = self.verbose
        self.verbose = True

        for c in l_attacks:
            startt = time.time()
            self.attacks_to_run = [c]
            adv[c] = self.run_standard_evaluation(bs=bs)
            if verbose_indiv:
                values = list(adv[c].values())
                acc_indiv = sum(values) / len(values)
                space = '\t \t' if c == 'fab' else '\t'
                self.logger.log(
                    'robust accuracy by {} {} {:.2%} \t (time attack: {:.1f} s)'.format(
                        c.upper(), space, acc_indiv,  time.time() - startt)
                )

        return adv


# # # # # # # # # # # # # # # # # loss function # # # # # # # # # # # # # # # #
def one_hot(gt, categories):
    size = [*gt.shape] + [categories]
    y = gt.view(-1, 1)
    gt = torch.FloatTensor(y.nelement(), categories).zero_().cuda()
    gt.scatter_(1, y, 1)
    gt = gt.view(size).permute(0, 4, 1, 2, 3).contiguous()
    return gt


class Dice(nn.Module):
    def __init__(self, eps=1):
        super(Dice, self).__init__()
        self.eps = eps

    def forward(self, inputs, targets, logits=True):
        if logits:
            inputs = torch.argmax(F.softmax(inputs, dim=1), dim=1)
        targets = targets.contiguous()
        targets = one_hot(targets, 2)
        inputs = one_hot(inputs, 2)

        dims = tuple(range(2, targets.ndimension()))
        tps = torch.sum(inputs * targets, dims)
        fps = torch.sum(inputs * (1 - targets), dims)
        fns = torch.sum((1 - inputs) * targets, dims)
        loss = (2 * tps) / (2 * tps + fps + fns + self.eps)
        return loss[:, 1:].mean(dim=1)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def attack_pgd(model, X, y, eps, rank, alpha=1./255., iters=5, restarts=5):
    # eps: magnitude of the attack
    # alpha: step size
    def clamp(X, lower_limit, upper_limit):
        return torch.max(torch.min(X, upper_limit), lower_limit)

    max_loss = torch.zeros(y.shape[0], 1, 1, 1).to(rank)
    max_delta = torch.zeros_like(X).to(rank)
    for _ in range(restarts):
        with torch.enable_grad():
            delta = torch.zeros_like(X).to(rank)
            # Delta for each channel
            for i in range(X.shape[1]):
                delta[:, i, :, :, :].uniform_(-eps[i][0][0][0].item(),
                                              eps[i][0][0][0].item())
            delta.requires_grad = True
            for _ in range(iters):
                output = model(X + delta)
                loss = F.cross_entropy(output, y.long())
                loss.backward()
                grad = delta.grad.detach()
                d = clamp(delta + alpha * torch.sign(grad), -eps, eps)
                d = clamp(d, 0 - X, 1 - X)
                delta.data = d
                delta.grad.zero_()

        # with torch.no_grad():
        final = model(X + delta)
        all_loss = F.cross_entropy(final, y.long(), reduction='none')
        idx = (all_loss >= max_loss).unsqueeze(1).repeat(
            1, X.shape[1], 1, 1, 1)
        max_delta[idx] = delta.detach()[idx]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta
