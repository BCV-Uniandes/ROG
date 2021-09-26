import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class APGDAttack():
    """
    Auto Projected Gradient Descent (Linf)
    """
    def __init__(self, model, dice_thresh, n_iter=20, n_restarts=1, eps=None,
                 seed=0, loss='bce', eot_iter=1, rho=.75, verbose=False,
                 device='cuda'):
        self.model = model
        self.n_iter = n_iter
        self.eps = eps
        self.n_restarts = n_restarts
        self.seed = seed
        self.loss = loss
        self.eot_iter = eot_iter
        self.thr_decr = rho
        self.verbose = verbose
        self.device = device
        # Dice loss
        self.dice_thresh = dice_thresh
        self.dice = Dice(eps=1e-5)

    def check_oscillation(self, x, j, k, y5, k3=0.75):
        t = np.zeros(x.shape[1])
        for counter5 in range(k):
            t += x[j - counter5] > x[j - counter5 - 1]

        return t <= (k * k3 * np.ones(t.shape))

    def check_shape(self, x):
        return x if len(x.shape) > 0 else np.expand_dims(x, 0)

    def dlr_loss(self, x, y):
        x_sorted, ind_sorted = x.sort(dim=1)
        ind = (ind_sorted[:, -1] == y).float()
        # numerator
        corr_logits = torch.gather(x, 1, index=y.unsqueeze(1)).squeeze()
        numerator = corr_logits - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind)
        # denominator
        denominator = x_sorted[:, -1] - x_sorted[:, -3] + 1e-12
        loss = -numerator / denominator
        loss = loss.mean(dim=(1, 2, 3))  # collapse volumetric dims
        return loss

    def attack_single_run(self, x_in, y_in):
        x = x_in.clone() if len(x_in.shape) == 5 else x_in.clone().unsqueeze(0)
        y = y_in.clone() if len(y_in.shape) == 4 else y_in.clone().unsqueeze(0)

        self.n_iter_2 = max(int(0.22 * self.n_iter), 1)
        self.n_iter_min = max(int(0.06 * self.n_iter), 1)
        self.size_decr = max(int(0.03 * self.n_iter), 1)
        if self.verbose:
            print(
                'parameters: ', self.n_iter, self.n_iter_2,
                self.n_iter_min, self.size_decr)

        t = 2 * torch.rand(x.shape).to(self.device).detach() - 1
        x_adv = x.detach() + self.eps * torch.ones(
            [x.shape[0], 1, 1, 1, 1]).to(self.device).detach()\
            * t / (t.reshape([t.shape[0], -1]).abs().max(
                dim=1, keepdim=True)[0].reshape([-1, 1, 1, 1, 1]))
        x_adv = x_adv.clamp(0., 1.)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        loss_steps = torch.zeros([self.n_iter, x.shape[0]])
        loss_best_steps = torch.zeros([self.n_iter + 1, x.shape[0]])

        if self.loss == 'ce':
            xent = nn.CrossEntropyLoss(reduce=False, reduction='none')
            criterion_indiv = lambda x, y: xent(x, y).mean(dim=(1, 2, 3))
        elif self.loss == 'dlr':
            criterion_indiv = self.dlr_loss
        else:
            raise ValueError('unknowkn loss')

        x_adv.requires_grad_()
        grad = torch.zeros_like(x)
        for _ in range(self.eot_iter):
            with torch.enable_grad():
                logits = self.model(x_adv)  # 1 forward pass (eot_iter = 1)
                loss_indiv = criterion_indiv(logits, y)
                loss = loss_indiv.sum()

            # 1 backward pass (eot_iter = 1)
            grad += torch.autograd.grad(loss, [x_adv])[0].detach()

        grad /= float(self.eot_iter)
        grad_best = grad.clone()

        dice_best = self.dice(logits.detach(), y)
        loss_best = loss_indiv.detach().clone()

        step_size = self.eps * torch.ones(
            [x.shape[0], 1, 1, 1, 1]).to(self.device).detach()\
            * torch.Tensor([2.0]).to(self.device).detach().reshape(
                [1, 1, 1, 1, 1])
        x_adv_old = x_adv.clone()
        k = self.n_iter_2 + 0
        u = np.arange(x.shape[0])
        counter3 = 0

        loss_best_last_check = loss_best.clone()
        reduced_last_check = np.zeros(loss_best.shape) == np.zeros(loss_best.shape)

        for i in range(self.n_iter):
            # # # gradient step
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()

                a = 0.75 if i > 0 else 1.0

                x_adv_1 = x_adv + step_size * torch.sign(grad)
                x_adv_1 = torch.clamp(
                    torch.min(
                        torch.max(x_adv_1, x - self.eps), x + self.eps),
                    0.0, 1.0)
                x_adv_1 = torch.clamp(torch.min(
                    torch.max(
                        x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a),
                        x - self.eps), x + self.eps),
                    0.0, 1.0)

                x_adv = x_adv_1 + 0.

            # # # get gradient
            x_adv.requires_grad_()
            grad = torch.zeros_like(x)
            for _ in range(self.eot_iter):
                with torch.enable_grad():
                    logits = self.model(x_adv)  # 1 forward pass (eot_iter = 1)
                    loss_indiv = criterion_indiv(logits, y)
                    loss = loss_indiv.sum()

                # 1 backward pass (eot_iter = 1)
                grad += torch.autograd.grad(loss, [x_adv])[0].detach()

            grad /= float(self.eot_iter)

            new_dice = self.dice(logits.detach(), y)
            index = (dice_best > new_dice).nonzero().squeeze()
            dice_best = torch.min(dice_best, new_dice)
            x_best_adv[index] = x_adv[index] + 0.
            if self.verbose:
                print('iteration: {} - Best loss: {:.6f}'.format(
                    i, loss_best.sum()))

            # # # check step size
            with torch.no_grad():
                y1 = loss_indiv.detach().clone()
                loss_steps[i] = y1.cpu() + 0
                ind = (y1 > loss_best).nonzero().squeeze()
                x_best[ind] = x_adv[ind].clone()
                grad_best[ind] = grad[ind].clone()
                loss_best[ind] = y1[ind] + 0
                loss_best_steps[i + 1] = loss_best + 0

                counter3 += 1

                if counter3 == k:
                    fl_oscillation = self.check_oscillation(
                        loss_steps.detach().cpu().numpy(), i, k,
                        loss_best.detach().cpu().numpy(), k3=self.thr_decr)
                    fl_reduce_no_impr = (~reduced_last_check) * (loss_best_last_check.cpu().numpy() >= loss_best.cpu().numpy())
                    fl_oscillation = ~(~fl_oscillation * ~fl_reduce_no_impr)
                    reduced_last_check = np.copy(fl_oscillation)
                    loss_best_last_check = loss_best.clone()

                    if np.sum(fl_oscillation) > 0:
                        step_size[u[fl_oscillation]] /= 2.0

                        fl_oscillation = np.where(fl_oscillation)

                        x_adv[fl_oscillation] = x_best[fl_oscillation].clone()
                        grad[fl_oscillation] = grad_best[fl_oscillation].clone()

                    counter3 = 0
                    k = np.maximum(k - self.size_decr, self.n_iter_min)

        return x_best, dice_best, loss_best, x_best_adv

    def perturb(self, x_in, y_in):
        x = x_in.clone() if len(x_in.shape) == 5 else x_in.clone().unsqueeze(0)
        y = y_in.clone() if len(y_in.shape) == 4 else y_in.clone().unsqueeze(0)
        adv = x.clone()
        results = self.dice(self.model(x), y)
        if self.verbose:
            print('---- running {}-attack with epsilon {:.4f} ----'.format(
                self.norm, self.eps))
            print('initial accuracy: {:.2%}'.format(results.float().mean()))
        startt = time.time()

        torch.random.manual_seed(self.seed)
        torch.cuda.random.manual_seed(self.seed)

        for counter in range(self.n_restarts):
            ind_to_fool = (results > self.dice_thresh).nonzero().squeeze()
            if len(ind_to_fool.shape) == 0:
                ind_to_fool = ind_to_fool.unsqueeze(0)

            if ind_to_fool.numel() != 0:
                x_to_fool = x[ind_to_fool].clone()
                y_to_fool = y[ind_to_fool].clone()
                _, results_curr, _, adv_curr = self.attack_single_run(
                    x_to_fool, y_to_fool)

                ind_curr = (results[ind_to_fool] > results_curr).nonzero().squeeze()
                results[ind_to_fool[ind_curr]] = results_curr[ind_curr]
                adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
                if self.verbose:
                    print('restart {} - robust accuracy: {:.2%} - cum. time: {:.1f} s'.format(
                        counter, results.float().mean(), time.time() - startt))
        return results, adv


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
