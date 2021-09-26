# Copyright (c) 2019-present, Francesco Croce
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class FABAttack():
    """
    Targeted Fast Adaptive Boundary Attack (Linf)
    https://arxiv.org/abs/1907.02044

    :param predict:       forward pass function
    :param n_restarts:    number of random restarts
    :param n_iter:        number of iterations
    :param eps:           epsilon for the random restarts
    :param alpha_max:     alpha_max
    :param eta:           overshooting
    :param beta:          backward step
    """

    def __init__(self, predict, dice_thresh, n_restarts=1, n_iter=10, eps=None,
                 alpha_max=0.1, eta=1.05, beta=0.9, loss_fn=None,
                 verbose=False, seed=0, device=None, n_target_classes=2):
        """ FAB-attack implementation in pytorch """

        self.predict = predict
        self.n_restarts = n_restarts
        self.n_iter = n_iter
        self.eps = eps
        self.alpha_max = alpha_max
        self.eta = eta
        self.beta = beta
        self.verbose = verbose
        self.seed = seed
        self.target_class = None
        self.device = device
        self.n_target_classes = n_target_classes
        # our stuff
        self.dice = Dice(eps=1e-5)  # alpha == 1 --> Dice score
        self.dice_thresh = dice_thresh

    def _get_predicted_label(self, x):
        with torch.no_grad():
            outputs = self.predict(x)
        _, y = torch.max(outputs, dim=1)
        return outputs

    def check_shape(self, x):
        return x if len(x.shape) > 0 else x.unsqueeze(0)

    def get_diff_logits_grads_batch(self, imgs, la, la_target):
        im = imgs.clone().requires_grad_()
        with torch.enable_grad():
            y = self.predict(im)
            term1 = torch.gather(y, 1, index=la.unsqueeze(1))
            term2 = torch.gather(y, 1, index=la_target.unsqueeze(1))
            diffy = -(term1 - term2).mean(dim=(1, 2, 3, 4))
            sumdiffy = diffy.sum()

        if im.grad is not None:
            im.grad.zero_()

        sumdiffy.backward()
        graddiffy = im.grad.data
        df = diffy.detach().unsqueeze(1)
        dg = graddiffy.unsqueeze(1)

        return df, dg

    def projection_linf(self, points_to_project, w_hyperplane, b_hyperplane):
        t = points_to_project.clone()
        w = w_hyperplane.clone()
        b = b_hyperplane.clone()

        ind2 = ((w * t).sum(1) - b < 0).nonzero().squeeze()
        ind2 = self.check_shape(ind2)
        w[ind2] *= -1
        b[ind2] *= -1

        c5 = (w < 0).float()
        a = torch.ones(t.shape).to(self.device)
        d = (a * c5 - t) * (w != 0).float()
        a -= a * (1 - c5)

        p = torch.ones(t.shape).to(self.device) * c5 - t * (2 * c5 - 1)
        indp = torch.argsort(p, dim=1)

        b = b - (w * t).sum(1)
        b0 = (w * d).sum(1)
        b1 = b0.clone()

        counter = 0
        indp2 = indp.unsqueeze(-1).flip(dims=(1, 2)).squeeze()
        u = torch.arange(0, w.shape[0])
        ws = w[u.unsqueeze(1), indp2]
        bs2 = - ws * d[u.unsqueeze(1), indp2]

        s = torch.cumsum(ws.abs(), dim=1)
        sb = torch.cumsum(bs2, dim=1) + b0.unsqueeze(1)

        c = b - b1 > 0
        b2 = sb[u, -1] - s[u, -1] * p[u, indp[u, 0]]
        c_l = (b - b2 > 0).nonzero().squeeze()
        c2 = ((b - b1 > 0) * (b - b2 <= 0)).nonzero().squeeze()
        c_l = self.check_shape(c_l)
        c2 = self.check_shape(c2)

        lb = torch.zeros(c2.shape[0])
        ub = torch.ones(c2.shape[0]) * (w.shape[1] - 1)
        nitermax = torch.ceil(torch.log2(torch.tensor(w.shape[1]).float()))
        counter2 = torch.zeros(lb.shape).long()

        while counter < nitermax:
            counter4 = torch.floor((lb + ub) / 2)
            counter2 = counter4.long()
            indcurr = indp[c2, -counter2 - 1]
            b2 = sb[c2, counter2] - s[c2, counter2] * p[c2, indcurr]
            c = b[c2] - b2 > 0
            ind3 = c.nonzero().squeeze()
            ind32 = (~c).nonzero().squeeze()
            ind3 = self.check_shape(ind3)
            ind32 = self.check_shape(ind32)
            lb[ind3] = counter4[ind3]
            ub[ind32] = counter4[ind32]
            counter += 1

        lb = lb.long()
        counter2 = 0

        if c_l.nelement != 0:
            lmbd_opt = (torch.max((b[c_l] - sb[c_l, -1]) / (-s[c_l, -1]),
                                  torch.zeros(sb[c_l, -1].shape)
                                  .to(self.device))).unsqueeze(-1)
            d[c_l] = (2 * a[c_l] - 1) * lmbd_opt

        lmbd_opt = (torch.max((b[c2] - sb[c2, lb]) / (-s[c2, lb]),
                              torch.zeros(sb[c2, lb].shape)
                              .to(self.device))).unsqueeze(-1)
        d[c2] = torch.min(lmbd_opt, d[c2]) * c5[c2]\
            + torch.max(-lmbd_opt, d[c2]) * (1 - c5[c2])

        return d * (w != 0).float()

    def attack_single_run(self, x, y=None, use_rand_start=False):
        """
        :param x:    clean images
        :param y:    clean labels, if None we use the predicted labels
        """

        if self.device is None:
            self.device = x.device
        self.orig_dim = list(x.shape[1:])
        self.ndims = len(self.orig_dim)

        x = x.detach().clone().float().to(self.device)

        y_pred = self._get_predicted_label(x)
        if y is None:
            y = y_pred.detach().clone().long().to(self.device)
        else:
            y = y.detach().clone().long().to(self.device)
        pred = self.dice(y_pred.detach(), y) > self.dice_thresh
        corr_classified = pred.float().sum()
        if self.verbose:
            print('Clean accuracy: {:.2%}'.format(pred.float().mean()))
        if pred.sum() == 0:
            return x
        pred = self.check_shape(pred.nonzero().squeeze())

        output = self.predict(x)
        la_target = output.sort(dim=1)[1][:, -self.target_class]

        startt = time.time()
        # runs the attack only on correctly classified points
        im2 = x[pred].detach().clone()
        la2 = y[pred].detach().clone()
        la_target2 = la_target[pred].detach().clone()
        if len(im2.shape) == self.ndims:
            im2 = im2.unsqueeze(0)
        bs = im2.shape[0]
        u1 = torch.arange(bs)
        adv = im2.clone()
        adv_c = x.clone()
        res2 = 1e10 * torch.ones([bs]).to(self.device)
        res_c = torch.zeros([x.shape[0]]).to(self.device)
        x1 = im2.clone()
        x0 = im2.clone().reshape([bs, -1])
        counter_restarts = 0

        while counter_restarts < 1:
            if use_rand_start:
                t = 2 * torch.rand(x1.shape).to(self.device) - 1
                x1 = im2 + (torch.min(res2,
                                      self.eps * torch.ones(res2.shape)
                                      .to(self.device)
                                      ).reshape([-1, *[1]*self.ndims])
                            ) * t / (t.reshape([t.shape[0], -1]).abs()
                                     .max(dim=1, keepdim=True)[0]
                                     .reshape([-1, *[1]*self.ndims])) * .5

                x1 = x1.clamp(0.0, 1.0)

            counter_iter = 0
            while counter_iter < self.n_iter:
                with torch.no_grad():
                    df, dg = self.get_diff_logits_grads_batch(
                        x1, la2, la_target2)
                    dist1 = df.abs() / (1e-12 +
                                        dg.abs()
                                        .view(dg.shape[0], dg.shape[1], -1)
                                        .sum(dim=-1))
                    ind = dist1.min(dim=1)[1]
                    dg2 = dg[u1, ind]
                    b = (- df[u1, ind] +
                         (dg2 * x1).view(x1.shape[0], -1).sum(dim=-1))
                    w = dg2.reshape([bs, -1])

                    d3 = self.projection_linf(
                        torch.cat((x1.reshape([bs, -1]), x0), 0),
                        torch.cat((w, w), 0),
                        torch.cat((b, b), 0))
                    d1 = torch.reshape(d3[:bs], x1.shape)
                    d2 = torch.reshape(d3[-bs:], x1.shape)
                    a0 = d3.abs().max(dim=1, keepdim=True)[0]\
                        .view(-1, *[1]*self.ndims)
                    a0 = torch.max(a0, 1e-8 * torch.ones(
                        a0.shape).to(self.device))
                    a1 = a0[:bs]
                    a2 = a0[-bs:]
                    alpha = torch.min(torch.max(a1 / (a1 + a2),
                                                torch.zeros(a1.shape)
                                                .to(self.device)),
                                      self.alpha_max * torch.ones(a1.shape)
                                      .to(self.device))

                    x1 = ((x1 + self.eta * d1) * (1 - alpha) +
                          (im2 + d2 * self.eta) * alpha).clamp(0.0, 1.0)

                    is_adv = self.dice(
                        self._get_predicted_label(x1).detach(),
                        la2) < self.dice_thresh
                    if is_adv.sum() > 0:
                        ind_adv = is_adv.nonzero().squeeze()
                        ind_adv = self.check_shape(ind_adv)
                        t = (x1[ind_adv] - im2[ind_adv]).reshape(
                            [ind_adv.shape[0], -1]).abs().max(dim=1)[0]

                        adv[ind_adv] = x1[ind_adv] * (t < res2[ind_adv]).\
                            float().reshape([-1, *[1]*self.ndims]) + adv[ind_adv]\
                            * (t >= res2[ind_adv]).float().reshape(
                            [-1, *[1]*self.ndims])
                        res2[ind_adv] = t * (t < res2[ind_adv]).float()\
                            + res2[ind_adv] * (t >= res2[ind_adv]).float()
                        x1[ind_adv] = im2[ind_adv] + (
                            x1[ind_adv] - im2[ind_adv]) * self.beta
                    counter_iter += 1

            counter_restarts += 1

        ind_succ = res2 < 1e10
        if self.verbose:
            print('success rate: {:.0f}/{:.0f}'
                  .format(ind_succ.float().sum(), corr_classified) +
                  ' (on correctly classified points) in {:.1f} s'
                  .format(time.time() - startt))

        res_c[pred] = res2 * ind_succ.float() + 1e10 * (1 - ind_succ.float())
        ind_succ = self.check_shape(ind_succ.nonzero().squeeze())
        adv_c[pred[ind_succ]] = adv[ind_succ].clone()

        return adv_c

    def perturb(self, x, y):
        adv = x.clone()
        with torch.no_grad():
            acc = self.dice(self.predict(x).detach(), y) > self.dice_thresh

            startt = time.time()

            torch.random.manual_seed(self.seed)
            torch.cuda.random.manual_seed(self.seed)

            for target_class in range(2, self.n_target_classes + 2):
                self.target_class = target_class
                for counter in range(self.n_restarts):
                    ind_to_fool = acc.nonzero().squeeze()
                    if len(ind_to_fool.shape) == 0:
                        ind_to_fool = ind_to_fool.unsqueeze(0)
                    if ind_to_fool.numel() != 0:
                        x_to_fool, y_to_fool = x[ind_to_fool].clone(), y[ind_to_fool].clone()
                        adv_curr = self.attack_single_run(
                            x_to_fool, y_to_fool, use_rand_start=(counter > 0))

                        acc_curr = self.dice(
                            self.predict(adv_curr).detach(),
                            y_to_fool) > self.dice_thresh
                        res = (x_to_fool - adv_curr).abs().view(
                            x_to_fool.shape[0], -1).max(1)[0]
                        acc_curr = torch.max(acc_curr, res > self.eps)

                        ind_curr = (acc_curr == 0).nonzero().squeeze()
                        acc[ind_to_fool[ind_curr]] = 0
                        adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()

                        if self.verbose:
                            print('restart {} - target_class {} - robust accuracy: {:.2%} at eps = {:.5f} - cum. time: {:.1f} s'.format(
                                counter, self.target_class, acc.float().mean(),
                                self.eps, time.time() - startt))
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
    """
        Calculates the Tversky loss of the Foreground categories.
        if alpha == 1 --> Dice score
        alpha: controls the penalty for false positives.
        beta: controls the penalty for false negatives.
    """
    def __init__(self, eps=1):
        super(Dice, self).__init__()
        self.alpha = 1
        self.beta = 2 - self.alpha
        self.eps = eps

    def forward(self, inputs, targets):
        targets = targets.contiguous()
        targets = one_hot(targets, inputs.shape[1])
        inputs = torch.argmax(F.softmax(inputs, dim=1), dim=1)
        inputs = one_hot(inputs, targets.shape[1])

        dims = tuple(range(2, targets.ndimension()))
        tps = torch.sum(inputs * targets, dims)
        fps = torch.sum(inputs * (1 - targets), dims) * self.alpha
        fns = torch.sum((1 - inputs) * targets, dims) * self.beta
        loss = (2 * tps) / (2 * tps + fps + fns + self.eps)
        # loss = torch.mean(loss, dim=0)
        return loss[:, 1:].mean(dim=1)  # loss[1:].mean()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
