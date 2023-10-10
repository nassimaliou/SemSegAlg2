# Copyright (c) 2020-present
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import time
import math
import torch.nn.functional as F

import numpy as np
import copy
import sys
from utils import Logger
import os


class RSAttack():   
    def __init__(
            self,
            predict,
            norm='L0',
            n_queries=10,
            eps=None,
            p_init=.5,
            n_restarts=1,
            seed=0,
            verbose=False,
            targeted=False,
            loss='margin',
            resc_schedule=True,
            device=None,
            log_path=None,
            resample_loc=None,
            data_loader=None,
            update_loc_period=None):
        
        self.predict = predict
        self.norm = norm
        self.n_queries = n_queries
        self.eps = eps
        self.p_init = p_init
        self.n_restarts = n_restarts
        self.seed = seed
        self.verbose = verbose
        self.targeted = targeted
        self.loss = loss
        self.rescale_schedule = resc_schedule
        self.device = device
        self.logger = Logger(log_path)
        #self.resample_loc = n_queries // 10 if resample_loc is None else resample_loc
        self.data_loader = data_loader
        self.update_loc_period = update_loc_period if not update_loc_period is None else 4 if not targeted else 10
        
    
    def margin_and_loss(self , x, y):

        max_class_probs = torch.argmax(x, dim =1).float ()

        xent = torch.sqrt(( max_class_probs - y.float ()))

        correct_class_probs = x[x == y.float ()]. clone ()

        max_other_class_probs = x.max(dim=-1)[0]
        if self.loss == 'ce':
            return correct_class_probs - max_other_class_probs , -1. * xent

        elif self.loss == 'margin ':

            return correct_class_probs - max_other_class_probs ,correct_class_probs - max_other_class_probs

    def init_hyperparam(self, x):
        assert self.norm in ['L0']
        assert not self.eps is None
        assert self.loss in ['margin']

        if self.device is None:
            self.device = x.device
        self.orig_dim = list(x.shape[1:])
        self.ndims = len(self.orig_dim)
        if self.seed is None:
            self.seed = time.time()
        
    def random_target_classes(self, y_pred, n_classes):
        y = torch.zeros_like(y_pred)
        for counter in range(y_pred.shape[0]):
            l = list(range(n_classes))
            l.remove(y_pred[counter])
            t = self.random_int(0, len(l))
            y[counter] = l[t]

        return y.long().to(self.device)

    def check_shape(self, x):
        return x if len(x.shape) == (self.ndims + 1) else x.unsqueeze(0)

    def random_choice(self, shape):
        t = 2 * torch.rand(shape).to(self.device) - 1
        return torch.sign(t)

    def random_int(self, low=0, high=1, shape=[1]):
        t = low + (high - low) * torch.rand(shape).to(self.device)
        return t.long()

    def normalize(self, x):
        if self.norm == 'Linf':
            t = x.abs().view(x.shape[0], -1).max(1)[0]
            return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)

        elif self.norm == 'L2':
            t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
            return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)

    def lp_norm(self, x):
        if self.norm == 'L2':
            t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
            return t.view(-1, *([1] * self.ndims))

    def p_selection(self, it):
        
        if 'L0' in self.norm:
            if 0 < it <= 50:
                p = self.p_init / it
            elif 50 < it <= 200:
                p = self.p_init / it
            elif 200 < it <= 500:
                p = self.p_init / it
            elif 500 < it <= 1000:
                p = self.p_init / it
            elif 1000 < it <= 2000:
                p = self.p_init / it
            elif 2000 < it <= 4000:
                p = self.p_init / it
            elif 4000 < it <= 6000:
                p = self.p_init / it
            elif 6000 < it <= 8000:
                p = self.p_init / it
            elif 8000 < it:
                p = self.p_init / it
            else:
                p = self.p_init
        
            if self.constant_schedule:
                p = self.p_init / 2
        
        return p

    def sh_selection(self, it):
        """ schedule to decrease the parameter p """

        t = max((float(self.n_queries - it) / self.n_queries - .0) ** 1., 0) * .75

        return t
    
    def get_init_patch(self, c, s, n_iter=1000):
        if self.init_patches == 'stripes':
            patch_univ = torch.zeros([1, c, s, s]).to(self.device) + self.random_choice(
                [1, c, 1, s]).clamp(0., 1.)
        elif self.init_patches == 'uniform':
            patch_univ = torch.zeros([1, c, s, s]).to(self.device) + self.random_choice(
                [1, c, 1, 1]).clamp(0., 1.)
        elif self.init_patches == 'random':
            patch_univ = self.random_choice([1, c, s, s]).clamp(0., 1.)
        elif self.init_patches == 'random_squares':
            patch_univ = torch.zeros([1, c, s, s]).to(self.device)
            for _ in range(n_iter):
                size_init = torch.randint(low=1, high=math.ceil(s ** .5), size=[1]).item()
                loc_init = torch.randint(s - size_init + 1, size=[2])
                patch_univ[0, :, loc_init[0]:loc_init[0] + size_init, loc_init[1]:loc_init[1] + size_init] = 0.
                patch_univ[0, :, loc_init[0]:loc_init[0] + size_init, loc_init[1]:loc_init[1] + size_init
                    ] += self.random_choice([c, 1, 1]).clamp(0., 1.)
        elif self.init_patches == 'sh':
            patch_univ = torch.ones([1, c, s, s]).to(self.device)
        
        return patch_univ.clamp(0., 1.)
    
    def attack(self , x, y):

        adv = x.clone ()
        c, h, w = x.shape
        n_pixels = c * h * w
        theta = self.theta

        F, E = torch.zeros(theta).long(), torch.zeros(n_pixels -theta).long()
        margin_min , loss_min = self.margin_and_loss(x, y)

        n_queries = torch.ones(x.shape [0]).to(self.device)

        for it in range(1, self.n_queries):
            idx_to_fool = (margin_min > 0.).nonzero ().squeeze ()
            x_curr = self.check_shape(x[idx_to_fool ])
            x_best_curr = self.check_shape(x[idx_to_fool ])

            y_curr = y[idx_to_fool]
            margin_min_curr = margin_min[idx_to_fool]
            loss_min_curr = loss_min[idx_to_fool]

            b_curr , be_curr = F[idx_to_fool], E[idx_to_fool]

            x_new = x_best_curr.clone ()

            theta_it = max(int(self.p_selection(it) * theta), 1)

            ind_p = torch.randperm(theta)[: theta_it]
            ind_np = torch.randperm(n_pixels - theta)[: theta_it]

            p_set = b_curr[ind_p]
            np_set = be_curr[ind_np]
            x_new[:, p_set // w, p_set % w] = x_curr[:, p_set // w, p_set % w].clone ()

            margin , loss = self.margin_and_loss(x_new , y_curr)

            n_queries[idx_to_fool] += 1

            # update best solution
            idx_improved = (loss < loss_min_curr).float ()
            idx_to_update = (idx_improved > 0.).nonzero ().squeeze ()
            loss_min[idx_to_fool[idx_to_update ]] = loss[idx_to_update]

            t = b_curr[idx_improved ]. clone ()
            te = be_curr[idx_improved ]. clone ()

            F[idx_to_fool[idx_improved ]] = t.clone ()
            E[idx_to_fool[idx_improved ]] = te.clone ()
        return margin , x

    def perturb(self, x, y=None):

        self.init_hyperparam(x)

        adv = x.clone()
        qr = torch.zeros([x.shape[0]]).to(self.device)
        if y is None:
            if not self.targeted:
                with torch.no_grad():
                    output = self.predict(x)
                    y_pred = output.max(1)[1]
                    y = y_pred.detach().clone().long().to(self.device)
            else:
                with torch.no_grad():
                    output = self.predict(x)
                    n_classes = output.shape[-1]
                    y_pred = output.max(1)[1]
                    y = self.random_target_classes(y_pred, n_classes)
        else:
            y = y.detach().clone().long().to(self.device)

        if not self.targeted:
            acc = self.predict(x).max(1)[1] == y
        else:
            acc = self.predict(x).max(1)[1] != y

        startt = time.time()

        torch.random.manual_seed(self.seed)
        torch.cuda.random.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        for counter in range(self.n_restarts):
            ind_to_fool = acc.nonzero().squeeze()
            if len(ind_to_fool.shape) == 0:
                ind_to_fool = ind_to_fool.unsqueeze(0)
            if ind_to_fool.numel() != 0:
                x_to_fool = x[ind_to_fool].clone()
                y_to_fool = y[ind_to_fool].clone()

                qr_curr, adv_curr = self.attack_single_run(x_to_fool, y_to_fool)

                output_curr = self.predict(adv_curr)
                if not self.targeted:
                    acc_curr = output_curr.max(1)[1] == y_to_fool
                else:
                    acc_curr = output_curr.max(1)[1] != y_to_fool
                ind_curr = (acc_curr == 0).nonzero().squeeze()

                acc[ind_to_fool[ind_curr]] = 0
                adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
                qr[ind_to_fool[ind_curr]] = qr_curr[ind_curr].clone()

        return qr, adv

