######################################################################################################
# Adapted from https://github.com/Ping-C/certifiedpatchdefense/blob/master/attacks/patch_attacker.py
######################################################################################################

import torch
import numpy as np



class PatchAttacker:
    def __init__(self, model, mean, std, image_size=244,epsilon=1,steps=500,step_size=0.05,patch_size=31,random_start=True):

        mean,std = torch.tensor(mean),torch.tensor(std)
        self.epsilon = epsilon / std
        self.epsilon_cuda=self.epsilon[None, :, None, None].cuda()
        self.steps = steps
        self.step_size = step_size / std
        self.step_size=self.step_size[None, :, None, None].cuda()
        self.model = model.cuda()
        self.mean = mean
        self.std = std
        self.random_start = random_start
        self.image_size = image_size
        self.lb = (-mean / std)
        self.lb=self.lb[None, :, None, None].cuda()
        self.ub = (1 - mean) / std
        self.ub=self.ub[None, :, None, None].cuda()
        self.patch_w = patch_size
        self.patch_l = patch_size

        self.criterion = torch.nn.CrossEntropyLoss()

    def perturb(self, inputs, labels, loc=None,random_count=1):
        worst_x = None
        worst_loss = None
        
        for _ in range(random_count):
            # generate random patch center for each image
            #print(inputs.shape)
            idx = torch.arange(inputs.shape[0])[:, None]
            zero_idx = torch.zeros((inputs.shape[0],1), dtype=torch.long)
            if loc is not None: #specified locations
                w_idx = torch.ones([inputs.shape[0],1],dtype=torch.int64)*loc[0]
                l_idx = torch.ones([inputs.shape[0],1],dtype=torch.int64)*loc[1]
            else: #random locations
                w_idx = torch.randint(0 , inputs.shape[2]-self.patch_w , (inputs.shape[0],1))
                l_idx = torch.randint(0 , inputs.shape[3]-self.patch_l , (inputs.shape[0],1))

            idx = torch.cat([idx,zero_idx, w_idx, l_idx], dim=1)
            idx_list = [idx]
            for w in range(self.patch_w):
                for l in range(self.patch_l):
                    idx_list.append(idx + torch.tensor([0,0,w,l]))
            idx_list = torch.cat(idx_list, dim =0)

            # create mask
            mask = torch.zeros([inputs.shape[0], 1, inputs.shape[2], inputs.shape[3]],
                               dtype=torch.bool).cuda()
            mask[idx_list[:,0],idx_list[:,1],idx_list[:,2],idx_list[:,3]] = True
            #print('mask has {} values'.format(True in mask))
            # There are true values in mask
            maskint = mask.long()
            maskint = torch.where(maskint<1, 0.0, 255.0)
            '''
            maskint = torch.clone(mask)
            maskint[maskint[True]] = 255.0
            maskint[maskint[False]] = 0.0
            '''
            #print('maskint is {}'.format(maskint))

            # Terence edit
            #print(mask.shape)

            if self.random_start:
                init_delta = np.random.uniform(-self.epsilon, self.epsilon,
                                               [inputs.shape[0]*inputs.shape[2]*inputs.shape[3], inputs.shape[1]])
                init_delta = init_delta.reshape(inputs.shape[0],inputs.shape[2],inputs.shape[3], inputs.shape[1])
                init_delta = init_delta.swapaxes(1,3).swapaxes(2,3)
                x = inputs + torch.where(mask, torch.Tensor(init_delta).to('cuda'), torch.tensor(0.).cuda())

                x = torch.min(torch.max(x, self.lb), self.ub).detach()  # ensure valid pixel range
            else:
                x = inputs.data.detach().clone()

            x_init = inputs.data.detach().clone()

            for step in range(self.steps+1): # default 500 steps
                x.requires_grad_()
                output = self.model(torch.where(mask, x, x_init)) # noisy x only at mask, the other parts of image use x_init (clean_
                loss_ind = torch.nn.functional.cross_entropy(input=output, target=labels,reduction='none')
                loss = loss_ind.sum()
                grads = torch.autograd.grad(loss, x,retain_graph=False)[0]

                if step % 10 ==0:
                    if worst_loss is None:
                        worst_loss = loss_ind.detach().clone() # lowest current loss
                        worst_x = x.detach().clone() # most perturbed x
                    else:
                        tmp_loss = loss_ind.detach().clone()
                        tmp_x = x.detach().clone()
                        filter_tmp=worst_loss.ge(tmp_loss).detach().clone()
                        worst_x = torch.where(filter_tmp.reshape([inputs.shape[0],1,1,1]), worst_x, tmp_x).detach().clone()
                        worst_loss = torch.where(filter_tmp, worst_loss, tmp_loss).detach().clone()
                        #print(worst_loss)
                        #del tmp_loss
                        #del tmp_x
                        #del filter_tmp
                signed_grad_x = torch.sign(grads).detach() # take sign of elements (FSGM)
                delta = signed_grad_x * self.step_size # perturbation only
                x = delta + x
                #del loss
                #del loss_ind
                #del grads
                # Project back into constraints ball and correct range
                x = torch.max(torch.min(x, x_init + self.epsilon_cuda), x_init - self.epsilon_cuda)#.detach()
                x = torch.min(torch.max(x, self.lb), self.ub).detach().clone()
        # Terence edit
        return worst_x.detach().clone(), torch.cat([w_idx, l_idx], dim=1).detach().clone(), x.detach().clone(), maskint.detach().clone()

