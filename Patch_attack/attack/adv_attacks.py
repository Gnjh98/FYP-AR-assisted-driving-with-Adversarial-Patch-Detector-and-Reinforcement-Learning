import random
from skimage.draw import random_shapes
import torch
import torch.nn.functional as F
from linf_sgd import Linf_SGD
from torch import nn
from torch.autograd import Variable

def add_shape(patchW, patchH, patchMin, patchMax):
    """ return a Tensor with a random shape
        1 for the shape
        0 for the background
        The size is controle by args.patch_size
    """
    image, _ = random_shapes((patchW,patchH), min_shapes=6, max_shapes=10,
                             intensity_range=(0, 50),  min_size=patchMin,
                             max_size=patchMax, allow_overlap=True, num_channels=1)
    image = torch.round(1 - torch.FloatTensor(image)/255.)
    return image.squeeze().to('cuda')

# CW attack
'''
def cw(x_in, y_true, net, mask, steps, eps, n_class):
    if eps == 0:
        return x_in
    training = net.training
    if training:
        net.eval()
    index = y_true.cpu().view(-1, 1)
    label_onehot = torch.FloatTensor(x_in.size(0), n_class).zero_().scatter_(1, index, 1).cuda()
    x_adv = x_in.clone().zero_().requires_grad_()
    optimizer = torch.optim.Adam([x_adv], lr=1.0e-2)
    zero = torch.FloatTensor([0]).cuda()
    print('===============')
        for _ in range(steps):
            a = 1/2*(nn.Tanh()(w) + 1)
    
            loss1 = nn.MSELoss(reduction='sum')(a, images)
            loss2 = torch.sum(c*f(a))
    
            cost = loss1 + loss2
    
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
    
            # Early Stop when loss does not converge.
            if step % (max_iter//10) == 0 :
                if cost > prev :
                    print('Attack Stopped due to CONVERGENCE....')
                    return a
                prev = cost
    
            print('- Learning Progress : %2.2f %%        ' %((step+1)/max_iter*100), end='\r')
    
        attack_images = 1/2*(nn.Tanh()(w) + 1)
    return x_adv
'''
#cw(images, target, net, mask, steps, args.epsilon, nclass)

def cw(images, labels, net, masks, max_iter, c=1e-4, kappa=0, targeted=False, learning_rate=0.01) :
    images = images.to('cuda')
    labels = labels.to('cuda')
    # Define f-function
    def f(x) :
        #outputs = model(x)
        outputs = net(torch.where(masks>0.1,x,images))
        one_hot_labels = torch.eye(len(outputs[0]))[labels].to('cuda')
        i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_labels.byte())
        # If targeted, optimize for making the other class most likely
        if targeted :
            return torch.clamp(i-j, min=-kappa)
        # If untargeted, optimize for making the other class most likely
        else :
            return torch.clamp(j-i, min=-kappa)
    # w = torch.where(masks>0.1, torch.zeros_like(images, requires_grad=True).to('cuda'), images)
    # w = torch.where(masks>0.1, torch.atanh(images*2-1), images)
    w = torch.zeros_like(images, requires_grad=True).to('cuda')
    # w.requires_grad = True
    optimizer = torch.optim.Adam([w], lr=learning_rate)
    prev = 1e10
    for step in range(max_iter):
        a = 1/2*(nn.Tanh()(w) + 1)
        loss1 = nn.MSELoss(reduction='sum')(a, images)
        loss2 = torch.sum(c*f(a))
        cost = loss1 + loss2
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        # Early Stop when loss does not converge.
        if step % (max_iter//10) == 0 :
            if cost > prev :
                print('Attack Stopped due to CONVERGENCE....')
                return a
            prev = cost
        print('- Learning Progress : %2.2f %%        ' %((step+1)/max_iter*100), end='\r')
    attack_images = torch.where(masks>0.1, 1/2*(nn.Tanh()(w) + 1), images)
    return attack_images



# performs L2-constraint PGD attack w/o noise
# @epsilon: radius of L2-norm ball
def L2_PGD(x_in, y_true, net, mask, steps, eps):

    if eps == 0:
        return x_in
    training = net.training
    if training:
        net.eval()
    x_adv = x_in.clone()
    x_adv += 0.2* torch.randn_like(x_adv) * mask
    x_adv.requires_grad = True
    #x_adv = x_in.clone().requires_grad_()
    optimizer = torch.optim.Adam([x_adv], lr=0.01)
    eps = torch.tensor(eps).view(1,1,1,1).cuda()
    #print('====================')
    for _ in range(steps):
        optimizer.zero_grad()
        net.zero_grad()
        # out,_ = net(torch.where(mask>0.1, x_adv, x_in))
        out = net(torch.where(mask>0.1, x_adv, x_in))
        loss = -F.cross_entropy(out, y_true)
        loss.backward()
        #print(loss.item())
        optimizer.step()
        diff = torch.where(mask>0.1, x_adv - x_in, torch.tensor(0.).to('cuda'))
        norm = torch.sqrt(torch.sum(diff * diff, (1, 2, 3)))
        norm = norm.view(norm.size(0), 1, 1, 1)
        norm_out = torch.min(norm, eps)
        diff = diff / norm * norm_out
        x_adv.detach().copy_((diff + x_in).clamp_(0, 1))
    net.zero_grad()
    # reset to the original state
    if training :
        net.train()
    return x_adv

# performs Linf-constraint PGD attack w/o noise
# @epsilon: radius of Linf-norm ball

def GooglePatch(x_in, target, net, mask, mode, steps, nclass):

    bsize, channel, width, height = x_in.size()
    x_in += 0.1 * torch.randn_like(x_in) * mask     # add some noise for more diverse attack
    x_adv = x_in.clone()
    #patch = 0.2 * torch.randn_like(x_adv) * mask
    x_adv = torch.where(mask>0.1, x_adv, x_in)
    #x_adv += 0.2 * torch.randn_like(x_adv) * mask
    # x_adv += 0.2 * torch.randn_like(x_adv) * mask
    #x_adv.requires_grad = True
    x_adv = Variable(x_adv.data, requires_grad=True)
    training = net.training
    if training:
        net.eval()
    optimizer = torch.optim.SGD([x_adv], lr=0.1, momentum=0.9)
    for _ in range(steps):
        x_adv = Variable(x_adv.data, requires_grad=True)
        optimizer.zero_grad()
        net.zero_grad()
        net_pred = net(x_adv)
        if mode == "max":
            fake_target = torch.randint(nclass - 1, size=(bsize, 1)).expand_as(target).to('cuda')
            loss = F.cross_entropy(net_pred, fake_target.reshape(-1))
        elif mode == "min":
            loss = F.cross_entropy(net_pred, target.long().reshape(-1))
        loss.backward()
        optimizer.step()
        # data_grad = x_adv.grad
        #print(x_adv.grad.type)
        data_grad = x_adv.grad.data.clone()
        #print('grad obtained')
        x_adv.grad.data.zero_()
        #patch = x_adv * mask
        with torch.no_grad():
            x_adv = 10*data_grad + x_adv
            x_adv = torch.mul((1-mask),x_in) + torch.mul(mask,x_adv)
    net.zero_grad()
    # reset to the original state
    if training:
        net.train()
    return x_adv

    '''
        data_grad = x_adv.grad.data
        #sign_data_grad = data_grad.sign()
        sign_data_grad = x_adv.grad.sign()
        sign_data_grad *= mask # ensure the gradient is localized to the mask only and nowhere else in image
        if mode == "max":
            perturbed = -0.5 * sign_data_grad
        elif mode == "min":
            perturbed = 0.5 * sign_data_grad
        # x_adv.data = torch.clamp(x_adv + perturbed, -3, 3)
        x_adv.data = torch.clamp(x_adv + perturbed, 0, 1)
    net.zero_grad()
    # reset to the original state

    if training:
        net.train()

    return x_adv.detach()
    '''

def NA(x_in):
    # x_in += 0.03 * torch.randn_like(x_in) * mask     # add some noise for more diverse attack
    x_adv = x_in.clone()
    mask = torch.zeros_like(x_adv)
    return x_adv, mask

def Linf_PGD(x_in, y_true, net, mask, steps, eps):
    # x_in += 0.03 * torch.randn_like(x_in) * mask     # add some noise for more diverse attack
    x_adv = x_in.clone()
    x_adv += 0.15 * torch.randn_like(x_adv) * mask
    x_adv.requires_grad = True
    if eps == 0:
        return x_in
    training = net.training
    if training:
        net.eval()
    # optimizer = torch.optim.SGD([x_adv], lr=0.1, momentum=0.9)
    optimizer = Linf_SGD([x_adv], lr=0.007)
    for _ in range(steps):
        optimizer.zero_grad()
        net.zero_grad()
        out = net(torch.where(mask>0.1, x_adv, x_in)) # ensure maximise attack wrt patch, and nothing else in the image
        loss = -F.cross_entropy(out, y_true)
        loss.backward()
        optimizer.step()
        diff = torch.where(mask>0.1, x_adv - x_in, torch.tensor(0.).to('cuda'))
        diff.clamp_(-eps, eps)
        x_adv.detach().copy_((diff + x_in).clamp_(0, 1))
    net.zero_grad()
    # reset to the original state
    if training:
        net.train()
    return x_adv

def fgsm_attack(x_in, target, net, mask, mode, steps, nclass, args):
    """ Perform a FGSM attack on the image
        images -> Tensor: (b,c,w,h) the batch of the images
        target -> Tensor: the label
        segnet -> torch.Module: The segmentation network
        mask   -> Tensor: (b,1,w,h) binary mask where to perform the attack
        mode   -> str: either to minimize the True class or maximize a random different class
        args   ->  Argparse: global arguments
    return
        images       -> Tensor: the image attacked
        perturbation -> Tensor: the perturbation
    """
    bsize, channel, width, height = x_in.size()
    x_in += 1 * torch.randn_like(x_in) * mask     # add some noise for more diverse attack
    x_adv = x_in.clone()
    # x_adv += 0.2 * torch.randn_like(x_adv) * mask
    # x_adv += 0.2 * torch.randn_like(x_adv) * mask
    x_adv.requires_grad = True

    training = net.training
    if training:
        net.eval()

    #optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    optimizer = torch.optim.SGD([x_adv], lr=0.1, momentum=0.9)
    for _ in range(steps):
        optimizer.zero_grad()
        net.zero_grad()
        net_pred = net(torch.where(mask>0.1, x_adv, x_in))
        if mode == "max":
            fake_target = torch.randint(nclass - 1, size=(bsize, 1)).expand_as(target).to('cuda')
            loss = F.cross_entropy(net_pred, fake_target.reshape(-1))
        elif mode == "min":
            loss = F.cross_entropy(net_pred, target.long().reshape(-1))
        loss.backward()
        optimizer.step()
        data_grad = x_adv.grad.data
        #sign_data_grad = data_grad.sign()
        sign_data_grad = x_adv.grad.sign()
        sign_data_grad *= mask # ensure the gradient is localized to the mask only and nowhere else in image
        if mode == "max":
            perturbed = - args.epsilon * sign_data_grad
        elif mode == "min":
            perturbed = args.epsilon * sign_data_grad
        # x_adv.data = torch.clamp(x_adv + perturbed, -3, 3)
        x_adv.data = torch.clamp(x_adv + perturbed, 0, 1)
    net.zero_grad()
    # reset to the original state

    if training:
        net.train()

    return x_adv.detach()

def generate_mask_square_patch(images, patchW, patchH, f, counter):
    """ Generate a mask to attack a random square patch on the images
        images -> Tensor: (b,c,w,h) the batch of images
        args   -> Argparse: global arguments
    return:
        mask -> the mask where to perform the attack """

    bsize, channel, width, height = images.size()
    mask = torch.zeros(bsize, width, height).to('cuda')
    print("/printing to text")
    for i in range(len(mask)):
        x = random.randint(0, width - patchW)
        y = random.randint(0, height - patchH)
        h = int(patchH)
        w = int(patchW)
        print(x)
        mask[i, x:x + w, y:y + h] = 1.
        imagenumber = '00000000' + str(counter)
        imagenumber = imagenumber[-8:]
        image_name = imagenumber + '.png'
        f_two = open("./cifarData/advimage/annotations/" + imagenumber + '.txt',"w")
        #f.write(image_name + " " + str(y) + "," + str(x) + "," + str(y+w) + "," + str(x+h) + "," + "1\n")
        f.write(str(0) + " " + str((y+y+w)/2/341) + " " + str((x+x+h)/2/192) + " " + str(32/341) + " " + str(32/192))
        f_two.write(str(0) + " " + str((y+y+w)/2/height) + " " + str((x+x+h)/2/width) + " " + str(w/height) + " " + str(h/width))
        f_two.close()
        counter += 1
    return mask.view(bsize, 1, width, height).expand_as(images)


def generate_mask_random_patch(images, patchW, patchH, patchMin, patchMax):
    """ Generate a mask to attack a random shape on the images
        images -> Tensor: (b,c,w,h) the batch of images
        args   -> Argparse: global arguments
    return:
        mask -> the mask where to perform the attack """
    bsize, channel, width, height = images.size()
    mask = torch.zeros(bsize, width, height).to('cuda')
    for i in range(len(mask)):
        x = random.randint(0, width - patchW)
        y = random.randint(0, height - patchH)
        h = int(patchH)
        w = int(patchW)
        shape = add_shape(patchW, patchH, patchMin, patchMax)
        mask[i, x:x + w, y:y + h] = shape == 1.
    return mask.view(bsize, 1, width, height)


def select_attack(images, target, net, steps, nclass, args, f, counter):
    """ Select the right attack given args.adv """
    if args.adv.endswith("square_patch"):
        mask = generate_mask_square_patch(images, args.patchW, args.patchH, f, counter)
    elif args.adv.endswith("random_patch"):
        mask = generate_mask_random_patch(images, args.patchW, args.patchH, args.patchMax, args.patchMin)
    else:
        raise NameError('Unknown attacks, please check args.adv arguments')
    if args.attack == 'Linf_PGD':
        return Linf_PGD(images, target, net, mask, steps, args.epsilon), mask
    elif args.attack == 'FGSM':
        return fgsm_attack(images, target, net, mask, "min", steps, nclass, args), mask
    elif args.attack == 'L2_PGD':
        return L2_PGD(images, target, net, mask, steps, args.epsilon), mask
    elif args.attack == 'GoogleP':
        return GooglePatch(images, target, net, mask, "min", steps, nclass), mask
    elif args.attack == 'cw':
        return cw(images, target, net, mask, steps, c=1e-4, kappa=0, targeted=False, learning_rate=0.01), mask
    elif args.attack == 'NA':
        return NA(images)