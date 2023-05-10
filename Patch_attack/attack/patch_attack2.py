import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
import nets.resnet
#from utils.defense_utils import *
import os
import argparse
from tqdm import tqdm
import PIL
from PatchAttacker import PatchAttacker
from torchvision.utils import save_image
from adv_attacks import select_attack
from torchvision import models

parser = argparse.ArgumentParser()
parser.add_argument("--dump_dir",default='patch_adv',type=str,help="directory to save attack results")
parser.add_argument("--model_dir",default='checkpoints',type=str,help="path to checkpoints")
parser.add_argument('--data_dir', default='data', type=str,help="path to data")
parser.add_argument('--dataset', default='cityscape', choices=('cityscape_adv','GTSRB','Food101','flower'),type=str,help="dataset")
parser.add_argument("--model",default='resnet50',type=str,help="model name")
parser.add_argument("--clip",default=-1,type=int,help="clipping value; do clipping when this argument is set to positive")
parser.add_argument("--aggr",default='mean',type=str,help="aggregation methods. set to none for local feature")
parser.add_argument("--patchW",type=int,help="size of the adversarial patch width")
parser.add_argument("--patchH",type=int,help="size of the adversarial patch height")
parser.add_argument("--patchMin",type=int,help="size of the adversarial patch min")
parser.add_argument("--patchMax",type=int,help="size of the adversarial patch max")
parser.add_argument("--adv",type=str,help="size of the adversarial patch")
parser.add_argument("--epsilon", type=float, default=0.1,help="epsilon for adversarial attacks")
parser.add_argument("--attack",type=str,choices=('Linf_PGD','FGSM','random','L2_PGD','GoogleP','cw','NA'),help="Attack mode")

args = parser.parse_args()

MODEL_DIR=os.path.join('.',args.model_dir)
DATA_DIR=os.path.join(args.data_dir,args.dataset)
DATASET = args.dataset

if DATASET == 'flower':
    mean_vec = [0.4914, 0.4822, 0.4465]
    std_vec = [0.2023, 0.1994, 0.2010]
    ds_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])
    val_dataset = datasets.Flowers102(root=DATA_DIR, download=True, transform=ds_transforms)
    class_names = 102

elif DATASET == 'cityscape':
    mean_vec = [0.4914, 0.4822, 0.4465]
    std_vec = [0.2023, 0.1994, 0.2010]
    ds_transforms = ds_transforms = transforms.Compose([
        transforms.Resize(384, interpolation=PIL.Image.BICUBIC),
        transforms.ToTensor(),
    ])
    val_dataset = datasets.ImageFolder(root='./Cityscape/', transform=ds_transforms)
    class_names = val_dataset.classes

elif DATASET == 'GTSRB':
    #DATA_DIR = os.path(os.getcwd(),'checkpoints')
    mean_vec = [0.4914, 0.4822, 0.4465]
    std_vec = [0.2023, 0.1994, 0.2010]
    ds_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])
    val_dataset = datasets.GTSRB(root=DATA_DIR, download=True, transform=ds_transforms)
    class_names = 43

elif DATASET == 'Food101':
    mean_vec = [0.4914, 0.4822, 0.4465]
    std_vec = [0.2023, 0.1994, 0.2010]
    ds_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor()
    ])
    val_dataset = datasets.Food101(root=DATA_DIR, download=True,transform=ds_transforms)
    class_names = 101

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8,shuffle=True)

#build and initialize model
device = 'cuda'

if args.clip > 0:
    clip_range = [0,args.clip]
else:
    clip_range = None

if 'resnet50' in args.model:
    model = nets.resnet.resnet50(pretrained=True,clip_range=clip_range,aggregation=args.aggr)

if  DATASET == 'cityscape':
    checkpoint = torch.load(os.path.join(MODEL_DIR,args.model+'.pth'))
    model.load_state_dict(checkpoint)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
elif DATASET == 'GTSRB':
    checkpoint = torch.load(os.path.join(MODEL_DIR,args.model+'.pth'))
    model.load_state_dict(checkpoint)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 43)
elif DATASET == 'Food101':
    checkpoint = torch.load(os.path.join(MODEL_DIR,args.model+'.pth'))
    model.load_state_dict(checkpoint)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 101)

model = model.to(device)
print("model loaded")
model.eval()
cudnn.benchmark = True

print("attack started")
# intialization of PatchAttacker
print("attack initialized")

# terence edit
counter = 0
steps = 20
if args.attack not in ['FGSM','L2_PGD','Linf_PGD','GoogleP','cw','NA']:
    print("error! wrong attacker")
    attacker = PatchAttacker(model, mean_vec, std_vec,patch_size=args.patchW,step_size=0.05,steps=500)
    counter = 0
    for data,labels in tqdm(val_loader):
        data,labels=data.to(device),labels.to(device) # cifar10 is 32x32, hence poor resolution when scales
        data_adv,patch_loc,x,mask = attacker.perturb(data, labels)

        for i in range(len(data_adv)):
            counter += 1
            imges = data_adv[i]
            masker = mask[i]

            if DATASET == 'cityscape':
                # Terence edit
                imagenumber = '00000000' + str(counter)
                imagenumber = imagenumber[-8:]
                save_image(imges, os.getcwd() + '/cityscape_adv/advimage/imges'+ imagenumber + '.png')
                save_image(masker, os.getcwd() + '/cityscape_adv/advmask/imges'+ imagenumber + '.png')

            if DATASET == 'GTSRB':
                # Terence edit
                imagenumber = '00000000' + str(counter)
                imagenumber = imagenumber[-8:]
                save_image(imges, os.getcwd() + '/GTSRBData/advimage/imges'+ imagenumber + '.png')
                save_image(masker, os.getcwd() + '/GTSRBData/advmask/imges'+ imagenumber + '.png')

            if DATASET == 'Food101':
                # Terence edit
                imagenumber = '00000000' + str(counter)
                imagenumber = imagenumber[-8:]
                save_image(imges, os.getcwd() + '/Food101Data/advimage/imges'+ imagenumber + '.png')
                save_image(masker, os.getcwd() + '/Food101Data/advmask/imges'+ imagenumber + '.png')

            if DATASET == 'flower':
                # Terence edit
                imagenumber = '00000000' + str(counter)
                imagenumber = imagenumber[-8:]
                save_image(imges, os.getcwd() + '/flower102Data/advimage/imges'+ imagenumber + '.png')
                save_image(masker, os.getcwd() + '/flower102Data/advmask/imges'+ imagenumber + '.png')
else:
    print('proper attacks')
    f = open("train.txt","w")
    image_number = 1
    for data,labels in tqdm(val_loader):
        data,labels=data.to(device),labels.to(device) # cifar10 is 32x32, hence poor resolution when scales
        data_adv, mask = select_attack(data, labels, model, steps, class_names, args, f, image_number)

        for i in range(len(data_adv)):
            counter += 1
            # print(counter)
            imges = data_adv[i]
            masker = mask[i]

            if DATASET == 'cityscape':
                # Terence edit
                imagenumber = '00000000' + str(counter)
                imagenumber = imagenumber[-8:]
                save_image(imges, os.getcwd() + '/cityscape_adv/advimage/'+ imagenumber + '.png')
                save_image(masker, os.getcwd() + '/cityscape_adv/advmask/'+ imagenumber + '.png')

            if DATASET == 'GTSRB':
                # Terence edit
                imagenumber = '00000000' + str(counter)
                imagenumber = imagenumber[-8:]
                save_image(imges, os.getcwd() + '/GTSRBData/advimage/imges'+ imagenumber + '.png')
                save_image(masker, os.getcwd() + '/GTSRBData/advmask/imges'+ imagenumber + '.png')

            if DATASET == 'Food101':
                # Terence edit
                imagenumber = '00000000' + str(counter)
                imagenumber = imagenumber[-8:]
                save_image(imges, os.getcwd() + '/Food101Data/advimage/imges'+ imagenumber + '.png')
                save_image(masker, os.getcwd() + '/Food101Data/advmask/imges'+ imagenumber + '.png')

            if DATASET == 'flower':
                # Terence edit
                imagenumber = '00000000' + str(counter)
                imagenumber = imagenumber[-8:]
                save_image(imges, os.getcwd() + '/flower102Data/advimage/imges'+ imagenumber + '.png')
                save_image(masker, os.getcwd() + '/flower102Data/advmask/imges'+ imagenumber + '.png')
        image_number += 8
    f.close()
