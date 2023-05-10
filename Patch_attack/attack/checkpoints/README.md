## Checkpoints
### overview
Model checkpoints used in the paper can be downloaded from [link](https://drive.google.com/drive/folders/1u5RsCuZNf7ddWW0utI4OrgWGmJCUDCuT?usp=sharing).

The checkpoints from the google drive is obtained with "provable adversarial training" (add feature masks during the training).

Model training should be very easy with the provided training scripts. 

### checkpoints for bagnet/resnet trained on imagenet
two model checkpoints trained with "provable adversarial training" are available now! bagnet17_net.pth will give the results reported in our paper. PS: the clean accuracy for resnet50 (note that resnet50 is not used in our defense!) reported in the paper uses the pretrained weights from torchvision.

- bagnet33_net.pth
- bagnet17_net.pth

### checkpoints for bagnet/resnet trained on imagenette
- resnet50_nette.pth
- bagnet33_nette.pth
- bagnet17_nette.pth
- bagnet9_nette.pth

### checkpoints for bagnet/resnet trained on cifar
- resnet50_192_cifar.pth
- bagnet33_192_cifar.pth
- bagnet17_192_cifar.pth
- bagnet9_192_cifar.pth

### checkpoints for ds-resnet on different datasets
- ds_net.pth
- ds_nette.pth
- ds_cifar.pth

Training scripts for ds-resnet are not provided in this repository, but can be found be found in [patchSmoothing](https://github.com/alevine0/patchSmoothing)
