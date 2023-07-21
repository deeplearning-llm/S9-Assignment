from torchvision import datasets, transforms
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2



# Custom dataset class with Albumentations transformation
class AlbumentationsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        image, target = self.dataset.data[index],self.dataset.targets[index]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, target

    def __len__(self):
        return len(self.dataset)



def cifar10_train_test_data(batch_size_=128):

    train_transforms =A.Compose([A.HorizontalFlip(),
                            # A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), always_apply=False, p=0.5),
                                       A.ShiftScaleRotate(shift_limit=0.05,scale_limit= 0.05,rotate_limit = 15),
                                      A.CoarseDropout(min_holes=1,max_holes=1,max_height =16,max_width=16,fill_value = (0.485, 0.456, 0.406)),
                             A.Resize(height=32, width=32),
                                       A.Normalize (mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                      ToTensorV2(),
                                       ])
# Test Phase transformations
    test_transforms = A.Compose([
                                       A.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                      ToTensorV2()
                                       ])

                                           
    train = datasets.CIFAR10('./data', train=True, download=True)
    test = datasets.CIFAR10('./data', train=False, download=True)
    
    SEED = 1
    # CUDA?
    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)

    # For reproducibility
    torch.manual_seed(SEED)

    if cuda:
        torch.cuda.manual_seed(SEED)

    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = dict(shuffle=True, batch_size=512, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)



    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        AlbumentationsDataset(train, transform=train_transforms),
        **dataloader_args
    )

    test_loader = torch.utils.data.DataLoader(
        AlbumentationsDataset(test, transform=test_transforms),
        **dataloader_args
    )
    
    return train_loader,test_loader

