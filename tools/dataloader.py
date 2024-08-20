# This code is mostly taken from BTS; author: Jin Han Lee
# and modified by Mohammed Chaouki ZIARA, 
# For Ph.D. Project @ RCAM Laboratory, Djilali Liabes University - Algeria 
# it contains the dataloader used to read, split, preprosess and build the dataloaders
# please Modify the args on the args_****.txt files to controle the Batch size and the number of images used for training 
# Contact: medchaoukiziara@gmail.com || me@chaouki.pro

import os
import random
import numpy as np
import torch
import torch.utils.data.distributed
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def data_extractor(path_txt, path_images, path_depths):
    file  = open(path_txt)
    lines = file.readlines()
    file.close()
    path_tr_img = f"{path_images}train/"
    path_tr_dep = f"{path_depths}train/"
    
    path_ts_img = f"{path_images}val/"
    path_ts_dep = f"{path_depths}val/"
    
    
    imgs = []
    deps = []
    
    for i in lines :
        image, depth = i.strip().split(' ')[:-1]
        image = f"{image.split('/')[1]}/{image}" 
        if os.path.isfile(path_tr_img+image) and os.path.isfile(path_tr_dep+depth):
            imgs.append(path_tr_img+image)
            deps.append(path_tr_dep+depth)

        elif os.path.isfile(path_ts_img+image) and os.path.isfile(path_ts_dep+depth):
            imgs.append(path_ts_img+image)
            deps.append(path_ts_dep+depth)
        else : 
            print(f"{image} or {depth} don't exist")
            
    return imgs, deps



def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

class NoOpTransform:
    def __call__(self, x):
        return x
    
def preprocessing_transforms(mode):
    return transforms.Compose([
        ToTensor(mode=mode)])


class DepthDataLoader(object):
    def __init__(self, args, images, depths, mode, num_threads = 5 ):
        if mode == 'train':
            self.training_samples = DataLoadPreprocess(images, depths, mode = mode, transform=preprocessing_transforms(mode), 
                                                       is_for_online_eval=False,  degree = 5.5,  do_kb_crop = True, 
                                                       do_random_rotate = True, dataset = args.dataset,
                                                       input_height = args.image_height, input_width = args.image_width
)
            self.train_sampler = None
            self.data = DataLoader(self.training_samples, args.bs,
                                   shuffle=(self.train_sampler is None),
                                   num_workers= num_threads,
                                   pin_memory=True,
                                   sampler=self.train_sampler)

        elif mode == 'online_eval':
            self.testing_samples = DataLoadPreprocess(images, depths, mode = mode, transform=preprocessing_transforms(mode), 
                                                       is_for_online_eval=True,  degree = 2.5,  do_kb_crop = False, 
                                                       do_random_rotate = False, dataset = args.dataset,
                                                       input_height = args.image_height, input_width = args.image_width
)
            self.eval_sampler = None
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=False,
                                   num_workers=1,
                                   pin_memory=False,
                                   sampler=self.eval_sampler)

        elif mode == 'test':
            self.testing_samples = DataLoadPreprocess(images, depths, mode = mode, transform=preprocessing_transforms(mode), 
                                                       is_for_online_eval=False,  degree = 2.5,  do_kb_crop = False, 
                                                       do_random_rotate = False, dataset = args.dataset,
                                                       input_height = args.image_height, input_width = args.image_width
)
            self.data = DataLoader(self.testing_samples, 1, shuffle=False, num_workers=1)

        else:
            print('mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))


def remove_leading_slash(s):
    if s[0] == '/' or s[0] == '\\':
        return s[1:]
    return s


class DataLoadPreprocess(Dataset):
    def __init__(self,  images, depths, mode = 'train', transform=None, 
                 is_for_online_eval=False,  degree = 2.5, 
                 do_kb_crop = True, do_random_rotate = True, dataset = 'kitti',
                 input_height = 352, input_width = 704 ):
        
        self.images             = images
        self.depths             = depths
        self.do_kb_crop         = do_kb_crop
        self.do_random_rotate   = do_random_rotate
        self.dataset            = dataset
        self.mode               = mode
        self.transform          = transform
        #self.to_tensor         = ToTensor
        self.is_for_online_eval = is_for_online_eval
        self.degree             = degree
        self.input_height       = input_height
        self.input_width        = input_width

    def __getitem__(self, idx):
        #sample_path = self.filenames[idx]
        #focal = float(sample_path.split()[2])

        if self.mode == 'train':

            image = Image.open(self.images[idx])
            depth_gt = Image.open(self.depths[idx])

            if ((self.do_kb_crop is True) and (self.dataset == 'kitti')):
                height = image.height
                width = image.width
                top_margin = int(height - self.input_height )
                left_margin = int((width - self.input_width) / 2)
                depth_gt = depth_gt.crop((left_margin, top_margin, left_margin + self.input_width, top_margin + self.input_height ))
                image = image.crop((left_margin, top_margin, left_margin + self.input_width, top_margin + self.input_height ))

            # To avoid blank boundaries due to pixel registration
            if self.dataset == 'nyu':
                depth_gt = depth_gt.crop((43, 45, 608, 472))
                image = image.crop((43, 45, 608, 472))

            if self.do_random_rotate is True:
                random_angle = (random.random() - 0.5) * 2 * self.degree
                image = self.rotate_image(image, random_angle)
                depth_gt = self.rotate_image(depth_gt, random_angle, flag=Image.NEAREST)

            image = np.asarray(image, dtype=np.float32) / 255.0
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)

            if self.dataset == 'nyu':
                depth_gt = depth_gt / 1000.0
            else:
                depth_gt = depth_gt / 256.0

            image, depth_gt = self.random_crop(image, depth_gt, image.shape[0], image.shape[1])
            image, depth_gt = self.train_preprocess(image, depth_gt)
            sample = {'image': image, 'depth': depth_gt}

        else:
            image = Image.open(self.images[idx])
            if self.dataset == 'nyu' : image = image.crop((43, 45, 608, 472))
            image = np.asarray(image, dtype=np.float32) / 255.0

            if self.mode == 'online_eval':
                
                depth_path = self.depths[idx]
                has_valid_depth = False
                try:
                    depth_gt = Image.open(depth_path)
                    if self.dataset == 'nyu' : 
                        depth_gt = depth_gt.crop((43, 45, 608, 472))
                    has_valid_depth = True
                except IOError:
                    depth_gt = False
                    # print('Missing gt for {}'.format(image_path))

                if has_valid_depth:
                    depth_gt = np.asarray(depth_gt, dtype=np.float32)
                    depth_gt = np.expand_dims(depth_gt, axis=2)
                    if self.dataset == 'nyu':
                        depth_gt = depth_gt / 1000.0
                    else:
                        depth_gt = depth_gt / 256.0

            if ((self.do_kb_crop is True) and (self.dataset == "kitti")):
                height = image.shape[0]
                width = image.shape[1]
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                image = image[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
                if self.mode == 'online_eval' and has_valid_depth:
                    depth_gt = depth_gt[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]

            if self.mode == 'online_eval':
                 # To avoid blank boundaries due to pixel registration
            
                sample = {'image': image, 'depth': depth_gt, 'has_valid_depth': has_valid_depth}
            else:
                sample = {'image': image}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def random_crop(self, img, depth, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]
        return img, depth

    def train_preprocess(self, image, depth_gt):
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()

        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)

        return image, depth_gt

    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        if self.dataset == 'nyu':
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

    def __len__(self):
        return len(self.images)


class ToTensor(object):
    def __init__(self, mode):
        self.mode = mode
        #self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.normalize  = transforms.Compose([NoOpTransform()])

    def __call__(self, sample):
        image = sample['image']
        image = self.to_tensor(image)
        image = self.normalize(image)

        if self.mode == 'test':
            return {'image': image}

        depth = sample['depth']
        if self.mode == 'train':
            depth = self.to_tensor(depth)
            return {'image': image, 'depth': depth}
        else:
            has_valid_depth = sample['has_valid_depth']
            return {'image': image, 'depth': depth,  'has_valid_depth': has_valid_depth}

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img