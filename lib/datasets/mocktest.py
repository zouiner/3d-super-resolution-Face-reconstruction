import os, sys
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cv2
import scipy
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp, resize, rescale
from glob import glob
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import json
from PIL import Image


def list_images(workspace_path, folder_path):
    image_files = []
    data_list_path = os.path.join(workspace_path, 'data_list.npy')
    
    if os.path.exists(data_list_path):
        image_files = np.load(data_list_path)
    else:
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']  # Add more if needed
        for file in os.listdir(folder_path):
            if os.path.isfile(os.path.join(folder_path, file)):
                _, ext = os.path.splitext(file)
                if ext.lower() in image_extensions:
                    image_files.append(file)
        image_files.sort()
        if image_files:  # Check if list is not empty before saving
            np.save(data_list_path, image_files)
    
    return image_files


def transform_augment(img_list, split='val', min_max=(0, 1)):    
    totensor = torchvision.transforms.ToTensor()
    hflip = torchvision.transforms.RandomHorizontalFlip()
    
    imgs = [totensor(img) for img in img_list]
    if split == 'train':
        imgs = torch.stack(imgs, 0)
        imgs = hflip(imgs)
        imgs = torch.unbind(imgs, dim=0)
    ret_img = [img * (min_max[1] - min_max[0]) + min_max[0] for img in imgs]
    return ret_img

class mockDataset(Dataset):
    def __init__(self, K, image_size, l_resolution, r_resolution, scale, trans_scale = 0, isEval = False, isTest = False, need_LR = False, split = 'train', data_len = None):
        '''
        K must be less than 6
        '''
        self.K = K
        self.image_size = image_size
        self.lr_size = l_resolution
        self.hr_size = r_resolution
        self.sr_size = self.hr_size
        self.workspace_path = '/shared/storage/cs/staffstore/ps1510/Tutorial/Image-Super-Resolution-via-Iterative-Refinement/contents/vgg_face2'
        if isEval:
            self.workspace_path = self.workspace_path + '_val' 
        else:
            self.workspace_path = self.workspace_path + '_train' 
        
        self.workspace_path = self.workspace_path + '_' + str(self.lr_size) + '_' + str(self.hr_size)
            
        # self.imagefolder = self.workspace_path + '/data'
        # self.kptfolder = self.workspace_path + '/labels/landmark'
        # self.segfolder = self.workspace_path + '/labels/skinmask'
        # ### Create this one
        # datafile = '/ps/scratch/face2d3d/texture_in_the_wild_code/VGGFace2_cleaning_codes/ringnetpp_training_lists/second_cleaning/vggface2_train_list_max_normal_100_ring_5_1_serial.npy' # !!!!
        # self.data_lines = np.load(datafile).astype('str') ### with line 35
        # file for order list of images 
        # self.no_face = np.load(os.path.join(self.workspace_path, 'labels/no_face_detect.npy')).astype('str')
        # self.img_paths = self.image_path(self.imagefolder)
        # self.data_lines = [str(i) for i in range(len(self.img_paths))]
        # self.data_lines = [x for x in self.data_lines if x not in self.no_face]
        
        # if os.path.exists(os.path.join(self.workspace_path, 'labels/dataline.npy')):
        #     self.data_lines = np.load(os.path.join(self.workspace_path, 'labels/dataline.npy')).astype('str')
        # else:
        #     self.data_lines = image_path(self.imagefolder)
        
        # From original
        
        
        self.lr_imagefolder = self.workspace_path + '/lr_' + str(self.lr_size) 
        self.sr_imagefolder = self.workspace_path + '/sr_' + str(self.lr_size) +  '_' + str(self.sr_size)
        self.hr_imagefolder = self.workspace_path + '/hr_' + str(self.hr_size)
        # self.kptfolder = self.workspace_path + '/labels/landmark/'
        # self.segfolder = self.workspace_path + '/labels/skinmask/'
        
        # datafile = '/shared/storage/cs/staffstore/ps1510/Tutorial/Dataset/VGG-Face2/DECA_setting/train_set.npy'
        # if isEval or isTest:
        #     datafile = '/shared/storage/cs/staffstore/ps1510/Tutorial/Dataset/VGG-Face2/DECA_setting/eval_set.npy'
        # self.data_lines = np.load(datafile).astype('str')
        
        self.data_lines = list_images(self.workspace_path, self.sr_imagefolder)
        # I will do a new list file and data_line parallelly by instead of the new name.
        if data_len:
            self.data_lines = self.data_lines[:data_len]
        
        self.scale = scale #[scale_min, scale_max]
        self.trans_scale = trans_scale #[dx, dy]
        
        self.need_LR = need_LR
        
        self.split = split
        

    def __len__(self):
        return len(self.data_lines)

    def __getitem__(self, idx):
        sr_images_list = []; hr_images_list = []; kpt_list = []; mask_list = []

        random_ind = [0]
        # random_ind = np.random.permutation(5)[:self.K]
        for i in random_ind:
            # name = self.data_lines[idx, i]
            # name = os.path.splitext(str(name))[0]
            # name = name[len(self.imagefolder)+1:]
            
            name = self.data_lines[idx]
            lr_image_path = os.path.join(self.lr_imagefolder, name)
            sr_image_path = os.path.join(self.sr_imagefolder, name) # + '.jpg')
            hr_image_path = os.path.join(self.hr_imagefolder, name)
            # seg_path = os.path.join(self.segfolder, name + '.npy')  
            # kpt_path = os.path.join(self.kptfolder, name + '.npy')
                                            
            # image = imread(sr_image_path)/255.
            # kpt = np.load(kpt_path)[0]
            # mask = self.load_mask(seg_path, image.shape[0], image.shape[1])

            # ### crop information
            # tform = self.crop(image, kpt)
            # ## crop 
            # cropped_image = warp(image, tform.inverse, output_shape=(self.image_size, self.image_size))
            # cropped_mask = warp(mask, tform.inverse, output_shape=(self.image_size, self.image_size), order = 0) 
            # # # add by patipol
            # # cropped_mask = np.where(cropped_mask != 0., 1, 0)
            # # # add by patipol
            # cropped_kpt = np.dot(tform.params, np.hstack([kpt, np.ones([kpt.shape[0],1])]).T).T # np.linalg.inv(tform.params)


            # # normalized kpt
            # cropped_kpt[:,:2] = cropped_kpt[:,:2]/self.image_size * 2  - 1

            # # Visualise cropped_img, cropped_mask, and overlay the cropped_ktp onto cropped_img

            # images_list.append(cropped_image.transpose(2,0,1))
            # kpt_list.append(cropped_kpt)
            # mask_list.append(cropped_mask)

        
        ### 
        # images_array = torch.from_numpy(np.array(images_list)).type(dtype = torch.float32) #K,224,224,3
        # kpt_array =  torch.from_numpy(np.array(kpt_list)).type(dtype = torch.float32) #K,224,224,3
        # mask_array = torch.from_numpy(np.array(mask_list)).type(dtype = torch.float32) #K,224,224,3
        
                 
        # data_dict = {
        #     'image': images_array,
        #     'landmark': kpt_array,
        #     'mask': mask_array
        # }
        
        # return data_dict
        
        img_HR = Image.open(hr_image_path).convert("RGB")
        img_SR = Image.open(sr_image_path).convert("RGB")
        if self.need_LR:
            img_LR = Image.open(lr_image_path).convert("RGB")
        if self.need_LR:
            [img_LR, img_SR, img_HR] = transform_augment(
                [img_LR, img_SR, img_HR], split=self.split, min_max=(-1, 1))
            return {'LR': img_LR, 'HR': img_HR, 'SR': img_SR, 'Index': idx}
        else:
            [img_SR, img_HR] = transform_augment(
                [img_SR, img_HR], split=self.split, min_max=(-1, 1))
            return {'HR': img_HR, 'SR': img_SR, 'Index': idx}
        
        
    
    def crop(self, image, kpt):
        left = np.min(kpt[:,0]); right = np.max(kpt[:,0]); 
        top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])

        h, w, _ = image.shape
        old_size = (right - left + bottom - top)/2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])#+ old_size*0.1])
        # translate center
        trans_scale = (np.random.rand(2)*2 -1) * self.trans_scale
        center = center + trans_scale*old_size # 0.5

        scale = np.random.rand() * (self.scale[1] - self.scale[0]) + self.scale[0]
        size = int(old_size*scale)

        # crop image
        src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
        DST_PTS = np.array([[0,0], [0,self.image_size - 1], [self.image_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)
        
        # cropped_image = warp(image, tform.inverse, output_shape=(self.image_size, self.image_size))
        # # change kpt accordingly
        # cropped_kpt = np.dot(tform.params, np.hstack([kpt, np.ones([kpt.shape[0],1])]).T).T # np.linalg.inv(tform.params)
        return tform
    
    def load_mask(self, maskpath, h, w):
        # print(maskpath)
        if os.path.isfile(maskpath):
            vis_parsing_anno = np.load(maskpath)
            # atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
            #     'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
            mask = np.zeros_like(vis_parsing_anno)
            # for i in range(1, 16):
            mask[vis_parsing_anno>0.5] = 1.
        else:
            mask = np.ones((h, w))
            print(maskpath)
        return mask
    
    
    

