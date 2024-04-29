import os, sys
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import scipy
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp, resize, rescale
from glob import glob
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import json

def check_workspace(image_path):
    
    # Find the index of "data" in the path
    data_index = image_path.find("data")

    if data_index != -1:
        # Extract the base directory dynamically
        base_directory = image_path[:data_index + len("data")]

        # Get the relative path
        b = os.path.relpath(image_path, base_directory)

        # print("Base directory:", base_directory)
        # print("Relative path:", b)
        
        return base_directory, b
    
    else:
        print("No 'data' directory found in the path. Your list file already clean a workspace path.")
        
        return None, None

def image_path(directory, amount = 1, random_set = False, extensions=['*.jpg', '*.png', '*.jpeg']):
        
        savepath = directory[:-5]    
        if os.path.exists(savepath + '/data_list.npy'):
            image_paths = np.load(savepath + '/data_list.npy')
            base_directory, _ = check_workspace(image_paths[0])
            if base_directory == None:
                if base_directory not in savepath:
                    new_list = []
                    for i in image_paths:
                        name = remove_workspace_path(i, base_directory)
                        new_list.append(os.path.join(directory, name))
                    image_paths = new_list
            print('Found: list of images')
        else:
            print('Not Found: list of images')
            # Use glob to find image file paths in the directory and its subdirectories
            image_paths = []
            for extension in extensions:
                image_paths.extend(glob(os.path.join(savepath, '**', extension), recursive=True))
            image_paths.sort()
            np.save(savepath + '/data_list.npy', image_paths)

        if random_set:
            
            if os.path.exists(savepath + '/data_list_fixamount.npy'):
                image_paths = np.load(savepath + '/data_list_fixamount.npy')
            else:
                # Randomly shuffle the subfolders
                random.shuffle(image_paths)
                np.save(savepath + '/data_list_fixamount.npy', image_paths)
                
        if amount != 1:
            image_paths = image_paths[:int(len(image_paths) * amount)]
            
        
        print("Found image paths:", len(image_paths), 'image(s)')
        
        return image_paths

class mockDataset(Dataset):
    def __init__(self, K, image_size, scale, trans_scale = 0, isEval = False, isTest = False):
        '''
        K must be less than 6
        '''
        self.K = K
        self.image_size = image_size
        self.workspace_path = '/shared/storage/cs/staffstore/ps1510/Tutorial/Dataset/VGG-Face2'
        # if isEval:
        #     self.workspace_path = '/shared/storage/cs/staffstore/ps1510/Tutorial/Dataset/CelebA_mock_split/val'
        # if isTest:
        #     self.workspace_path = '/shared/storage/cs/staffstore/ps1510/Tutorial/Dataset/CelebA_mock_split/test'
            
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
        
        self.imagefolder = self.workspace_path + '/data/train'
        self.kptfolder = self.workspace_path + '/labels/landmark/train'
        self.segfolder = self.workspace_path + '/labels/skinmask/train'
        
        datafile = '/shared/storage/cs/staffstore/ps1510/Tutorial/Dataset/VGG-Face2/DECA_setting/train_set.npy'
        if isEval or isTest:
            datafile = '/shared/storage/cs/staffstore/ps1510/Tutorial/Dataset/VGG-Face2/DECA_setting/eval_set.npy'
        self.data_lines = np.load(datafile).astype('str')
        
        self.scale = scale #[scale_min, scale_max]
        self.trans_scale = trans_scale #[dx, dy]
        

    def __len__(self):
        return len(self.data_lines)

    def __getitem__(self, idx):
        images_list = []; kpt_list = []; mask_list = []

        # random_ind = [0]
        random_ind = np.random.permutation(5)[:self.K]
        for i in random_ind:
            name = self.data_lines[idx, i]
            # name = os.path.splitext(str(name))[0]
            # name = name[len(self.imagefolder)+1:]
            
            image_path = os.path.join(self.imagefolder, name + '.jpg')  
            seg_path = os.path.join(self.segfolder, name + '.npy')  
            kpt_path = os.path.join(self.kptfolder, name + '.npy')
                                            
            image = imread(image_path)/255.
            kpt = np.load(kpt_path)[0]
            mask = self.load_mask(seg_path, image.shape[0], image.shape[1])

            ### crop information
            tform = self.crop(image, kpt)
            ## crop 
            cropped_image = warp(image, tform.inverse, output_shape=(self.image_size, self.image_size))
            cropped_mask = warp(mask, tform.inverse, output_shape=(self.image_size, self.image_size), order = 0) 
            # # add by patipol
            # cropped_mask = np.where(cropped_mask != 0., 1, 0)
            # # add by patipol
            cropped_kpt = np.dot(tform.params, np.hstack([kpt, np.ones([kpt.shape[0],1])]).T).T # np.linalg.inv(tform.params)


            # normalized kpt
            cropped_kpt[:,:2] = cropped_kpt[:,:2]/self.image_size * 2  - 1

            # Visualise cropped_img, cropped_mask, and overlay the cropped_ktp onto cropped_img

            images_list.append(cropped_image.transpose(2,0,1))
            # images_list.append(cropped_image)
            kpt_list.append(cropped_kpt)
            mask_list.append(cropped_mask)

        
        ### Check actual shape is?
        images_array = torch.from_numpy(np.array(images_list)).type(dtype = torch.float32) #K,224,224,3
        kpt_array =  torch.from_numpy(np.array(kpt_list)).type(dtype = torch.float32) #K,224,224,3
        mask_array = torch.from_numpy(np.array(mask_list)).type(dtype = torch.float32) #K,224,224,3
        
                 
        data_dict = {
            'image': images_array,
            'landmark': kpt_array,
            'mask': mask_array
        }
        
        return data_dict
    
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
    
    
    

