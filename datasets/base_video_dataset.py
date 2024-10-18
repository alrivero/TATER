import torch.utils.data
from skimage.transform import estimate_transform, warp
from vidaug import augmentors as va
import albumentations as A
import numpy as np
from skimage import transform as trans
import cv2


class BaseVideoDataset(torch.utils.data.Dataset):
    def __init__(self, video_data, split_idxs, config, test=False):
        self.video_data = video_data
        self.split_idxs = split_idxs
        self.config = config
        self.test = test
        self.image_size = config.image_size
        self.prescale = config.train.stored_scale

        if not self.test:
            self.scale = [config.train.train_scale_min, config.train.train_scale_max]
        else:
            self.scale = config.train.test_scale

        self.transform = A.ReplayCompose([
            # resize
            A.Resize(self.image_size, self.image_size),
            # color ones
            A.RandomBrightnessContrast(p=0.5),
            A.RandomGamma(p=0.5),
            A.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05, p=0.25),
            A.CLAHE(p=0.255),
            A.RGBShift(p=0.25),
            A.Blur(p=0.1),
            A.GaussNoise(p=0.5),
            # affine ones
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, border_mode=0, p=0.9),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),  additional_targets={'mediapipe_keypoints': 'keypoints'})

        self.resize = A.Compose([A.Resize(self.image_size, self.image_size)])

    @staticmethod
    def crop_face(frame, landmarks, scale=1.0, image_size=224):
        left = np.min(landmarks[:, 0])
        right = np.max(landmarks[:, 0])
        top = np.min(landmarks[:, 1])
        bottom = np.max(landmarks[:, 1])

        h, w, _ = frame.shape
        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])

        size = int(old_size * scale)

        # crop image
        src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                            [center[0] + size / 2, center[1] - size / 2]])
        DST_PTS = np.array([[0, 0], [0, image_size - 1], [image_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)

        return tform
        
    def __getitem__(self, index):
        landmarks_not_checked = True
        while landmarks_not_checked:
            try:
                data_dict = self.__getitem_aux__(index)
                # check if landmarks are not None
                if data_dict is not None:
                    landmarks = data_dict['landmarks_fan']
                    if landmarks is not None and (landmarks.shape[-2] == 68):
                        landmarks_not_checked = False
                        break
                #else:
                print("Error in loading data. Trying again...")
                index = np.random.randint(0, len(self.split_idxs))
            except Exception as e:
                # raise e
                print('Error in loading data. Trying again...', e)
                index = np.random.randint(0, len(self.split_idxs))
        
        return data_dict
