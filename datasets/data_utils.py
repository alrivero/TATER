import torch
from datasets.iHiTOP_dataset import get_datasets_iHiTOP
from datasets.mixed_dataset_sampler import MixedDatasetBatchSampler
import os

import torch

# def iHiTOP_collate(batch):
#     # Extract the "img" tensors and other keys
#     imgs = [item['img'] for item in batch]
#     max_len = max([img.shape[0] for img in imgs])  # Get the max number of tokens

#     # Initialize tensors for the batch
#     img_batch = torch.zeros((len(batch), max_len, *imgs[0].shape[1:]), dtype=imgs[0].dtype)
#     attention_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)

#     # Process each item in the batch
#     for i, item in enumerate(batch):
#         img_len = item['img'].shape[0]
#         img_batch[i, :img_len] = item['img']  # Add image tensor with padding if necessary
#         attention_mask[i, :img_len] = 1  # Set the attention mask to True for valid tokens

#     # Now, combine any additional keys
#     combined_batch = {'img': img_batch, 'attention_mask': attention_mask}

#     list_keys = ['text', 'removed_frames', 'text', 'tokens', 'flag_landmarks_fan']
#     for key in batch[0].keys():
#         if key == 'img':
#             continue  # Already handled the 'img' key
#         elif key in list_keys:
#             combined_batch[key] = [item[key] for item in batch]
#         elif key == 'audio':
#             # Handle the audio key
#             audios = [item['audio'] for item in batch]
#             audio_lens = [audio.shape[0] for audio in audios]
#             max_audio_len = max(audio_lens)  # Get the max audio length

#             # Pad audio tensors to max_audio_len and store their lengths
#             audio_batch = torch.zeros((len(batch), max_audio_len), dtype=audios[0].dtype)
#             for i, audio in enumerate(audios):
#                 audio_len = audio.shape[0]
#                 audio_batch[i, :audio_len] = audio  # Pad audio if necessary

#             combined_batch['audio'] = audio_batch
#             combined_batch['audio_len'] = torch.tensor(audio_lens)  # Store original audio lengths

#         else:
#             values = [item[key][None] for item in batch]

#             # Pad the first dimension if necessary
#             padded_values = []
#             for value in values:
#                 padding = (0,) * (2 * (value.ndim - 1)) + (0, max_len - value.shape[0])
#                 padded_value = torch.nn.functional.pad(value, padding)
#                 padded_values.append(padded_value)
#             stacked = torch.stack(padded_values, dim=0)

#             combined_batch[key] = stacked

#     return combined_batch

def load_dataloaders(config):
    # ----------------------- initialize datasets ----------------------- #
    train_dataset_iHiTOP, val_dataset_iHiTOP, test_dataset_iHiTOP = get_datasets_iHiTOP(config)
    dataset_percentages = {
        'iHiTOP': 1.0
    }
    
    train_dataset = train_dataset_iHiTOP
    sampler = MixedDatasetBatchSampler([
                                        len(train_dataset_iHiTOP)
                                        ], 
                                       list(dataset_percentages.values()), 
                                       config.train.batch_size, len(train_dataset_iHiTOP))
    def collate_fn(batch):
        combined_batch = {}
        for key in batch[0].keys():
            combined_batch[key] = [b[key] for b in batch]

        return combined_batch
    
    val_dataset = torch.utils.data.ConcatDataset([val_dataset_iHiTOP])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=sampler, num_workers=config.train.num_workers, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.train.batch_size,
                                                num_workers=config.train.num_workers, shuffle=False, drop_last=True, collate_fn=collate_fn)
    return train_loader, val_loader

def linear_interpolate(landmarks, start_idx, stop_idx):
    """linear_interpolate.

    :param landmarks: ndarray, input landmarks to be interpolated.
    :param start_idx: int, the start index for linear interpolation.
    :param stop_idx: int, the stop for linear interpolation.
    """
    start_landmarks = landmarks[start_idx]
    stop_landmarks = landmarks[stop_idx]
    delta = stop_landmarks - start_landmarks
    for idx in range(1, stop_idx-start_idx):
        landmarks[start_idx+idx] = start_landmarks + idx/float(stop_idx-start_idx) * delta
    return landmarks

def landmarks_interpolate(landmarks):
    """landmarks_interpolate.

    :param landmarks: List, the raw landmark (in-place)

    """
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    if not valid_frames_idx:
        return None
    for idx in range(1, len(valid_frames_idx)):
        if valid_frames_idx[idx] - valid_frames_idx[idx - 1] == 1:
            continue
        else:
            landmarks = linear_interpolate(landmarks, valid_frames_idx[idx - 1], valid_frames_idx[idx])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    # -- Corner case: keep frames at the beginning or at the end failed to be detected.
    if valid_frames_idx:
        landmarks[:valid_frames_idx[0]] = [landmarks[valid_frames_idx[0]]] * valid_frames_idx[0]
        landmarks[valid_frames_idx[-1]:] = [landmarks[valid_frames_idx[-1]]] * (len(landmarks) - valid_frames_idx[-1])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    assert len(valid_frames_idx) == len(landmarks), "not every frame has landmark"
    return landmarks




def create_LRS3_lists(lrs3_path, lrs3_landmarks_path):
    from sklearn.model_selection import train_test_split
    import pickle
    trainval_folder_list = list(os.listdir(f"{lrs3_path}/trainval"))
    train_folder_list, val_folder_list = train_test_split(trainval_folder_list, test_size=0.2, random_state=42)
    test_folder_list = list(os.listdir(f"{lrs3_path}/test"))



    def gather_LRS3_split(folder_list, split="trainval"):
        list_ = []
        for folder in folder_list:
            for file in os.listdir(os.path.join(f"{lrs3_path}/{split}", folder)):
                if file.endswith(".txt"):
                    file_without_extension = file.split(".")[0]
                    file_inner_path = f"{split}/{folder}/{file_without_extension}"

                    landmarks_filename = os.path.join(lrs3_landmarks_path, file_inner_path+".pkl")

                    valid = True
                    with open(landmarks_filename, "rb") as pkl_file:
                        landmarks = pickle.load(pkl_file)
                        preprocessed_landmarks = landmarks_interpolate(landmarks)
                        if preprocessed_landmarks is None:
                            valid = False

                    mediapipe_landmarks_filepath = os.path.join(lrs3_path, file_inner_path+".npy")
                    if not os.path.exists(mediapipe_landmarks_filepath):
                        valid = False
                    if os.path.exists(landmarks_filename) and valid:
                        subject = folder
                        list_.append([os.path.join(lrs3_path, file_inner_path + ".mp4"), os.path.join(lrs3_landmarks_path, file_inner_path+".pkl"), 
                                      mediapipe_landmarks_filepath,
                                      subject])
        return list_

    train_list = gather_LRS3_split(train_folder_list, split="trainval")
    val_list = gather_LRS3_split(val_folder_list, split="trainval")
    test_list = gather_LRS3_split(test_folder_list, split="test")

    print(len(train_list), len(val_list), len(test_list))

    pickle.dump([train_list,val_list,test_list], open(f"assets/LRS3_lists.pkl", "wb"))
