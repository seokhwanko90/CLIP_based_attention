import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from PIL import Image
import openslide
import argparse
import tifffile

import timm
from model.model import PatchAttention, PathologyCLIP
from typing import Dict, List, Tuple
import json


Image.MAX_IMAGE_PIXELS = 30000000000
torch.manual_seed(42)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

L=3
thum_x = 1024

# Custom Dataset for pathology images and captions
class PathologyDataset(Dataset):
    def __init__(self, data: dict, embed_dim: int = 512):
        #with open(json_path, 'r') as f:
        #    self.data = json.load(f)
        self.data = data
        self.sample_ids = list(self.data.keys())
        self.embed_dim = embed_dim
    
    def __len__(self) -> int:
        return len(self.sample_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
        sample_id = self.sample_ids[idx]
        sample = self.data[sample_id]

        thum_embedding = torch.tensor(sample.get('thumbnail', [0]*self.embed_dim), dtype=torch.float32)
        # Get patch embeddings as a tensor (num_patches, embed_dim)
        patch_ids = list(sample['patches'].keys())
        patch_embeddings = torch.tensor([sample['patches'][patch] for patch in sample['patches']], dtype=torch.float32)
        
        # Handle text embedding and caption
        if 'text' in sample.keys():
            text_embedding = torch.tensor(sample.get('text', [0]*self.embed_dim), dtype=torch.float32)
            text_orgp_embedding = torch.tensor(sample.get('text_organ_proc', [0]*self.embed_dim), dtype=torch.float32)
            text_hist_embedding = torch.tensor(sample.get('text_hist_detail', [0]*self.embed_dim), dtype=torch.float32)
    
            list_text_embs = [text_embedding, text_orgp_embedding, text_hist_embedding]
            list_text_embs = torch.stack(list_text_embs)
            #print(list_text_embs.shape)
            
            caption = sample.get('text_str', '')
        else:
            list_text_embs = []
            caption = ''
        
        return thum_embedding, patch_embeddings, list_text_embs, patch_ids, sample_id, caption
        
def img_transform(img):
    np_img = np.array(img)
    torch_input = torch.Tensor(np_img)/255.
    torch_input = torch_input.permute(2,1,0)
    torch_input = torch_input.unsqueeze(0)

    return torch_input
    
def vis_emb_gen(vis_model, vis_input):
    vis_model.eval()
    
    with torch.no_grad():
        emb = vis_model(vis_input.to(device))

    return emb

def center_crop_pil(im, new_size=(1024,1024)):
    width, height = im.size   # Get dimensions
    new_width, new_height = new_size

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    # Crop the center of the image
    im = im.crop((left, top, right, bottom))

    return im

def patch_filter(arr, white_threshold=(235, 235, 235), max_white_ratio=0.95):
    """
    Check if the patch contains more than min_pixel_count purple or pink pixels based on RGB thresholds.
    Purple: High R and B, low G. Pink: High R, moderate G and B.
    """
    pixels = np.array(arr)
    if len(pixels.shape) != 3 or pixels.shape[2] != 3:
        return False  # Not an RGB image
    
    # Identify white pixels (R, G, B all >= threshold)
    white_condition = (
        (pixels[:, :, 0] >= white_threshold[0]) &
        (pixels[:, :, 1] >= white_threshold[1]) &
        (pixels[:, :, 2] >= white_threshold[2])
    )
    
    # Calculate the proportion of white pixels
    white_pixel_count = np.sum(white_condition)
    total_pixels = pixels.shape[0] * pixels.shape[1]
    white_ratio = white_pixel_count / total_pixels
    
    # Return True if white pixels exceed the max_white_ratio
    return white_ratio > max_white_ratio
    
def crop_image_to_patches(vis_model, image_path, patch_size=256):

    pid = image_path.split('/')[-1].split('.')[0]
    print(image_path)

    # Open the image
    try:
        t = openslide.OpenSlide(image_path)
        t_img = t.read_region((0,0),0,t.dimensions).convert('RGB')
    except:
        try:
            t = tifffile.imread(image_path)
            t_img = Image.fromarray(t)
        except:
            try:
                t_img = Image.open(image_path)
                t_res_test = t_img.resize((100,100))
            except:            
                print('Corrupted:', pid)
                return False
    
    l0_len_x = t_img.size[0]
    l0_len_y = t_img.size[1]
    
    down_len_x = round(l0_len_x/(2**L))
    down_len_y = round(l0_len_y/(2**L))

    thum_len_x = thum_x
    thum_len_y = round((down_len_y/down_len_x)*thum_x)
    thum_target_size = ((thum_len_x//16)*16 , (thum_len_y//16)*16)

    
    print('resized :',t_img.size,'->',down_len_x,down_len_y)
    print('thumnail:',(down_len_x,down_len_y),'->',(thum_len_x,thum_len_y))
    print('c_crop  :',(thum_len_x,thum_len_y),thum_target_size)

    resized_img = t_img.resize((down_len_x,down_len_y))
    thumbnail_img = resized_img.resize((thum_len_x,thum_len_y))
    thum_cr_img = center_crop_pil(thumbnail_img, thum_target_size)


    width, height = resized_img.size
       
    # Calculate number of patches
    patches_x = width // patch_size
    patches_y = height // patch_size

    # Crop and save patches
    loc_patch = {}
    list_patch = []
    white_patch = []

    one_pid = {}
    ### Save the thum
    vis_emb = vis_emb_gen(vis_model, img_transform(thum_cr_img)).cpu().squeeze().numpy()
    one_pid['thumbnail'] = vis_emb

    patches_dict = {}
    for y in range(patches_y):
        for x in range(patches_x):
            left = x * patch_size
            upper = y * patch_size
            right = left + patch_size
            lower = upper + patch_size
            
            # Crop the patch
            patch = resized_img.crop((left, upper, right, lower))

            if patch_filter(patch):
                pass
            else:                
                ### Save the patch
                #patch.save(patch_path)
                patch_name = f"{pid}_{left}_{upper}.png"
                vis_emb = vis_emb_gen(vis_model, img_transform(patch)).cpu().squeeze().numpy()
                patches_dict[patch_name] = vis_emb
    
    one_pid['patches'] = patches_dict


    return one_pid


def predict_caption(model: nn.Module, test_loader: DataLoader, text_embeddings: torch.Tensor, train_captions: List, device: torch.device, log: bool = False) -> List[Dict]:
    model.eval()
    predictions = []
    
    # Load train text embeddings for caption matching
    train_text_embeddings = text_embeddings.to(device)
    
    with torch.no_grad():
        for thum_embs, patch_embs, _, sample_ids, captions, patch_ids in test_loader:
            # Encode images through model (handles attention internally)
            (logits_per_thum, logits_per_patch), (_, _), attention_weights = model(thum_embs.to(device), patch_embs, train_text_embeddings)  # Reuse train_text_embeddings for feature extraction
           
            similarities = (logits_per_patch+logits_per_thum)/2
            
            # Get top caption for each image in the batch
            best_indices = similarities.argmax(dim=1)
            for i, best_idx in enumerate(best_indices):
                predicted_caption = train_captions[best_idx.item()]
                predictions.append({
                    'id': sample_ids[i],
                    'report': predicted_caption,
                })
                if log:
                    print('-------', sample_ids[i])
                    print('gth:', captions[i])
                    print('prd:', predicted_caption)
    
    return predictions


def custom_collate_fn(batch):
    thum_embeddings, patch_embeddings, text_embeddings, patches_ids, sample_ids, captions = [], [], [], [], [], []
    
    for thum_emb, patch_emb, list_text_emb, patch_id, sample_id, caption in batch:
        thum_embeddings.append(thum_emb)
        patch_embeddings.append(patch_emb)  # Keep as list of (num_patches, embed_dim)
        #text_embeddings.append(list_text_emb)   # (embed_dim,)
        patches_ids.append(patch_id)       # List of patch IDs
        captions.append(caption)
        sample_ids.append(sample_id)

        #print(list_text_emb[0].shape, list_text_emb[1].shape, list_text_emb[2].shape)
    
    # Stack fixed-size tensors
    thum_embeddings = torch.stack(thum_embeddings)  # (batch_size, embed_dim)
    #text_embeddings = torch.stack(text_embeddings)  # (batch_size, embed_dim)
    
    return thum_embeddings, patch_embeddings, _, sample_ids, captions, patches_ids

if __name__ == "__main__":
    input_path = '/input/images/he-staining'
    output_path = '/output/text-report.json'

    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    list_ori = os.listdir(input_path)

    #'''
    ### vision encoder
    
    uni_enc_path = './pretrained/pytorch_model.bin'
    vis_model = timm.create_model(
        "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
    )
    vis_model = vis_model.to(device)
    
    state_dict = torch.load(uni_enc_path, map_location=torch.device(device))
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    
    vis_model.load_state_dict(state_dict, strict=True)


    total_pids = {}
    for _id in list_ori:
        
        p = os.path.join(input_path, _id)
        print(_id)
        one_pid = crop_image_to_patches(vis_model, p)
        
        total_pids[_id] = one_pid
        
    #'''
    
    test_dataset = PathologyDataset(total_pids)

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=custom_collate_fn
    )
  
    model = PathologyCLIP(embed_dim=512, device=device).to(device)

    ### load model
    model_path = './model_checkpoint.pth'
    loaded_checkpoint = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(loaded_checkpoint['model_state_dict'])
    
    text_embs = loaded_checkpoint['text_embs'].to(device)
    text_capt = loaded_checkpoint['text_capt']

    pred_gth = predict_caption(model, test_loader, text_embs,text_capt, device, log=False)

    with open(output_path, 'w') as f:
        json.dump(pred_gth, f, indent=4)
