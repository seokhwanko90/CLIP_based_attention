import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple

torch.manual_seed(42)

# Attention module for patch embeddings
class PatchAttention(nn.Module):
    def __init__(self, embed_dim: int):
        super(PatchAttention, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = 1.0 / np.sqrt(embed_dim)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, patch_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        patch_embeddings: (batch_size, num_patches, embed_dim)
        Returns: weighted embedding (batch_size, embed_dim), attention weights (batch_size, num_patches)
        """
        query = self.query(patch_embeddings)  # (batch_size, num_patches, embed_dim)
        key = self.key(patch_embeddings)      # (batch_size, num_patches, embed_dim)
        value = self.value(patch_embeddings)  # (batch_size, num_patches, embed_dim)
        
        # Compute attention scores
        scores = torch.bmm(query, key.transpose(-2, -1)) * self.scale  # (batch_size, num_patches, num_patches)
        attention_weights = self.softmax(scores)  # (batch_size, num_patches, num_patches)
        
        # Compute weighted sum of values
        weighted_embedding = torch.bmm(attention_weights, value)  # (batch_size, num_patches, embed_dim)
        weighted_embedding = weighted_embedding.mean(dim=1)       # (batch_size, embed_dim)
        
        return weighted_embedding, attention_weights.mean(dim=1)  # (batch_size, embed_dim), (batch_size, num_patches)

# CLIP-like model architecture with integrated patch attention
class PathologyCLIP(nn.Module):
    def __init__(self, embed_dim: int = 512, device: str = 'cuda'):
        super(PathologyCLIP, self).__init__()

        self.device = device
        # Patch attention module
        self.patch_attention = PatchAttention(embed_dim=1024)
        
        # Image encoder (for weighted patch embeddings)
        self.patch_encoder = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, embed_dim),
            nn.ReLU()
        )
        
        # Text encoder
        self.text_encoder = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, embed_dim),
            nn.ReLU()
        )
        # Text encoder
        self.text_orgp_encoder = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, embed_dim),
            nn.ReLU()
        )
        # Text encoder
        self.text_hist_encoder = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, embed_dim),
            nn.ReLU()
        )

        self.thum_encoder = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, embed_dim),
            nn.ReLU()
        )
        
        # Temperature parameter for contrastive loss
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def forward(self, thum_embeddings: torch.Tensor, patch_embeddings: List[torch.Tensor], text_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        patch_embeddings: List of (num_patches, embed_dim) tensors for each sample in batch
        text_embeddings: (batch_size, embed_dim)
        Returns: logits_per_image, logits_per_text, attention_weights (list of (num_patches,) tensors)
        """
        # Process patch embeddings through attention
        weighted_embeddings = []
        attention_weights_list = []
        for patches in patch_embeddings:
            patches = patches.unsqueeze(0)  # (1, num_patches, embed_dim)
            weighted_emb, attn_weights = self.patch_attention(patches.to(self.device))  # (1, embed_dim), (1, num_patches)           
            weighted_embeddings.append(weighted_emb.squeeze(0))  # (embed_dim,)
            attention_weights_list.append(attn_weights.squeeze(0))  # (num_patches,)
        
        weighted_embeddings = torch.stack(weighted_embeddings)  # (batch_size, embed_dim)
        #print(weighted_embeddings.shape)

        # Encode images and texts
        patch_features = self.patch_encoder(weighted_embeddings)
        thum_features = self.thum_encoder(thum_embeddings)
        
        text_features = self.text_encoder(text_embeddings[:,0])
        text_orgp_features = self.text_encoder(text_embeddings[:,1])
        text_hist_features = self.text_encoder(text_embeddings[:,2])

        
        # Normalize features
        patch_features = patch_features / patch_features.norm(dim=-1, keepdim=True)
        thum_features = thum_features / thum_features.norm(dim=-1, keepdim=True)
        
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_orgp_features = text_orgp_features / text_orgp_features.norm(dim=-1, keepdim=True)
        text_hist_features = text_hist_features / text_hist_features.norm(dim=-1, keepdim=True)

        # Compute logits
        logit_scale = self.logit_scale.exp()
        
        #logits_per_image = logit_scale * image_features @ text_features.t()
        #logits_per_text = logits_per_image.t()

        logits_per_patch = logit_scale * patch_features @ text_hist_features.t()
        logits_per_text_p = logits_per_patch.t()

        logits_per_thum = logit_scale * thum_features @ text_orgp_features.t()
        logits_per_text_t = logits_per_thum.t()
        
        return (logits_per_thum, logits_per_patch), (logits_per_text_t, logits_per_text_p), attention_weights_list
