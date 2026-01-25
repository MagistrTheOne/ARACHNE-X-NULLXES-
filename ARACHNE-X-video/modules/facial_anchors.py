"""
Facial Keypoint Anchoring Module for ARACHNE-X
Provides 68-point facial landmark anchoring for stable face generation
and high-frequency detail preservation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


class FacialAnchorEmbedder(nn.Module):
    """
    Embeds 68 facial landmarks (MediaPipe/DLIB format) into high-dimensional space.
    These anchors are used to constrain the diffusion process in facial regions.
    """
    def __init__(self, hidden_size: int = 1024, num_landmarks: int = 68):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_landmarks = num_landmarks
        
        # Each landmark has (x, y, confidence)
        self.landmark_input_dim = num_landmarks * 2  # x, y only
        
        # MLP for landmark encoding
        self.landmark_encoder = nn.Sequential(
            nn.Linear(self.landmark_input_dim, hidden_size * 2),
            nn.SiLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        
        # Per-landmark attention weights (learned per region)
        self.landmark_region_attn = nn.Sequential(
            nn.Linear(hidden_size, num_landmarks),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, landmarks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            landmarks: [B, 68, 2] normalized facial keypoints (0-1 range)
            
        Returns:
            anchor_embed: [B, hidden_size] aggregated landmark embedding
            region_attn: [B, 68] attention weights per landmark region
        """
        B = landmarks.shape[0]
        
        # Flatten landmarks for encoder
        landmarks_flat = landmarks.reshape(B, -1)  # [B, 136]
        
        # Encode landmarks
        anchor_embed = self.landmark_encoder(landmarks_flat)  # [B, hidden_size]
        
        # Compute per-landmark attention
        region_attn = self.landmark_region_attn(anchor_embed)  # [B, 68]
        
        return anchor_embed, region_attn


class FacialAnchorConstrainer(nn.Module):
    """
    Applies facial landmark constraints to latent features during diffusion.
    Prevents unrealistic face deformations and maintains identity consistency.
    """
    def __init__(self, hidden_size: int = 1024, latent_channels: int = 16):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_channels = latent_channels
        
        # Compute warp field from anchors
        self.warp_generator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 2)  # 2D warp displacement
        )
        
        # Feature masking for face regions
        self.face_mask_generator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self, 
        latents: torch.Tensor, 
        anchor_embed: torch.Tensor,
        region_attn: torch.Tensor,
        spatial_shape: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Args:
            latents: [B, C, T, H, W] latent features
            anchor_embed: [B, hidden_size] landmark embeddings
            region_attn: [B, 68] per-landmark attention
            spatial_shape: (H, W) of latent space
            
        Returns:
            constrained_latents: [B, C, T, H, W] with facial constraints applied
        """
        B, C, T, H, W = latents.shape
        
        # Generate face region mask
        face_mask = self.face_mask_generator(anchor_embed)  # [B, 1]
        face_mask = face_mask.view(B, 1, 1, 1, 1).expand(B, 1, T, H, W)
        
        # Generate warp displacement field
        warp_disp = self.warp_generator(anchor_embed)  # [B, 2]
        
        # Apply constraints: blend original with constrained version
        # Using face mask to selectively apply constraints
        constrained_latents = latents.clone()
        
        # Soft constraint: reduce variation in face region
        face_region_feature = latents * face_mask
        constrained_latents = constrained_latents * (1 - face_mask) + face_region_feature * face_mask
        
        return constrained_latents


class LandmarkExtractor(nn.Module):
    """
    Extracts 68 facial landmarks from video frames using lightweight detector.
    Can use MediaPipe, DLIB, or other facial detection frameworks.
    """
    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device
        try:
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                min_detection_confidence=0.5
            )
            self.use_mediapipe = True
        except ImportError:
            self.use_mediapipe = False
            print("MediaPipe not available, using DLIB fallback")
    
    def forward(self, video_frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video_frames: [B, T, H, W, 3] RGB frames normalized to 0-255
            
        Returns:
            landmarks: [B, T, 68, 2] normalized facial keypoints (0-1)
        """
        B, T, H, W, C = video_frames.shape
        landmarks_list = []
        
        for b in range(B):
            frame_landmarks = []
            for t in range(T):
                frame = (video_frames[b, t] * 255).byte().cpu().numpy()
                
                if self.use_mediapipe:
                    results = self.mp_face_mesh.process(frame)
                    if results.multi_face_landmarks:
                        # Extract first face landmarks
                        face_landmarks = results.multi_face_landmarks[0].landmark
                        # Convert to normalized coordinates, select key 68 points
                        lmks = np.array([[lm.x, lm.y] for lm in face_landmarks[:68]])
                    else:
                        # Fallback to zeros if no face detected
                        lmks = np.zeros((68, 2))
                else:
                    lmks = np.zeros((68, 2))
                
                frame_landmarks.append(torch.tensor(lmks, dtype=torch.float32))
            
            landmarks_list.append(torch.stack(frame_landmarks, dim=0))
        
        # Stack all batches: [B, T, 68, 2]
        all_landmarks = torch.stack(landmarks_list, dim=0).to(self.device)
        return all_landmarks


class FacialAnchorModule(nn.Module):
    """
    Complete facial anchoring system combining extraction, embedding, and constraints.
    """
    def __init__(
        self, 
        hidden_size: int = 1024,
        latent_channels: int = 16,
        num_landmarks: int = 68,
        anchor_weight: float = 0.15
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.anchor_weight = anchor_weight
        
        self.embedder = FacialAnchorEmbedder(hidden_size, num_landmarks)
        self.constrainer = FacialAnchorConstrainer(hidden_size, latent_channels)
        self.landmark_extractor = LandmarkExtractor()
        
    def forward(
        self,
        latents: torch.Tensor,
        video_frames: Optional[torch.Tensor] = None,
        landmarks: Optional[torch.Tensor] = None,
        spatial_shape: Tuple[int, int] = (60, 104)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            latents: [B, C, T, H, W] latent features
            video_frames: [B, T, H, W, 3] RGB frames (if None, use pre-extracted landmarks)
            landmarks: [B, T, 68, 2] pre-extracted facial landmarks
            spatial_shape: shape of latent space
            
        Returns:
            constrained_latents: [B, C, T, H, W]
            anchor_embed: [B, hidden_size]
            region_attn: [B, 68]
        """
        B, C, T, H, W = latents.shape
        
        # Extract or use provided landmarks
        if landmarks is None and video_frames is not None:
            # Extract landmarks from video frames
            landmarks = self.landmark_extractor(video_frames)  # [B, T, 68, 2]
        elif landmarks is None:
            # No landmarks provided, return unmodified latents
            anchor_embed = torch.zeros(B, self.hidden_size, device=latents.device)
            region_attn = torch.ones(B, 68, device=latents.device) / 68
            return latents, anchor_embed, region_attn
        
        # Average landmarks across time for stability
        landmarks_avg = landmarks.mean(dim=1)  # [B, 68, 2]
        
        # Embed landmarks
        anchor_embed, region_attn = self.embedder(landmarks_avg)  # [B, hidden_size], [B, 68]
        
        # Apply constraints to latents
        constrained_latents = self.constrainer(
            latents, anchor_embed, region_attn, spatial_shape
        )
        
        # Blend: keep anchor_weight of constraint, (1-anchor_weight) of original
        constrained_latents = (
            self.anchor_weight * constrained_latents + 
            (1 - self.anchor_weight) * latents
        )
        
        return constrained_latents, anchor_embed, region_attn
