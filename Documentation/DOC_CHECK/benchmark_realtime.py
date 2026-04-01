#!/usr/bin/env python3
"""
ARACHNE-X Real-Time Performance Benchmark
Measures FPS, latency, memory usage on H200
"""

import time
import torch
import numpy as np
from typing import Dict, List
import statistics

from arachne_x.loader import load_avatar_pipeline


class RealtimeBenchmark:
    """Benchmark real-time inference performance."""
    
    def __init__(self, pipeline, device: str = 'cuda'):
        self.pipeline = pipeline
        self.device = device
        self.metrics: Dict[str, List[float]] = {
            'frame_times': [],
            'denoise_times': [],
            'vae_decode_times': [],
            'audio_encode_times': [],
            'total_times': [],
        }
    
    def benchmark_single_frame(self, 
                               latents: torch.Tensor,
                               audio_emb: torch.Tensor,
                               prompt_embeds: torch.Tensor,
                               prompt_mask: torch.Tensor,
                               timestep: torch.Tensor,
                               num_runs: int = 10) -> Dict[str, float]:
        """
        Benchmark single frame generation.
        
        Returns:
            {
                'avg_fps': float,
                'p95_latency_ms': float,
                'p99_latency_ms': float,
                'min_latency_ms': float,
                'max_latency_ms': float,
            }
        """
        
        frame_times = []
        
        # Warmup (Avatar DiT: num_cond_latents=1 for image conditioning)
        with torch.inference_mode():
            for _ in range(2):
                _ = self.pipeline.dit(
                    hidden_states=latents,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_mask,
                    num_cond_latents=1,
                    audio_embs=audio_emb
                )
        
        # Benchmark
        torch.cuda.synchronize(self.device)
        
        for _ in range(num_runs):
            start = time.perf_counter()
            
            with torch.inference_mode():
                _ = self.pipeline.dit(
                    hidden_states=latents,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_mask,
                    num_cond_latents=1,
                    audio_embs=audio_emb
                )
            
            torch.cuda.synchronize(self.device)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            frame_times.append(elapsed)
        
        sorted_times = sorted(frame_times)
        avg_time = statistics.mean(frame_times)
        
        return {
            'avg_fps': 1000.0 / avg_time,
            'avg_latency_ms': avg_time,
            'p95_latency_ms': sorted_times[int(len(sorted_times) * 0.95)],
            'p99_latency_ms': sorted_times[int(len(sorted_times) * 0.99)],
            'min_latency_ms': min(frame_times),
            'max_latency_ms': max(frame_times),
        }
    
    def benchmark_streaming_pipeline(self,
                                     num_frames: int = 24,
                                     latents_shape: tuple = (1, 16, 24, 60, 104),
                                     ) -> Dict[str, float]:
        """
        Benchmark full pipeline: denoise + VAE decode per frame.
        Uses Avatar DiT audio_embs format [B, T, W, S, C].
        """
        num_latent = latents_shape[2]
        latents = torch.randn(latents_shape, dtype=torch.bfloat16, device=self.device)
        audio_emb = torch.randn(1, num_latent, 5, 12, 768, dtype=torch.bfloat16, device=self.device)
        prompt_embeds = torch.randn(1, 1, 512, 768, dtype=torch.bfloat16, device=self.device)
        prompt_mask = torch.ones(1, 512, dtype=torch.int64, device=self.device)
        timestep = torch.tensor([500.0], dtype=torch.bfloat16, device=self.device)
        
        # Full pipeline
        frame_times = []
        torch.cuda.synchronize(self.device)
        
        start_total = time.perf_counter()
        
        for frame_idx in range(num_frames):
            frame_start = time.perf_counter()
            
            with torch.inference_mode():
                # Denoise step (Avatar: num_cond_latents=1)
                _ = self.pipeline.dit(
                    hidden_states=latents,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_mask,
                    num_cond_latents=1,
                    audio_embs=audio_emb
                )
                
                # VAE decode
                _ = self.pipeline.vae.decode(
                    latents[:, :, frame_idx:frame_idx+1],
                    return_dict=False
                )
            
            torch.cuda.synchronize(self.device)
            frame_time = (time.perf_counter() - frame_start) * 1000
            frame_times.append(frame_time)
        
        total_time = (time.perf_counter() - start_total)
        
        sorted_times = sorted(frame_times)
        
        return {
            'num_frames': num_frames,
            'total_time_sec': total_time,
            'avg_fps': num_frames / total_time,
            'avg_frame_latency_ms': statistics.mean(frame_times),
            'p95_frame_latency_ms': sorted_times[int(len(sorted_times) * 0.95)],
            'p99_frame_latency_ms': sorted_times[int(len(sorted_times) * 0.99)],
            'min_frame_latency_ms': min(frame_times),
            'max_frame_latency_ms': max(frame_times),
        }
    
    def benchmark_memory_usage(self) -> Dict[str, int]:
        """Get peak memory usage."""
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.synchronize(self.device)
        
        latents = torch.randn(1, 16, 24, 60, 104, dtype=torch.bfloat16, device=self.device)
        
        with torch.inference_mode():
            _ = self.pipeline.vae.decode(latents, return_dict=False)
        
        torch.cuda.synchronize(self.device)
        
        peak_memory = torch.cuda.max_memory_allocated(self.device)
        current_memory = torch.cuda.memory_allocated(self.device)
        
        return {
            'peak_memory_gb': peak_memory / (1024**3),
            'current_memory_gb': current_memory / (1024**3),
        }


def run_benchmark(pipeline, hardware: str = "H200") -> Dict:
    """Run full benchmark suite."""
    
    print(f"\n{'='*60}")
    print(f"ARACHNE-X Real-Time Benchmark ({hardware})")
    print(f"{'='*60}\n")
    
    benchmark = RealtimeBenchmark(pipeline)
    
    # Single frame benchmark (Avatar DiT expects audio_embs [B, T, W, S, C])
    print("[*] Benchmarking single frame denoise...")
    latents_shape = (1, 16, 24, 60, 104)  # B, C, T_latent, H, W
    num_latent = latents_shape[2]
    latents = torch.randn(*latents_shape, dtype=torch.bfloat16, device=pipeline.device)
    audio_emb = torch.randn(1, num_latent, 5, 12, 768, dtype=torch.bfloat16, device=pipeline.device)
    prompt_embeds = torch.randn(1, 1, 512, 768, dtype=torch.bfloat16, device=pipeline.device)
    prompt_mask = torch.ones(1, 512, dtype=torch.int64, device=pipeline.device)
    timestep = torch.tensor([500.0], dtype=torch.bfloat16, device=pipeline.device)
    
    single_frame_metrics = benchmark.benchmark_single_frame(
        latents, audio_emb, prompt_embeds, prompt_mask, timestep, num_runs=10
    )
    
    print(f"  ✓ Single frame:")
    print(f"    FPS: {single_frame_metrics['avg_fps']:.1f}")
    print(f"    Latency: {single_frame_metrics['avg_latency_ms']:.2f}ms")
    print(f"    P95: {single_frame_metrics['p95_latency_ms']:.2f}ms")
    print(f"    P99: {single_frame_metrics['p99_latency_ms']:.2f}ms")
    
    # Full pipeline benchmark (short)
    print("\n[*] Benchmarking full pipeline (10 frames)...")
    pipeline_metrics = benchmark.benchmark_streaming_pipeline(num_frames=10)
    
    print(f"  ✓ Pipeline metrics:")
    print(f"    FPS: {pipeline_metrics['avg_fps']:.1f}")
    print(f"    Latency: {pipeline_metrics['avg_frame_latency_ms']:.2f}ms")
    print(f"    P95: {pipeline_metrics['p95_frame_latency_ms']:.2f}ms")
    print(f"    Total time: {pipeline_metrics['total_time_sec']:.2f}s")
    
    # Memory benchmark
    print("\n[*] Benchmarking memory usage...")
    memory_metrics = benchmark.benchmark_memory_usage()
    
    print(f"  ✓ Memory:")
    print(f"    Peak: {memory_metrics['peak_memory_gb']:.1f} GB")
    print(f"    Current: {memory_metrics['current_memory_gb']:.1f} GB")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Target FPS: 30 | Achieved FPS: {pipeline_metrics['avg_fps']:.1f}")
    print(f"Latency Budget (33ms) | Actual: {pipeline_metrics['avg_frame_latency_ms']:.2f}ms")
    
    if pipeline_metrics['avg_fps'] >= 30:
        print("✓ REAL-TIME CAPABLE ✓")
    elif pipeline_metrics['avg_fps'] >= 15:
        print("✓ NEAR REAL-TIME (~15 FPS)")
    else:
        print("⚠ OFFLINE MODE (<15 FPS)")
    
    print(f"{'='*60}\n")
    
    return {
        'single_frame': single_frame_metrics,
        'pipeline': pipeline_metrics,
        'memory': memory_metrics,
    }


if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser(description="ARACHNE-X Benchmark")
    parser.add_argument('--checkpoint_dir', type=str, default='./weights/ARACHNE-X-Avatar')
    parser.add_argument('--hardware', type=str, default='H200', choices=['H200', 'H100', 'A100'])
    parser.add_argument('--no_load', action='store_true', help='Skip pipeline load (for testing with mock)')
    args = parser.parse_args()

    if not args.no_load:
        if not torch.cuda.is_available():
            print("[!] CUDA not available. Benchmark requires GPU.")
            exit(1)
        if not os.path.isdir(args.checkpoint_dir):
            print(f"[!] Checkpoint not found: {args.checkpoint_dir}")
            print("    Run with --checkpoint_dir pointing to ARACHNE-X-Avatar weights.")
            exit(1)
        print("[*] Loading models...")
        pipeline = load_avatar_pipeline(args.checkpoint_dir, device="cuda")
        print("[*] Ready for benchmark")
        run_benchmark(pipeline, hardware=args.hardware)
    else:
        print("[*] Skipping load (--no_load). Provide pipeline manually for run_benchmark().")

