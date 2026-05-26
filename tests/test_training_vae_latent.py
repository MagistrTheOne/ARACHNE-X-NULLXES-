"""Unit tests for VAE latent normalize/denorm parity with pipeline."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import torch

from arachne_x.training_vae_latent import (
    denormalize_vae_latents,
    estimate_z0_from_flow_match,
    normalize_vae_latents,
)


class _FakeVaeConfig:
    z_dim = 2
    latents_mean = [0.1, -0.2]
    latents_std = [0.5, 1.25]


class VaeLatentRoundtripTests(unittest.TestCase):
    def test_normalize_denormalize_roundtrip(self):
        vae = MagicMock()
        vae.config = _FakeVaeConfig()
        z = torch.randn(1, 2, 3, 4, 4)
        recovered = denormalize_vae_latents(vae, normalize_vae_latents(vae, z))
        self.assertTrue(torch.allclose(z.float(), recovered, atol=1e-5))

    def test_denorm_matches_pipeline_formula(self):
        vae = MagicMock()
        vae.config = _FakeVaeConfig()
        norm = torch.tensor([1.0, 2.0]).view(1, 2, 1, 1, 1)
        out = denormalize_vae_latents(vae, norm)
        expected = norm * torch.tensor([0.5, 1.25]).view(1, 2, 1, 1, 1) + torch.tensor(
            [0.1, -0.2]
        ).view(1, 2, 1, 1, 1)
        self.assertTrue(torch.allclose(out, expected))


class Z0EstimateTests(unittest.TestCase):
    def test_flow_match_inversion(self):
        scheduler = MagicMock()
        scheduler.timesteps = torch.tensor([1000.0, 500.0, 0.0])
        scheduler.sigmas = torch.tensor([1.0, 0.5, 0.0])
        scheduler.index_for_timestep = lambda t, schedule_t: 1

        z0 = torch.ones(1, 4, 2, 8, 8)
        eps = torch.full_like(z0, 0.5)
        sigma = 0.5
        x_t = sigma * eps + (1.0 - sigma) * z0
        t = torch.tensor([500.0])

        z0_hat = estimate_z0_from_flow_match(x_t, eps, t, scheduler)
        self.assertTrue(torch.allclose(z0_hat, z0, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
