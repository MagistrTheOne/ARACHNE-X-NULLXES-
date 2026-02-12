# ARACHNE-X Avatar Upgrade Checklist

Updated: 2026-02-12

## Step 1. Temporal Compression Memory
- [x] Added temporal KV-cache compression in `LongCatVideoAvatarPipeline`.
- [x] Preserved reference latents (`num_ref_latents`) in cache compression path.
- [x] Added sliding recent window + summarized historical memory.
- [x] Switched `generate_avc` to use `effective_num_cond_latents` for KV path.
- [x] Kept output conditioning replacement logic stable for final latents.
- [x] Added metrics for before/after cache latent count.

## Step 2. Identity Token Bank
- [x] Define identity token schema and persistence format.
- [x] Inject identity tokens into avatar DiT conditioning path.
- [x] Add training/inference controls for identity strength.
- [x] Add regression metrics for identity consistency.

## Step 3. Phoneme-Conditioned Head
- [x] Add phoneme timeline extraction/alignment pipeline.
- [x] Integrate phoneme stream into audio conditioning path.
- [x] Add phoneme alignment losses and monitoring.
- [x] Keep wav2vec fallback path for robustness.

## Step 4. Emotion Control Channel
- [x] Add explicit affective conditioning channel.
- [x] Add API knobs for emotion class and intensity.
- [x] Integrate emotion guidance into denoising path.
- [x] Add quality checks to avoid lip-sync regression.

## Step 5. Hybrid Renderer (Controlled Mouth Zone)
- [x] Add mouth-zone controlled rendering branch.
- [x] Add seam-safe blending with global renderer.
- [x] Add temporal stabilization for mouth boundary.
- [x] Validate artifact/flicker budget under long sequences.
