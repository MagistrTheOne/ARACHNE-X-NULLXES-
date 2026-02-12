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
- [ ] Define identity token schema and persistence format.
- [ ] Inject identity tokens into avatar DiT conditioning path.
- [ ] Add training/inference controls for identity strength.
- [ ] Add regression metrics for identity consistency.

## Step 3. Phoneme-Conditioned Head
- [ ] Add phoneme timeline extraction/alignment pipeline.
- [ ] Integrate phoneme stream into audio conditioning path.
- [ ] Add phoneme alignment losses and monitoring.
- [ ] Keep wav2vec fallback path for robustness.

## Step 4. Emotion Control Channel
- [ ] Add explicit affective conditioning channel.
- [ ] Add API knobs for emotion class and intensity.
- [ ] Integrate emotion guidance into denoising path.
- [ ] Add quality checks to avoid lip-sync regression.

## Step 5. Hybrid Renderer (Controlled Mouth Zone)
- [ ] Add mouth-zone controlled rendering branch.
- [ ] Add seam-safe blending with global renderer.
- [ ] Add temporal stabilization for mouth boundary.
- [ ] Validate artifact/flicker budget under long sequences.

