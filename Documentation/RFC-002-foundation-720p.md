# RFC-002 — Remove 480p from the Foundation Video Stack

**Status:** Draft / not started (deferred epic)
**Owner:** NULLXES core
**Depends on:** Avatar runtime 720p canonicalization (shipped — see [`ARCHITECTURE.md` → Resolution & restoration policy](../ARCHITECTURE.md))
**Risk class:** High (latent-cache + checkpoint compatibility)

---

## 1. Context

Avatar runtime is already canonical **720p** (modes `ai2v / at2v / avc / streaming_ai2v / enroll_identity`).
That change was surgical and avatar-only: it never touches the bucket tables, training export, or foundation
video pipelines.

This RFC tracks the *separate, deferred* work of removing 480p from the **foundation video stack**
(`t2v / i2v / vc`, audio-conditioned video adapters, and training/latent export). It is intentionally
NOT bundled with the avatar change because it is destructive to existing artifacts.

**Doctrine restated:** resolution policy ≠ restoration policy. Dropping a *generation* tier (480p) is a
data/compatibility migration, not a quality decision. Upscaling to 1080p+ remains a post-processing concern
(`arachne_x/runtime/frame_post_processing.py`), independent of this RFC.

---

## 2. Current 480p footprint (foundation)

| File | Location | What |
|------|----------|------|
| `arachne_x/utils/bukcet_config.py` | `ASPECT_RATIO_627`, `_F64`, `_F128`, `_F256` + `get_bucket_config()` 480p branch | 480p aspect-ratio bucket tables (shared) |
| `arachne_x/pipeline_arachne_x_video.py` | `generate_i2v`, `generate_vc` (`Literal["480p","720p"]="480p"`) | foundation video entrypoints default 480p |
| `arachne_x/pipeline_audio_i2v.py` | `generate_audio_i2v` (`Literal["480p","720p"]="480p"`) | lab audio→video adapter |
| `arachne_x/training_latent_export.py` | `resolution="480p"` (×2) | latent export default tier |
| `arachne_x/training_latent_export_base.py` | `resolution="480p"` (×2) | latent export base default tier |
| `scripts/infer.py` | `--resolution default="480p" choices=["480p","720p"]` | **shared** CLI (video + avatar) |

> The avatar contour no longer reaches any of these with `480p` — `inference_engine` forces `720p` for avatar
> modes before `get_hw_for_resolution`, and `bucket_config` 480p buckets are simply unreachable from avatar.

---

## 3. Why this is high-risk (do NOT just delete)

1. **Latent caches.** Any latents exported via `training_latent_export*` at 480p encode the
   `ASPECT_RATIO_627*` geometry. Deleting the tables makes those caches unreadable and silently changes
   bucket selection for re-exports.
2. **Checkpoint geometry.** Weights/LoRAs trained on 480p buckets assume those resolutions; a hard removal
   without a retrain/compat shim can degrade or break foundation `t2v/i2v/vc`.
3. **Shared CLI default.** `scripts/infer.py --resolution` is shared with avatar modes. Flipping its global
   default affects foundation behavior; it must be decoupled or both domains move together.
4. **No silent fallback.** Removing 480p must `raise` on a 480p request (loud), never re-bucket to the
   nearest tier behind the operator's back.

---

## 4. Goals / non-goals

**Goals**
- Single canonical generation tier (720p) across foundation video, matching avatar runtime.
- Deterministic, loud failure for any residual 480p request.
- A documented compatibility matrix for existing latents/checkpoints.

**Non-goals**
- Upscaling/restoration (owned by `frame_post_processing.py`).
- Avatar runtime changes (already shipped).

---

## 5. Migration plan (phased)

### Phase A — Inventory & freeze
- Enumerate all 480p latent caches and the checkpoints/LoRAs trained against them.
- Tag a compatibility matrix: `{artifact → bucket tier → still needed?}`.
- Stop producing *new* 480p latents: flip `training_latent_export*` defaults to `720p` (additive, non-breaking
  for reads).

### Phase B — Read-compat shim
- Keep `ASPECT_RATIO_627*` tables behind a `legacy_480p_buckets=True` read path; new exports never select them.
- `get_bucket_config('480p', ...)` emits a deprecation warning (still functional) for one release.

### Phase C — Re-export / retrain
- Re-export required 480p datasets at 720p; retrain/re-fit affected checkpoints or confirm they are 720p-native.
- Validate foundation `t2v/i2v/vc` parity at 720p.

### Phase D — Removal (the destructive step)
- Delete `ASPECT_RATIO_627*` and the `480p` branch; `get_bucket_config('480p', ...)` raises `ValueError`.
- Foundation pipelines: `Literal["480p","720p"]="480p"` → `Literal["720p"]="720p"`.
- `scripts/infer.py`: drop `480p` from `choices`, default `720p`.
- Grep gate in CI: no `480p` / `ASPECT_RATIO_627` references remain.

---

## 6. Rollback

Phases A–C are reversible (additive defaults + shims). Phase D is the irreversible cut — gate it behind an
explicit go and a tagged release so the `627` tables can be restored from VCS if a latent dependency surfaces.

---

## 7. Decision required

- [ ] Confirm 480p latents/checkpoints that must survive (drives Phase C cost).
- [ ] Approve flipping `training_latent_export*` defaults to 720p (Phase A, low risk).
- [ ] Schedule Phase D against a release tag.
