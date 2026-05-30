"""
Shared DiT runtime machinery for the NULLXES / ARACHNE-X transformers.

``DiTLoRABSAMixin`` holds the LoRA attach/enable/disable lifecycle, the BSA
toggle, and ``unpatchify`` — logic that is identical between the base video DiT
(:mod:`arachne_x.modules.arachne_video_dit`) and the avatar DiT
(:mod:`arachne_x.modules.avatar.arachne_avatar_dit`).

The mixin defines no ``__init__`` and registers no parameters/buffers, so it
does not affect ``state_dict`` keys, ``register_to_config``, or the
``ModelMixin`` / ``ConfigMixin`` MRO. Concrete classes must initialize
``self.lora_dict``, ``self.active_loras``, ``self.blocks``, ``self.patch_size``
and ``self.out_channels`` in their own ``__init__``.
"""

from __future__ import annotations

import loguru
from einops import rearrange
from safetensors.torch import load_file

from .lora_utils import create_lora_network


class DiTLoRABSAMixin:
    """LoRA lifecycle + BSA toggle + unpatchify shared across DiT variants."""

    def load_lora(self, lora_path, lora_key, multiplier=1.0, lora_network_dim=128, lora_network_alpha=64):
        lora_network_state_dict_loaded = load_file(lora_path, device="cpu")
        lora_network = create_lora_network(
            transformer=self,
            lora_network_state_dict_loaded=lora_network_state_dict_loaded,
            multiplier=multiplier,
            network_dim=lora_network_dim,
            network_alpha=lora_network_alpha,
        )

        incompatible = lora_network.load_state_dict(
            lora_network_state_dict_loaded, strict=False
        )
        if incompatible.missing_keys or incompatible.unexpected_keys:
            loguru.logger.warning(
                "LoRA load_state_dict non-strict for key={}: missing={} unexpected={}",
                lora_key,
                len(incompatible.missing_keys),
                len(incompatible.unexpected_keys),
            )

        self.lora_dict[lora_key] = lora_network

    def enable_loras(self, lora_key_list=None):
        if lora_key_list is None:
            lora_key_list = []
        self.disable_all_loras()

        module_loras = {}  # {module_name: [lora1, lora2, ...]}
        model_device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype

        for lora_key in lora_key_list:
            if lora_key in self.lora_dict:
                for lora in self.lora_dict[lora_key].loras:
                    lora.to(model_device, dtype=model_dtype, non_blocking=True)
                    module_name = lora.lora_name.replace("lora___lorahyphen___", "").replace("___lorahyphen___", ".")
                    if module_name not in module_loras:
                        module_loras[module_name] = []
                    module_loras[module_name].append(lora)
                self.active_loras.append(lora_key)

        for module_name, loras in module_loras.items():
            module = self._get_module_by_name(module_name)
            if not hasattr(module, 'org_forward'):
                module.org_forward = module.forward
            module.forward = self._create_multi_lora_forward(module, loras)

    def _create_multi_lora_forward(self, module, loras):
        def multi_lora_forward(x, *args, **kwargs):
            weight_dtype = x.dtype
            org_output = module.org_forward(x, *args, **kwargs)

            total_lora_output = 0
            for lora in loras:
                if lora.use_lora:
                    lx = lora.lora_down(x.to(lora.lora_down.weight.dtype))
                    lx = lora.lora_up(lx)
                    lora_output = lx.to(weight_dtype) * lora.multiplier * lora.alpha_scale
                    total_lora_output += lora_output

            return org_output + total_lora_output

        return multi_lora_forward

    def _get_module_by_name(self, module_name):
        try:
            module = self
            for part in module_name.split('.'):
                module = getattr(module, part)
            return module
        except AttributeError as e:
            raise ValueError(f"Cannot find module: {module_name}, error: {e}")

    def disable_all_loras(self):
        for name, module in self.named_modules():
            if hasattr(module, 'org_forward'):
                module.forward = module.org_forward
                delattr(module, 'org_forward')

        for lora_key, lora_network in self.lora_dict.items():
            for lora in lora_network.loras:
                lora.to("cpu")

        self.active_loras.clear()

    def enable_bsa(self):
        for block in self.blocks:
            block.attn.enable_bsa = True

    def disable_bsa(self):
        for block in self.blocks:
            block.attn.enable_bsa = False

    def unpatchify(self, x, N_t, N_h, N_w):
        """
        Args:
            x (torch.Tensor): of shape [B, N, C]

        Return:
            x (torch.Tensor): of shape [B, C_out, T, H, W]
        """
        T_p, H_p, W_p = self.patch_size
        x = rearrange(
            x,
            "B (N_t N_h N_w) (T_p H_p W_p C_out) -> B C_out (N_t T_p) (N_h H_p) (N_w W_p)",
            N_t=N_t,
            N_h=N_h,
            N_w=N_w,
            T_p=T_p,
            H_p=H_p,
            W_p=W_p,
            C_out=self.out_channels,
        )
        return x
