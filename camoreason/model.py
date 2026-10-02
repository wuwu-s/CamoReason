"""CamoReason assembly.

Frozen: the SAM3 image encoder (ViT trunk + FPN neck). Trainable:
  - image_encoder...trunk.adapter.*   MultimodalAdapter — the complete
                                      adaptation package: SemanticProjector
                                      (z+ / z-), frequency branch + band
                                      selector, stage/boundary/texture tokens,
                                      32x BottleneckAdapter + 32x FiLM head,
                                      RegionAdaptiveFusion (3-modality
                                      gating, paper Eq.16-23)
  - mask_decoder / pe_layer / no_mask_embed
                                      SAM3 MaskDecoder (K=3), fine-tuned as the
                                      task head (SAM-Adapter / MSA practice)

Weight loading happens OUTSIDE this module (train.py mirrors the original
repo's remap: detector.backbone.* -> image_encoder.*, tracker.sam_mask_decoder.*
-> mask_decoder.*, pe_layer / no_mask_embed), so building the model is pure
structure. At eval time the full state_dict comes from model_epoch_best.pth.

forward(image, v, t_pos, t_neg, gt=None): with gt -> returns the total loss
(components stored in self.loss_log, original-repo convention); without gt ->
returns the output dict (logits/iou/...). infer() returns the upsampled
logits of the quality-selected hypothesis k+ = argmax_k q_hat — callers apply
torch.sigmoid, exactly like the original repo's eval_psnr.
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from camoreason.adapter import MultimodalAdapter
from camoreason.losses import (
    gate_reg,
    info_nce,
    iou_of,
    seg_loss_per_mask,
)


class CamoReason(nn.Module):
    def __init__(
        self,
        inp_size=1008,
        use_adapter=True,
        d_multi=512,
        vit_depth=32,
        vit_stages=4,
        bottleneck=32,
        film_hidden=256,
        ff_ch=128,
        n_fpn_levels=3,
        lambda_iou=1.0,
        lambda_con=0.1,
        lambda_gate=0.1,
        lambda_tv=0.1,
        tau=0.07,
        focal_alpha=0.25,
        focal_gamma=2.0,
        ablation=None,
        r_freq=0.25,
        prompt_type='high',
        use_hp=False,
    ):
        super().__init__()
        self.use_adapter = use_adapter
        # ablation switches (paper Table 2 + Table 3): None / 'no_image' / 'no_freq' /
        # 'no_gate' / 'no_text_freq' (single-modal: visual branch only, Table 3)
        assert ablation in (None, 'no_image', 'no_freq', 'no_gate', 'no_text_freq'), ablation
        self.ablation = ablation
        self.inp_size = inp_size
        self.grid = inp_size // 14  # patch grid (72 for 1008)
        self.lambda_iou = lambda_iou
        self.lambda_con = lambda_con
        self.lambda_gate = lambda_gate
        self.lambda_tv = lambda_tv
        self.tau = tau
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.loss_log = {}

        from models.sam3.build_sam3 import build_sam3_image_encoder_only

        # frozen SAM3 encoder. use_adapter=True attaches the complete multimodal
        # adapter (semantic projection, in-trunk injection, region-modal fusion,
        # mask-text projection) to the trunk. use_adapter=False leaves
        # trunk.adapter = None -> the ViT runs the native SAM3 path and only the
        # mask decoder is trainable: the baseline_sam3 protocol.
        self.image_encoder = build_sam3_image_encoder_only()
        if not use_adapter:
            # The decoder-only SAM3 baseline is trained in
            # /root/SAM3-Adapter-Pytorch (official SAM1 recipe), not through
            # this class — fail with guidance instead.
            raise NotImplementedError(
                "use_adapter=False (baseline_sam3 protocol) is not implemented "
                "in camoreason.model; train the decoder-only baseline with "
                "/root/SAM3-Adapter-Pytorch (run_baseline.sh / ft_train.py)."
            )
        if use_adapter:
            trunk = self.image_encoder.vision_backbone.trunk
            trunk.adapter = MultimodalAdapter(
                img_size=inp_size,
                embed_dim=trunk.blocks[0].norm1.normalized_shape[0],
                d_multi=d_multi,
                depth=vit_depth,
                n_stages=vit_stages,
                bottleneck=bottleneck,
                film_hidden=film_hidden,
                ff_ch=ff_ch,
                n_fpn_levels=n_fpn_levels,
                use_freq=ablation not in ('no_freq', 'no_text_freq'),
                use_image=ablation != 'no_image',
                use_text=ablation != 'no_text_freq',
                fusion_mode='fixed' if ablation == 'no_gate' else 'gate',
                r_freq=r_freq,
                prompt_type=prompt_type,
                use_hp=use_hp,
            )

        # SAM3 MaskDecoder (K=3 mask tokens + IoU token), pretrained weights
        # arrive through train.py's remap (tracker.sam_mask_decoder.*)
        from models.mmseg.models.sam import MaskDecoder, TwoWayTransformer
        from models.sam3.sam.prompt_encoder import PositionEmbeddingRandom

        self.mask_decoder = MaskDecoder(
            transformer_dim=256,
            transformer=TwoWayTransformer(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8),
            num_multimask_outputs=3,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=True,
            pred_obj_scores=True,
            pred_obj_scores_mlp=True,
        )
        self.pe_layer = PositionEmbeddingRandom(128)
        self.no_mask_embed = nn.Embedding(1, 256)

    # ------------------------------------------------------------------ #
    def forward(self, image, v, t_pos, t_neg, gt=None):
        """image: [B,3,S,S] (SAM3-normalized); v/t_pos/t_neg: [B,768] cached CLIP
        embeddings; gt: [B,1,S,S] {0,1} (optional)."""
        B = image.shape[0]
        adapter = self.image_encoder.vision_backbone.trunk.adapter
        z_pos, z_neg = adapter.project_semantic(v, t_pos, t_neg)

        adapter.set_conditioning(z_pos)
        feat = self.image_encoder(image)  # dict: vision_features / vision_pos_enc / backbone_fpn
        V_list = feat["backbone_fpn"]  # 3 levels, 256ch: 288 / 144 / 72

        F_fuse, alphas = adapter.fuse(V_list, z_pos)

        # two-way transformer decoder, K=3 mask tokens + IoU token
        image_emb = F_fuse[2]  # 72x72 (stride-14 level)
        image_pe = self.pe_layer((self.grid, self.grid)).unsqueeze(0)  # [1,256,72,72]
        dense = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(B, -1, self.grid, self.grid)
        sparse = self.no_mask_embed.weight[None].expand(B, 1, -1)  # null prompt convention
        hr = [self.mask_decoder.conv_s0(F_fuse[0]), self.mask_decoder.conv_s1(F_fuse[1])]
        masks, iou, _, _ = self.mask_decoder(
            image_embeddings=image_emb,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=True,
            repeat_image=False,
            high_res_features=hr,
        )
        # masks: [B,3,288,288] logits; iou: [B,3] logits

        if gt is None:
            return {
                "logits": masks,
                "iou": iou,
                "z_pos": z_pos,
                "z_neg": z_neg,
                "alphas": alphas,
                "t_pos": t_pos,
                "t_neg": t_neg,
            }

        # ------------------------- loss (Eq.31 + extensions) ------------------------- #
        logits_up = F.interpolate(
            masks, size=(self.inp_size, self.inp_size), mode="bilinear", align_corners=False
        )
        # Eq.29: oracle selection k* = argmin_k L_seg(S_k, Y)
        per = seg_loss_per_mask(logits_up, gt)  # [B,3]
        k_star = per.argmin(dim=1)
        loss_seg = per[torch.arange(B, device=per.device), k_star].mean()
        # Eq.30-31: IoU regression (MSE between q_hat and hard-IoU q_gt)
        q_gt = iou_of(logits_up.detach(), gt)  # [B,3]
        loss_iou = F.mse_loss(iou, q_gt)
        # Eq.7: target-background contrastive (negatives = z-_i U in-batch t+/t-)
        # skipped for the Table 3 single-modal variant: text is unused there and
        # z- would degenerate to a constant (W_n bias)
        loss_con = (
            info_nce(z_pos, z_neg, t_pos, t_neg,
                     adapter.semantic.W_p, adapter.semantic.W_n, tau=self.tau)
            if self.ablation != 'no_text_freq' else loss_seg.new_zeros(())
        )
        # Eq.23: gate regularization (skipped in fixed-fusion mode: no alphas)
        loss_gate = (
            gate_reg(alphas, lambda_tv=self.lambda_tv)
            if alphas is not None else loss_con.new_zeros(())
        )

        # Eq.31: L = L_seg + lambda_iou L_iou + lambda_con L_con + lambda_gate L_gate
        loss = (
            loss_seg
            + self.lambda_iou * loss_iou
            + self.lambda_con * loss_con
            + self.lambda_gate * loss_gate
        )
        self.loss_log = {
            "seg": loss_seg.detach(),
            "iou": loss_iou.detach(),
            "con": loss_con.detach(),
            "gate": loss_gate.detach(),
            "total": loss.detach(),
        }
        return loss

    # ------------------------------------------------------------------ #
    def infer(self, image, v, t_pos, t_neg):
        """k+ = argmax_k q_hat. Returns the upsampled LOGITS [B,1,S,S] of the
        selected hypothesis — callers apply torch.sigmoid (original-repo
        eval_psnr convention: metric_fn(sigmoid(infer(inp)), gt))."""
        out = self.forward(image, v, t_pos, t_neg)
        k = out["iou"].argmax(dim=1)  # [B]
        logits = out["logits"][torch.arange(out["logits"].shape[0]), k]  # [B,288,288]
        logits = logits.unsqueeze(1)  # [B,1,288,288]
        return F.interpolate(
            logits, size=(self.inp_size, self.inp_size), mode="bilinear", align_corners=False
        )
