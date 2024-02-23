import torch
import torch.nn as nn
from .backbone import ResNetFPN_8_2
from .position_embedding import PositionEmbeddingSine
from .transformer import LocalFeatureTransformer
from loftr_pytorch.matcher.coarse_matching import CoarseMatcher
from loftr_pytorch.model.coarse_to_fine import CoarseToFine
from loftr_pytorch.matcher.fine_matching import FineMatcher


class LoFTR(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.backbone = ResNetFPN_8_2(config["backbone"])
        self.position_embedding = PositionEmbeddingSine(
            config["transformer"]["position_embedding"]
        )
        self.transformer_coarse = LocalFeatureTransformer(
            config["transformer"]["coarse"]
        )
        self.transformer_fine = LocalFeatureTransformer(config["transformer"]["fine"])
        self.coarse_matcher = CoarseMatcher(config["matcher"]["coarse"])
        self.coarse_to_fine = CoarseToFine(config["coarse_to_fine"])
        self.fine_matcher = FineMatcher()

    def forward(self, data):
        """
        Args:
            data (dict): {
                    'image0': (torch.Tensor): (B, 1, H, W)
                    'image1': (torch.Tensor): (B, 1, H, W)
                    'mask0'(optional) : (torch.Tensor): (B, hc, wc) '0' indicates a padded position
                    'mask1'(optional) : (torch.Tensor): (B, hc, wc)
                }
        """
        image0, image1 = data["image0"], data["image1"]
        hw0_i = image0.shape[2:]
        hw1_i = image1.shape[2:]

        if hw0_i == hw1_i:  # faster & better BN convergence
            feats_c, feats_f = self.backbone(torch.cat([image0, image1], dim=0))
            (feat_c0, feat_c1), (feat_f0, feat_f1) = torch.chunk(
                feats_c, 2, dim=0
            ), torch.chunk(feats_f, 2, dim=0)
        else:  # handle different input shapes
            (feat_c0, feat_f0), (feat_c1, feat_f1) = self.backbone(
                image0
            ), self.backbone(image1)

        hw0_f = feat_f0.shape[2:]

        B, _, h0_c, w0_c = feat_c0.shape
        feat_c0 = (
            self.position_embedding(feat_c0)
            .permute(0, 2, 3, 1)
            .view(B, h0_c * w0_c, -1)
        )
        B, _, h1_c, w1_c = feat_c1.shape
        feat_c1 = (
            self.position_embedding(feat_c1)
            .permute(0, 2, 3, 1)
            .view(B, h1_c * w1_c, -1)
        )
        mask0 = data["mask0"] if "mask0" in data else None
        mask1 = data["mask1"] if "mask1" in data else None

        mask_c0 = mask_c1 = None
        if mask0:
            mask_c0, mask_c1 = mask0.flatten(-2), mask1.flatten(-2)
        feat_c0, feat_c1 = self.transformer_coarse(feat_c0, feat_c1, mask_c0, mask_c1)

        hw0_c = (h0_c, w0_c)
        hw1_c = (h1_c, w1_c)

        batch_ids, l_ids, s_ids, gt_mask, m_bids, mkpts0_c, mkpts1_c, mconf = (
            self.coarse_matcher(feat_c0, feat_c1, hw0_c, hw1_c, mask_c0, mask_c1)
        )

        scale = hw0_i[0] / hw0_c[0]
        scale0 = scale * data["scale0"] if "scale0" in data else scale
        scale1 = scale * data["scale1"] if "scale1" in data else scale
        mkpts0_c, mkpts1_c = self.coarse_matcher.rescale_mkpts_to_image(
            mkpts0_c, mkpts1_c, scale0, scale1
        )

        feat_f0_unfold, feat_f1_unfold = self.coarse_to_fine(
            feat_f0, feat_f1, feat_c0, feat_c1, batch_ids, l_ids, s_ids
        )
        if feat_f0_unfold.shape[0] != 0:
            feat_f0_unfold, feat_f1_unfold = self.transformer_fine(
                feat_f0_unfold, feat_f1_unfold
            )

        mkpts0_f, mkpts1_f = self.fine_matcher(
            feat_f0_unfold, feat_f1_unfold, mkpts0_c, mkpts1_c
        )

        scale = hw0_i[0] / hw0_f[0]  # TODO:: scale multiplication
        scale1 = data["scale1"][batch_ids] if "scale1" in data else scale
        return self.fine_matcher.rescale_mkpts_to_image(mkpts0_f, mkpts1_f, 1, scale1)
