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

    def _preprocess_features(self, feat_c0, feat_c1):

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

        return feat_c0, feat_c1

    def forward(self, data, coarse_gt):
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
        data["hw0_i"] = image0.shape[-2:]
        data["hw1_i"] = image1.shape[-2:]

        if data["hw0_i"] == data["hw1_i"]:  # faster & better BN convergence
            feats_c, feats_f = self.backbone(torch.cat([image0, image1], dim=0))
            (feat_c0, feat_c1), (feat_f0, feat_f1) = torch.chunk(
                feats_c, 2, dim=0
            ), torch.chunk(feats_f, 2, dim=0)
        else:  # handle different input shapes
            (feat_c0, feat_f0), (feat_c1, feat_f1) = self.backbone(
                image0
            ), self.backbone(image1)

        data["hw0_c"] = feat_c0.shape[2:]
        data["hw1_c"] = feat_c1.shape[2:]
        data["hw0_f"] = feat_f0.shape[2:]
        data["hw1_f"] = feat_f1.shape[2:]

        # position embedding and flatten to (B, L, C), (B, S, C)
        feat_c0, feat_c1 = self._preprocess_features(feat_c0, feat_c1)

        mask0 = data["mask0"] if "mask0" in data else None
        mask1 = data["mask1"] if "mask1" in data else None

        if "mask0" in data:
            mask0, mask1 = mask0.flatten(-2), mask1.flatten(-2)

        feat_c0, feat_c1 = self.transformer_coarse(feat_c0, feat_c1, mask0, mask1)

        coarse_prediction = self.coarse_matcher(
            feat_c0, feat_c1, data, coarse_gt, mask0, mask1
        )

        feat_f0_unfold, feat_f1_unfold = self.coarse_to_fine(
            feat_f0, feat_f1, feat_c0, feat_c1, coarse_prediction
        )

        if feat_f0_unfold.shape[0] != 0:
            feat_f0_unfold, feat_f1_unfold = self.transformer_fine(
                feat_f0_unfold, feat_f1_unfold
            )

        fine_prediction = self.fine_matcher(
            feat_f0_unfold, feat_f1_unfold, coarse_prediction, data
        )

        result = {"coarse": coarse_prediction, "fine": fine_prediction}
        return result
