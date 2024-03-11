import torch
import torch.nn as nn


class LoFTRLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config["trainer"]["loss"]
        self.coarse_config = self.config["coarse"]
        self.fine_config = self.config["fine"]

    def forward(self, data, coarse_prediction, fine_prediction, coarse_gt, fine_gt):
        """
        Args:
        Returns:
        """

        coarse_loss = self.compute_coarse_loss(
            coarse_prediction["conf_matrix"],
            coarse_gt["conf_matrix_gt"],
            coarse_gt["spv_scores"],
            data,
        )
        fine_loss = self.compute_fine_loss(
            fine_prediction["expec_f"], fine_gt["expec_f_gt"]
        )

        coarse_weight, fine_weight = (
            self.coarse_config["weight"],
            self.fine_config["weight"],
        )
        loss = coarse_weight * coarse_loss + fine_weight * fine_loss

        loss_dict = {
            "coarse_loss": coarse_loss.item(),
            "fine_loss": fine_loss.item(),
            "loss": loss,
        }
        return loss_dict

    def compute_coarse_loss(self, conf_matrix, conf_matrix_gt, spv_scores, data):
        """
        Args:
        Returns:
        """

        pos_weight, neg_weight = (
            self.coarse_config["pos_weight"],
            self.coarse_config["neg_weight"],
        )

        weight_mask = None
        if "mask0" in data:
            weight_mask = (
                data["mask0"].flatten(-2)[:, :, None]
                * data["mask1"].flatten(-2)[:, None, :]
            )  # (B, H0*W0, H1*W1)

        pos_mask, neg_mask = (
            conf_matrix_gt == 1,
            conf_matrix_gt == 0,
        )  # (B, H0*W0, H1*W1)

        spv_score_mask = spv_scores >= self.coarse_config["spv_score_thr"]  # (B,)

        pos_mask = pos_mask * spv_score_mask[:, None, None]
        neg_mask = neg_mask * spv_score_mask[:, None, None]
        conf_matrix = torch.clamp(conf_matrix, 1e-6, 1 - 1e-6)

        if pos_mask.sum() == 0:
            pos_mask[0, 0, 0] = True
            if "mask0" in data:
                weight_mask[0, 0, 0] = 0
            pos_weight = 0
            print("No positive mask found")
        if neg_mask.sum() == 0:
            neg_mask[0, 0, 0] = True
            if "mask0" in data:
                weight_mask[0, 0, 0] = 0
            neg_weight = 0
            print("No negative mask found")

        if self.coarse_config["type"] == "cross_entropy":
            loss_pos = -torch.log(conf_matrix[pos_mask])
            loss_neg = -torch.log(1 - conf_matrix[neg_mask])
            if weight_mask is not None:
                loss_pos = loss_pos * weight_mask[pos_mask]
                loss_neg = loss_neg * weight_mask[neg_mask]
            return pos_weight * loss_pos.mean() + neg_weight * loss_neg.mean()
        elif self.coarse_config["type"] == "focal":
            alpha = self.coarse_config["focal_alpha"]
            gamma = self.coarse_config["focal_gamma"]
            loss_pos = (
                -alpha
                * (1 - conf_matrix[pos_mask]) ** gamma
                * torch.log(conf_matrix[pos_mask])
            )
            loss_neg = (
                -alpha
                * conf_matrix[neg_mask] ** gamma
                * torch.log(1 - conf_matrix[neg_mask])
            )
            if weight_mask is not None:
                loss_pos = loss_pos * weight_mask[pos_mask]
                loss_neg = loss_neg * weight_mask[neg_mask]
            return pos_weight * loss_pos.mean() + neg_weight * loss_neg.mean()
        else:
            raise ValueError(f"Unknown coarse loss type {self.coarse_config['type']}")

    def compute_fine_loss(self, expec_f, expec_f_gt):
        """
        Args:
            expect_f (torch.Tensor): Expected fine-level normalized xy coordinates. (M, 2)
            expect_f_gt (torch.Tensor): Groundtruth expected fine-level normalized xy coordinates. (M, 2)
        """
        if self.fine_config["type"] == "l2":
            return self._compute_fine_l2_loss(expec_f, expec_f_gt)
        elif self.fine_config["type"] == "l2_with_std":
            return self._compute_fine_l2_with_std_loss(expec_f, expec_f_gt)
        else:
            raise ValueError(f"Unknown fine loss type {self.fine_config['type']}")

    def _compute_fine_l2_loss(self, expec_f, expec_f_gt):
        """
        Args:
            expect_f (torch.Tensor): Expected fine-level normalized xy coordinates. (M, 2)
            expect_f_gt (torch.Tensor): Groundtruth expected fine-level normalized xy coordinates. (M, 2)
        """
        correct_mask = (
            torch.linalg.norm(expec_f_gt, ord=float("inf"), dim=1)
            < self.fine_config["correct_thr"]
        )
        assert correct_mask.sum() > 0, "No correct fine matches found"
        l2_loss = ((expec_f[correct_mask] - expec_f_gt[correct_mask]) ** 2).sum(-1)
        return l2_loss.mean()

    def _compute_fine_l2_with_std_loss(self, expec_f, expec_f_gt):
        """
        Args:
            expect_f (torch.Tensor): Expected fine-level normalized xy coordinates. (M, 2)
            expect_f_gt (torch.Tensor): Groundtruth expected fine-level normalized xy coordinates. (M, 2)
        """
        correct_mask = (
            torch.linalg.norm(expec_f_gt, ord=float("inf"), dim=1)
            < self.fine_config["correct_thr"]
        )
        std = expec_f[:, 2]  # (M,)
        inverse_std = 1 / std
        weight = (inverse_std / inverse_std.mean()).detach()
        if correct_mask.sum() == 0:
            correct_mask[0] = True
            weight[0] = 0
        l2_loss = ((expec_f[correct_mask, :2] - expec_f_gt[correct_mask]) ** 2).sum(-1)
        return (l2_loss * weight[correct_mask]).mean()
