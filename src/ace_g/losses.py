# Copyright © Niantic Spatial, Inc. 2025. All rights reserved.
"""Losses for the ACE-G model."""

from __future__ import annotations

import collections
import math
from typing import Literal

import numpy as np
import rerun as rr
import torch

from ace_g import utils

dataset_stats = collections.defaultdict(int)


def compute_loss(
    pred_coords: torch.Tensor,
    pred_uncertainties: torch.Tensor | None,
    w2c_b34: torch.Tensor | None,
    image_from_camera_b33: torch.Tensor | None,
    target_pixels: torch.Tensor | None,
    target_coords: torch.Tensor | None,
    supervision_type: Literal["2d", "3d"],
    target_mask: torch.Tensor | None = None,
    target_mask_2d: torch.Tensor | None = None,
    target_mask_3d: torch.Tensor | None = None,
    weight_2d: float | torch.Tensor = 1.0,
    weight_3d: float | torch.Tensor = 1.0,
    use_depth_as_prior: torch.Tensor | bool = False,
    const_depth_prior: float = 10.0,
    depth_min: float = 0.1,
    depth_max: float = 1000,
    max_reprojection_error: float = 1000,
    loss_fn_2d: ReproLoss | str | None = None,
    loss_fn_3d: ReproLoss | str | None = None,
    iteration: int | None = None,
) -> tuple:
    """Compute loss from predicted scene coordinates and target pixels or target scene coordinates.

    A ground-truth scene coordinate is ignored if target_mask is False or all three coordinates are zero.
    A ground-truth pixel is ignored only if target_mask is False.

    To disable any kind of 3D supervision for a specific pixel, set the target scene coordinate to (0, 0, 0).

    Args:
        pred_coords: Predicted scene coordinates. Shape (..., 3).
        pred_uncertainties:
            Predicted uncertainties. Interpreted as standard deviation of the 3D distance.
            If not None, probabilistic loss is used. Shape (..., 1).
        w2c_b34: World-to-camera transformation. Only required for 2D supervision. Shape (..., 3, 4).
        target_pixels: Target pixels. Only required for 2D supervision. Shape (..., 2).
        target_coords: Target scene coordinates. Shape (..., 3).
        target_mask:
            Target mask. Only True values will be supervised. Will be and combined with 2d and 3d masks. Shape (..., ).
        target_mask_2d:
            Target mask for 2D only. Allows to enable / disable 2D supervision per pixel.
            Shape must broadcast to batch dimensions.
        target_mask_3d:
            Target mask. Allows to enable / disable 3D supervision per pixel.
            Shape must broadcast to batch dimensions.
        supervision_type:
            The type of supervision to use.
            2D: Only 2D supervision is used.
            3D: Only 3D supervision is used.
        probabilistic_loss_type:
            If pred_uncertainties is not None, the type of probabilistic loss to use.
            In all cases uncertainty is interpreted as the standard deviation.
        weight_2d: The weight to apply to the 2D loss.
        weight_3d: The weight to apply to the 3D loss.
        use_depth_as_prior:
            Whether to use ground-truth depth as the prior for invalid pixels.
            Either boolean or a tensor of any shape that broadcasts to the batch dimensions, which controls this
            attribute per pixel. If False, a constant depth prior is used even if ground-truth depth is available.
            See const_depth_prior.
        const_depth_prior:
            The constant depth prior used for outliers instead of reprojection error.
        depth_min:
            The minimum depth value to clamp to for 2D supervision. Predictions outside will not be used in the loss.
        depth_max:
            The maximum depth value to clamp to for 2D supervision. Predictions outside will not be used in the loss.
        max_reprojection_error: The maximum reprojection error to consider a pixel valid. In pixels.
        iteration: The current iteration of the training process. Forwarded to the valid_2d_loss.
        valid_2d_loss:
            The loss function to use for valid 2D pixels.
            If str, eval will be used with x being the 2D distances tensor.
            If None, 2D distance is used.
            Not used if pred_uncertainties is not None, in which case the probabilistic loss is used.
        valid_3d_loss:
            The loss function to use for valid 3D coordinates.
            If str, eval will be used with x being the 3D distances tensor.
            If None, 2D distance is used.
            Not used if pred_uncertainties is not None, in which case the probabilistic loss is used.
        quantile_type:
            At which level to compute and apply the quantile threshold.

    Returns:
        loss: The computed loss as a scalar. None if all targets are masked out.
        losses_3d: The individual 3D losses. Before reduction and weighing.
        losses_2d: The individual 2D losses. Before reduction and weighing. Shape (num_losses_2d,).
        distances_3d: The individual valid 3D distances. In meters. Shape (num_valid_target_coords,). None, if target_coords is None.
        distances_2d: The individual valid 2D distances. In pixels. Shape (num_pixels,). None, if target_pixels is None.
    """

    def eval_loss(loss_fn: str, x: torch.Tensor):
        return eval(loss_fn)

    valid_distances_3d = all_distances_2d = None
    losses_3d = losses_2d = None

    if isinstance(use_depth_as_prior, bool):
        use_depth_as_prior = torch.tensor(use_depth_as_prior, dtype=torch.bool)

    if target_mask is None:
        target_mask = torch.ones_like(pred_coords[..., 0], dtype=torch.bool)
    target_mask_2d = target_mask if target_mask_2d is None else target_mask & target_mask_2d
    target_mask_3d = target_mask if target_mask_3d is None else target_mask & target_mask_3d

    loss = 0.0

    # Compute 3D loss.
    if target_coords is not None:
        valid_target_coords_mask = target_coords.abs().sum(dim=-1) > 0.00001
        used_in_loss_mask = target_mask_3d & valid_target_coords_mask
        all_distances_3d = torch.linalg.norm(pred_coords - target_coords, dim=-1, ord=2)
        valid_distances_3d = all_distances_3d[valid_target_coords_mask]
        loss_distances_3d = all_distances_3d[used_in_loss_mask]
        if pred_uncertainties is None or target_mask_3d.sum() == 0:
            if isinstance(loss_fn_3d, ReproLoss):
                losses_3d = loss_fn_3d.compute(loss_distances_3d, iteration)
            elif isinstance(loss_fn_3d, str):
                losses_3d = eval_loss(loss_fn_3d, loss_distances_3d)
            else:
                losses_3d = loss_distances_3d
        elif pred_uncertainties is not None:
            pred_uncertainties_3d = pred_uncertainties[used_in_loss_mask].squeeze(-1)
            pred_uncertainties_3d = torch.clamp(pred_uncertainties_3d, min=1e-4)

            # Laplace NLL
            losses_3d = torch.log(pred_uncertainties_3d) + math.sqrt(2) * loss_distances_3d / pred_uncertainties_3d

        loss_3d = losses_3d.mean()

        if supervision_type == "3d":
            loss = loss + weight_3d * torch.nan_to_num(loss_3d)

    # Compute 2D loss.
    if target_pixels is not None and w2c_b34 is not None and image_from_camera_b33 is not None:
        # use w to denote world coordinates and c to denote camera coordinates
        pred_coords_w_b31 = pred_coords.unsqueeze(-1)  # b could represent any number of batch dimensions.
        pred_coords_w_b41 = utils.to_homogeneous(pred_coords_w_b31, dim=-2)
        pred_coords_c_b31 = torch.matmul(w2c_b34, pred_coords_w_b41)
        pred_coords_c_b3 = pred_coords_c_b31.squeeze(-1)
        pred_pixels_b31 = torch.matmul(image_from_camera_b33, pred_coords_c_b31)

        # Avoid division by zero.
        # Note: negative values are also clamped at +self.options.depth_min. The predicted pixel would be wrong,
        # but that's fine since we mask them out later.
        pred_pixels_b31[..., 2, :].clamp_(min=depth_min)

        # Dehomogenise.
        pred_pixels_b21 = pred_pixels_b31[..., :2, :] / pred_pixels_b31[..., 2, None, :]

        # Measure reprojection error.
        all_distances_2d = torch.norm(pred_pixels_b21.squeeze(-1) - target_pixels, dim=-1, p=1)

        # Compute mask for which depth prior should be used instead
        invalid_min_depth_b = pred_coords_c_b3[..., 2] < depth_min  # behind or close to camera plane
        invalid_repro_b = all_distances_2d > max_reprojection_error  # Large reprojection errors.
        invalid_max_depth_b = pred_coords_c_b3[..., 2] > depth_max  # too far away
        invalid_mask_b = invalid_min_depth_b | invalid_repro_b | invalid_max_depth_b
        # if we have access to target coord, further add a check a check on ground-truth coordinate
        target_coords_available_b = False
        if use_depth_as_prior.any() and target_coords is not None:
            invalid_target_crds_b = torch.linalg.norm(target_coords - pred_coords_w_b31.squeeze(-1), dim=-1) > 0.1
            # in the previous mask, ignore pixels without GT scene coordinates (all zeros) or when depth should not be
            # used
            target_coords_available_b = target_coords.abs().sum(dim=-1) > 0.00001

            invalid_mask_b = invalid_mask_b | (invalid_target_crds_b & target_coords_available_b & use_depth_as_prior)

        valid_mask_b = ~invalid_mask_b

        # Mask out pixels that should be skipped according to tarket mask (removed from the losses tensor later)
        valid_mask_b = valid_mask_b & target_mask_2d
        invalid_mask_b = invalid_mask_b & target_mask_2d
        losses_2d_mask_b = valid_mask_b | invalid_mask_b

        losses_2d_b = torch.zeros_like(all_distances_2d)
        valid_distances_2d = all_distances_2d[valid_mask_b]
        all_distances_2d = all_distances_2d.flatten()

        if valid_mask_b.any():
            if pred_uncertainties is None:
                if isinstance(loss_fn_2d, ReproLoss):
                    losses_2d_b[valid_mask_b] = loss_fn_2d.compute(valid_distances_2d, iteration)
                elif isinstance(loss_fn_2d, str):
                    losses_2d_b[valid_mask_b] = eval_loss(loss_fn_2d, valid_distances_2d)
                else:
                    losses_2d_b[valid_mask_b] = valid_distances_2d  # just use the mean distances (akin to 3D case)
            else:
                # approximately project the 3D uncertainties (in m) to 2D uncertainties (in px)
                uncertainties_2d = (pred_uncertainties[..., 0] * image_from_camera_b33[..., 0, 0])[
                    valid_mask_b
                ] / pred_pixels_b31[valid_mask_b][..., 2, 0]

                uncertainties_2d = torch.clamp(uncertainties_2d, min=1e-4)

                # Laplace NLL
                losses_2d_b[valid_mask_b] = (
                    torch.log(uncertainties_2d) + math.sqrt(2) * valid_distances_2d / uncertainties_2d
                )

        # Handle the invalid predictions
        if invalid_mask_b.sum() > 0:
            depth_prior_mask = invalid_mask_b & use_depth_as_prior & target_coords_available_b
            const_depth_prior_mask = invalid_mask_b & ~depth_prior_mask

            if depth_prior_mask.any():
                assert target_coords is not None, "Target coordinates are required for depth prior"
                distances_to_prior = torch.linalg.norm(pred_coords_c_b31.squeeze(-1) - target_coords, dim=-1)
                losses_2d_b[depth_prior_mask] = distances_to_prior[depth_prior_mask]

            if const_depth_prior_mask.any():
                # generate proxy coordinate targets with constant depth assumption.
                pixel_grid_crop_b31 = utils.to_homogeneous(target_pixels.unsqueeze(-1), dim=-2)
                prior_targets_b31 = const_depth_prior * torch.matmul(
                    image_from_camera_b33.inverse(), pixel_grid_crop_b31
                )

                # Compute the distance to target camera coordinates.
                distances_to_prior = torch.linalg.norm(
                    pred_coords_c_b31.squeeze(-1) - prior_targets_b31.squeeze(-1), dim=-1
                )
                if pred_uncertainties is None:
                    losses_2d_b[const_depth_prior_mask] = distances_to_prior[const_depth_prior_mask]
                else:
                    depth_prior_uncertainties = pred_uncertainties[..., 0][const_depth_prior_mask]

                    # Laplace NLL
                    losses_2d_b[const_depth_prior_mask] = (
                        torch.log(depth_prior_uncertainties)
                        + math.sqrt(2) * distances_to_prior[const_depth_prior_mask] / depth_prior_uncertainties
                    )

        losses_2d = losses_2d_b[losses_2d_mask_b]

        loss_2d = losses_2d.mean()

        if supervision_type == "2d":
            loss = loss + weight_2d * torch.nan_to_num(loss_2d)

    return (
        loss,
        losses_3d,
        losses_2d,
        valid_distances_3d,
        all_distances_2d,
    )


def log_metrics(
    loss: torch.Tensor | None,
    losses_3d: torch.Tensor | None,
    losses_2d: torch.Tensor | None,
    distances_3d: torch.Tensor | None,
    distances_2d: torch.Tensor | None,
) -> None:
    """Log metrics to Rerun."""
    if loss is not None:
        rr.log("loss", rr.Scalars(loss.item()))
        if loss > 0:
            rr.log("log_loss", rr.Scalars(math.log(loss.item())))

    if losses_3d is not None:
        loss_3d = losses_3d.mean()
        if torch.isfinite(loss_3d):
            rr.log("loss_3d", rr.Scalars(loss_3d.item()))

    if losses_2d is not None:
        loss_2d = losses_2d.mean().item()
        rr.log("loss_2d", rr.Scalars(loss_2d))

    if distances_3d is not None:
        inlier_thresholds_3d = [5, 10, 20]  # cm
        for inlier_threshold in inlier_thresholds_3d:
            inliers = distances_3d < (inlier_threshold * 0.01)
            inlier_ratio = inliers.float().mean()
            if torch.isfinite(inlier_ratio):
                rr.log(f"inliers_{inlier_threshold}cm", rr.Scalars(inlier_ratio.item()))

    if distances_2d is not None:
        inlier_thresholds_2d = [5, 10, 20]  # px
        for inlier_threshold in inlier_thresholds_2d:
            inliers = distances_2d < inlier_threshold
            inlier_ratio = inliers.float().mean().item()
            rr.log(f"inliers_{inlier_threshold}px", rr.Scalars(inlier_ratio))

        quantiles = [0.1, 0.2, 0.5]
        for quantile in quantiles:
            if distances_2d.numel() == 0:
                continue
            quantile_value = torch.quantile(distances_2d, quantile).item()
            rr.log(f"quantile_2d_{quantile:.1f}", rr.Scalars(quantile_value))

        rr.log("mean_distance_2d", rr.Scalars(distances_2d.mean().item()))


def weighted_tanh(repro_errs: torch.Tensor, weight: float) -> torch.Tensor:
    """Compute weighted tanh.

    Args:
        repro_errs: A tensor containing the reprojection errors.
        weight: The weight to use in the tanh loss.

    Returns: The weighted tanh errors.
    """
    return weight * torch.tanh(repro_errs / weight)


class ReproLoss:
    """Compute per-pixel reprojection loss using different configurable approaches.

    - tanh:     tanh loss with a constant scale factor given by the `soft_clamp` parameter (when a pixel's reprojection
                error is equal to `soft_clamp`, its loss is equal to `soft_clamp * tanh(1)`).
    - dyntanh:  Used in the paper, similar to the tanh loss above, but the scaling factor decreases during the course of
                the training from `soft_clamp` to `soft_clamp_min`. The decrease is linear, unless `circle_schedule`
                is True (default), in which case it applies a circular scheduling. See paper for details.
    - l1:       Standard L1 loss, computed only on those pixels having an error lower than `soft_clamp`
    - l1+sqrt:  L1 loss for pixels with reprojection error smaller than `soft_clamp` and
                `sqrt(soft_clamp * reprojection_error)` for pixels with a higher error.
    - l1+logl1: Similar to the above, but using log L1 for pixels with high reprojection error.
    """

    def __init__(
        self,
        total_iterations: int,
        soft_clamp: float,
        soft_clamp_min: float,
        type: Literal["tanh", "dyntanh", "l1", "l1+sqrt", "l1+logl1"] = "dyntanh",
        circle_schedule: bool = True,
    ):
        """Initialize the ReproLoss class."""
        self.total_iterations = total_iterations
        self.soft_clamp = soft_clamp
        self.soft_clamp_min = soft_clamp_min
        self.type = type
        self.circle_schedule = circle_schedule

    def compute(self, errors: torch.Tensor, iteration: int) -> torch.Tensor:
        """Compute the reprojection loss based on the specified type of loss function.

        The types of loss function available are: 'tanh', 'dyntanh', 'l1', 'l1+sqrt', and 'l1+logl1'.

        Args:
            errors: A tensor containing the reprojection errors. Any shape.
            iteration: The current iteration of the training process.

        Returns: The computed per-element losses. Same shape as the input tensor.
        """
        # If there are no elements in the reprojection errors tensor, return 0
        if errors.nelement() == 0:
            return torch.Tensor([])

        # Compute the simple tanh loss
        if self.type == "tanh":
            return weighted_tanh(errors, self.soft_clamp)

        # Compute the dynamic tanh loss
        elif self.type == "dyntanh":
            # Compute the progress over the training process.
            schedule_weight = iteration / self.total_iterations

            # Optionally scale it using the circular schedule.
            if self.circle_schedule:
                schedule_weight = 1 - np.sqrt(1 - schedule_weight**2)

            # Compute the weight to use in the tanh loss.
            loss_weight = (1 - schedule_weight) * self.soft_clamp + self.soft_clamp_min

            # Compute actual loss.
            return weighted_tanh(errors, loss_weight)

        # Compute the L1 loss
        elif self.type == "l1":
            raise NotImplementedError("L1 not checked since update.")
            # L1 loss on all pixels with small-enough error.
            softclamp_mask_b1 = errors > self.soft_clamp
            return errors[~softclamp_mask_b1].sum()

        # Compute the L1 loss for small errors and sqrt loss for larger errors
        elif self.type == "l1+sqrt":
            raise NotImplementedError("L1+sqrt not checked since update.")

            softclamp_mask_b1 = errors > self.soft_clamp
            loss_l1 = errors[~softclamp_mask_b1].sum()
            loss_sqrt = torch.sqrt(self.soft_clamp * errors[softclamp_mask_b1]).sum()

            return loss_l1 + loss_sqrt

        # Compute the L1 loss for small errors and log L1 loss for larger errors
        else:
            softclamp_mask_b1 = errors > self.soft_clamp
            loss_l1 = errors[~softclamp_mask_b1].sum()
            loss_logl1 = torch.log(1 + (self.soft_clamp * errors[softclamp_mask_b1])).sum()

            return loss_l1 + loss_logl1
