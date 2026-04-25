# Copyright © Niantic Spatial, Inc. 2025. All rights reserved.
"""Training schedule for single-scene optimization."""

import dataclasses
import logging
from typing import Iterable, Literal

import torch
from torch import optim

_logger = logging.getLogger(__name__)


class ScheduleACE:
    """Handles the training schedule and optimization for the ACE model."""

    @dataclasses.dataclass
    class Config:
        """Configuration for the training schedule (optimizer + scheduler configuration).

        Attributes:
            learning_rate_min: Lowest learning rate of 1 cycle learning rate schedule.
            learning_rate_max: Highest learning rate of 1 cycle learning rate schedule.
            learning_rate_schedule: The learning rate schedule to use.
            learning_rate_warmup: Learning rate at the beginning of the warmup phase.
            learning_rate_warmup_iterations: Number of iterations for the warmup phase.
            learning_rate_cooldown_trigger_percent_threshold: Min percentage of inliers for early cool down.
            learning_rate_cooldown_iterations: Length of the cooldown period.
            learning_rate_div_factor: Initial div factor of cycle scheduler.
            learning_rate_final_div_factor: Final div factor of cycle scheduler.
            optimizer: The optimizer to use.
            weight_decay: The weight decay for the optimizer.
        """

        learning_rate_min: float = 0.0005
        learning_rate_max: float = 0.005
        learning_rate_schedule: Literal["circle", "cycle", "constant", "1cyclepoly", "schedulefree"] = "circle"
        learning_rate_warmup: float = 0.0005
        learning_rate_warmup_iterations: int = 1_000
        learning_rate_cooldown_trigger_percent_threshold: float = 0.7
        learning_rate_cooldown_iterations: int = 5_000
        learning_rate_div_factor: float = 25
        learning_rate_final_div_factor: float = 10000
        optimizer: Literal["adam", "adamw"] = "adamw"
        weight_decay: float = 0.01

    def __init__(self, parameters: Iterable, use_scaler: bool, num_iterations: int, config: Config) -> None:
        """Initializes the training schedule and optimizer.

        Args:
            parameters:
                The parameters that will be optimized. This class does not change requires_grad of the parameters.
            use_scaler:
                Whether to use the gradient scaler (should be enabled for half precision training).
            num_iterations:
                The total number of iterations for training.
            options: The training options.
        """
        self.config = config
        if self.config.optimizer == "adam":
            optimizer_cls = optim.Adam
        elif self.config.optimizer == "adamw":
            optimizer_cls = optim.AdamW
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer=}")

        if self.config.learning_rate_schedule not in ["circle", "cycle", "constant", "1cyclepoly", "schedulefree"]:
            raise ValueError(f"Unknown learning rate schedule: {self.config.learning_rate_schedule}")

        self.max_iterations = num_iterations

        is_param_groups = isinstance(parameters, list) and len(parameters) > 0 and isinstance(parameters[0], dict)
        if is_param_groups:
            max_lrs = [g.get('max_lr', self.config.learning_rate_max) for g in parameters]
            min_lrs = [g.get('min_lr', self.config.learning_rate_min) for g in parameters]
            for g, max_lr in zip(parameters, max_lrs):
                if 'lr' not in g:
                    g['lr'] = max_lr
        else:
            max_lrs = self.config.learning_rate_max
            min_lrs = self.config.learning_rate_min

        # Setup learning rate scheduler
        if self.config.learning_rate_schedule == "constant":
            self.optimizer = optimizer_cls(parameters, lr=self.config.learning_rate_min)
            # No schedule. Use constant learning rate.
            self.scheduler = None

        elif self.config.learning_rate_schedule == "schedulefree":
            import schedulefree  # type: ignore

            self.scheduler = None
            # self.optimizer = schedulefree.AdamWScheduleFree(
            #     parameters, lr=self.learning_rate_max, warmup_steps=self.max_iterations // 5
            # )
            self.optimizer = schedulefree.SGDScheduleFree(
                parameters, lr=self.config.learning_rate_max, warmup_steps=self.max_iterations // 5
            )

        elif self.config.learning_rate_schedule == "1cyclepoly":
            # Approximate 1cycle learning rate schedule with linear warmup and cooldown.
            self.optimizer = optimizer_cls(parameters, lr=self.config.learning_rate_max)

            # Warmup phase. Increase from warmup learning rate to max learning rate.
            self.warmup_iterations = self.config.learning_rate_warmup_iterations
            lr_factor_warmup = self.config.learning_rate_warmup / self.config.learning_rate_max
            scheduler_warmup = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=lr_factor_warmup,
                total_iters=self.warmup_iterations,
            )

            # Cooldown phase. Decrease from max learning rate to min learning rate.
            self.cooldown_trigger_percent_threshold = self.config.learning_rate_cooldown_trigger_percent_threshold
            self.cooldown_iterations = self.config.learning_rate_cooldown_iterations

            lr_factor_cooldown = self.config.learning_rate_min / self.config.learning_rate_max
            self.scheduler_cooldown = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1,
                end_factor=lr_factor_cooldown,
                total_iters=self.cooldown_iterations,
            )

            # Set the scheduler to the warmup phase.
            # We will switch to cooldown scheduler when the cooldown criteria is met.
            self.scheduler = scheduler_warmup

            # Flag indicating whether we are in the final cooldown phase.
            self.in_cooldown_phase = False

            # Rolling buffer holding statistics for the cooldown criteria.
            self.cooldown_criterium_buffer = []
            # Max size of the buffer
            self.cooldown_buffer_size = 100

        elif self.config.learning_rate_schedule == "cycle":
            self.optimizer = optimizer_cls(parameters, lr=self.config.learning_rate_max)
            self.scheduler = optim.lr_scheduler.CyclicLR(
                self.optimizer,
                base_lr=min_lrs,
                max_lr=max_lrs,
                step_size_up=1000,
                step_size_down=1000,
                cycle_momentum=False,
                mode="triangular",
            )

        elif self.config.learning_rate_schedule == "circle":
            # 1 Cycle learning rate schedule.
            self.optimizer = optimizer_cls(parameters, lr=self.config.learning_rate_min)
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=max_lrs,
                total_steps=self.max_iterations,
                cycle_momentum=False,
                div_factor=self.config.learning_rate_div_factor,
                final_div_factor=self.config.learning_rate_final_div_factor,
            )

        else:
            raise ValueError(f"Unknown learning rate schedule: {self.config.learning_rate_schedule}")

        # Gradient scaler in case we train with half precision.
        self.scaler = torch.GradScaler("cuda", enabled=use_scaler)

    def check_and_set_cooldown(self, iteration):

        # cool down only supported by 1cyclepoly lr schedule
        if self.config.learning_rate_schedule != "1cyclepoly":
            return

        # check whether we are already in cool down
        if self.in_cooldown_phase:
            return

        # check whether warmup has finished, we do not want to cooldown earlier than that
        if iteration < self.warmup_iterations:
            return

        # check whether we should go into cool down according to max training duration
        start_cooldown_max_duration = iteration >= (self.max_iterations - self.cooldown_iterations)

        # check whether we should go into cool down according to dynamic criterion

        start_cooldown_dynamic = min(self.cooldown_criterium_buffer) > self.cooldown_trigger_percent_threshold

        if start_cooldown_max_duration or start_cooldown_dynamic:
            _logger.info(
                f"Starting learning rate cooldown. "
                f"(Reason: max duration {start_cooldown_max_duration}, dynamic {start_cooldown_dynamic})"
            )
            _logger.info(f"Training scheduled to stop in {self.cooldown_iterations} iterations.")

            self.scheduler = self.scheduler_cooldown
            self.max_iterations = iteration + self.cooldown_iterations
            self.in_cooldown_phase = True

    def zero_grad(self, set_to_none=False):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def backward(self, loss):
        self.scaler.scale(loss).backward()

    def step(self):
        # prev_scale = self.scaler.get_scale()

        # Parameter update
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # new_scale = self.scaler.get_scale()
        # scaler_stepped = new_scale != prev_scale

        # Schedule update
        if self.scheduler is not None:
            self.scheduler.step()

            if self.config.learning_rate_schedule == "1cyclepoly":
                raise NotImplementedError("1cyclepoly not supported with new loss function yet")

                # # keep track of cooldown trigger statistic over the last n batches
                # self.cooldown_criterium_buffer.append(batch_inliers)

                # # trim buffer size
                # if len(self.cooldown_criterium_buffer) > self.cooldown_buffer_size:
                #     self.cooldown_criterium_buffer = self.cooldown_criterium_buffer[1:]
