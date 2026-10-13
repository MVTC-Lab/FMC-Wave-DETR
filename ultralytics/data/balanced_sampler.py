from __future__ import annotations

import math
import random
from typing import Iterator

import torch
import torch.distributed as dist
from torch.utils.data import Sampler


class PositivePreservedNegativeBatchSampler(Sampler[list[int]]):
    """
    SODA-A 正图全保留、空图动态采样的分布式批采样器。

    默认目标：
        - 每个 epoch 保留全部正图
        - 每个 epoch 无放回抽取 6907 张空图
        - 空图比例约为 25%
        - 4 GPU、global batch=8 时，每个全局 batch 为 6 正 + 2 空
        - 保证每个 rank 的 local batch 至少含一张正图

    参数
    ----
    dataset:
        Ultralytics 训练 Dataset，需要包含 dataset.labels。
    local_batch_size:
        每张 GPU 上的 batch size。
        global batch=8、world_size=4 时，该值应为 2。
    negative_per_epoch:
        每个 epoch 抽取的空图数量。
    negative_ratio:
        每个全局 batch 中空图目标比例。
    seed:
        基础随机种子。
    """

    is_sodaa_balanced_empty_sampler = True
    expected_positive_count = 20_720
    expected_empty_count = 17_260
    expected_negative_per_epoch = 6_907
    expected_negative_ratio = 0.25
    expected_global_batch_size = 8
    supported_world_sizes = {1, 4}

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        negative_per_epoch: int = 6907,
        negative_ratio: float = 0.25,
        seed: int = 42,
        num_replicas: int | None = None,
        rank: int | None = None,
    ) -> None:
        if num_replicas is None:
            num_replicas = (
                dist.get_world_size()
                if dist.is_available() and dist.is_initialized()
                else 1
            )

        if rank is None:
            rank = (
                dist.get_rank()
                if dist.is_available() and dist.is_initialized()
                else 0
            )

        self.dataset = dataset
        self.local_batch_size = int(local_batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.negative_per_epoch = int(negative_per_epoch)
        self.negative_ratio = float(negative_ratio)
        self.seed = int(seed)
        self.epoch = 0

        if self.num_replicas not in self.supported_world_sizes:
            raise ValueError(
                f"SODA-A balanced sampler supports world_size 1 or 4, got {self.num_replicas}."
            )
        if not 0 <= self.rank < self.num_replicas:
            raise ValueError(f"rank must be in [0, {self.num_replicas}), got {self.rank}.")
        if self.local_batch_size <= 0:
            raise ValueError(f"local_batch_size must be positive, got {self.local_batch_size}.")
        self.global_batch_size = self.local_batch_size * self.num_replicas
        if self.global_batch_size != self.expected_global_batch_size:
            raise ValueError(
                f"SODA-A balanced sampler requires global batch 8, got {self.global_batch_size} "
                f"({self.local_batch_size} x {self.num_replicas})."
            )
        if self.negative_per_epoch != self.expected_negative_per_epoch:
            raise ValueError(
                f"SODA-A balanced sampler requires negative_per_epoch=6907, got {self.negative_per_epoch}."
            )
        if not math.isclose(self.negative_ratio, self.expected_negative_ratio, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"SODA-A balanced sampler requires negative_ratio=0.25, got {self.negative_ratio}.")
        if not hasattr(dataset, "labels"):
            raise ValueError("SODA-A training dataset must expose a 'labels' list.")

        self.positive_indices: list[int] = []
        self.negative_indices: list[int] = []

        for index, label in enumerate(dataset.labels):
            if not isinstance(label, dict):
                raise ValueError(f"dataset.labels[{index}] must be a dict, got {type(label).__name__}.")
            cls = label.get("cls")
            try:
                if cls is None:
                    num_objects = 0
                elif isinstance(cls, torch.Tensor):
                    num_objects = int(cls.numel())
                else:
                    num_objects = int(len(cls))
            except TypeError as error:
                raise ValueError(f"dataset.labels[{index}]['cls'] has no valid length.") from error
            target = self.positive_indices if num_objects > 0 else self.negative_indices
            target.append(index)

        actual_positive = len(self.positive_indices)
        actual_empty = len(self.negative_indices)
        if (actual_positive, actual_empty) != (self.expected_positive_count, self.expected_empty_count):
            raise ValueError(
                "SODA-A train split mismatch: expected positive=20720 and empty=17260, "
                f"got positive={actual_positive} and empty={actual_empty}."
            )

        if not self.positive_indices:
            raise RuntimeError("No positive training images were found.")

        if self.negative_per_epoch > len(self.negative_indices):
            raise ValueError(
                "negative_per_epoch exceeds available empty images: "
                f"{self.negative_per_epoch} > {len(self.negative_indices)}"
            )

        self.negatives_per_global_batch = max(
            1,
            int(round(
                self.global_batch_size * self.negative_ratio
            )),
        )
        self.positives_per_global_batch = (
            self.global_batch_size
            - self.negatives_per_global_batch
        )

        # 为保证每个 rank 至少有一张正图，
        # 每个全局 batch 的正图数不能少于 rank 数。
        if self.positives_per_global_batch < self.num_replicas:
            raise ValueError(
                "Too many negatives per global batch to guarantee "
                "one positive image per rank. "
                f"global_batch={self.global_batch_size}, "
                f"positive_per_batch={self.positives_per_global_batch}, "
                f"world_size={self.num_replicas}"
            )

        # local batch 中至少保留一个正图，因此每个 rank 最多只能
        # 放 local_batch_size - 1 个空图。
        max_negatives_without_empty_rank = (
            self.num_replicas
            * max(self.local_batch_size - 1, 0)
        )

        if (
            self.negatives_per_global_batch
            > max_negatives_without_empty_rank
        ):
            raise ValueError(
                "Current local batch is too small to distribute negatives "
                "while keeping at least one positive on every rank. "
                f"negatives_per_global_batch="
                f"{self.negatives_per_global_batch}, "
                f"maximum={max_negatives_without_empty_rank}"
            )

        positive_batches = math.ceil(
            len(self.positive_indices)
            / self.positives_per_global_batch
        )
        negative_batches = math.ceil(
            self.negative_per_epoch
            / self.negatives_per_global_batch
        )

        self.num_global_batches = max(
            positive_batches,
            negative_batches,
        )

        self.required_positive_count = (
            self.num_global_batches
            * self.positives_per_global_batch
        )
        self.required_negative_count = (
            self.num_global_batches
            * self.negatives_per_global_batch
        )

    def set_epoch(self, epoch: int) -> None:
        """设置当前 epoch，使每轮抽取不同但可复现的空图。"""
        self.epoch = int(epoch)

    @staticmethod
    def _pad_to_length(
        indices: list[int],
        required_length: int,
    ) -> list[int]:
        """
        将索引补齐到固定长度。

        只在最后一个不完整 batch 出现极少量重复，不会丢掉正图。
        """
        if len(indices) >= required_length:
            return indices[:required_length]

        if not indices:
            raise RuntimeError("Cannot pad an empty index list.")

        result = list(indices)
        need = required_length - len(result)

        while need > 0:
            take = min(need, len(indices))
            result.extend(indices[:take])
            need -= take

        return result

    def _build_epoch_indices(
        self,
    ) -> tuple[list[int], list[int]]:
        """
        生成当前 epoch 的：
            - 全部随机排列后的正图
            - 随机抽取的空图
        """
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        positive_perm = torch.randperm(
            len(self.positive_indices),
            generator=generator,
        ).tolist()

        negative_perm = torch.randperm(
            len(self.negative_indices),
            generator=generator,
        )[:self.negative_per_epoch].tolist()

        selected_positive = [
            self.positive_indices[i]
            for i in positive_perm
        ]
        selected_negative = [
            self.negative_indices[i]
            for i in negative_perm
        ]

        selected_positive = self._pad_to_length(
            selected_positive,
            self.required_positive_count,
        )
        selected_negative = self._pad_to_length(
            selected_negative,
            self.required_negative_count,
        )

        return selected_positive, selected_negative

    def __iter__(self) -> Iterator[list[int]]:
        positive, negative = self._build_epoch_indices()

        pos_cursor = 0
        neg_cursor = 0

        python_rng = random.Random(self.seed + self.epoch)

        for batch_idx in range(self.num_global_batches):
            pos_chunk = positive[
                pos_cursor:
                pos_cursor + self.positives_per_global_batch
            ]
            neg_chunk = negative[
                neg_cursor:
                neg_cursor + self.negatives_per_global_batch
            ]

            pos_cursor += self.positives_per_global_batch
            neg_cursor += self.negatives_per_global_batch

            local_batches: list[list[int | None]] = [
                [None] * self.local_batch_size
                for _ in range(self.num_replicas)
            ]

            # 每个 global batch 轮换 rank 顺序，
            # 避免总是同一批 GPU 接收空图。
            rank_order = [
                (batch_idx + offset) % self.num_replicas
                for offset in range(self.num_replicas)
            ]

            pos_idx = 0

            # 第一个位置全部填正图，保证每个 rank 至少一张正图。
            for rank_id in rank_order:
                local_batches[rank_id][0] = pos_chunk[pos_idx]
                pos_idx += 1

            # 构建各 rank 的剩余槽位。
            remaining_slots: list[tuple[int, int]] = []

            for slot_id in range(
                1,
                self.local_batch_size,
            ):
                for rank_id in rank_order:
                    remaining_slots.append(
                        (rank_id, slot_id)
                    )

            # 先把空图放入不同 rank 的剩余槽位。
            for neg_index, (rank_id, slot_id) in zip(
                neg_chunk,
                remaining_slots[
                    :self.negatives_per_global_batch
                ],
            ):
                local_batches[rank_id][slot_id] = neg_index

            # 其余槽位全部填正图。
            for rank_id, slot_id in remaining_slots[
                self.negatives_per_global_batch:
            ]:
                local_batches[rank_id][slot_id] = pos_chunk[pos_idx]
                pos_idx += 1

            local_batch = local_batches[self.rank]

            if any(index is None for index in local_batch):
                raise RuntimeError(
                    "Batch construction left an unfilled slot."
                )

            # local batch 内部再轻微打乱，避免固定正/负位置。
            final_batch = [int(index) for index in local_batch]
            python_rng.shuffle(final_batch)

            yield final_batch

    def summary(self) -> str:
        """Return one concise line describing the active SODA-A sampling contract."""
        padding_duplicates = (
            self.required_positive_count
            + self.required_negative_count
            - len(self.positive_indices)
            - self.negative_per_epoch
        )
        return (
            "SODA-A balanced sampler: enabled=1, "
            f"epoch={self.epoch}, positive={len(self.positive_indices)}, "
            f"empty_pool={len(self.negative_indices)}, empty_unique={self.negative_per_epoch}, "
            f"world_size={self.num_replicas}, local_batch={self.local_batch_size}, "
            f"global_batch={self.global_batch_size}, "
            f"batch_mix={self.positives_per_global_batch}POS+{self.negatives_per_global_batch}EMPTY, "
            f"batches={self.num_global_batches}, "
            f"padded_total={self.num_global_batches * self.global_batch_size}, "
            f"padding_duplicates={padding_duplicates}, seed={self.seed}"
        )

    def __len__(self) -> int:
        """每个 rank 每个 epoch 的 batch 数。"""
        return self.num_global_batches
