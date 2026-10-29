# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch


def terminal_rewards_from_scores(scores: list[float] | torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """Place each scalar score on the last generated token of its response.

    ``response_mask`` has shape (batch_size, response_length), with 1 for
    assistant tokens and 0 for tool observations or padding. It may have holes
    between assistant turns. Rows with no assistant tokens receive no reward.
    """
    rewards = torch.zeros_like(response_mask, dtype=torch.float32)
    if response_mask.size(1) == 0:
        return rewards

    # A finish tool reply can outlast the final assistant turn. GAE skips these
    # observation tokens, so the score must land on the last mask=1 position.
    positions = torch.arange(response_mask.size(1), device=response_mask.device).expand_as(response_mask)
    last_positions = positions.masked_fill(~response_mask.bool(), -1).amax(dim=1)
    valid_rows = last_positions >= 0
    rows = torch.arange(response_mask.size(0), device=response_mask.device)[valid_rows]
    scores = torch.as_tensor(scores, dtype=rewards.dtype, device=rewards.device)
    rewards[rows, last_positions[valid_rows]] = scores[valid_rows]
    return rewards
