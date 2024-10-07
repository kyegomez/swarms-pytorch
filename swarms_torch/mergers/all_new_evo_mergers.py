import torch
from loguru import logger
from typing import List

# Initialize the logger
logger.add("evolutionary_merge.log", rotation="10 MB")


def hyperslice_merge(
    models: List[torch.nn.Module], slice_indices: List[int]
) -> torch.nn.Module:
    """
    Merge models by selecting specific slices of weight tensors across the models.

    Args:
        models (List[torch.nn.Module]): List of pre-trained models to merge.
        slice_indices (List[int]): Indices of the slices to be merged from each model's weights.

    Returns:
        torch.nn.Module: Merged model with selected slices combined.
    """
    logger.info("Starting HyperSlice merging with {} models", len(models))
    merged_model = models[0]

    for key in merged_model.state_dict().keys():
        tensors = [model.state_dict()[key] for model in models]
        slices = [
            tensor.index_select(0, torch.tensor(slice_indices))
            for tensor in tensors
        ]
        merged_slices = torch.mean(torch.stack(slices), dim=0)
        remaining = tensors[0].index_select(
            0,
            torch.tensor(
                [i for i in range(tensors[0].size(0)) if i not in slice_indices]
            ),
        )
        merged_weights = torch.cat((merged_slices, remaining), dim=0)
        merged_model.state_dict()[key].copy_(merged_weights)

    logger.info("HyperSlice merging completed")
    return merged_model


def random_subspace_merge(
    models: List[torch.nn.Module], subspace_fraction: float = 0.5
) -> torch.nn.Module:
    """
    Merge models by randomly selecting subspaces of weights and averaging them.

    Args:
        models (List[torch.nn.Module]): List of pre-trained models to merge.
        subspace_fraction (float): Fraction of the weight tensor to select and merge.

    Returns:
        torch.nn.Module: Merged model with selected subspaces combined.
    """
    logger.info(
        "Starting Random Subspace merging with subspace fraction of {}",
        subspace_fraction,
    )
    merged_model = models[0]

    for key in merged_model.state_dict().keys():
        layer_shape = models[0].state_dict()[key].shape
        subspace_size = int(subspace_fraction * layer_shape[0])
        subspace_indices = torch.randperm(layer_shape[0])[:subspace_size]

        subspace_weights = torch.mean(
            torch.stack(
                [model.state_dict()[key][subspace_indices] for model in models]
            ),
            dim=0,
        )
        remaining_weights = models[0].state_dict()[key][subspace_size:]

        merged_weights = torch.cat((subspace_weights, remaining_weights), dim=0)
        merged_model.state_dict()[key].copy_(merged_weights)

    logger.info("Random Subspace merging completed")
    return merged_model


def dimensional_cross_fusion(
    models: List[torch.nn.Module], cross_axis: int = 0
) -> torch.nn.Module:
    """
    Merge models by fusing weights along a specific axis (dimension) across models.

    Args:
        models (List[torch.nn.Module]): List of pre-trained models to merge.
        cross_axis (int): Axis along which to perform cross-fusion (0 for rows, 1 for columns).

    Returns:
        torch.nn.Module: Merged model with cross-fused weights.
    """
    logger.info(
        "Starting Dimensional Cross-fusion merging along axis {}", cross_axis
    )
    merged_model = models[0]

    for key in merged_model.state_dict().keys():
        tensors = [model.state_dict()[key] for model in models]

        if cross_axis == 0:
            merged_weights = torch.cat(
                [
                    tensors[0][: tensors[0].size(0) // 2],
                    tensors[1][tensors[1].size(0) // 2 :],
                ],
                dim=0,
            )
        elif cross_axis == 1:
            merged_weights = torch.cat(
                [
                    tensors[0][:, : tensors[0].size(1) // 2],
                    tensors[1][:, tensors[1].size(1) // 2 :],
                ],
                dim=1,
            )
        else:
            logger.error("Invalid cross-axis specified: {}", cross_axis)
            raise ValueError("Cross-axis must be 0 or 1")

        merged_model.state_dict()[key].copy_(merged_weights)

    logger.info("Dimensional Cross-fusion merging completed")
    return merged_model


def weighted_evolutionary_crossover(
    models: List[torch.nn.Module], performance_scores: List[float]
) -> torch.nn.Module:
    """
    Merge models using a weighted crossover based on performance scores.

    Args:
        models (List[torch.nn.Module]): List of pre-trained models to merge.
        performance_scores (List[float]): List of performance scores corresponding to the models.

    Returns:
        torch.nn.Module: Merged model with weights combined using weighted crossovers.
    """
    logger.info(
        "Starting Weighted Evolutionary Crossover with performance scores: {}",
        performance_scores,
    )
    merged_model = models[0]

    for key in merged_model.state_dict().keys():
        weights = torch.tensor(performance_scores, dtype=torch.float32) / sum(
            performance_scores
        )
        merged_weights = torch.sum(
            torch.stack(
                [
                    weights[i] * models[i].state_dict()[key]
                    for i in range(len(models))
                ]
            ),
            dim=0,
        )
        merged_model.state_dict()[key].copy_(merged_weights)

    logger.info("Weighted Evolutionary Crossover completed")
    return merged_model


def permutation_weight_swapping(
    models: List[torch.nn.Module], permutation_seed: int = 42
) -> torch.nn.Module:
    """
    Merge models by permuting weight matrices and swapping them between models.

    Args:
        models (List[torch.nn.Module]): List of pre-trained models to merge.
        permutation_seed (int): Random seed for permutation consistency.

    Returns:
        torch.nn.Module: Merged model with permuted weight swapping.
    """
    logger.info(
        "Starting Permutation-based Weight Swapping with seed {}",
        permutation_seed,
    )
    merged_model = models[0]
    torch.manual_seed(permutation_seed)

    for key in merged_model.state_dict().keys():
        tensors = [model.state_dict()[key] for model in models]

        # Ensure the permutation respects the tensor size
        perm = torch.randperm(tensors[0].numel())
        perm = perm[: tensors[0].numel()].reshape(tensors[0].shape)

        # Ensure that swapped weights stay within the bounds of the tensor shape
        swapped_weights = tensors[1].reshape(-1)[perm].reshape(tensors[1].shape)
        merged_model.state_dict()[key].copy_(swapped_weights)

    logger.info("Permutation-based Weight Swapping completed")
    return merged_model


# Example of how to use the logger and merge methods

if __name__ == "__main__":
    # Example models, replace with actual model instances
    model_1 = torch.nn.Linear(10, 10)
    model_2 = torch.nn.Linear(10, 10)
    model_3 = torch.nn.Linear(10, 10)

    # Perform HyperSlice merge
    merged_model_hs = hyperslice_merge(
        [model_1, model_2, model_3], slice_indices=[0, 2, 4]
    )

    # Perform Random Subspace merge
    merged_model_rs = random_subspace_merge(
        [model_1, model_2, model_3], subspace_fraction=0.5
    )

    # Perform Dimensional Cross-fusion merge
    merged_model_dc = dimensional_cross_fusion([model_1, model_2], cross_axis=0)

    # Perform Weighted Evolutionary Crossover merge
    merged_model_wc = weighted_evolutionary_crossover(
        [model_1, model_2, model_3], performance_scores=[0.7, 0.85, 0.65]
    )

    # Perform Permutation-based Weight Swapping
    merged_model_pw = permutation_weight_swapping(
        [model_1, model_2], permutation_seed=42
    )
