import torch
from typing import List, Tuple
from loguru import logger

# Initialize the logger
logger.add("multimodal_mergers.log", rotation="10 MB")


def modality_weighted_merge(
    models: List[torch.nn.Module], modality_weights: List[float]
) -> torch.nn.Module:
    """
    Merge multi-modal models by weighting their contributions based on modality importance.

    Args:
        models (List[torch.nn.Module]): List of multi-modal models to merge.
        modality_weights (List[float]): Weights assigned to each model's modality (e.g., 0.7 for image, 0.3 for text).

    Returns:
        torch.nn.Module: Merged multi-modal model with modality-weighted parameters.
    """
    logger.info("Starting Modality-Weighted Merge with {} models", len(models))
    merged_model = models[0]

    for key in merged_model.state_dict().keys():
        weighted_sum = torch.sum(
            torch.stack(
                [
                    modality_weights[i] * models[i].state_dict()[key]
                    for i in range(len(models))
                ]
            ),
            dim=0,
        )
        merged_model.state_dict()[key].copy_(weighted_sum)

    logger.info("Modality-Weighted Merge completed")
    return merged_model


def modality_specific_layer_swap(
    models: List[torch.nn.Module], modality_layer_map: List[str]
) -> torch.nn.Module:
    """
    Swap layers between models based on modality-specific features (e.g., swap image layers from one model with text layers from another).

    Args:
        models (List[torch.nn.Module]): List of multi-modal models to merge.
        modality_layer_map (List[str]): List of model names to use for specific layers (e.g., ['image', 'text']).

    Returns:
        torch.nn.Module: Merged multi-modal model with swapped modality-specific layers.
    """
    logger.info("Starting Modality-Specific Layer Swap")
    merged_model = models[0]

    for key in merged_model.state_dict().keys():
        modality_choice = (
            modality_layer_map[0] if "image" in key else modality_layer_map[1]
        )
        selected_model = models[modality_layer_map.index(modality_choice)]
        merged_model.state_dict()[key].copy_(selected_model.state_dict()[key])

    logger.info("Modality-Specific Layer Swap completed")
    return merged_model


def cross_modality_weight_crossover(
    models: List[torch.nn.Module],
    modality_pairs: List[Tuple[int, int]],
    crossover_fraction: float,
) -> torch.nn.Module:
    """
    Perform cross-modality weight crossover by merging specific modalities from different models.

    Args:
        models (List[torch.nn.Module]): List of multi-modal models to merge.
        modality_pairs (List[Tuple[int, int]]): List of tuples indicating modality pairs to cross over.
        crossover_fraction (float): Fraction of the layer's weights to be swapped between modalities.

    Returns:
        torch.nn.Module: Merged multi-modal model with cross-modality weight crossover.
    """
    logger.info(
        "Starting Cross-Modality Weight Crossover with crossover fraction {}",
        crossover_fraction,
    )
    merged_model = models[0]

    for key in merged_model.state_dict().keys():
        if len(key.split(".")) > 1:  # Avoid non-weight parameters
            for i, j in modality_pairs:
                layer_size = models[i].state_dict()[key].shape[0]
                crossover_size = int(crossover_fraction * layer_size)
                (
                    models[i].state_dict()[key][:crossover_size],
                    models[j].state_dict()[key][:crossover_size],
                ) = (
                    models[j].state_dict()[key][:crossover_size],
                    models[i].state_dict()[key][:crossover_size],
                )

            merged_model.state_dict()[key].copy_(models[i].state_dict()[key])

    logger.info("Cross-Modality Weight Crossover completed")
    return merged_model


def hierarchical_modality_fusion(
    models: List[torch.nn.Module], modality_hierarchy: List[List[int]]
) -> torch.nn.Module:
    """
    Merge models based on a hierarchical structure of modalities, where lower layers are modality-specific and upper layers are fused across modalities.

    Args:
        models (List[torch.nn.Module]): List of multi-modal models to merge.
        modality_hierarchy (List[List[int]]): A hierarchy defining which modalities contribute to lower and higher layers.

    Returns:
        torch.nn.Module: Merged multi-modal model based on hierarchical modality fusion.
    """
    logger.info("Starting Hierarchical Modality Fusion")
    merged_model = models[0]

    for key in merged_model.state_dict().keys():
        layer_hierarchy = (
            modality_hierarchy[0] if "lower" in key else modality_hierarchy[1]
        )
        merged_weights = torch.mean(
            torch.stack(
                [models[idx].state_dict()[key] for idx in layer_hierarchy]
            ),
            dim=0,
        )
        merged_model.state_dict()[key].copy_(merged_weights)

    logger.info("Hierarchical Modality Fusion completed")
    return merged_model


def modality_mutation_merge(
    models: List[torch.nn.Module], mutation_rate: float
) -> torch.nn.Module:
    """
    Merge models by introducing mutations (small random changes) into the weights of specific modalities to encourage diversity.

    Args:
        models (List[torch.nn.Module]): List of multi-modal models to merge.
        mutation_rate (float): The rate at which mutations are applied to the model weights.

    Returns:
        torch.nn.Module: Merged multi-modal model with modality-specific mutations.
    """
    logger.info(
        "Starting Modality Mutation Merge with mutation rate {}", mutation_rate
    )
    merged_model = models[0]

    for key in merged_model.state_dict().keys():
        mutation_tensor = (
            torch.randn_like(merged_model.state_dict()[key]) * mutation_rate
        )
        modality_weights = torch.mean(
            torch.stack([model.state_dict()[key] for model in models]), dim=0
        )
        mutated_weights = modality_weights + mutation_tensor
        merged_model.state_dict()[key].copy_(mutated_weights)

    logger.info("Modality Mutation Merge completed")
    return merged_model


# # Example of how to use the logger and multi-modal merge methods

# if __name__ == "__main__":
#     # Example models, replace with actual multi-modal model instances
#     model_1 = torch.nn.Linear(
#         100, 50
#     )  # Assume multi-modal model (e.g., image + text)
#     model_2 = torch.nn.Linear(100, 50)
#     model_3 = torch.nn.Linear(100, 50)

#     # Perform Modality-Weighted Merge
#     merged_model_wm = modality_weighted_merge(
#         [model_1, model_2, model_3], modality_weights=[0.6, 0.3, 0.1]
#     )

#     # Perform Modality-Specific Layer Swap
#     merged_model_ls = modality_specific_layer_swap(
#         [model_1, model_2], modality_layer_map=["image", "text"]
#     )

#     # Perform Cross-Modality Weight Crossover
#     merged_model_cm = cross_modality_weight_crossover(
#         [model_1, model_2], modality_pairs=[(0, 1)], crossover_fraction=0.5
#     )

#     # Perform Hierarchical Modality Fusion
#     merged_model_hf = hierarchical_modality_fusion(
#         [model_1, model_2, model_3], modality_hierarchy=[[0], [1, 2]]
#     )

#     # Perform Modality Mutation Merge
#     merged_model_mm = modality_mutation_merge(
#         [model_1, model_2, model_3], mutation_rate=0.01
#     )
