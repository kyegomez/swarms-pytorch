import torch
from swarms_torch.mergers.mm_mergers import (
    modality_weighted_merge,
    modality_specific_layer_swap,
    cross_modality_weight_crossover,
    hierarchical_modality_fusion,
    modality_mutation_merge,
)

if __name__ == "__main__":
    # Example models, replace with actual multi-modal model instances
    model_1 = torch.nn.Linear(
        100, 50
    )  # Assume multi-modal model (e.g., image + text)
    model_2 = torch.nn.Linear(100, 50)
    model_3 = torch.nn.Linear(100, 50)

    # Perform Modality-Weighted Merge
    merged_model_wm = modality_weighted_merge(
        [model_1, model_2, model_3], modality_weights=[0.6, 0.3, 0.1]
    )

    # Perform Modality-Specific Layer Swap
    merged_model_ls = modality_specific_layer_swap(
        [model_1, model_2], modality_layer_map=["image", "text"]
    )

    # Perform Cross-Modality Weight Crossover
    merged_model_cm = cross_modality_weight_crossover(
        [model_1, model_2], modality_pairs=[(0, 1)], crossover_fraction=0.5
    )

    # Perform Hierarchical Modality Fusion
    merged_model_hf = hierarchical_modality_fusion(
        [model_1, model_2, model_3], modality_hierarchy=[[0], [1, 2]]
    )

    # Perform Modality Mutation Merge
    merged_model_mm = modality_mutation_merge(
        [model_1, model_2, model_3], mutation_rate=0.01
    )
