import torch 
from swarms_torch.mergers.all_new_evo_mergers import (
    hyperslice_merge,
    random_subspace_merge,
    dimensional_cross_fusion,
    weighted_evolutionary_crossover,
    permutation_weight_swapping,
)

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
