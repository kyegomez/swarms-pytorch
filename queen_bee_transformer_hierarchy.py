"""
Queen Bee Transformer Hierarchy (QBTH) Implementation

This module implements a hierarchical transformer architecture where one "queen" transformer
dominates decision-making while worker transformers specialize in different tasks.

The architecture combines:
- Hierarchical transformer organization with queen-worker dynamics
- Dynamic role assignment using genetic algorithm-based evolution
- Adaptive specialization of worker transformers
- Queen-worker communication protocols
- Genetic mutation of transformer weights

Reference:
----------
Inspired by biological bee colony organization and genetic algorithms for
neural architecture optimization.

Author: Swarms PyTorch Team
License: MIT
"""

import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from loguru import logger
except ImportError:
    # Simple logging replacement if loguru not available
    class SimpleLogger:
        def info(self, msg: str): print(f"INFO: {msg}")
        def debug(self, msg: str): print(f"DEBUG: {msg}")
    logger = SimpleLogger()


class SimpleTransformerLayer(nn.Module):
    """Simple transformer layer without external dependencies."""
    
    def __init__(self, dim: int, heads: int, dim_head: int, dropout: float = 0.1):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        
        inner_dim = heads * dim_head
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, dim = x.shape
        
        # Self attention
        residual = x
        x = self.norm1(x)
        
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(batch_size, seq_len, self.heads, self.dim_head).transpose(1, 2), qkv)
        
        # Attention
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        out = self.to_out(out)
        
        x = residual + self.dropout(out)
        
        # Feed forward
        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = residual + x
        
        return x


class SimpleDecoder(nn.Module):
    """Simple transformer decoder without external dependencies."""
    
    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            SimpleTransformerLayer(dim, heads, dim_head, dropout)
            for _ in range(depth)
        ])
        
    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# Replace the zeta imports with our simple implementation
Decoder = SimpleDecoder
Transformer = SimpleDecoder


class GeneticTransformerEvolution(nn.Module):
    """
    Genetic Algorithm for evolving transformer weights and structure.
    
    This component handles the evolutionary aspects of the QBTH system,
    including mutation, crossover, and fitness evaluation of transformer weights.
    
    Args:
        mutation_rate (float): Base mutation rate for weight perturbation
        strong_mutation_rate (float): Rate for applying strong mutations
        elite_ratio (float): Proportion of top performers to preserve
        crossover_rate (float): Probability of crossover operations
        
    Example:
        >>> genetic_evolver = GeneticTransformerEvolution(
        ...     mutation_rate=0.01,
        ...     strong_mutation_rate=0.1,
        ...     elite_ratio=0.2,
        ...     crossover_rate=0.8
        ... )
    """
    
    def __init__(
        self,
        mutation_rate: float = 0.01,
        strong_mutation_rate: float = 0.1, 
        elite_ratio: float = 0.2,
        crossover_rate: float = 0.8,
    ):
        super().__init__()
        self.mutation_rate = mutation_rate
        self.strong_mutation_rate = strong_mutation_rate
        self.elite_ratio = elite_ratio
        self.crossover_rate = crossover_rate
        
        logger.info(
            f"Initialized GeneticTransformerEvolution with mutation_rate={mutation_rate}, "
            f"strong_mutation_rate={strong_mutation_rate}, elite_ratio={elite_ratio}"
        )
        
    def mutate_weights(
        self, 
        weights: Tensor, 
        fitness_score: float,
        generation: int
    ) -> Tensor:
        """
        Apply genetic mutations to transformer weights based on fitness.
        
        Args:
            weights: Transformer weights to mutate
            fitness_score: Current fitness score (higher is better)
            generation: Current generation number
            
        Returns:
            Mutated weights tensor
        """
        # Adaptive mutation rate based on fitness and generation
        adaptive_rate = self.mutation_rate * (1.0 / (1.0 + fitness_score)) * (1.0 + 0.1 * generation)
        
        # Create mutation mask
        mutation_mask = torch.rand_like(weights) < adaptive_rate
        
        # Generate noise for mutations
        noise = torch.randn_like(weights) * 0.1
        
        # Apply mutations
        mutated_weights = torch.where(mutation_mask, weights + noise, weights)
        
        # Apply strong mutations for exploration
        if random.random() < self.strong_mutation_rate:
            strong_mask = torch.rand_like(weights) < (adaptive_rate * 2)
            strong_noise = torch.randn_like(weights) * 0.5
            mutated_weights = torch.where(strong_mask, weights + strong_noise, mutated_weights)
            
        logger.debug(f"Applied mutations with adaptive_rate={adaptive_rate:.4f}")
        return mutated_weights
        
    def crossover_weights(
        self, 
        parent1_weights: Tensor, 
        parent2_weights: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Perform crossover operation between two parent weight tensors.
        
        Args:
            parent1_weights: First parent weights
            parent2_weights: Second parent weights
            
        Returns:
            Tuple of two offspring weight tensors
        """
        if random.random() > self.crossover_rate:
            return parent1_weights, parent2_weights
            
        # Single-point crossover for transformer weights
        crossover_point = random.randint(1, parent1_weights.numel() - 1)
        
        flat_p1 = parent1_weights.flatten()
        flat_p2 = parent2_weights.flatten()
        
        # Create offspring
        offspring1 = torch.cat([flat_p1[:crossover_point], flat_p2[crossover_point:]])
        offspring2 = torch.cat([flat_p2[:crossover_point], flat_p1[crossover_point:]])
        
        # Reshape back to original shape
        offspring1 = offspring1.reshape(parent1_weights.shape)
        offspring2 = offspring2.reshape(parent2_weights.shape)
        
        logger.debug(f"Performed crossover at point {crossover_point}")
        return offspring1, offspring2


class QueenWorkerCommunication(nn.Module):
    """
    Communication protocol between queen and worker transformers.
    
    This module implements the information exchange mechanism that allows
    the queen transformer to coordinate with worker transformers and
    aggregate their specialized outputs.
    
    Args:
        queen_dim (int): Dimension of queen transformer
        worker_dim (int): Dimension of worker transformers  
        num_workers (int): Number of worker transformers
        communication_heads (int): Number of attention heads for communication
        
    Example:
        >>> comm = QueenWorkerCommunication(
        ...     queen_dim=512,
        ...     worker_dim=256, 
        ...     num_workers=4,
        ...     communication_heads=8
        ... )
    """
    
    def __init__(
        self,
        queen_dim: int,
        worker_dim: int,
        num_workers: int,
        communication_heads: int = 8,
    ):
        super().__init__()
        self.queen_dim = queen_dim
        self.worker_dim = worker_dim
        self.num_workers = num_workers
        self.communication_heads = communication_heads
        
        # Queen-to-worker communication
        self.queen_to_worker = nn.MultiheadAttention(
            embed_dim=worker_dim,
            num_heads=communication_heads,
            batch_first=True
        )
        
        # Worker-to-queen communication
        self.worker_to_queen = nn.MultiheadAttention(
            embed_dim=queen_dim,
            num_heads=communication_heads, 
            batch_first=True
        )
        
        # Projection layers for dimension alignment
        self.queen_projection = nn.Linear(queen_dim, worker_dim)
        self.worker_projection = nn.Linear(worker_dim, queen_dim)
        
        # Gating mechanism for controlling information flow
        self.communication_gate = nn.Sequential(
            nn.Linear(queen_dim + worker_dim * num_workers, 128),
            nn.ReLU(),
            nn.Linear(128, num_workers),
            nn.Sigmoid()
        )
        
        logger.info(
            f"Initialized QueenWorkerCommunication with queen_dim={queen_dim}, "
            f"worker_dim={worker_dim}, num_workers={num_workers}"
        )
        
    def forward(
        self,
        queen_state: Tensor,
        worker_states: List[Tensor]
    ) -> Tuple[Tensor, List[Tensor]]:
        """
        Execute communication between queen and workers.
        
        Args:
            queen_state: Queen transformer hidden state [batch, seq_len, queen_dim]
            worker_states: List of worker hidden states [batch, seq_len, worker_dim]
            
        Returns:
            Tuple of updated (queen_state, worker_states)
        """
        batch_size, seq_len = queen_state.shape[:2]
        
        # Project queen state for worker communication
        queen_projected = self.queen_projection(queen_state)
        
        # Queen broadcasts instructions to workers
        updated_workers = []
        for i, worker_state in enumerate(worker_states):
            # Use queen state as query, worker state as key/value
            attended_worker, _ = self.queen_to_worker(
                queen_projected,
                worker_state, 
                worker_state
            )
            updated_workers.append(attended_worker)
            
        # Workers report back to queen - keep in original worker dimension for concatenation
        worker_reports = torch.cat(updated_workers, dim=-1)  # [batch, seq_len, worker_dim * num_workers]
        combined_state = torch.cat([queen_state, worker_reports], dim=-1)  # [batch, seq_len, queen_dim + worker_dim * num_workers]
        
        # Compute communication gates
        gates = self.communication_gate(combined_state.mean(dim=1))  # [batch, num_workers]
        
        # Apply gates to worker contributions - simplified gating
        gated_workers = []
        for i, worker in enumerate(updated_workers):
            # worker shape: [batch, seq_len, worker_dim]
            # gates[:, i] shape: [batch]
            gate = gates[:, i].unsqueeze(1).unsqueeze(2)  # [batch, 1, 1]
            gated_worker = worker * gate  # Broadcasting: [batch, seq_len, worker_dim] * [batch, 1, 1]
            gated_workers.append(gated_worker)
        
        # Aggregate worker information for queen update - project to queen dimension here
        # Project each worker to queen dimension, then average them
        projected_workers = [self.worker_projection(w) for w in gated_workers]
        worker_aggregate = torch.mean(torch.stack(projected_workers, dim=0), dim=0)  # [batch, seq_len, queen_dim]
        
        # Update queen state with worker information
        updated_queen, _ = self.worker_to_queen(
            queen_state,
            worker_aggregate,
            worker_aggregate
        )
        
        logger.debug(f"Communication completed with gates: {gates.mean().item():.4f}")
        return updated_queen, gated_workers


class WorkerTransformer(nn.Module):
    """
    Specialized worker transformer for the QBTH hierarchy.
    
    Each worker transformer specializes in specific aspects of the task
    through genetic evolution and queen guidance.
    
    Args:
        dim (int): Model dimension
        depth (int): Number of transformer layers
        heads (int): Number of attention heads
        dim_head (int): Dimension per attention head
        max_seq_len (int): Maximum sequence length
        specialization_id (int): Unique identifier for specialization
        dropout (float): Dropout rate
        
    Example:
        >>> worker = WorkerTransformer(
        ...     dim=256,
        ...     depth=4,
        ...     heads=8,
        ...     dim_head=32,
        ...     max_seq_len=1024,
        ...     specialization_id=1
        ... )
    """
    
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        max_seq_len: int,
        specialization_id: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head
        self.max_seq_len = max_seq_len
        self.specialization_id = specialization_id
        
        # Core transformer architecture
        self.transformer = Decoder(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout
        )
        
        # Specialization embeddings
        self.specialization_embedding = nn.Parameter(torch.randn(1, 1, dim))
        
        # Fitness tracking for genetic evolution
        self.register_buffer('fitness_score', torch.tensor(0.0))
        self.register_buffer('performance_history', torch.zeros(100))  # Last 100 performances
        self.register_buffer('generation', torch.tensor(0))
        
        logger.info(f"Initialized WorkerTransformer {specialization_id} with dim={dim}, depth={depth}")
        
    def forward(self, x: Tensor, queen_guidance: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass of worker transformer.
        
        Args:
            x: Input tensor [batch, seq_len, dim]
            queen_guidance: Optional guidance from queen [batch, seq_len, dim]
            
        Returns:
            Worker output tensor [batch, seq_len, dim]
        """
        batch_size, seq_len = x.shape[:2]
        
        # Add specialization embedding
        spec_embed = self.specialization_embedding.expand(batch_size, seq_len, -1)
        x = x + spec_embed
        
        # Incorporate queen guidance if provided
        if queen_guidance is not None:
            x = x + 0.1 * queen_guidance  # Weighted integration
            
        # Apply transformer
        output = self.transformer(x)
        
        return output
        
    def update_fitness(self, performance: float):
        """Update fitness score based on recent performance."""
        # Update performance history
        self.performance_history = torch.roll(self.performance_history, 1)
        self.performance_history[0] = performance
        
        # Compute exponential moving average fitness
        alpha = 0.1
        self.fitness_score = (1 - alpha) * self.fitness_score + alpha * performance
        
        logger.debug(f"Worker {self.specialization_id} fitness updated to {self.fitness_score.item():.4f}")
        
    def get_fitness(self) -> float:
        """Get current fitness score."""
        return self.fitness_score.item()


class QueenBeeTransformerHierarchy(nn.Module):
    """
    Queen Bee Transformer Hierarchy (QBTH) - Main Architecture
    
    Implements a hierarchical transformer where one "queen" transformer dominates 
    decision-making while worker transformers specialize in different tasks.
    
    Key Features:
    - Hierarchical transformer organization with queen-worker dynamics
    - Dynamic role assignment using genetic algorithm-based evolution  
    - Adaptive specialization of worker transformers
    - Queen-worker communication protocols
    - Genetic mutation of transformer weights
    
    Args:
        num_tokens (int): Vocabulary size
        max_seq_len (int): Maximum sequence length
        queen_dim (int): Queen transformer dimension
        worker_dim (int): Worker transformer dimension  
        queen_depth (int): Queen transformer depth
        worker_depth (int): Worker transformer depth
        queen_heads (int): Queen attention heads
        worker_heads (int): Worker attention heads
        num_workers (int): Number of worker transformers
        dim_head (int): Dimension per attention head
        dropout (float): Dropout rate
        evolution_frequency (int): Generations between evolution steps
        mutation_rate (float): Base mutation rate for evolution
        
    Example:
        >>> model = QueenBeeTransformerHierarchy(
        ...     num_tokens=50000,
        ...     max_seq_len=1024,
        ...     queen_dim=512,
        ...     worker_dim=256,
        ...     queen_depth=12,
        ...     worker_depth=6,
        ...     queen_heads=16,
        ...     worker_heads=8,
        ...     num_workers=4
        ... )
        >>> 
        >>> x = torch.randint(0, 50000, (2, 1024))
        >>> output = model(x)
        >>> print(output['logits'].shape)  # torch.Size([2, 1024, 50000])
    """
    
    def __init__(
        self,
        num_tokens: int,
        max_seq_len: int = 1024,
        queen_dim: int = 512,
        worker_dim: int = 256,
        queen_depth: int = 12,
        worker_depth: int = 6, 
        queen_heads: int = 16,
        worker_heads: int = 8,
        num_workers: int = 4,
        dim_head: int = 64,
        dropout: float = 0.1,
        evolution_frequency: int = 10,
        mutation_rate: float = 0.01,
    ):
        super().__init__()
        
        # Architecture parameters
        self.num_tokens = num_tokens
        self.max_seq_len = max_seq_len
        self.queen_dim = queen_dim
        self.worker_dim = worker_dim
        self.num_workers = num_workers
        self.evolution_frequency = evolution_frequency
        
        # Token embeddings
        self.token_embedding = nn.Embedding(num_tokens, queen_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, queen_dim)
        
        # Queen transformer decoder only (without final projection to vocab)
        self.queen_decoder = Decoder(
            dim=queen_dim,
            depth=queen_depth,
            heads=queen_heads,
            dim_head=dim_head,
            dropout=dropout
        )
        
        # Worker transformers - specialized processors
        self.workers = nn.ModuleList([
            WorkerTransformer(
                dim=worker_dim,
                depth=worker_depth,
                heads=worker_heads,
                dim_head=dim_head,
                max_seq_len=max_seq_len,
                specialization_id=i,
                dropout=dropout
            )
            for i in range(num_workers)
        ])
        
        # Communication system
        self.communication = QueenWorkerCommunication(
            queen_dim=queen_dim,
            worker_dim=worker_dim,
            num_workers=num_workers
        )
        
        # Genetic evolution system
        self.genetic_evolver = GeneticTransformerEvolution(
            mutation_rate=mutation_rate
        )
        
        # Dimension alignment layers
        self.queen_to_worker_proj = nn.Linear(queen_dim, worker_dim)
        self.worker_to_queen_proj = nn.Linear(worker_dim, queen_dim)
        
        # Output projection and decision aggregation
        self.worker_aggregate = nn.Linear(worker_dim * num_workers, queen_dim)
        self.final_decision = nn.Sequential(
            nn.Linear(queen_dim * 2, queen_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(queen_dim, num_tokens)
        )
        
        # Evolution tracking
        self.register_buffer('generation_count', torch.tensor(0))
        self.register_buffer('evolution_history', torch.zeros(1000))  # Track evolution progress
        
        logger.info(
            f"Initialized QBTH with queen_dim={queen_dim}, worker_dim={worker_dim}, "
            f"num_workers={num_workers}, evolution_frequency={evolution_frequency}"
        )
        
    def forward(self, x: Tensor, targets: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """
        Forward pass through the Queen Bee Transformer Hierarchy.
        
        Args:
            x: Input token indices [batch, seq_len]
            targets: Target tokens for training [batch, seq_len]
            
        Returns:
            Dictionary containing:
                - logits: Output logits [batch, seq_len, num_tokens]
                - queen_output: Queen transformer output
                - worker_outputs: List of worker outputs
                - fitness_scores: Current fitness scores of workers
        """
        batch_size, seq_len = x.shape
        
        # Token and position embeddings
        token_emb = self.token_embedding(x)
        pos_emb = self.pos_embedding(torch.arange(seq_len, device=x.device))
        queen_input = token_emb + pos_emb
        
        # Queen transformer processing - high-level decision making
        # Use decoder directly to get hidden states, not logits
        queen_output = self.queen_decoder(queen_input)
        
        # Project queen guidance for workers
        queen_guidance = self.queen_to_worker_proj(queen_output)
        
        # Worker transformer processing - specialized tasks
        worker_outputs = []
        for worker in self.workers:
            # Workers process the projected queen guidance
            worker_out = worker(queen_guidance, queen_guidance)
            worker_outputs.append(worker_out)
            
        # Queen-worker communication
        updated_queen, updated_workers = self.communication(queen_output, worker_outputs)
        
        # Aggregate worker decisions
        worker_concat = torch.cat(updated_workers, dim=-1)
        worker_aggregate = self.worker_aggregate(worker_concat)
        
        # Final decision combining queen and worker insights
        combined_representation = torch.cat([updated_queen, worker_aggregate], dim=-1)
        logits = self.final_decision(combined_representation)
        
        # Compute fitness scores if targets provided
        fitness_scores = [worker.get_fitness() for worker in self.workers]
        
        if targets is not None:
            self._update_fitness_scores(logits, targets)
            
        # Perform evolution if needed
        if self.generation_count % self.evolution_frequency == 0 and self.training:
            self._evolve_workers()
            
        self.generation_count += 1
        
        return {
            'logits': logits,
            'queen_output': updated_queen,
            'worker_outputs': updated_workers,
            'fitness_scores': fitness_scores,
        }
        
    def _update_fitness_scores(self, logits: Tensor, targets: Tensor):
        """Update fitness scores for workers based on performance."""
        # Compute loss for each worker's contribution
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        
        for i, worker in enumerate(self.workers):
            # Get worker-specific predictions (simplified metric)
            worker_logits = logits  # In practice, you'd isolate worker contribution
            worker_loss = loss_fn(
                worker_logits.reshape(-1, self.num_tokens),
                targets.reshape(-1)
            ).mean()
            
            # Convert loss to fitness (lower loss = higher fitness)
            fitness = 1.0 / (1.0 + worker_loss.item())
            worker.update_fitness(fitness)
            
        logger.debug(f"Updated fitness scores: {[w.get_fitness() for w in self.workers]}")
        
    def _evolve_workers(self):
        """Perform genetic evolution on worker transformers."""
        logger.info(f"Performing evolution at generation {self.generation_count}")
        
        # Get fitness scores
        fitness_scores = [worker.get_fitness() for worker in self.workers]
        
        # Sort workers by fitness
        sorted_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)
        
        # Evolve bottom-performing workers
        num_evolve = len(self.workers) // 2
        
        for i in range(num_evolve):
            worst_idx = sorted_indices[-(i+1)]
            best_idx = sorted_indices[i % (len(self.workers) - num_evolve)]
            
            # Mutate worst performer based on best performer
            with torch.no_grad():
                for param_worst, param_best in zip(
                    self.workers[worst_idx].parameters(),
                    self.workers[best_idx].parameters()
                ):
                    mutated = self.genetic_evolver.mutate_weights(
                        param_best.data,
                        fitness_scores[best_idx],
                        self.generation_count.item()
                    )
                    param_worst.data.copy_(mutated)
                    
        logger.info(f"Evolution completed. Fitness range: {min(fitness_scores):.4f} - {max(fitness_scores):.4f}")
        
    def get_hierarchy_stats(self) -> Dict[str, float]:
        """Get statistics about the current hierarchy state."""
        fitness_scores = [worker.get_fitness() for worker in self.workers]
        
        return {
            'generation': self.generation_count.item(),
            'avg_worker_fitness': sum(fitness_scores) / len(fitness_scores),
            'best_worker_fitness': max(fitness_scores),
            'worst_worker_fitness': min(fitness_scores),
            'fitness_std': torch.tensor(fitness_scores).std().item(),
        }


# Example usage and testing
if __name__ == "__main__":
    # Create model
    model = QueenBeeTransformerHierarchy(
        num_tokens=1000,
        max_seq_len=128,
        queen_dim=256,
        worker_dim=128,
        queen_depth=6,
        worker_depth=3,
        queen_heads=8,
        worker_heads=4,
        num_workers=3,
        evolution_frequency=5
    )
    
    # Test forward pass
    x = torch.randint(0, 1000, (2, 128))
    targets = torch.randint(0, 1000, (2, 128))
    
    with torch.no_grad():
        output = model(x, targets)
        
    logger.info(f"Model output shape: {output['logits'].shape}")
    logger.info(f"Hierarchy stats: {model.get_hierarchy_stats()}")
    
    print("Queen Bee Transformer Hierarchy test completed successfully!") 