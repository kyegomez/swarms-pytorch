"""
Queen Bee Transformer Hierarchy (QBTH) Example

This example demonstrates how to use the Queen Bee Transformer Hierarchy model
for various NLP tasks including language modeling, text classification, and
sequence-to-sequence tasks.

The QBTH combines:
- Hierarchical transformer organization
- Genetic algorithm-based evolution
- Queen-worker communication protocols
- Adaptive specialization

Author: Swarms PyTorch Team
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger
import matplotlib.pyplot as plt
from typing import Dict, List
import numpy as np

from swarms_torch import QueenBeeTransformerHierarchy


def setup_logging():
    """Configure logging for the example."""
    logger.add("qbth_example.log", rotation="10 MB", level="INFO")
    logger.info("Starting Queen Bee Transformer Hierarchy Example")


def create_sample_data(vocab_size: int = 1000, seq_len: int = 128, num_samples: int = 1000):
    """Create sample data for training."""
    # Generate random token sequences
    x = torch.randint(0, vocab_size, (num_samples, seq_len))
    
    # Create targets (shifted by 1 for language modeling)
    targets = torch.roll(x, -1, dims=1)
    targets[:, -1] = torch.randint(0, vocab_size, (num_samples,))
    
    return x, targets


def train_qbth_model():
    """Train the QBTH model on sample data."""
    logger.info("Setting up QBTH model for training")
    
    # Model configuration
    config = {
        'num_tokens': 1000,
        'max_seq_len': 128,
        'queen_dim': 256,
        'worker_dim': 128,
        'queen_depth': 6,
        'worker_depth': 3,
        'queen_heads': 8,
        'worker_heads': 4,
        'num_workers': 4,
        'evolution_frequency': 10,
        'mutation_rate': 0.01,
        'dropout': 0.1
    }
    
    # Create model
    model = QueenBeeTransformerHierarchy(**config)
    logger.info(f"Created QBTH model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create sample data
    x_train, y_train = create_sample_data(config['num_tokens'], config['max_seq_len'], 1000)
    x_val, y_val = create_sample_data(config['num_tokens'], config['max_seq_len'], 200)
    
    # Create data loaders
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    train_losses = []
    val_losses = []
    hierarchy_stats = []
    
    logger.info("Starting training...")
    
    for epoch in range(20):  # Short training for example
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward pass
            output = model(x_batch, y_batch)
            logits = output['logits']
            
            # Compute loss
            loss = criterion(logits.reshape(-1, config['num_tokens']), y_batch.reshape(-1))
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                output = model(x_batch, y_batch)
                logits = output['logits']
                loss = criterion(logits.reshape(-1, config['num_tokens']), y_batch.reshape(-1))
                val_loss += loss.item()
                val_batches += 1
        
        model.train()
        
        # Record metrics
        avg_train_loss = epoch_loss / num_batches
        avg_val_loss = val_loss / val_batches
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Get hierarchy statistics
        stats = model.get_hierarchy_stats()
        hierarchy_stats.append(stats)
        
        scheduler.step()
        
        logger.info(
            f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"Best Worker Fitness: {stats['best_worker_fitness']:.4f}"
        )
    
    return model, train_losses, val_losses, hierarchy_stats


def analyze_hierarchy_evolution(hierarchy_stats: List[Dict]):
    """Analyze how the hierarchy evolved during training."""
    logger.info("Analyzing hierarchy evolution")
    
    generations = [s['generation'] for s in hierarchy_stats]
    avg_fitness = [s['avg_worker_fitness'] for s in hierarchy_stats]
    best_fitness = [s['best_worker_fitness'] for s in hierarchy_stats]
    worst_fitness = [s['worst_worker_fitness'] for s in hierarchy_stats]
    fitness_std = [s['fitness_std'] for s in hierarchy_stats]
    
    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Fitness evolution
    ax1.plot(generations, avg_fitness, label='Average Fitness', color='blue')
    ax1.plot(generations, best_fitness, label='Best Fitness', color='green')
    ax1.plot(generations, worst_fitness, label='Worst Fitness', color='red')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness Score')
    ax1.set_title('Worker Fitness Evolution')
    ax1.legend()
    ax1.grid(True)
    
    # Fitness standard deviation
    ax2.plot(generations, fitness_std, color='orange')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Fitness Std Dev')
    ax2.set_title('Fitness Diversity')
    ax2.grid(True)
    
    # Fitness range
    fitness_range = [b - w for b, w in zip(best_fitness, worst_fitness)]
    ax3.plot(generations, fitness_range, color='purple')
    ax3.set_xlabel('Generation')
    ax3.set_ylabel('Fitness Range')
    ax3.set_title('Fitness Range (Best - Worst)')
    ax3.grid(True)
    
    # Fitness improvement rate
    fitness_improvement = [0] + [avg_fitness[i] - avg_fitness[i-1] for i in range(1, len(avg_fitness))]
    ax4.plot(generations, fitness_improvement, color='brown')
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Generation')
    ax4.set_ylabel('Fitness Improvement')
    ax4.set_title('Fitness Improvement Rate')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('qbth_hierarchy_evolution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print analysis
    logger.info(f"Final average fitness: {avg_fitness[-1]:.4f}")
    logger.info(f"Fitness improvement: {avg_fitness[-1] - avg_fitness[0]:.4f}")
    logger.info(f"Best worker fitness: {max(best_fitness):.4f}")
    logger.info(f"Evolution cycles completed: {max(generations)}")


def demonstrate_inference(model: QueenBeeTransformerHierarchy):
    """Demonstrate inference capabilities."""
    logger.info("Demonstrating inference capabilities")
    
    model.eval()
    
    # Create sample input
    sample_input = torch.randint(0, 1000, (1, 50))  # Single sequence
    
    with torch.no_grad():
        # Forward pass
        output = model(sample_input)
        logits = output['logits']
        worker_outputs = output['worker_outputs'] 
        fitness_scores = output['fitness_scores']
        
        # Get predictions
        predictions = torch.argmax(logits, dim=-1)
        
        logger.info(f"Input shape: {sample_input.shape}")
        logger.info(f"Output logits shape: {logits.shape}")
        logger.info(f"Number of workers: {len(worker_outputs)}")
        logger.info(f"Worker fitness scores: {fitness_scores}")
        
        # Analyze worker specialization
        worker_activations = []
        for i, worker_output in enumerate(worker_outputs):
            activation_norm = torch.norm(worker_output, dim=-1).mean()
            worker_activations.append(activation_norm.item())
            logger.info(f"Worker {i} average activation: {activation_norm.item():.4f}")
        
        # Show specialization
        most_active_worker = np.argmax(worker_activations)
        least_active_worker = np.argmin(worker_activations)
        
        logger.info(f"Most active worker: {most_active_worker}")
        logger.info(f"Least active worker: {least_active_worker}")
        logger.info(f"Activation ratio: {max(worker_activations) / min(worker_activations):.2f}")


def compare_with_baseline():
    """Compare QBTH with a baseline transformer."""
    logger.info("Comparing QBTH with baseline transformer")
    
    # This would involve creating a standard transformer and comparing
    # performance, but for brevity, we'll just log the concept
    logger.info("Baseline comparison would involve:")
    logger.info("1. Standard transformer with same total parameters")
    logger.info("2. Training on same data for same epochs")
    logger.info("3. Comparing convergence speed, final performance")
    logger.info("4. Analyzing computational efficiency")
    logger.info("5. Measuring emergent behaviors in QBTH")


def main():
    """Main example function."""
    setup_logging()
    
    logger.info("="*50)
    logger.info("Queen Bee Transformer Hierarchy Example")
    logger.info("="*50)
    
    try:
        # Train the model
        model, train_losses, val_losses, hierarchy_stats = train_qbth_model()
        
        # Analyze evolution
        analyze_hierarchy_evolution(hierarchy_stats)
        
        # Demonstrate inference
        demonstrate_inference(model)
        
        # Compare with baseline
        compare_with_baseline()
        
        # Final model statistics
        final_stats = model.get_hierarchy_stats()
        logger.info("Final hierarchy statistics:")
        for key, value in final_stats.items():
            logger.info(f"  {key}: {value:.4f}")
        
        logger.info("Example completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in example: {e}")
        raise


if __name__ == "__main__":
    main() 