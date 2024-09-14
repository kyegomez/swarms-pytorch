from swarms_torch import SPO

# Example Usage
goal_str = "Attention is all you need"
optimizer = SPO(goal_str)
optimizer.optimize()
print(f"Optimized String: {optimizer.best_string()}")
