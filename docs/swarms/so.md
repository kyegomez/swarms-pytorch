# `SPO` Class


The `SPO` class implements the Spiral Optimization (SPO) algorithm. This algorithm is used for optimization towards a target string.

## Attributes
----------

-   `goal` (torch.Tensor): The goal string to be optimized.
-   `m` (int): Number of search points.
-   `k_max` (int): Maximum number of iterations.
-   `n_dim` (int): Length of the goal string.
-   `points` (torch.Tensor): The search points.
-   `center` (torch.Tensor): The center point.

## Methods
-------

### `__init__(self, goal: str = None, m: int = 10, k_max: int = 1000)`

The constructor for the `SPO` class. Initializes the search points and the center.

#### Parameters

-   `goal` (str, optional): The goal string to be optimized.
-   `m` (int, optional): Number of search points. Default is 10.
-   `k_max` (int, optional): Maximum number of iterations. Default is 1000.

#### Example

```
spo = SPO(goal="Hello SPO", m=100, k_max=1000)
```


### `_step_rate(self, k)`

Defines the step rate function.

#### Parameters

-   `k` (int): Current iteration.

#### Returns

-   `step_rate` (float): Step rate for the current iteration.

#### Example

```
step_rate = spo._step_rate(k)
```


### `_update_points(self, k)`

Updates the search points based on the spiral model.

#### Parameters

-   `k` (int): Current iteration.

#### Example

```
spo._update_points(k)
```


### `_update_center(self)`

Finds the best search point and sets it as the new center.

#### Example

```
spo._update_center()
```


### `optimize(self)`

Runs the optimization loop. Updates the search points and the center for a given number of iterations.

#### Example

```
spo.optimize()
```


### `best_string(self)`

Converts the best found point to its string representation.

#### Returns

-   `best_string` (str): The best string.

#### Example

```
best_string = spo.best_string()
print("Best Matched String:", best_string)
```


## Usage Examples
--------------

### Example 1: Optimize a String

In this example, we will optimize the string "Attention is all you need" using 100 search points and 1000 iterations.

```python
spo = SPO(goal="Attention is all you need", m=100, k_max=1000)
spo.optimize()
print("Best Matched String:", spo.best_string())
```


### Example 2: Using a Different Goal String

In this example, we will optimize the string "Hello, World!" using 100 search points and 1000 iterations.

```python
spo = SPO(goal="Hello, World!", m=100, k_max=1000)
spo.optimize()
print("Best Matched String:", spo.best_string())
```


### Example 3: Using a Different Number of Search Points

In this example, we will optimize the string "Attention is all you need" using 200 search points and 1000 iterations.

```python
spo = SPO(goal="Attention is all you need", m=200, k_max=1000)
spo.optimize()
print("Best Matched String:", spo.best_string())
```
