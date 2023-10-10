import torch
from torch import nn
from torch.nn import Transformer, CrossEntropyLoss
from torch.optim import Adam


class Fish(nn.Module):
    """
    A fish is a transformer model with a negative loss as food.

    Parameters
    ----------
    dim : int
        The number of expected features in the input (required).
    heads : int
        The number of heads in the multiheadattention models (required).
    depth : int
        The number of sub-encoder-layers in the encoder (required).

    Attributes

    model : torch.nn.Transformer
        The transformer model.
    food : float
        The fish's food, which is the negative loss of the model.

    Methods
    =======
    train(src, tgt, labels)
        Train the model with the given source, target, and labels.


    Usage:
    >>> fish = Fish(512, 8, 6)
    >>> fish.train(src, tgt, labels)
    >>> fish.food
    -0.123456789

    """

    def __init__(
        self,
        dim,
        heads,
        depth,
    ):
        super().__init__()
        self.model = Transformer(
            d_model=dim, nhead=heads, num_encoder_layers=depth, num_decoder_layers=depth
        )
        self.food = 0

    def train(self, src, tgt, labels):
        """Trains the fish school"""
        outputs = self.model(src, tgt)
        loss = CrossEntropyLoss()
        loss = loss(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        loss.backward()
        optimizer = Adam(self.model.parameters())
        optimizer.step()
        self.food = -loss.item()  # use negative loss as food


class FishSchool(nn.Module):
    """
    Fish School is a collection of fish.

    Parameters
    ----------
    num_fish : int
        The number of fish in the school.
    dim : int
        The number of expected features in the input (required).
    heads : int
        The number of heads in the multiheadattention models (required).
    depth : int
        The number of sub-encoder-layers in the encoder (required).
    num_iter : int
        The number of iterations to train the fish school.


    Usage:
    >>> school = FishSchool(10, 512, 8, 6, 100)
    >>> school.train(src, tgt, labels)
    >>> school.fish[0].food

    """

    def __init__(self, num_fish, dim, heads, depth, num_iter):
        super().__init__()
        self.fish = [Fish(dim, heads, depth) for _ in range(num_fish)]
        self.num_iter = num_iter

    def optimizer(self, src, tgt, labels):
        for _ in range(self.num_iter):
            total_food = 0
            for fish in self.fish:
                fish.train(src, tgt, labels)
                total_food += fish.food
            # adjust schoold behavior on total food
            avg_food = total_food / len(self.fish)
            for fish in self.fish:
                if fish.food < avg_food:
                    # transformer weights from the best performing fish
                    best_fish = max(self.fish, key=lambda f: f.food)
                    fish.model.load_state_dict(best_fish.model.state_dict())

