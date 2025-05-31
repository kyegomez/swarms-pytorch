import torch
from torch import nn
from torch.nn import CrossEntropyLoss, Transformer
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


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



    Example2
    # Create random source and target sequences
    src = torch.randn(10, 32, 512)
    tgt = torch.randn(10, 32, 512)

    # Create random labels
    labels = torch.randint(0, 512, (10, 32))

    # Create a fish and train it on the random data
    fish = Fish(512, 8, 6)
    fish.train(src, tgt, labels)
    print(fish.food)  # Print the fish's food

    # Create a fish school and optimize it on the random data
    school = FishSchool(10, 512, 8, 6, 100)
    school.forward(src, tgt, labels)
    print(school.fish[0].food)  # Print the first fish's food


    """

    def __init__(
        self,
        dim,
        heads,
        depth,
        dynamic_learning_rate=False,
        early_stopping=False,
        complexity_regularization=False,
        max_patience=None,
        alpha=0.1,
    ):
        super().__init__()
        self.model = Transformer(
            d_model=dim,
            nhead=heads,
            num_encoder_layers=depth,
            num_decoder_layers=depth,
        )
        self.optimizer = Adam(self.parameters())
        self.scheduler = ReduceLROnPlateau(self.optimizer, "min")

        self.complexity_regularization = complexity_regularization
        self.dynamic_learning_rate = dynamic_learning_rate
        self.early_stopping = early_stopping
        self.food = 0
        self.best_food = float("inf")
        self.patience = 0
        self.max_patience = max_patience
        self.alpha = alpha

    def train(self, src, tgt, labels):
        """Trains the fish school"""
        self.model.train()

        self.optimizer.zero_grad()

        outputs = self.model(src, tgt)

        # cross entropy loss
        loss = CrossEntropyLoss()(
            outputs.view(-1, outputs.size(-1)), labels.view(-1)
        )

        # complexity regularization by adding the sum of the squares of the
        # weights
        if self.complexity_regularization:
            # complexity regularization
            loss += self.alpha * sum(
                p.pow(2.0).sum() for p in self.model.parameters()
            )

        # backpropagation
        loss.backward()

        # dynamic learning rate
        if self.dynamic_learning_rate:
            self.scheduler.step(loss)
        else:
            forward = self.optimizer
            forward.step()

        # # use negative loss as food
        self.food = -loss.item()

        # early stopping if the fish is not improving
        if self.early_stopping:
            if loss < self.best_food:
                self.best_food = loss
                self.patience = 0
            else:
                self.patience += 1

    def forward(self, src, tgt):
        """Forward pass of the fish school"""
        return self.model(src, tgt)

    def early_stopping(self, food):
        """
        Early stopping if the fish is not improving.
        """
        if food < self.best_food:
            self.best_food = food
            self.patience = 0
        else:
            self.patience += 1
            if self.patience > 5:
                return True
        return False

    def generate(self, src, tgt):
        """
        Generate a sequence using the fish's model.
        """
        return self.model.generate(src, tgt)

    def save(self, path):
        """
        Save the fish's model.
        """
        torch.save(self.model.state_dict(), path)


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

    def __init__(
        self,
        num_fish,
        dim,
        heads,
        depth,
        num_iter,
        num_top_fish=None,
        complex_school=False,
        max_seq_len=8192,
    ):
        super().__init__()
        self.fish = [Fish(dim, heads, depth) for _ in range(num_fish)]
        self.num_iter = num_iter
        self.num_top_fish = num_top_fish
        self.complex_school = complex_school
        self.max_seq_len = max_seq_len

    def forward(self, src, tgt, labels):
        for _ in range(self.num_iter):
            total_food = 0
            for fish in self.fish:
                fish.train(src, tgt, labels)
                total_food += fish.food
            # adjust schoold behavior on total food
            avg_food = total_food / len(self.fish)

            # complex school behavior => fish with lower food learn from fish
            # with higher food
            if self.complex_school:
                for fish in self.fish:
                    neighbor = self.fish[
                        torch.randint(0, len(self.fish), (1,)).item()
                    ]
                    if neighbor.food > fish.food:
                        fish.model.load_state_dict(neighbor.model.state_dict())

            # adjust schoold behavior on average food
            for fish in self.fish:
                if fish.food < avg_food:
                    # transformer weights from the best performing fish
                    best_fish = max(self.fish, key=lambda f: f.food)
                    fish.model.load_state_dict(best_fish.model.state_dict())

    def generate(self, src, tgt):
        """
        Generate a sequence using the fish school's model.
        """
        return self.fish[0].generate(src, tgt)

    def predict(self, src, tgt):
        """
        Ensemble learning => enseble prediction of top performing models

        averages outputs of the top peforming models
        """
        top_fish = sorted(self.fish, key=lambda f: f.food, reverse=True)[
            : self.num_top_fish
        ]

        self.model.eval()

        with torch.no_grad():
            outputs = torch.stack([fish.model(src, tgt) for fish in top_fish])

        return sum(outputs) / len(outputs)

    def save(self, path):
        """
        Save the fish school's models.
        """
        for i, fish in enumerate(self.fish):
            fish.save(path + f"fish_{i}.pt")

    def load(self, path):
        """
        Load the fish school's models.
        """
        for i, fish in enumerate(self.fish):
            fish.model.load_state_dict(torch.load(path + f"fish_{i}.pt"))

    def early_stopping(self):
        """
        Early stopping if the fish school is not improving.
        """
        for fish in self.fish:
            if fish.early_stopping(fish.food):
                return True
        return False

    def dynamic_learning_rate(self):
        """
        Dynamic learning rate for the fish school.
        """
        for fish in self.fish:
            fish.dynamic_learning_rate = True

    def complexity_regularization(self):
        """
        Complexity regularization for the fish school.
        """
        for fish in self.fish:
            fish.complexity_regularization = True

    def reset(self):
        """
        Reset the fish school's food.
        """
        for fish in self.fish:
            fish.food = 0

    def __getitem__(self, index):
        """Get the fish at the given index"""
        return self.fish[index]

    def __len__(self):
        """Get the number of fish in the school"""
        return len(self.fish)

    def __iter__(self):
        """Iterate over the fish in the school"""
        return iter(self.fish)

    def __next__(self):
        """Get the next fish in the school"""
        return next(self.fish)

    def __str__(self):
        """Get the string representation of the fish school"""
        return str(self.fish)
