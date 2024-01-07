import torch
from swarms_torch.fish_school import Fish, FishSchool


def test_fish_initialization():
    fish = Fish(dim=512, heads=8, depth=6)
    assert isinstance(fish, Fish)


def test_fish_train():
    fish = Fish(dim=512, heads=8, depth=6)
    src = torch.randn(10, 32, 512)
    tgt = torch.randn(10, 32, 512)
    labels = torch.randint(0, 512, (10, 32))
    fish.train(src, tgt, labels)
    assert isinstance(fish.food, float)


def test_fishschool_initialization():
    fishschool = FishSchool(
        num_fish=10, dim=512, heads=8, depth=6, num_iter=100
    )
    assert isinstance(fishschool, FishSchool)
    assert len(fishschool.fish) == 10


def test_fishschool_forward():
    fishschool = FishSchool(
        num_fish=10, dim=512, heads=8, depth=6, num_iter=100
    )
    src = torch.randn(10, 32, 512)
    tgt = torch.randn(10, 32, 512)
    labels = torch.randint(0, 512, (10, 32))
    fishschool.forward(src, tgt, labels)
    assert isinstance(fishschool.fish[0].food, float)
