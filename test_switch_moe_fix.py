import torch
from swarms_torch.structs.switch_moe import SwitchMoE


def test_switch_moe_aux_loss():
    """Test that SwitchMoE works with auxiliary loss enabled."""

    # Set up test parameters
    batch_size = 32
    seq_len = 128
    dim = 512
    num_experts = 8

    # Create model with auxiliary loss enabled
    model = SwitchMoE(
        dim=dim,
        hidden_dim=dim,
        output_dim=dim,
        num_experts=num_experts,
        use_aux_loss=True,
    )

    # Create test input
    x = torch.randn(batch_size, dim)

    try:
        # Forward pass
        output, loss = model(x)

        print("✅ Success! No runtime error occurred.")
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Auxiliary loss: {loss.item() if loss is not None else 'None'}")

        # Verify shapes
        assert (
            output.shape == x.shape
        ), f"Output shape {output.shape} doesn't match input shape {x.shape}"
        assert (
            loss is not None
        ), "Loss should not be None when use_aux_loss=True"
        assert torch.isfinite(loss), "Loss should be finite"

        print("✅ All assertions passed!")

    except Exception as e:
        print(f"❌ Error occurred: {e}")
        raise e


if __name__ == "__main__":
    test_switch_moe_aux_loss()
