# import torch
# import torch.nn as nn
# from loguru import logger
# from typing import Tuple, List, Dict


# class TextEncoder(nn.Module):
#     """Encodes text input into latent representation using a Transformer-based encoder."""
#     def __init__(self, hidden_dim: int, num_layers: int, num_heads: int):
#         super(TextEncoder, self).__init__()
#         self.transformer = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(
#                 d_model=hidden_dim,
#                 nhead=num_heads),
#             num_layers=num_layers
#         )
#         self.embedding = nn.EmbeddingBag(10000, hidden_dim)  # Assuming a vocab size of 10k

#     def forward(self, text_input: torch.Tensor) -> torch.Tensor:
#         """Encodes text input.

#         Args:
#             text_input (torch.Tensor): Tensor of tokenized text input.

#         Returns:
#             torch.Tensor: Latent text representation.
#         """
#         embedded = self.embedding(text_input)
#         encoded = self.transformer(embedded)
#         logger.info(f"Text input encoded with shape: {encoded.shape}")
#         return encoded


# class SpeechEncoder(nn.Module):
#     """Encodes speech/audio input into latent representation using CNN + Transformer."""
#     def __init__(self, hidden_dim: int, num_layers: int, num_heads: int):
#         super(SpeechEncoder, self).__init__()
#         self.cnn = nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=3, stride=2, padding=1)
#         self.transformer = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(
#                 d_model=hidden_dim,
#                 nhead=num_heads),
#             num_layers=num_layers
#         )

#     def forward(self, audio_input: torch.Tensor) -> torch.Tensor:
#         """Encodes speech input.

#         Args:
#             audio_input (torch.Tensor): Tensor of raw speech input.

#         Returns:
#             torch.Tensor: Latent speech representation.
#         """
#         batch_size = audio_input.size(0)

#         cnn_out = self.cnn(audio_input)  # Shape: [batch_size, hidden_dim, sequence_length]
#         sequence_length = cnn_out.size(2)

#         # Flatten the CNN output: (batch_size, hidden_dim * sequence_length)
#         flattened_cnn_out = cnn_out.view(batch_size, -1)

#         # Dynamically create the linear layer based on the flattened size
#         self.fc = nn.Linear(flattened_cnn_out.size(1), hidden_dim)  # Adjust based on output

#         cnn_out = self.fc(flattened_cnn_out)  # Ensure correct dimensions for transformer input
#         cnn_out = cnn_out.unsqueeze(1)  # Add the sequence length for transformer (batch_size, seq_len, hidden_dim)
#         encoded = self.transformer(cnn_out)

#         logger.info(f"Speech input encoded with shape: {encoded.shape}")
#         return encoded

# class ImageEncoder(nn.Module):
#     """Encodes image input using a ResNet-based backbone."""
#     def __init__(self, hidden_dim: int):
#         super(ImageEncoder, self).__init__()
#         self.cnn = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
#             nn.Conv2d(64, hidden_dim, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(hidden_dim),
#             nn.ReLU(),
#         )

#     def forward(self, image_input: torch.Tensor) -> torch.Tensor:
#         """Encodes image input.

#         Args:
#             image_input (torch.Tensor): Tensor of images.

#         Returns:
#             torch.Tensor: Latent image representation.
#         """
#         encoded = self.cnn(image_input)
#         logger.info(f"Image input encoded with shape: {encoded.shape}")
#         return encoded


# class VideoEncoder(nn.Module):
#     """Encodes video input using 3D CNN."""
#     def __init__(self, hidden_dim: int):
#         super(VideoEncoder, self).__init__()
#         self.cnn = nn.Sequential(
#             nn.Conv3d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm3d(64),
#             nn.ReLU(),
#             nn.Conv3d(64, hidden_dim, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm3d(hidden_dim),
#             nn.ReLU(),
#         )

#     def forward(self, video_input: torch.Tensor) -> torch.Tensor:
#         """Encodes video input.

#         Args:
#             video_input (torch.Tensor): Tensor of video frames.

#         Returns:
#             torch.Tensor: Latent video representation.
#         """
#         encoded = self.cnn(video_input)
#         logger.info(f"Video input encoded with shape: {encoded.shape}")
#         return encoded


# class MultimodalFusion(nn.Module):
#     """Multimodal fusion network that combines text, speech, vision, and video representations."""
#     def __init__(self, hidden_dim: int, num_layers: int, num_heads: int):
#         super(MultimodalFusion, self).__init__()
#         self.transformer = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(
#                 d_model=hidden_dim,
#                 nhead=num_heads),
#             num_layers=num_layers
#         )

#     def forward(self, encodings: List[torch.Tensor]) -> torch.Tensor:
#         """Fuses multiple input modalities.

#         Args:
#             encodings (List[torch.Tensor]): List of encoded modalities (text, speech, image, video).

#         Returns:
#             torch.Tensor: Fused multimodal representation.
#         """
#         concatenated = torch.cat(encodings, dim=1)
#         fused_representation = self.transformer(concatenated)
#         logger.info(f"Fused representation shape: {fused_representation.shape}")
#         return fused_representation


# class ActionDecoder(nn.Module):
#     """Decodes fused multimodal representations into actions (e.g., joint angles or velocities)."""
#     def __init__(self, hidden_dim: int, num_actions: int):
#         super(ActionDecoder, self).__init__()
#         self.fc = nn.Linear(hidden_dim, num_actions)

#     def forward(self, fused_representation: torch.Tensor) -> torch.Tensor:
#         """Generates control signals (actions) for the humanoid robot.

#         Args:
#             fused_representation (torch.Tensor): Fused multimodal latent vector.

#         Returns:
#             torch.Tensor: Control actions for the humanoid robot.
#         """
#         actions = self.fc(fused_representation)
#         logger.info(f"Generated actions shape: {actions.shape}")
#         return actions


# class SpeechDecoder(nn.Module):
#     """Decodes fused multimodal representations into speech (e.g., text-to-speech)."""
#     def __init__(self, hidden_dim: int, vocab_size: int):
#         super(SpeechDecoder, self).__init__()
#         self.fc = nn.Linear(hidden_dim, vocab_size)

#     def forward(self, fused_representation: torch.Tensor) -> torch.Tensor:
#         """Generates speech output.

#         Args:
#             fused_representation (torch.Tensor): Fused multimodal latent vector.

#         Returns:
#             torch.Tensor: Generated speech (text output).
#         """
#         speech_output = self.fc(fused_representation)
#         logger.info(f"Generated speech output shape: {speech_output.shape}")
#         return speech_output


# class HumanoidMultimodalModel(nn.Module):
#     """Multimodal Foundation Model for humanoid robotics."""
#     def __init__(self, hidden_dim: int, num_actions: int, vocab_size: int, num_layers: int, num_heads: int):
#         super(HumanoidMultimodalModel, self).__init__()
#         # Encoders for each modality
#         self.text_encoder = TextEncoder(hidden_dim, num_layers, num_heads)
#         self.speech_encoder = SpeechEncoder(hidden_dim, num_layers, num_heads)
#         self.image_encoder = ImageEncoder(hidden_dim)
#         self.video_encoder = VideoEncoder(hidden_dim)

#         # Multimodal fusion
#         self.fusion = MultimodalFusion(hidden_dim, num_layers, num_heads)

#         # Decoders
#         self.action_decoder = ActionDecoder(hidden_dim, num_actions)
#         self.speech_decoder = SpeechDecoder(hidden_dim, vocab_size)

#     def forward(
#         self,
#         text_input: torch.Tensor,
#         speech_input: torch.Tensor,
#         image_input: torch.Tensor,
#         video_input: torch.Tensor
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """Forward pass for the multimodal humanoid model.

#         Args:
#             text_input (torch.Tensor): Textual input (tokenized).
#             speech_input (torch.Tensor): Speech input (raw audio).
#             image_input (torch.Tensor): Image input.
#             video_input (torch.Tensor): Video input.

#         Returns:
#             Tuple[torch.Tensor, torch.Tensor]: Output actions for the robot and speech (text) output.
#         """
#         # Encode each modality
#         text_encoding = self.text_encoder(text_input)
#         speech_encoding = self.speech_encoder(speech_input)
#         image_encoding = self.image_encoder(image_input)
#         video_encoding = self.video_encoder(video_input)

#         # Fuse modalities
#         fused_representation = self.fusion([text_encoding, speech_encoding, image_encoding, video_encoding])

#         # Decode into actions and speech
#         actions = self.action_decoder(fused_representation)
#         speech_output = self.speech_decoder(fused_representation)

#         return actions, speech_output


# if __name__ == "__main__":
#     # Sample input sizes (to be adjusted for real data)
#     batch_size = 4
#     hidden_dim = 512
#     num_actions = 32  # Example: 32 controllable joints
#     vocab_size = 10000  # Example: 10k word vocab for speech
#     num_layers = 4
#     num_heads = 8

#     # Initialize the model
#     model = HumanoidMultimodalModel(hidden_dim, num_actions, vocab_size, num_layers, num_heads)

#     # Sample inputs (random tensors)
#     text_input = torch.randint(0, vocab_size, (batch_size, 10))  # Tokenized text input
#     speech_input = torch.randn(batch_size, 1, 16000)  # Example raw audio input
#     image_input = torch.randn(batch_size, 3, 224, 224)  # Image input (224x224 resolution)
#     video_input = torch.randn(batch_size, 3, 16, 224, 224)  # Video input (16 frames, 224x224 resolution)

#     # Forward pass
#     actions, speech_output = model(text_input, speech_input, image_input, video_input)
#     logger.info(f"Actions: {actions.shape}, Speech Output: {speech_output.shape}")

import torch
import torch.nn as nn
from loguru import logger
from typing import Tuple, List
from transformers import AutoModel, Wav2Vec2Processor, Wav2Vec2Model
import timm


class PretrainedTextEncoder(nn.Module):
    """Encodes text input using a pretrained HuggingFace Transformer model."""

    def __init__(self, model_name: str = "bert-base-uncased"):
        super(PretrainedTextEncoder, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.hidden_dim = self.model.config.hidden_size

    def forward(self, text_input: torch.Tensor) -> torch.Tensor:
        """Encodes text input into latent representation.

        Args:
            text_input (torch.Tensor): Tensor of tokenized text input.

        Returns:
            torch.Tensor: Latent text representation.
        """
        outputs = self.model(text_input).last_hidden_state
        logger.info(f"Text input encoded with shape: {outputs.shape}")
        return outputs


class PretrainedSpeechEncoder(nn.Module):
    """Encodes speech input using HuggingFace Wav2Vec2."""

    def __init__(self, model_name: str = "facebook/wav2vec2-base-960h"):
        super(PretrainedSpeechEncoder, self).__init__()
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.hidden_dim = self.model.config.hidden_size

    def forward(self, speech_input: torch.Tensor) -> torch.Tensor:
        """Encodes speech input into latent representation.

        Args:
            speech_input (torch.Tensor): Tensor of raw speech input.

        Returns:
            torch.Tensor: Latent speech representation.
        """
        outputs = self.model(speech_input).last_hidden_state
        logger.info(f"Speech input encoded with shape: {outputs.shape}")
        return outputs


class PretrainedImageEncoder(nn.Module):
    """Encodes image input using the best model from timm (Swin Transformer V2)."""

    def __init__(self, model_name: str = "swinv2_base_window12_192_22k"):
        super(PretrainedImageEncoder, self).__init__()
        # Load the best vision model from timm
        self.model = timm.create_model(
            model_name, pretrained=True, num_classes=0
        )  # Remove classification head
        self.hidden_dim = (
            self.model.num_features
        )  # Hidden dimension of the model output

    def forward(self, image_input: torch.Tensor) -> torch.Tensor:
        """Encodes image input into latent representation.

        Args:
            image_input (torch.Tensor): Tensor of images.

        Returns:
            torch.Tensor: Latent image representation.
        """
        outputs = self.model(image_input)  # Pass through Swin Transformer V2
        logger.info(f"Image input encoded with shape: {outputs.shape}")
        return outputs


class PretrainedVideoEncoder(nn.Module):
    """Encodes video input using a pretrained timm model (for images) by averaging video frames."""

    def __init__(self, model_name: str = "vit_base_patch16_224"):
        super(PretrainedVideoEncoder, self).__init__()
        self.model = timm.create_model(
            model_name, pretrained=True, num_classes=0
        )  # No classification head
        self.hidden_dim = self.model.num_features

    def forward(self, video_input: torch.Tensor) -> torch.Tensor:
        """Encodes video input by averaging frames and then using the timm model.

        Args:
            video_input (torch.Tensor): Tensor of video frames (Batch, Channels, Frames, Height, Width).

        Returns:
            torch.Tensor: Latent video representation.
        """
        # Average over the frames dimension (dim=2), resulting in (Batch, Channels, Height, Width)
        video_input = video_input.mean(dim=2)

        # Pass the averaged frame through the model (now in 4D format)
        outputs = self.model(video_input)

        logger.info(f"Video input encoded with shape: {outputs.shape}")
        return outputs


class MultimodalFusion(nn.Module):
    """Multimodal fusion network that combines text, speech, vision, and video representations."""

    def __init__(self, input_dim: int = 4352, hidden_dim: int = None):
        super(MultimodalFusion, self).__init__()
        # Updated the input size to match the total concatenated dimension (4352)
        self.fc = nn.Linear(
            input_dim, hidden_dim
        )  # 768 (text) + 768 (speech) + 2048 (image) + 768 (video)

    def forward(self, encodings: List[torch.Tensor]) -> torch.Tensor:
        """Fuses multiple input modalities.

        Args:
            encodings (List[torch.Tensor]): List of encoded modalities (text, speech, image, video).

        Returns:
            torch.Tensor: Fused multimodal representation.
        """
        concatenated = torch.cat(
            encodings, dim=1
        )  # Concatenate along the feature dimension
        fused_representation = self.fc(concatenated)
        logger.info(f"Fused representation shape: {fused_representation.shape}")
        return fused_representation


class ActionDecoder(nn.Module):
    """Decodes fused multimodal representations into a sequence of actions using a transformer."""

    def __init__(
        self,
        hidden_dim: int,
        num_actions: int = 64,
        num_layers: int = 6,
        num_heads: int = 32,
    ):
        super(ActionDecoder, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads),
            num_layers=num_layers,
        )
        self.fc = nn.Linear(
            hidden_dim, num_actions
        )  # Final linear layer to produce action outputs

    def forward(self, fused_representation: torch.Tensor) -> torch.Tensor:
        """Generates control signals (actions) for the humanoid robot using a transformer.

        Args:
            fused_representation (torch.Tensor): Fused multimodal latent vector.

        Returns:
            torch.Tensor: Control actions for the humanoid robot.
        """
        # Unsqueeze the fused representation to add a sequence dimension (batch_size, seq_len=1, hidden_dim)
        fused_representation = fused_representation.unsqueeze(1)

        # Pass through the transformer to generate action sequences
        transformer_output = self.transformer(fused_representation)

        # Flatten the output and pass through the fully connected layer to get the final actions
        actions = self.fc(transformer_output.squeeze(1))
        logger.info(f"Generated actions shape: {actions.shape}")
        return actions


class SpeechDecoder(nn.Module):
    """Decodes fused multimodal representations into speech (e.g., text-to-speech)."""

    def __init__(self, hidden_dim: int, vocab_size: int):
        super(SpeechDecoder, self).__init__()
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, fused_representation: torch.Tensor) -> torch.Tensor:
        """Generates speech output.

        Args:
            fused_representation (torch.Tensor): Fused multimodal latent vector.

        Returns:
            torch.Tensor: Generated speech (text output).
        """
        # b, d = fused_representation.shape
        # logger.info(f"{fused_representation.shape}")
        speech_output = self.fc(fused_representation)
        logger.info(f"Generated speech output shape: {speech_output.shape}")
        return speech_output


class HumanoidMultimodalModel(nn.Module):
    """Multimodal Foundation Model for humanoid robotics."""

    def __init__(self, hidden_dim: int, num_actions: int, vocab_size: int):
        super(HumanoidMultimodalModel, self).__init__()
        self.hidden_dim = hidden_dim
        # Pretrained Encoders for each modality
        self.text_encoder = PretrainedTextEncoder()
        self.speech_encoder = PretrainedSpeechEncoder()
        self.image_encoder = PretrainedImageEncoder()
        self.video_encoder = PretrainedVideoEncoder()

        # Decoders
        self.action_decoder = ActionDecoder(hidden_dim, num_actions)
        self.speech_decoder = SpeechDecoder(hidden_dim, vocab_size)

    def forward(
        self,
        text_input: torch.Tensor,
        speech_input: torch.Tensor,
        image_input: torch.Tensor,
        video_input: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for the multimodal humanoid model.

        Args:
            text_input (torch.Tensor): Textual input (tokenized).
            speech_input (torch.Tensor): Speech input (raw audio).
            image_input (torch.Tensor): Image input.
            video_input (torch.Tensor): Video input.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output actions for the robot and speech (text) output.
        """
        # Encode each modality
        text_encoding = self.text_encoder(
            text_input
        )  # Shape: [batch_size, seq_length, hidden_dim]
        speech_encoding = self.speech_encoder(
            speech_input
        )  # Shape: [batch_size, seq_length, hidden_dim]
        image_encoding = self.image_encoder(
            image_input
        )  # Shape: [batch_size, hidden_dim]
        video_encoding = self.video_encoder(
            video_input
        )  # Shape: [batch_size, hidden_dim]

        # Apply mean pooling to text and speech to reduce to [batch_size, hidden_dim]
        text_encoding = text_encoding.mean(dim=1)
        speech_encoding = speech_encoding.mean(dim=1)

        _, d_t = text_encoding.shape
        _, d_s = speech_encoding.shape
        _, d_i = image_encoding.shape
        _, d_v = video_encoding.shape

        # Fuse modalities
        # fused_representation = self.fusion([text_encoding, speech_encoding, image_encoding, video_encoding])
        logger.info(f"Input dim{d_t + d_s + d_i + d_v}")
        fused_representation = MultimodalFusion(
            d_t + d_s + d_i + d_v, self.hidden_dim
        )([text_encoding, speech_encoding, image_encoding, video_encoding])

        # Decode into actions and speech
        actions = self.action_decoder(fused_representation)
        speech_output = self.speech_decoder(fused_representation)

        return actions, speech_output


if __name__ == "__main__":
    # Example input sizes (to be adjusted based on real data)
    batch_size = 4
    hidden_dim = 768
    num_actions = 32  # Example: 32 controllable joints
    vocab_size = 10000  # Example: 10k word vocab for speech

    # Initialize the model
    model = HumanoidMultimodalModel(hidden_dim, num_actions, vocab_size)

    # Sample inputs (random tensors as placeholders)
    text_input = torch.randint(
        0, vocab_size, (batch_size, 10)
    )  # Tokenized text input
    speech_input = torch.randn(batch_size, 16000)  # Example raw audio input
    image_input = torch.randn(
        batch_size, 3, 192, 192
    )  # Image input (224x224 resolution)
    video_input = torch.randn(
        batch_size, 3, 16, 224, 224
    )  # Video input (16 frames, 224x224 resolution)

    # Forward pass
    actions, speech_output = model(
        text_input, speech_input, image_input, video_input
    )
    logger.info(
        f"Actions: {actions.shape}, Speech Output: {speech_output.shape}"
    )
