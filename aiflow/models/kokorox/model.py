import torch
import torch.nn as nn
import numpy as np
from typing import List

class TextChunker:
    """Split text into chunks by paragraphs or token count."""
    def __init__(self, chunk_type="paragraph", max_tokens_per_chunk=512):
        self.chunk_type = chunk_type
        self.max_tokens_per_chunk = max_tokens_per_chunk

    def split(self, text: str) -> List[str]:
        """Split text into chunks based on configuration."""
        if self.chunk_type == "paragraph":
            # Split by paragraphs (double newlines)
            chunks = [p.strip() for p in text.split("\n\n") if p.strip()]
            # Further split any paragraphs that are too long
            result = []
            for chunk in chunks:
                if len(chunk.split()) > self.max_tokens_per_chunk:
                    # Split by sentences if paragraph is too long
                    sentences = [s.strip() for s in chunk.split(". ")]
                    current = ""
                    for sentence in sentences:
                        if len((current + sentence).split()) <= self.max_tokens_per_chunk:
                            current += sentence + ". "
                        else:
                            if current:
                                result.append(current.strip())
                            current = sentence + ". "
                    if current:
                        result.append(current.strip())
                else:
                    result.append(chunk)
            return result
        else:
            # Split by token count
            words = text.split()
            return [" ".join(words[i:i+self.max_tokens_per_chunk]) 
                    for i in range(0, len(words), self.max_tokens_per_chunk)]

class ContentFilter(nn.Module):
    """Neural network for determining content relevance based on embeddings."""
    def __init__(self, config):
        super().__init__()
        self.embedding_dim = config["embedding_dimensions"]
        self.hidden_dim = config["filter_hidden_dim"]

        # Question-chunk interaction network
        self.interaction = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Emotional content enhancement
        self.emotion_enhancer = nn.Sequential(
            nn.Linear(config["num_emotions"], self.hidden_dim // 4),
            nn.LeakyReLU(0.1),
            nn.Linear(self.hidden_dim // 4, 1),
            nn.Sigmoid()
        )

        # Content quality assessor (combines relevance and emotion scores)
        self.quality_assessor = nn.Sequential(
            nn.Linear(2, self.hidden_dim // 4),
            nn.LeakyReLU(0.1),
            nn.Linear(self.hidden_dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, 
                question_embedding: torch.Tensor, 
                chunk_embedding: torch.Tensor,
                emotion_scores: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the filter model.

        Args:
            question_embedding: Tensor of shape [batch_size, embedding_dim]
            chunk_embedding: Tensor of shape [batch_size, embedding_dim]
            emotion_scores: Tensor of shape [batch_size, num_emotions]

        Returns:
            Tensor of shape [batch_size, 1] with relevance scores
        """
        # Combine question and chunk embeddings
        combined = torch.cat([question_embedding, chunk_embedding], dim=1)

        # Calculate relevance score
        relevance_score = self.interaction(combined)

        # Calculate emotion enhancement score
        emotion_score = self.emotion_enhancer(emotion_scores)

        # Combine scores for final quality assessment
        combined_scores = torch.cat([relevance_score, emotion_score], dim=1)
        quality_score = self.quality_assessor(combined_scores)

        return quality_score

class KokoroXProcessor:
    """Main processor that combines Kokoro emotional model with ContentFilter."""
    def __init__(self, config, kokoro_model, embedding_function):
        self.config = config
        self.kokoro_model = kokoro_model
        self.get_embedding = embedding_function
        self.chunker = TextChunker(
            chunk_type=config.get("chunk_type", "paragraph"),
            max_tokens_per_chunk=config.get("max_tokens_per_chunk", 512)
        )
        self.content_filter = ContentFilter(config)
        self.device = torch.device(config["device"])
        self.content_filter.to(self.device)
        self.target_token_limit = config.get("target_token_limit", 8192)
        self.emotion_names = config["emotion_names"]

    def process(self, 
                question: str, 
                data: str, 
                target_emotions: List[str], 
                max_iterations: int = 5) -> str:
        """
        Process input data to filter content based on emotions and relevance.

        Args:
            question: The user's question
            data: The raw text data to filter
            target_emotions: List of target emotions to prioritize
            max_iterations: Maximum number of filtering iterations

        Returns:
            Filtered text content
        """
        # 1. Chunk the data
        chunks = self.chunker.split(data)
        if not chunks:
            return "No content to process."

        # Stop if already under token limit
        if self._count_tokens("".join(chunks)) <= self.target_token_limit:
            return data

        # 2. Get question embedding
        question_embedding = torch.tensor(
            self.get_embedding(question),
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)  # Add batch dimension

        # 3. Begin iterative filtering
        for iteration in range(max_iterations):
            # Get embeddings for all chunks
            chunk_embeddings = []
            for chunk in chunks:
                emb = torch.tensor(
                    self.get_embedding(chunk),
                    dtype=torch.float32,
                    device=self.device
                ).unsqueeze(0)  # Add batch dimension
                chunk_embeddings.append(emb)

            # Get emotion scores for all chunks using Kokoro
            emotion_scores = []
            for i, chunk in enumerate(chunks):
                # Use Kokoro to get emotion values
                with torch.no_grad():
                    chunk_embedding = chunk_embeddings[i]
                    emotion_outputs, _ = self.kokoro_model(chunk_embedding)

                    # Convert emotion outputs to a tensor
                    emotion_tensor = torch.cat(
                        [emotion_outputs[name] for name in self.emotion_names],
                        dim=1
                    )

                    # Apply target emotion filtering
                    target_mask = torch.zeros(emotion_tensor.shape[1], device=self.device)
                    for emotion in target_emotions:
                        if emotion in self.emotion_names:
                            idx = self.emotion_names.index(emotion)
                            target_mask[idx] = 1.0

                    # Apply mask (or weight by target emotions)
                    emotion_tensor = emotion_tensor * target_mask + emotion_tensor * 0.1
                    emotion_scores.append(emotion_tensor)

            # Stack all embeddings and scores for batch processing
            stacked_embeddings = torch.cat(chunk_embeddings, dim=0)
            stacked_emotions = torch.cat(emotion_scores, dim=0)

            # Expand question embedding to match batch size
            question_expanded = question_embedding.expand(stacked_embeddings.shape[0], -1)

            # Get quality scores from content filter
            with torch.no_grad():
                quality_scores = self.content_filter(
                    question_expanded, 
                    stacked_embeddings,
                    stacked_emotions
                )

            # Convert to numpy for sorting
            quality_scores_np = quality_scores.cpu().numpy().flatten()

            # Calculate how many chunks to keep (50% reduction per iteration)
            keep_count = max(1, int(len(chunks) * 0.5))

            # Sort chunks by quality score and keep top chunks
            indices = np.argsort(quality_scores_np)[-keep_count:]
            chunks = [chunks[i] for i in indices]

            # Check if we're under token limit
            if self._count_tokens("".join(chunks)) <= self.target_token_limit:
                break

        # Join the final chunks and return
        return "\n\n".join(chunks)

    def _count_tokens(self, text: str) -> int:
        """Simple token counter based on whitespace splitting."""
        return len(text.split())

    def save(self, path: str) -> None:
        """Save the content filter model."""
        torch.save(self.content_filter.state_dict(), path)

    def load(self, path: str) -> None:
        """Load the content filter model."""
        self.content_filter.load_state_dict(torch.load(path, map_location=self.device))