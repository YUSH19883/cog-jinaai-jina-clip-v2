# Jina CLIP v2 - Multimodal Embeddings

A powerful multimodal embedding model that generates high-quality embeddings for both text and images using Jina AI's CLIP v2 architecture.

[![Replicate](https://replicate.com/zsxkib/jina-clip-v2/badge)](https://replicate.com/zsxkib/jina-clip-v2)

## Overview

Jina CLIP v2 is a state-of-the-art multimodal embedding model that can process both text and images to generate dense vector representations. This implementation provides optimized inference on Replicate with support for Matryoshka representations, allowing you to choose embedding dimensions from 64 to 1024.

### Key Features

- **Multimodal**: Process text, images, or both simultaneously
- **Matryoshka Representations**: Flexible embedding dimensions (64-1024)
- **Optimized Performance**: CUDA acceleration with normalized embeddings
- **Multiple Output Formats**: Base64 encoded or raw array outputs
- **High Token Limit**: Support for up to 8,192 text tokens
- **Optimal Image Processing**: Best performance with 512x512 images

## Inputs

- **`text`** (string, optional): Text content to embed (up to 8,192 tokens)
- **`image`** (file, optional): Image file to embed (optimal size: 512x512px)
- **`embedding_dim`** (integer, 64-1024, default: 64): Output embedding dimension using Matryoshka representations
- **`output_format`** (string, default: "base64"): Choose between "base64" or "array" format

*Note: At least one input (text or image) must be provided. If both are provided, you'll get a list with [text_embedding, image_embedding].*

## Outputs

The model returns embeddings in your chosen format:
- **Single input**: Returns one embedding
- **Both inputs**: Returns list of [text_embedding, image_embedding]
- **Base64 format**: Compact string representation
- **Array format**: Standard numerical arrays

## Usage Examples

Clone this repository and use `cog predict` to run the model locally:

### Text Only
```bash
cog predict -i text="A beautiful sunset over the mountains" -i embedding_dim=256 -i output_format="array"
```

### Image Only  
```bash
cog predict -i image=@my_image.jpg -i embedding_dim=512 -i output_format="base64"
```

### Text + Image (Multimodal)
```bash
cog predict -i text="A beautiful sunset over the mountains" -i image=@sunset.jpg -i embedding_dim=128 -i output_format="array"
```
*Returns: [text_embedding, image_embedding]*

### Quick Start
```bash
# Clone and run
git clone https://github.com/zsxkib/cog-jinaai-jina-clip-v2.git
cd cog-jinaai-jina-clip-v2
cog predict -i text="Hello world" -i embedding_dim=64
```

## Model Details

This implementation is based on Jina AI's `jinaai/jina-clip-v2` model, featuring:
- Vision encoder optimized for 512x512 images
- Text encoder supporting up to 8,192 tokens
- L2 normalized embeddings for better similarity computations
- Efficient GPU inference with mixed precision
- Matryoshka representation learning for flexible dimensions

## Performance Tips

1. **Image Size**: Use 512x512 images for optimal performance
2. **Batch Processing**: Process multiple items in separate API calls
3. **Dimension Selection**: Choose the minimum dimension needed for your use case
4. **Text Length**: Longer texts (up to 8,192 tokens) are supported but may take longer

## Use Cases

- **Image Search**: Create searchable image databases with text queries
- **Content Recommendation**: Find similar content across text and images  
- **Multimodal RAG**: Enhance retrieval systems with image understanding
- **Content Moderation**: Analyze text and visual content together
- **Cross-modal Retrieval**: Find images with text descriptions or vice versa

---

## Model Attribution

This model is based on [Jina AI's CLIP v2](https://huggingface.co/jinaai/jina-clip-v2). Please refer to their documentation for technical details and citation information.

---

⭐ Star this on [GitHub](https://github.com/zsxkib/cog-jinaai-jina-clip-v2)!

👋 Follow `zsxkib` on [Twitter/X](https://twitter.com/zsakib_) | Built with ❤️ by [zsxkib](https://github.com/zsxkib)
