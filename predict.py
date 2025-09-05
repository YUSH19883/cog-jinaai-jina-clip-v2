import subprocess
import time
import warnings
# Aggressively suppress all warnings
warnings.filterwarnings('ignore')

from cog import BasePredictor, Input, Path
from typing import List, Union
import os
import torch
from PIL import Image
import base64
import numpy as np
from transformers import AutoProcessor, AutoModel
import torchvision
import transformers

# Additional specific suppressions
torchvision.disable_beta_transforms_warning()
transformers.utils.logging.set_verbosity_error()
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

REPO_ID = "jinaai/jina-clip-v2"
MODEL_CACHE = "models"
BASE_URL = f"https://weights.replicate.delivery/default/jina-clip-v2/{MODEL_CACHE}/"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE



def map_to_b64(ndarray2d):
    return [base64.b64encode(x.tobytes()).decode('utf-8') for x in ndarray2d]

def map_to_list(ndarray2d):
    return ndarray2d.tolist()

FORMATS = [
    ("base64", map_to_b64),
    ("array", map_to_list),
]
FORMATS_MAP = dict(FORMATS)


def download_weights(url: str, dest: str) -> None:
    # NOTE WHEN YOU EXTRACT SPECIFY THE PARENT FOLDER
    start = time.time()
    print("[!] Initiating download from URL: ", url)
    print("[~] Destination path: ", dest)
    if ".tar" in dest:
        dest = os.path.dirname(dest)
    command = ["pget", "-vf" + ("x" if ".tar" in url else ""), url, dest]
    try:
        print(f"[~] Running command: {' '.join(command)}")
        subprocess.check_call(command, close_fds=False)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Failed to download weights. Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
        )
        raise
    print("[+] Download completed in: ", time.time() - start, "seconds")


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        model_files = [
            "models--jinaai--jina-clip-implementation.tar",
            "models--jinaai--xlm-roberta-flash-implementation.tar", 
            "models--jinaai--jina-clip-v2.tar",
            "modules.tar",
            "models--jinaai--jina-embeddings-v3.tar"
        ]
        os.makedirs(MODEL_CACHE, exist_ok=True)
        for model_file in model_files:
            url = BASE_URL + model_file
            filename = url.split("/")[-1]
            dest_path = os.path.join(MODEL_CACHE, filename)
            if not os.path.exists(dest_path.replace(".tar", "")):
                download_weights(url, dest_path)
        
        self.processor = AutoProcessor.from_pretrained(
            REPO_ID,
            local_files_only=True,
            trust_remote_code=True
        )
        
        self.model = AutoModel.from_pretrained(
            REPO_ID,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        ).cuda()
        
        self.model.eval()

        # Add model constants
        self.IMAGE_SIZE = 512  # Optimal size from docs
        self.MAX_TOKENS = 8192  # Max token length
        self.MIN_DIM = 64
        self.MAX_DIM = 1024

    def predict(
        self,
        text: str = Input(
            description="Text content to embed (up to 8192 tokens). If both text and image provided, text embedding will be first in returned list.",
            default=None,
        ),
        image: Path = Input(
            description="Image file to embed (optimal size: 512x512). If both text and image provided, image embedding will be second in returned list.",
            default=None,
        ),
        embedding_dim: int = Input(
            description="Matryoshka dimension - output embedding dimension (64-1024)",
            default=64,
            ge=64,
            le=1024
        ),
        output_format: str = Input(
            description="Format to use in outputs",
            default=FORMATS[0][0],
            choices=[k for (k, _v) in FORMATS],
        ),
    ) -> List[Union[str, List[float]]]:
        """Generate embeddings for text and/or images. 
        Returns:
            - Single embedding if only text or image provided
            - List of [text_embedding, image_embedding] if both provided
        """
        
        if not text and not image:
            raise ValueError("At least one input (text or image) must be provided")
            
        map_func = FORMATS_MAP[output_format]
        embeddings_list = []
        
        # Process text if provided
        if text:
            inputs = self.processor(
                text=text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.MAX_TOKENS,
            ).to("cuda")
            
            with torch.no_grad():
                # Use model's text encoder directly for better performance
                text_embeddings = self.model.get_text_features(**inputs)
                if embedding_dim < text_embeddings.shape[1]:
                    text_embeddings = text_embeddings[:, :embedding_dim]
                embeddings_list.append(text_embeddings)

        # Process image if provided
        if image:
            img = Image.open(image).convert('RGB')
            if img.size != (self.IMAGE_SIZE, self.IMAGE_SIZE):
                print(f"Warning: For optimal performance, resize image to {self.IMAGE_SIZE}x{self.IMAGE_SIZE}")
            
            inputs = self.processor(
                images=img,
                return_tensors="pt"
            ).to("cuda")
            
            with torch.no_grad():
                # Use model's image encoder directly for better performance
                image_embeddings = self.model.get_image_features(**inputs)
                if embedding_dim < image_embeddings.shape[1]:
                    image_embeddings = image_embeddings[:, :embedding_dim]
                embeddings_list.append(image_embeddings)
        
        # Combine embeddings if both inputs were provided
        if len(embeddings_list) > 1:
            embeddings = torch.cat(embeddings_list, dim=0)
        else:
            embeddings = embeddings_list[0]
            
        # Normalize embeddings (as recommended in docs)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        embeddings = embeddings.cpu().numpy()
        return map_func(embeddings)
        