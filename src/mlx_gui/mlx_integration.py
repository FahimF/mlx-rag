"""
MLX-LM integration layer for MLX-GUI.
Handles actual model loading, tokenization, and inference using MLX-LM.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, AsyncGenerator
from dataclasses import dataclass

import mlx.core as mx
from mlx_lm import load, generate, stream_generate
from mlx_lm.utils import load as load_utils
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from huggingface_hub import snapshot_download
import numpy as np

# Audio model imports (optional)
try:
    import mlx_whisper
    HAS_MLX_WHISPER = True
except ImportError:
    HAS_MLX_WHISPER = False

try:
    import mlx_audio
    HAS_MLX_AUDIO = True
except ImportError:
    HAS_MLX_AUDIO = False

try:
    import parakeet_mlx
    HAS_PARAKEET_MLX = True
except ImportError:
    HAS_PARAKEET_MLX = False

# Vision/multimodal model imports (optional)
try:
    import mlx_vlm
    HAS_MLX_VLM = True
except ImportError:
    HAS_MLX_VLM = False

from mlx_gui.huggingface_integration import get_huggingface_client

logger = logging.getLogger(__name__)

# Log library status after logger is defined
if not HAS_MLX_WHISPER:
    logger.warning("mlx-whisper not installed - Whisper models not supported")
if not HAS_MLX_AUDIO:
    logger.warning("mlx-audio not installed - Audio models not supported")
if not HAS_PARAKEET_MLX:
    logger.warning("parakeet-mlx not installed - Parakeet models not supported")
if not HAS_MLX_VLM:
    logger.warning("mlx-vlm not installed - Vision/multimodal models not supported")


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_tokens: int = 8192
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    repetition_context_size: int = 20
    seed: Optional[int] = None


@dataclass
class GenerationResult:
    """Result from text generation."""
    text: str
    prompt: str
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    generation_time_seconds: float
    tokens_per_second: float


class MLXModelWrapper:
    """Base wrapper for MLX models with unified interface."""

    def __init__(self, model, tokenizer, model_path: str, config: Dict[str, Any]):
        self.model = model
        self.tokenizer = tokenizer
        self.model_path = model_path
        self.config = config
        self.model_type = config.get("model_type", "text-generation")

        # Only apply tokenizer fixes if tokenizer is not None (audio models don't have tokenizers)
        if tokenizer is not None:
            # Workaround for processors (Gemma3Processor, Qwen3Processor, etc.) missing methods
            if not hasattr(tokenizer, 'eos_token_id'):
                # Try to get from wrapped tokenizer first
                if hasattr(tokenizer, 'tokenizer') and hasattr(tokenizer.tokenizer, 'eos_token_id'):
                    tokenizer.eos_token_id = tokenizer.tokenizer.eos_token_id
                    logger.debug(f"Set eos_token_id from wrapped tokenizer: {tokenizer.eos_token_id}")
                # Try to get from model config
                elif hasattr(tokenizer, 'vocab_size') and 'eos_token_id' in config:
                    tokenizer.eos_token_id = config['eos_token_id']
                    logger.debug(f"Set eos_token_id from model config: {tokenizer.eos_token_id}")
                # Check for common attribute names
                elif hasattr(tokenizer, 'eos_id'):
                    tokenizer.eos_token_id = tokenizer.eos_id
                    logger.debug(f"Set eos_token_id from eos_id: {tokenizer.eos_token_id}")
                else:
                    # Fallback: use common default
                    tokenizer.eos_token_id = 2
                    logger.warning(f"Using fallback eos_token_id: {tokenizer.eos_token_id}")

            # Comprehensive workaround for processors missing common tokenizer methods/attributes
            if hasattr(tokenizer, 'tokenizer'):
                inner_tokenizer = tokenizer.tokenizer

                # Add missing methods
                for method_name in ['encode', 'decode', 'apply_chat_template']:
                    if not hasattr(tokenizer, method_name) and hasattr(inner_tokenizer, method_name):
                        setattr(tokenizer, method_name, getattr(inner_tokenizer, method_name))
                        logger.debug(f"Set {method_name} method from wrapped tokenizer")

                # Add missing attributes
                for attr_name in ['vocab', 'vocab_size', 'bos_token_id', 'pad_token_id', 'bos_token', 'eos_token', 'pad_token']:
                    if not hasattr(tokenizer, attr_name) and hasattr(inner_tokenizer, attr_name):
                        setattr(tokenizer, attr_name, getattr(inner_tokenizer, attr_name))
                        logger.debug(f"Set {attr_name} attribute from wrapped tokenizer")

    def generate(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        """Generate text synchronously."""
        import time
        start_time = time.time()

        # Tokenize prompt
        prompt_tokens = self.tokenizer.encode(prompt)

        # Create sampler with temperature and top_p
        sampler = make_sampler(
            temp=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k
        )

        # Create logits processors for repetition penalty
        logits_processors = make_logits_processors(
            repetition_penalty=config.repetition_penalty,
            repetition_context_size=config.repetition_context_size
        )

        # Generate with MLX-LM
        response = generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_tokens=config.max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            verbose=False
        )

        end_time = time.time()
        generation_time = end_time - start_time

        # Debug logging
        logger.debug(f"Prompt: {repr(prompt)}")
        logger.debug(f"Full response: {repr(response)}")
        logger.debug(f"Response length: {len(response)}, Prompt length: {len(prompt)}")

        # Extract generated text (remove prompt)
        if response.startswith(prompt):
            generated_text = response[len(prompt):]
        else:
            # MLX-LM might return only the generated text
            generated_text = response
            logger.debug("Response doesn't start with prompt, using full response as generated text")

        # Count tokens
        completion_tokens = len(self.tokenizer.encode(generated_text))
        total_tokens = len(prompt_tokens) + completion_tokens
        tokens_per_second = completion_tokens / generation_time if generation_time > 0 else 0

        return GenerationResult(
            text=generated_text,
            prompt=prompt,
            total_tokens=total_tokens,
            prompt_tokens=len(prompt_tokens),
            completion_tokens=completion_tokens,
            generation_time_seconds=generation_time,
            tokens_per_second=tokens_per_second
        )

    async def generate_stream(self, prompt: str, config: GenerationConfig) -> AsyncGenerator[str, None]:
        """Generate text with streaming."""
        import asyncio

        # Create sampler and logits processors
        sampler = make_sampler(
            temp=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k
        )

        logits_processors = make_logits_processors(
            repetition_penalty=config.repetition_penalty,
            repetition_context_size=config.repetition_context_size
        )

        # Use MLX-LM stream_generate
        for response in stream_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_tokens=config.max_tokens,
            sampler=sampler,
            logits_processors=logits_processors
        ):
            yield response.text
            # Allow other coroutines to run
            await asyncio.sleep(0)

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        # This is a basic implementation using the model's forward pass
        # For proper embedding models, this should be overridden
        import time

        start_time = time.time()
        embeddings = []

        for text in texts:
            # Tokenize the text
            tokens = self.tokenizer.encode(text)

            # Convert to MLX array and add batch dimension
            input_ids = mx.array([tokens])

            # Get model outputs (hidden states)
            # For embedding models, we typically want the last hidden state
            # This is a simplified approach - real embedding models may need different handling
            try:
                # Forward pass through the model
                outputs = self.model(input_ids)

                # For embedding models, we typically take the mean of the last hidden states
                # or use the [CLS] token representation
                if hasattr(outputs, 'last_hidden_state'):
                    hidden_states = outputs.last_hidden_state
                elif isinstance(outputs, tuple) and len(outputs) > 0:
                    # Some models return tuple of (logits, hidden_states)
                    hidden_states = outputs[0]
                else:
                    # If model returns logits directly, we need to get hidden states differently
                    # This is a fallback that may not work for all models
                    hidden_states = outputs

                # Mean pooling across sequence length (dim=1)
                embedding = mx.mean(hidden_states, axis=1).squeeze()

                # Ensure we have a 1D array and convert to list
                if embedding.ndim == 0:
                    embedding = mx.expand_dims(embedding, axis=0)

                embeddings.append(embedding.tolist())

            except Exception as e:
                logger.error(f"Error generating embedding for text: {e}")
                # Return a zero embedding as fallback
                embeddings.append([0.0] * 768)  # Default embedding size

        end_time = time.time()
        logger.info(f"Generated {len(embeddings)} embeddings in {end_time - start_time:.2f}s")

        return embeddings


class MLXWhisperWrapper(MLXModelWrapper):
    """Specialized wrapper for MLX-Whisper models."""

    def __init__(self, model_path: str, config: Dict[str, Any]):
        # Whisper models don't use tokenizers in the same way
        super().__init__(None, None, model_path, config)
        self.model_type = "whisper"
        self.whisper_model = None

    def load_whisper_model(self):
        """Load the Whisper model using mlx-whisper."""
        if not HAS_MLX_WHISPER:
            raise ImportError("mlx-whisper is required for Whisper models. Install with: pip install mlx-whisper")

        try:
            # Extract model name from path for mlx-whisper
            model_name = self.model_path.split("/")[-1] if "/" in self.model_path else self.model_path
            # Remove common prefixes/suffixes
            if model_name.endswith("-mlx"):
                model_name = model_name[:-4]
            if model_name.startswith("whisper-"):
                model_name = model_name[8:]

            logger.info(f"Loading Whisper model: {model_name}")
            self.whisper_model = mlx_whisper.load_model(model_name, path=self.model_path)
            return True
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            return False

    def transcribe_audio(self, audio_file_path: str, **kwargs):
        """Transcribe audio file using MLX-Whisper."""
        if not self.whisper_model:
            if not self.load_whisper_model():
                raise RuntimeError("Whisper model not loaded")

        try:
            result = mlx_whisper.transcribe(
                audio=audio_file_path,
                model=self.whisper_model,
                **kwargs
            )
            return result
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            raise


class MLXParakeetWrapper(MLXModelWrapper):
    """Specialized wrapper for Parakeet models using parakeet-mlx."""

    def __init__(self, model_path: str, config: Dict[str, Any]):
        super().__init__(None, None, model_path, config)
        self.model_type = "audio"
        self.parakeet_model = None

    def load_parakeet_model(self):
        """Load the Parakeet model using parakeet-mlx."""
        if not HAS_PARAKEET_MLX:
            raise ImportError("parakeet-mlx is required for Parakeet models. Install with: pip install parakeet-mlx")

        try:
            import parakeet_mlx

            logger.info(f"Loading Parakeet model from: {self.model_path}")

            # Load the Parakeet model using from_pretrained
            self.parakeet_model = parakeet_mlx.from_pretrained(self.model_path)

            logger.info(f"Successfully loaded Parakeet model")
            return True
        except Exception as e:
            logger.error(f"Failed to load Parakeet model: {e}")
            return False

    def transcribe_audio(self, audio_file_path: str, **kwargs):
        """Transcribe audio using Parakeet model."""
        if not self.parakeet_model:
            if not self.load_parakeet_model():
                raise RuntimeError("Parakeet model not loaded")

        try:
            # Filter kwargs to only include supported parameters for Parakeet
            supported_params = {}
            if 'chunk_duration' in kwargs:
                supported_params['chunk_duration'] = kwargs['chunk_duration']
            if 'overlap_duration' in kwargs:
                supported_params['overlap_duration'] = kwargs['overlap_duration']

            # Use the Parakeet model to transcribe
            result = self.parakeet_model.transcribe(audio_file_path, **supported_params)

            # The result is an AlignedResult object, extract the text
            if hasattr(result, 'sentences'):
                # Combine all sentences into one text
                text_parts = []
                for sentence in result.sentences:
                    if hasattr(sentence, 'text'):
                        text_parts.append(sentence.text)
                    elif hasattr(sentence, 'content'):
                        text_parts.append(sentence.content)
                transcribed_text = " ".join(text_parts)
                return {"text": transcribed_text}
            elif hasattr(result, 'text'):
                return {"text": result.text}
            else:
                return {"text": str(result)}

        except Exception as e:
            logger.error(f"Parakeet transcription failed: {e}")
            raise


class MLXAudioWrapper(MLXModelWrapper):
    """Specialized wrapper for other MLX-Audio models (non-Parakeet)."""

    def __init__(self, model_path: str, config: Dict[str, Any]):
        super().__init__(None, None, model_path, config)
        self.model_type = "audio"
        self.audio_model = None

    def load_audio_model(self):
        """Load the audio model using mlx-audio."""
        if not HAS_MLX_AUDIO:
            raise ImportError("mlx-audio is required for audio models. Install with: pip install mlx-audio")

        try:
            logger.info(f"Audio model ready: {self.model_path}")
            self.model_subtype = "stt"
            self.audio_model = "loaded"  # Placeholder
            return True
        except Exception as e:
            logger.error(f"Failed to load audio model: {e}")
            return False

    def transcribe_audio(self, audio_file_path: str, **kwargs):
        """Transcribe audio using MLX-Audio STT models."""
        if not self.audio_model:
            if not self.load_audio_model():
                raise RuntimeError("Audio model not loaded")

        try:
            import tempfile
            import os
            from mlx_audio.stt.generate import generate

            # Create a temporary output file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
                temp_output_path = temp_file.name

            try:
                # Call the generate function with model path and audio file
                generate(
                    model_path=self.model_path,
                    audio_path=audio_file_path,
                    output_path=temp_output_path,
                    format="txt",
                    verbose=False
                )

                # Read the transcribed text from the output file
                with open(temp_output_path, 'r') as f:
                    transcribed_text = f.read().strip()

                return {"text": transcribed_text}

            finally:
                # Clean up the temporary file
                if os.path.exists(temp_output_path):
                    os.unlink(temp_output_path)

        except Exception as e:
            logger.error(f"Audio transcription failed: {e}")
            raise


class MLXVisionWrapper(MLXModelWrapper):
    """Specialized wrapper for MLX-VLM vision/multimodal models."""

    def __init__(self, model, processor, model_path: str, config: Dict[str, Any]):
        super().__init__(model, processor, model_path, config)
        self.model_type = "vision"
        self.processor = processor
        self.model_config = model.config  # Use model.config for MLX-VLM operations

    def generate(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        """Generate text for a vision model with text-only input."""
        # For text-only, the prompt is a string. Wrap it in the message structure.
        messages = [{"role": "user", "content": prompt}]
        return self.generate_with_images(messages, images=[], config=config)

    async def generate_stream(self, prompt: str, config: GenerationConfig) -> AsyncGenerator[str, None]:
        """Simulate streaming for vision models by running a full generation and yielding the single result."""
        import asyncio
        # Run the synchronous generate method for text-only. This will block, but it prevents crashing.
        result = self.generate(prompt, config)
        yield result.text
        await asyncio.sleep(0)  # Allow other tasks to run.

    def generate_with_images(self, messages: List[Dict[str, Any]], images: List[str], config: GenerationConfig) -> GenerationResult:
        """Generate text with optional image inputs using MLX-VLM, accepting structured messages."""
        # Ensure mlx-vlm is available
        if not HAS_MLX_VLM:
            raise ImportError("mlx-vlm is required for vision models. Install with: pip install mlx-vlm")

        import time
        from mlx_vlm.prompt_utils import apply_chat_template

        start_time = time.time()

        try:
            logger.info(f"Generating with {len(images)} images and {len(messages)} messages.")

            # ------------------------------------------------------------------
            # Processor sanity-patches (Gemma3/Qwen3 variants may miss methods)
            # ------------------------------------------------------------------
            # 1. Ensure a .tokenizer attribute exists so downstream utilities work
            if not hasattr(self.processor, "tokenizer") or self.processor.tokenizer is None:
                self.processor.tokenizer = self.processor  # Use self as a fallback

            # 2. Copy encode/decode from inner tokenizer if missing
            for _method in ("encode", "decode"):
                if not hasattr(self.processor, _method):
                    inner_tok = getattr(self.processor, "tokenizer", None)
                    if inner_tok is not None and hasattr(inner_tok, _method):
                        setattr(self.processor, _method, getattr(inner_tok, _method))
                        logger.debug(f"Patched processor with '{_method}' from inner tokenizer")

            # ------------------------------------------------------------------
            # Format prompt & generate
            # ------------------------------------------------------------------
            formatted_prompt = apply_chat_template(
                processor=self.processor,
                config=self.model_config,
                prompt=messages,
                num_images=len(images)
            )

            image_list = images  # already a list of file paths

            logger.info(f"Using images: {image_list if image_list else 'text-only'}")
            logger.debug(f"Formatted prompt: {repr(formatted_prompt)}")

            from mlx_vlm import generate as vlm_generate
            result = vlm_generate(
                model=self.model,
                processor=self.processor,
                prompt=formatted_prompt,
                image=image_list,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                verbose=True
            )

            logger.debug(f"MLX-VLM raw result: {repr(result)}")

            end_time = time.time()
            generation_time = end_time - start_time

            # Extract text and usage stats from the verbose result object
            generated_text = result.text.strip()
            prompt_tokens = getattr(result, 'prompt_tokens', 0)
            completion_tokens = getattr(result, 'generation_tokens', 0)
            total_tokens = getattr(result, 'total_tokens', prompt_tokens + completion_tokens)
            tokens_per_second = getattr(result, 'generation_tps', 0)

            # Fallback if attributes are not present or zero
            if total_tokens == 0:
                # Simple approximation for fallback
                prompt_tokens = sum(len(str(m.get("content", "")).split()) for m in messages)
                completion_tokens = len(generated_text.split())
                total_tokens = prompt_tokens + completion_tokens

            if tokens_per_second == 0 and generation_time > 0:
                tokens_per_second = completion_tokens / generation_time

            return GenerationResult(
                text=generated_text,
                prompt=str(messages),  # Store the structured messages as a string for logging
                total_tokens=total_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                generation_time_seconds=generation_time,
                tokens_per_second=tokens_per_second
            )

        except Exception as e:
            logger.error(f"MLX-VLM generation failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise


class MLXLoader:
    """Handles loading models with MLX-LM."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "mlx-gui")
        os.makedirs(self.cache_dir, exist_ok=True)

    def download_model(self, model_id: str, token: Optional[str] = None) -> str:
        """Download model from HuggingFace Hub."""
        try:
            logger.info(f"Downloading model {model_id}")

            # Increase file descriptor limit temporarily
            import resource
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            try:
                resource.setrlimit(resource.RLIMIT_NOFILE, (min(hard, 8192), hard))
            except ValueError:
                logger.warning("Could not increase file descriptor limit")

            local_path = snapshot_download(
                repo_id=model_id,
                cache_dir=self.cache_dir,
                token=token,
                local_files_only=False,
                max_workers=4  # Limit concurrent downloads
            )

            # Restore original limit
            try:
                resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
            except ValueError:
                pass

            logger.info(f"Model {model_id} downloaded to {local_path}")
            return local_path

        except Exception as e:
            logger.error(f"Error downloading model {model_id}: {e}")
            raise

    def load_model(self, model_path: str) -> MLXModelWrapper:
        """Load a model using appropriate MLX library."""
        try:
            logger.info(f"Loading MLX model from {model_path}")

            # Detect model type from path/config first (before checking if path exists)
            model_type = self._detect_model_type(model_path)
            logger.info(f"Detected model type: {model_type}")

            # For parakeet models, we can load directly from HuggingFace ID
            if model_type == "audio" and "parakeet" in model_path.lower():
                logger.info(f"Loading Parakeet model from HuggingFace ID: {model_path}")
                config = {"estimated_memory_gb": 2.0}  # Default estimate for parakeet
                wrapper = MLXParakeetWrapper(model_path=model_path, config=config)
                if not wrapper.load_parakeet_model():
                    raise RuntimeError("Failed to load Parakeet model")
                return wrapper

            # For vision models, we can load directly from HuggingFace ID
            if model_type == "vision" and not os.path.exists(model_path):
                logger.info(f"Loading Vision model from HuggingFace ID: {model_path}")
                try:
                    from mlx_vlm import load as vlm_load
                    model, processor = vlm_load(model_path)
                    config = {"estimated_memory_gb": 8.0}  # Default estimate for vision models
                    wrapper = MLXVisionWrapper(
                        model=model,
                        processor=processor,
                        model_path=model_path,
                        config=config
                    )
                    return wrapper
                except Exception as e:
                    logger.error(f"Failed to load vision model from HuggingFace: {e}")
                    raise RuntimeError(f"Failed to load vision model: {e}")

            # Check if path exists (for local models)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model path does not exist: {model_path}")

            # Load config first to get additional info
            config = {}
            config_path = os.path.join(model_path, "config.json")
            if os.path.exists(config_path):
                import json
                with open(config_path, 'r') as f:
                    config = json.load(f)

            # Estimate memory usage
            memory_usage = self._estimate_model_memory(model_path)
            config['estimated_memory_gb'] = memory_usage

            # Load appropriate model type
            if model_type == "whisper":
                logger.info("Loading as Whisper model")
                wrapper = MLXWhisperWrapper(model_path=model_path, config=config)
                if not wrapper.load_whisper_model():
                    raise RuntimeError("Failed to load Whisper model")

            elif model_type == "audio":
                if "parakeet" in model_path.lower():
                    logger.info("Loading as Parakeet model")
                    wrapper = MLXParakeetWrapper(model_path=model_path, config=config)
                    if not wrapper.load_parakeet_model():
                        raise RuntimeError("Failed to load Parakeet model")
                else:
                    logger.info("Loading as Audio model")
                    wrapper = MLXAudioWrapper(model_path=model_path, config=config)
                    if not wrapper.load_audio_model():
                        raise RuntimeError("Failed to load audio model")

            elif model_type == "vision":
                # Vision/multimodal model using MLX-VLM
                logger.info("Loading as Vision/VLM model")
                try:
                    # MLX-VLM needs both model and processor
                    from mlx_vlm import load as vlm_load
                    model, processor = vlm_load(model_path)
                    wrapper = MLXVisionWrapper(
                        model=model,
                        processor=processor,
                        model_path=model_path,
                        config=config
                    )
                except Exception as e:
                    # Fallback to regular text model if VLM loading fails
                    logger.warning(f"Failed to load as VLM model, trying as text model: {e}")
                    try:
                        model, tokenizer = load(model_path)
                        wrapper = MLXModelWrapper(
                            model=model,
                            tokenizer=tokenizer,
                            model_path=model_path,
                            config=config
                        )
                    except Exception as fallback_e:
                        raise RuntimeError(f"Failed to load as both VLM and text model. VLM error: {e}, Text error: {fallback_e}")

            elif model_type == "embedding":
                logger.info("Loading as embedding model")
                try:
                    model, tokenizer = load(model_path)
                except Exception as e:
                    logger.error(f"Failed to load embedding model via mlx-lm: {e}")
                    raise

                wrapper = MLXModelWrapper(
                    model=model,
                    tokenizer=tokenizer,
                    model_path=model_path,
                    config=config
                )

            else:
                # Standard text generation model
                logger.info("Loading as text generation model")
                try:
                    model, tokenizer = load(model_path)
                except KeyError as e:
                    if str(e) == "'model'":
                        # This is a known bug in MLX-LM's gemma3n implementation
                        raise ValueError(f"Model '{model_path}' has incompatible format for MLX-LM. This appears to be a gemma3n model with a known MLX-LM compatibility issue.")
                    else:
                        raise

                wrapper = MLXModelWrapper(
                    model=model,
                    tokenizer=tokenizer,
                    model_path=model_path,
                    config=config
                )

            logger.info(f"Successfully loaded {model_type} model from {model_path}")
            logger.info(f"Estimated memory usage: {memory_usage:.1f}GB")

            return wrapper

        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            raise

    def _detect_model_type(self, model_path: str) -> str:
        """Detect the type of model from path and contents."""
        # Check path for model type indicators
        path_lower = model_path.lower()

        # Whisper models
        if "whisper" in path_lower:
            return "whisper"

        # Parakeet and other audio models (including HuggingFace IDs)
        if any(keyword in path_lower for keyword in ["parakeet", "speech", "stt", "asr", "tdt"]):
            return "audio"

        # Embedding models
        if "embedding" in path_lower or path_lower.endswith("-emb"):
            return "embedding"

        # Vision/multimodal models - includes Gemma 3 vision variants
        if any(keyword in path_lower for keyword in [
            "vision", "vlm", "multimodal", "llava", "qwen2-vl", "idefics",
            "gemma-3n", "gemma3n"  # Gemma 3 vision variants use MLX-VLM
        ]):
            return "vision"

        # Check config.json for additional hints
        try:
            config_path = os.path.join(model_path, "config.json")
            if os.path.exists(config_path):
                import json
                with open(config_path, 'r') as f:
                    config = json.load(f)

                # Check architecture types
                arch = config.get("architectures", [])
                if arch:
                    arch_str = str(arch[0]).lower()
                    if "whisper" in arch_str:
                        return "whisper"
                    if any(keyword in arch_str for keyword in ["speech", "audio", "parakeet"]):
                        return "audio"
                    if any(keyword in arch_str for keyword in ["vision", "vlm", "multimodal", "llava", "qwen2vl", "idefics"]):
                        return "vision"

                # Check model type field
                model_type_field = config.get("model_type", "").lower()
                if "whisper" in model_type_field:
                    return "whisper"
                if any(keyword in model_type_field for keyword in ["speech", "audio"]):
                    return "audio"
                if any(keyword in model_type_field for keyword in ["vision", "multimodal"]):
                    return "vision"
                if "embedding" in model_type_field:
                    return "embedding"

        except Exception as e:
            logger.debug(f"Could not read config for model type detection: {e}")

        # Default to text generation
        return "text"

    def load_from_hub(self, model_id: str, token: Optional[str] = None) -> MLXModelWrapper:
        """Load a model directly from HuggingFace Hub."""
        try:
            # Check if model is MLX compatible
            hf_client = get_huggingface_client(token)
            model_info = hf_client.get_model_details(model_id)

            if not model_info or not model_info.mlx_compatible:
                raise ValueError(f"Model {model_id} is not MLX compatible")

            # Use MLX repo if available
            if model_info.mlx_repo_id:
                logger.info(f"Using MLX version: {model_info.mlx_repo_id}")
                model_id = model_info.mlx_repo_id

            # Download and load
            local_path = self.download_model(model_id, token)
            return self.load_model(local_path)

        except Exception as e:
            logger.error(f"Error loading model {model_id} from hub: {e}")
            raise

    def _estimate_model_memory(self, model_path: str) -> float:
        """Estimate model memory usage from model files."""
        total_size = 0

        try:
            for root, dirs, files in os.walk(model_path):
                for file in files:
                    if file.endswith(('.safetensors', '.bin', '.pth', '.pt')):
                        file_path = os.path.join(root, file)
                        total_size += os.path.getsize(file_path)

            # Convert to GB and add overhead (MLX typically needs less memory than the file size)
            memory_gb = (total_size / (1024**3)) * 0.8  # 80% of file size
            return max(memory_gb, 0.5)  # Minimum 0.5GB

        except Exception as e:
            logger.warning(f"Could not estimate memory usage: {e}")
            return 2.0  # Default estimate


class MLXInferenceEngine:
    """High-level inference engine for MLX models."""

    def __init__(self):
        self.loader = MLXLoader()
        self._loaded_models: Dict[str, MLXModelWrapper] = {}

    def load_model(self, model_name: str, model_path: str, token: Optional[str] = None) -> MLXModelWrapper:
        """Load a model by name."""
        if model_name in self._loaded_models:
            return self._loaded_models[model_name]

        # Check for HF token in environment if not provided
        if not token:
            import os
            token = os.getenv('HUGGINGFACE_HUB_TOKEN') or os.getenv('HF_TOKEN')

        # Determine if this is a local path or HuggingFace model ID
        if os.path.exists(model_path):
            wrapper = self.loader.load_model(model_path)
        else:
            # Assume it's a HuggingFace model ID
            wrapper = self.loader.load_from_hub(model_path, token)

        self._loaded_models[model_name] = wrapper
        return wrapper

    def unload_model(self, model_name: str):
        """Unload a model from memory."""
        if model_name in self._loaded_models:
            # MLX models are automatically garbage collected
            del self._loaded_models[model_name]

            # Force garbage collection
            import gc
            gc.collect()

            # Clear MLX memory
            mx.metal.clear_cache()

            logger.info(f"Unloaded model {model_name}")

    def generate(self, model_name: str, prompt: str, config: GenerationConfig) -> GenerationResult:
        """Generate text using a loaded model."""
        if model_name not in self._loaded_models:
            raise ValueError(f"Model {model_name} is not loaded")

        wrapper = self._loaded_models[model_name]
        return wrapper.generate(prompt, config)

    async def generate_stream(self, model_name: str, prompt: str, config: GenerationConfig) -> AsyncGenerator[str, None]:
        """Generate text with streaming."""
        if model_name not in self._loaded_models:
            raise ValueError(f"Model {model_name} is not loaded")

        wrapper = self._loaded_models[model_name]
        async for token in wrapper.generate_stream(prompt, config):
            yield token

    def get_loaded_models(self) -> List[str]:
        """Get list of loaded model names."""
        return list(self._loaded_models.keys())

    def is_model_loaded(self, model_name: str) -> bool:
        """Check if a model is loaded."""
        return model_name in self._loaded_models

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a loaded model."""
        if model_name not in self._loaded_models:
            return None

        wrapper = self._loaded_models[model_name]
        return {
            "model_path": wrapper.model_path,
            "model_type": wrapper.model_type,
            "config": wrapper.config,
            "estimated_memory_gb": wrapper.config.get("estimated_memory_gb", 0)
        }


# Global inference engine
_inference_engine: Optional[MLXInferenceEngine] = None


def get_inference_engine() -> MLXInferenceEngine:
    """Get the global MLX inference engine."""
    global _inference_engine
    if _inference_engine is None:
        _inference_engine = MLXInferenceEngine()
    return _inference_engine


def validate_mlx_installation():
    """Validate that MLX is properly installed and working."""
    try:
        # Test basic MLX functionality
        x = mx.array([1, 2, 3])
        y = mx.array([4, 5, 6])
        z = x + y
        result = z.tolist()

        if result == [5, 7, 9]:
            logger.info("MLX installation validated successfully")
            return True
        else:
            logger.error("MLX computation test failed")
            return False

    except Exception as e:
        logger.error(f"MLX validation failed: {e}")
        return False


def get_mlx_device_info() -> Dict[str, Any]:
    """Get information about the MLX device."""
    try:
        return {
            "device": "GPU" if mx.metal.is_available() else "CPU",
            "metal_available": mx.metal.is_available(),
            "memory_limit": mx.metal.get_memory_limit() if mx.metal.is_available() else None,
            "peak_memory": mx.metal.get_peak_memory() if mx.metal.is_available() else None,
            "cache_size": mx.metal.get_cache_memory() if mx.metal.is_available() else None
        }
    except Exception as e:
        logger.error(f"Error getting MLX device info: {e}")
        return {"error": str(e)}

# ------------------------------------------------------------------
#  Hot-patch for Gemma VLM checkpoints missing conv_stem bias
#  We monkey-patch mlx_vlm.utils.sanitize_weights so the patch applies
#  both in the dev virtual-env and inside PyInstaller (runtime hook
#  already patches in the bundled app). Idempotent.
# ------------------------------------------------------------------
def _patch_mlx_vlm_bias():
    """Ensure sanitize_weights injects a zero bias if Gemma conv stem bias is absent (robust)."""
    try:
        import mlx_vlm.utils as vlm_utils
        if getattr(vlm_utils, "__bias_patch_applied", False):
            return

        orig_sanitize = vlm_utils.sanitize_weights

        def _ensure_bias(weights):
            """Inject zero bias if missing and weight present."""
            bias_key = "vision_tower.timm_model.conv_stem.conv.bias"
            weight_key = "vision_tower.timm_model.conv_stem.conv.weight"
            if bias_key not in weights and weight_key in weights:
                try:
                    import mlx.core as mx
                    w = weights[weight_key]
                    out_channels = w.shape[0] if hasattr(w, "shape") else len(w)
                    dtype = getattr(w, "dtype", mx.float32)
                    weights[bias_key] = mx.zeros((out_channels,), dtype=dtype)
                    logger.debug("Injected zero conv_stem bias")
                except Exception as e:
                    logger.warning(f"Bias injection failed: {e}")

        def _patched_sanitize(model_obj, weights, config=None):
            # Pre-patch to avoid upstream missing-param error
            _ensure_bias(weights)
            try:
                weights = orig_sanitize(model_obj, weights, config)
            except Exception as first_err:
                # Retry once after ensuring bias (if not already)
                _ensure_bias(weights)
                try:
                    weights = orig_sanitize(model_obj, weights, config)
                except Exception:
                    # Give up – re-raise original
                    raise first_err
            return weights

        vlm_utils.sanitize_weights = _patched_sanitize
        vlm_utils.__bias_patch_applied = True
        logger.info("Applied Gemma VLM bias patch (development mode, robust)")
    except ImportError:
        # mlx_vlm not available yet – skip
        pass

# Apply patch immediately on module import
_patch_mlx_vlm_bias()