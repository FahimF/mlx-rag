"""
FastAPI server for MLX-RAG REST API.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Depends, HTTPException, status, Header, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional, Union, Annotated
import json
import tempfile
import os
import uuid

from mlx_rag.database import get_db_session, get_database_manager
from mlx_rag.models import Model, AppSettings, RAGCollection, ChatSession, ChatMessage
from mlx_rag.system_monitor import get_system_monitor
from mlx_rag.huggingface_integration import get_huggingface_client
from mlx_rag.model_manager import get_model_manager
from mlx_rag.rag_manager import get_rag_manager
from mlx_rag.mlx_integration import GenerationConfig, get_inference_engine
from mlx_rag.inference_queue_manager import get_inference_manager, QueuedRequest
from mlx_rag.queued_inference import queued_generate_text, queued_generate_text_stream, queued_transcribe_audio, queued_generate_speech, queued_generate_embeddings, queued_generate_vision
from mlx_rag import __version__
import re

logger = logging.getLogger(__name__)

# API Key validation (accepts any key for OpenAI compatibility)
security = HTTPBearer(auto_error=False)

def validate_api_key(
    authorization: Optional[HTTPAuthorizationCredentials] = Depends(security),
    x_api_key: Optional[str] = Header(None, alias="x-api-key")
) -> Optional[str]:
    """
    Validate API key for OpenAI compatibility.
    Accepts any key via Authorization header or x-api-key header.
    """
    # Check Authorization: Bearer <token>
    if authorization:
        return authorization.credentials

    # Check x-api-key header
    if x_api_key:
        return x_api_key

    # For OpenAI compatibility, we accept any key, so return None if no key provided
    # This allows both authenticated and unauthenticated access
    return None


# Pydantic models for OpenAI compatibility
from typing import Union, Any, Dict

class ChatMessageContent(BaseModel):
    """Content part for multimodal messages."""
    type: str  # "text" or "image_url"
    text: Optional[str] = None
    image_url: Optional[Union[Dict[str, str], str]] = None  # Can be dict or direct string


class ToolCall(BaseModel):
    """Tool call in a message."""
    id: str
    type: str = "function"
    function: Dict[str, Any]


class ToolChoice(BaseModel):
    """Tool choice for function calling."""
    type: str = "function"
    function: Optional[Dict[str, str]] = None


class Tool(BaseModel):
    """Tool definition for function calling."""
    type: str = "function"
    function: Dict[str, Any]


class ChatCompletionMessage(BaseModel):
    role: Optional[str] = None  # "system", "user", "assistant", "tool"
    content: Optional[Union[str, List[ChatMessageContent]]] = None
    # Alternative image fields that some clients might use
    image: Optional[str] = None  # Single image as base64 or URL
    images: Optional[List[str]] = None  # Multiple images array
    # Tool calling support
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None  # For tool response messages


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatCompletionMessage]
    max_tokens: Optional[int] = 8192
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = 0
    repetition_penalty: Optional[float] = 1.0
    seed: Optional[int] = None
    stream: Optional[bool] = False
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, ToolChoice]] = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionMessage
    finish_reason: str


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage


class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: dict  # Use dict instead of ChatMessage for flexibility
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionStreamChoice]


class ModelInstallRequest(BaseModel):
    model_id: str
    name: Optional[str] = None


# Audio API models
class AudioTranscriptionRequest(BaseModel):
    """Request model for audio transcription."""
    model: str = "whisper-1"
    language: Optional[str] = None
    prompt: Optional[str] = None
    response_format: Optional[str] = "json"  # json, text, srt, verbose_json, vtt
    temperature: Optional[float] = 0.0


class AudioTranscriptionResponse(BaseModel):
    """Response model for audio transcription."""
    text: str


class AudioSpeechRequest(BaseModel):
    """Request model for text-to-speech."""
    model: str = "tts-1"  # tts-1, tts-1-hd
    input: str
    voice: str = "alloy"  # alloy, echo, fable, onyx, nova, shimmer
    response_format: Optional[str] = "mp3"  # mp3, opus, aac, flac
    speed: Optional[float] = 1.0  # 0.25 to 4.0


class EmbeddingRequest(BaseModel):
    """Request model for text embeddings."""
    input: Union[str, List[str]]
    model: str
    encoding_format: Optional[str] = "float"  # float, base64
    dimensions: Optional[int] = None  # Optional output dimensions
    user: Optional[str] = None  # Optional user identifier


class EmbeddingData(BaseModel):
    """Single embedding data entry."""
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingUsage(BaseModel):
    """Token usage for embeddings."""
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    """Response model for embeddings."""
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: EmbeddingUsage


def _get_chat_template_from_hf(model_id: str) -> str:
    """Fetch chat template from HuggingFace model card."""
    try:
        from huggingface_hub import hf_hub_download
        import json

        # Download tokenizer_config.json which contains the chat template
        tokenizer_config_path = hf_hub_download(
            repo_id=model_id,
            filename="tokenizer_config.json",
            local_files_only=False
        )

        with open(tokenizer_config_path, 'r') as f:
            tokenizer_config = json.load(f)

        chat_template = tokenizer_config.get("chat_template")
        if chat_template:
            logger.info(f"Retrieved chat template from HF for {model_id}")
            return chat_template

    except Exception as e:
        logger.debug(f"Could not fetch chat template from HF for {model_id}: {e}")

    return None


def _format_chat_prompt(messages: List[ChatCompletionMessage]) -> tuple[List[Dict[str, Any]], List[str]]:
    """Convert API ChatMessage objects to a list of dictionaries and extract image URLs."""
    chat_messages = []
    all_images = []

    for message in messages:
        content = message.content
        role = message.role

        if isinstance(content, list):
            # Handle multimodal content
            text_parts = []
            images = []
            for part in content:
                if part.type == "text" and part.text:
                    text_parts.append(part.text)
                elif part.type == "image_url" and part.image_url:
                    # Try different possible field names for the image URL
                    image_url = ""
                    if isinstance(part.image_url, dict):
                        # Standard OpenAI format: {"url": "data:image/..."}
                        image_url = part.image_url.get("url", "")
                        # Alternative format: {"image": "data:image/..."}
                        if not image_url:
                            image_url = part.image_url.get("image", "")
                        # Another alternative: {"data": "data:image/..."}
                        if not image_url:
                            image_url = part.image_url.get("data", "")
                    elif isinstance(part.image_url, str):
                        # Direct string format
                        image_url = part.image_url

                    if image_url:
                        images.append(image_url)
                        all_images.append(image_url)
                        logger.debug(f"Found image URL: {image_url[:50]}{'...' if len(image_url) > 50 else ''}")

            # Reconstruct content for the message dictionary
            reconstructed_content = " ".join(text_parts)
            chat_messages.append({"role": role, "content": reconstructed_content})

        elif isinstance(content, str):
            # Handle standard text content
            chat_messages.append({"role": role, "content": content})

        # Check for alternative image fields in the message itself
        # Some clients send images as separate fields like "image", "images", etc.
        if hasattr(message, 'image') and message.image:
            # Single image field
            all_images.append(message.image)
            logger.debug(f"Found image in message.image field: {message.image[:50]}{'...' if len(message.image) > 50 else ''}")

        if hasattr(message, 'images') and message.images:
            # Multiple images field
            for img in message.images:
                all_images.append(img)
                logger.debug(f"Found image in message.images array: {img[:50]}{'...' if len(img) > 50 else ''}")

            logger.debug(f"🔧 Image collection complete: {len(all_images)} images found")
    for i, img in enumerate(all_images):
        logger.debug(f"🔧 Collected image {i+1}: {img[:100]}{'...' if len(img) > 100 else ''}")

    return chat_messages, all_images


async def _apply_chat_template(tokenizer: Any, messages: List[Dict[str, Any]], model_name: str) -> str:
    """Apply a chat template to a list of messages."""
    try:
        # Use the tokenizer's chat template
        if hasattr(tokenizer, 'apply_chat_template') and getattr(tokenizer, 'chat_template', None):
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            logger.debug(f"Used tokenizer chat template for {model_name}")
            return formatted_prompt
    except Exception as e:
        logger.warning(f"Could not apply chat template for {model_name}: {e}")

    # Fallback to manual formatting
    prompt_parts = []
    is_gemma = "gemma" in model_name.lower()
    is_phi = "phi" in model_name.lower()

    for message in messages:
        role = message["role"]
        content = message["content"]

        if is_gemma:
            if role == "system":
                prompt_parts.append(f"<bos><start_of_turn>system\n{content}<end_of_turn>")
            elif role == "user":
                prompt_parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
            elif role == "assistant":
                prompt_parts.append(f"<start_of_turn>model\n{content}<end_of_turn>")
        elif is_phi:
            if role == "system": prompt_parts.append(f"System: {content}")
            elif role == "user": prompt_parts.append(f"User: {content}")
            elif role == "assistant": prompt_parts.append(f"Assistant: {content}")
        else: # ChatML format
            prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

    if is_gemma:
        prompt_parts.append("<start_of_turn>model\n")
    elif is_phi:
        prompt_parts.append("Assistant:")
    else:
        prompt_parts.append("<|im_start|>assistant")

    formatted_prompt = "\n".join(prompt_parts)
    logger.info(f"Used fallback manual template for {model_name}")
    return formatted_prompt


async def _process_image_urls(image_urls: List[str]) -> List[str]:
    """Process image URLs (base64 data or URLs) and save to temporary files."""
    import base64
    import tempfile
    import os
    from urllib.parse import urlparse
    import httpx

    processed_images = []

    logger.error(f"🔧 _process_image_urls CALLED with {len(image_urls)} URLs")
    for i, url in enumerate(image_urls):
        logger.error(f"🔧 Image {i+1}: {url[:100]}{'...' if len(url) > 100 else ''}")

    for i, image_url in enumerate(image_urls):
        logger.debug(f"Processing image {i+1}/{len(image_urls)}: {image_url[:100]}...")

        try:
            if image_url.startswith("data:image/"):
                # Handle base64 encoded images
                logger.debug("Processing base64 image data")

                if "," not in image_url:
                    logger.error(f"Invalid base64 image format - no comma separator: {image_url[:50]}...")
                    continue

                header, data = image_url.split(",", 1)
                logger.debug(f"Base64 header: {header}")
                logger.debug(f"Base64 data length: {len(data)} characters")

                try:
                    image_data = base64.b64decode(data)
                    logger.debug(f"Decoded image data: {len(image_data)} bytes")
                except Exception as decode_error:
                    logger.error(f"Base64 decode failed: {decode_error}")
                    continue

                # Determine file extension from header
                if "jpeg" in header or "jpg" in header:
                    ext = ".jpg"
                elif "png" in header:
                    ext = ".png"
                elif "gif" in header:
                    ext = ".gif"
                elif "webp" in header:
                    ext = ".webp"
                else:
                    ext = ".jpg"  # Default

                logger.debug(f"Using file extension: {ext}")

                # Save to temporary file
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                        tmp.write(image_data)
                        temp_path = tmp.name

                    # Verify file was created and has content
                    if os.path.exists(temp_path):
                        file_size = os.path.getsize(temp_path)
                        logger.debug(f"Created temporary file: {temp_path} ({file_size} bytes)")
                        processed_images.append(temp_path)
                    else:
                        logger.error(f"Temporary file was not created: {temp_path}")

                except Exception as file_error:
                    logger.error(f"Failed to write temporary file: {file_error}")
                    continue

            elif image_url.startswith(("http://", "https://")):
                # Handle URL images - download them
                logger.debug(f"Downloading image from URL: {image_url}")

                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.get(image_url)
                        response.raise_for_status()

                        logger.debug(f"Downloaded {len(response.content)} bytes")

                        # Determine extension from content-type or URL
                        content_type = response.headers.get("content-type", "")
                        if "jpeg" in content_type or "jpg" in content_type:
                            ext = ".jpg"
                        elif "png" in content_type:
                            ext = ".png"
                        elif "gif" in content_type:
                            ext = ".gif"
                        elif "webp" in content_type:
                            ext = ".webp"
                        else:
                            # Try to get from URL
                            parsed = urlparse(image_url)
                            path_ext = os.path.splitext(parsed.path)[1].lower()
                            ext = path_ext if path_ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"] else ".jpg"

                        logger.debug(f"Using file extension: {ext}")

                        # Save to temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                            tmp.write(response.content)
                            temp_path = tmp.name

                        # Verify file was created
                        if os.path.exists(temp_path):
                            file_size = os.path.getsize(temp_path)
                            logger.debug(f"Created temporary file: {temp_path} ({file_size} bytes)")
                            processed_images.append(temp_path)
                        else:
                            logger.error(f"Temporary file was not created: {temp_path}")

                except Exception as download_error:
                    logger.error(f"Failed to download image from URL: {download_error}")
                    continue

            else:
                # Check if this is raw base64 data (no data:image/ prefix)
                logger.debug(f"Checking if raw base64 data: {image_url[:50]}...")

                # Try to decode as raw base64 - if it works, it's likely an image
                try:
                    # Raw base64 should only contain valid base64 characters
                    import re
                    if re.match(r'^[A-Za-z0-9+/]*={0,2}$', image_url) and len(image_url) > 100:
                        logger.debug("Detected raw base64 image data")

                        # Attempt to decode
                        image_data = base64.b64decode(image_url)
                        logger.debug(f"Successfully decoded raw base64: {len(image_data)} bytes")

                        # Try to detect image format from the binary data
                        ext = ".png"  # Default to PNG
                        if image_data.startswith(b'\xFF\xD8\xFF'):
                            ext = ".jpg"
                        elif image_data.startswith(b'\x89PNG'):
                            ext = ".png"
                        elif image_data.startswith(b'GIF'):
                            ext = ".gif"
                        elif image_data.startswith(b'RIFF'):
                            ext = ".webp"

                        logger.debug(f"Detected image format: {ext}")

                        # Save to temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                            tmp.write(image_data)
                            temp_path = tmp.name

                        # Verify file was created
                        if os.path.exists(temp_path):
                            file_size = os.path.getsize(temp_path)
                            logger.debug(f"Created temporary file from raw base64: {temp_path} ({file_size} bytes)")
                            processed_images.append(temp_path)
                        else:
                            logger.error(f"Temporary file was not created: {temp_path}")
                    else:
                        logger.warning(f"Unsupported image URL format: {image_url[:100]}...")
                        continue

                except Exception as base64_error:
                    logger.warning(f"Failed to decode as raw base64: {base64_error}")
                    logger.warning(f"Unsupported image URL format: {image_url[:100]}...")
                    continue

        except Exception as e:
            logger.error(f"Failed to process image URL {image_url[:100]}...: {e}")
            logger.debug(f"Exception type: {type(e)}, Args: {e.args}")
            continue

    logger.info(f"Successfully processed {len(processed_images)} out of {len(image_urls)} images")
    for i, path in enumerate(processed_images):
        logger.debug(f"Processed image {i+1}: {path}")

    return processed_images


def _parse_tool_calls_from_text(text: str) -> List[Dict[str, Any]]:
    """Parse tool calls from LLM response text.
    
    This function detects tool calls in various formats:
    - JSON function calls
    - XML-style tool tags
    - Structured tool call patterns
    - Qwen3-style TOOL_CALL format
    - Reasoning-based tool intentions (for models that use <think> tags)
    
    Includes aggressive early termination to prevent infinite tool call loops.
    """
    # AGGRESSIVE: If the text is extremely long and repetitive, truncate it early
    if len(text) > 10000:  # If response is longer than 10k characters
        # Count occurrences of "TOOL_CALL" to detect excessive repetition
        tool_call_count = text.count("TOOL_CALL:")
        if tool_call_count > 10:  # More than 10 tool calls detected
            logger.warning(f"🚨 EMERGENCY: Detected {tool_call_count} TOOL_CALL occurrences in {len(text)} char response. Truncating to prevent infinite loop.")
            # Find the position after the 3rd TOOL_CALL and truncate there
            call_positions = []
            pos = 0
            for _ in range(4):  # Find first 4 occurrences
                pos = text.find("TOOL_CALL:", pos)
                if pos == -1:
                    break
                call_positions.append(pos)
                pos += 1
            
            if len(call_positions) >= 3:
                # Truncate after the 3rd tool call
                truncation_point = call_positions[2] + 100  # Give some space after 3rd call
                text = text[:truncation_point]
                logger.warning(f"🚨 Text truncated from {len(text)} to {truncation_point} characters")
    """Parse tool calls from LLM response text.
    
    This function detects tool calls in various formats:
    - JSON function calls
    - XML-style tool tags
    - Structured tool call patterns
    - Qwen3-style TOOL_CALL format
    - Reasoning-based tool intentions (for models that use <think> tags)
    """
    tool_calls = []
    
    # Pattern 1: Qwen3-style TOOL_CALL format
    # Look for patterns like:
    # TOOL_CALL: function_name
    # PARAMETERS: {"key": "value"}
    qwen_pattern = r'TOOL_CALL:\s*(\w+)\s*\n\s*PARAMETERS:\s*(\{[^}]*\})'
    qwen_matches = re.finditer(qwen_pattern, text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
    
    for match in qwen_matches:
        function_name = match.group(1)
        parameters_str = match.group(2)
        
        try:
            arguments = json.loads(parameters_str)
            tool_call = {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": function_name,
                    "arguments": json.dumps(arguments)
                }
            }
            tool_calls.append(tool_call)
            logger.debug(f"Parsed Qwen3 tool call: {function_name}")
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse Qwen3 parameters: {parameters_str}")
    
    # Pattern 2: JSON-style function calls
    # Look for patterns like: {"function": "tool_name", "arguments": {...}}
    json_pattern = r'\{\s*"function"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^}]*\})\s*\}'
    json_matches = re.finditer(json_pattern, text, re.IGNORECASE | re.DOTALL)
    
    for match in json_matches:
        function_name = match.group(1)
        arguments_str = match.group(2)
        
        try:
            arguments = json.loads(arguments_str)
            tool_call = {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": function_name,
                    "arguments": json.dumps(arguments)
                }
            }
            tool_calls.append(tool_call)
            logger.debug(f"Parsed JSON tool call: {function_name}")
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON arguments: {arguments_str}")
    
    # Pattern 3: XML-style tool calls
    # Look for patterns like: <tool_call function="tool_name" args='{"key": "value"}'/>
    xml_pattern = r'<tool_call\s+function="([^"]+)"\s+args=\'([^\']*)\'/?>'
    xml_matches = re.finditer(xml_pattern, text, re.IGNORECASE | re.DOTALL)
    
    for match in xml_matches:
        function_name = match.group(1)
        arguments_str = match.group(2)
        
        try:
            arguments = json.loads(arguments_str) if arguments_str else {}
            tool_call = {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": function_name,
                    "arguments": json.dumps(arguments)
                }
            }
            tool_calls.append(tool_call)
            logger.debug(f"Parsed XML tool call: {function_name}")
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse XML arguments: {arguments_str}")
    
    # Pattern 4: Function call patterns with parentheses
    # Look for patterns like: tool_name({"key": "value"})
    func_pattern = r'(\w+)\s*\(\s*(\{[^}]*\})\s*\)'
    func_matches = re.finditer(func_pattern, text, re.IGNORECASE | re.DOTALL)
    
    # List of known tool names to avoid false positives
    known_tools = {'list_directory', 'read_file', 'search_files', 'write_file', 'edit_file'}
    
    for match in func_matches:
        function_name = match.group(1)
        arguments_str = match.group(2)
        
        # Only process if it's a known tool name
        if function_name in known_tools:
            try:
                arguments = json.loads(arguments_str)
                tool_call = {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": json.dumps(arguments)
                    }
                }
                tool_calls.append(tool_call)
                logger.debug(f"Parsed function call: {function_name}")
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse function arguments: {arguments_str}")
    
    # Pattern 5: Natural language tool detection (Enhanced for Qwen3-Coder-30B)
    # Look for common phrases that indicate tool usage
    if not tool_calls:
        # Look for phrases like "let me explore", "I need to find", "First, let me list", etc.
        natural_patterns = [
            # Direct action patterns
            (r'(?:let me|I need to|I will|I should|First,? let me)\s+(?:explore|list|check)\s+(?:the\s+)?(?:directory|folder)(?:\s+structure)?', 'list_directory', '.'),
            (r'(?:let me|I need to|I will|I should|First,? let me)\s+(?:find|locate|search for)\s+(?:the\s+)?(?:main\.dart|[\w\.]+\.dart|file)', 'search_files', 'main.dart'),
            (r'(?:let me|I need to|I will|I should|First,? let me)\s+(?:read|examine|look at|check)\s+(?:the\s+)?(?:main\.dart|[\w\.]+\.dart|file)', 'read_file', 'main.dart'),
            (r'(?:explore|list|check)\s+(?:the\s+)?(?:directory|folder)(?:\s+structure)?', 'list_directory', '.'),
            (r'(?:find|locate|search for)\s+(?:the\s+)?(?:main\.dart|[\w\.]+\.dart)', 'search_files', 'main.dart'),
            (r'(?:read|examine|look at|check)\s+(?:the\s+)?(?:main\.dart|[\w\.]+\.dart)', 'read_file', 'main.dart'),
            
            # Qwen3-specific patterns
            (r'To\s+(?:read|examine|check)\s+(?:the\s+)?([\w\.\-/]+\.\w+)', 'read_file', None),
            (r'(?:I\'ll|Let me)\s+(?:start by|first)\s+(?:reading|examining|checking)\s+(?:the\s+)?([\w\.\-/]+\.\w+)', 'read_file', None),
            (r'(?:To understand|To analyze|To examine)\s+(?:the\s+)?([\w\.\-/]+\.\w+)', 'read_file', None),
            (r'(?:I need to|I should)\s+(?:read|examine|look at)\s+(?:the\s+)?([\w\.\-/]+\.\w+)', 'read_file', None),
            (r'(?:Reading|Examining|Checking)\s+(?:the\s+)?([\w\.\-/]+\.\w+)', 'read_file', None),
            (r'First,?\s+(?:I\'ll|let me|I will)\s+(?:read|examine|check)\s+(?:the\s+)?([\w\.\-/]+\.\w+)', 'read_file', None),
            
            # Directory exploration patterns
            (r'(?:I\'ll|Let me)\s+(?:start by|first)\s+(?:exploring|listing|checking)\s+(?:the\s+)?(?:directory|folder|structure)', 'list_directory', '.'),
            (r'(?:explore|list|check)\s+(?:the\s+)?(?:current\s+)?(?:directory|folder)', 'list_directory', '.'),
            
            # Search patterns
            (r'(?:I need to|I\'ll)\s+(?:search for|find|locate)\s+([\w\.\-/]+)', 'search_files', None),
            (r'(?:searching for|looking for|finding)\s+([\w\.\-/]+)', 'search_files', None)
        ]
        
        for pattern, tool_name, default_arg in natural_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Create appropriate arguments based on the tool
                if tool_name == 'list_directory':
                    arguments = {"path": "."}
                elif tool_name == 'search_files':
                    # Try to extract filename from the match or use default
                    filename = default_arg
                    match_text = match.group(0).lower()
                    # Look for specific filenames in the match
                    file_match = re.search(r'([\w\-\.]+\.(?:dart|py|js|ts|java|cpp|c|h|json|yaml|yml|xml|html|css|md|txt))', match_text)
                    if file_match:
                        filename = file_match.group(1)
                    arguments = {"query": filename, "path": "."}
                elif tool_name == 'read_file':
                    # Try to extract filename from the match or use default
                    filename = default_arg
                    match_text = match.group(0).lower()
                    file_match = re.search(r'([\w\-\.]+\.(?:dart|py|js|ts|java|cpp|c|h|json|yaml|yml|xml|html|css|md|txt))', match_text)
                    if file_match:
                        filename = file_match.group(1)
                    arguments = {"path": filename}
                else:
                    continue
                
                tool_call = {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(arguments)
                    }
                }
                tool_calls.append(tool_call)
                logger.info(f"Inferred tool call from natural language: {tool_name} with {arguments}")
                break  # Only create one tool call per natural language detection
        
        # If still no tools found, try the more complex reasoning extraction
        if not tool_calls:
            tool_calls.extend(_extract_tool_intentions_from_reasoning(text, known_tools))
    
    # Early detection: check for excessive repetition in raw text before processing
    if len(tool_calls) > 20:  # If we found more than 20 tool calls, likely a repetition issue
        logger.warning(f"Detected {len(tool_calls)} tool calls, checking for repetition patterns...")
        
        # Count function names to detect excessive repetition
        function_counts = {}
        for call in tool_calls:
            func_name = call["function"]["name"]
            function_counts[func_name] = function_counts.get(func_name, 0) + 1
        
        # Check if any function appears excessively
        max_count = max(function_counts.values()) if function_counts else 0
        if max_count > 10:  # If any function appears more than 10 times
            most_repeated_func = max(function_counts, key=function_counts.get)
            logger.error(f"Detected excessive repetition: '{most_repeated_func}' appears {max_count} times. Limiting to first few calls.")
            # Keep only the first few calls to avoid the repetition
            tool_calls = tool_calls[:5]
        
        # Also check for alternating patterns in large sets
        if len(tool_calls) > 10:
            call_names = [call["function"]["name"] for call in tool_calls]
            # Check if we have clear alternating pattern (A-B-A-B...)
            if len(set(call_names[:10])) <= 2:  # Only 1-2 unique functions in first 10 calls
                pattern_funcs = list(set(call_names[:4]))  # Get unique functions from first 4
                if len(pattern_funcs) == 2:
                    # Check if it's truly alternating
                    expected_pattern = (pattern_funcs * 10)[:20]  # Create expected alternating pattern
                    actual_pattern = call_names[:20]
                    
                    if actual_pattern[:len(expected_pattern)] == expected_pattern[:len(actual_pattern)]:
                        logger.error(f"Detected clear alternating pattern: {pattern_funcs[0]} <-> {pattern_funcs[1]}. Limiting to first few calls.")
                        # Keep just one of each to break the pattern
                        tool_calls = tool_calls[:4]  # Keep first 4 (2 of each type)
    
    # Remove duplicates based on function name and arguments
    seen = set()
    unique_tool_calls = []
    call_counts = {}  # Track repetition counts
    
    for call in tool_calls:
        call_key = (call["function"]["name"], call["function"]["arguments"])
        
        # Count occurrences of this specific call
        if call_key not in call_counts:
            call_counts[call_key] = 0
        call_counts[call_key] += 1
        
        # If this call appears more than 2 times, stop processing more calls (reduced from 3)
        if call_counts[call_key] > 2:
            logger.error(f"Detected repetitive tool call pattern: {call['function']['name']} appears {call_counts[call_key]} times. Truncating tool call list.")
            break
            
        if call_key not in seen:
            seen.add(call_key)
            unique_tool_calls.append(call)
    
    # Additional safety: limit total unique tool calls to prevent overwhelming execution
    if len(unique_tool_calls) > 5:  # Reduced from 10 to 5
        logger.warning(f"Too many tool calls detected ({len(unique_tool_calls)}), limiting to first 5")
        unique_tool_calls = unique_tool_calls[:5]
    
    # Enhanced alternating pattern detection
    if len(unique_tool_calls) >= 4:
        call_names = [call["function"]["name"] for call in unique_tool_calls]
        
        # Check for simple A-B-A-B patterns
        for i in range(len(call_names) - 3):
            if call_names[i] == call_names[i + 2] and call_names[i + 1] == call_names[i + 3]:
                logger.error(f"Detected alternating tool call pattern: {call_names[i]} <-> {call_names[i + 1]}. Truncating to first 2 calls.")
                # Keep only the first 2 calls to break the pattern
                unique_tool_calls = unique_tool_calls[:2]
                break
        
        # Also check for A-A-B-B patterns (pairs)
        if len(call_names) >= 4:
            for i in range(len(call_names) - 3):
                if (call_names[i] == call_names[i + 1] and 
                    call_names[i + 2] == call_names[i + 3] and 
                    call_names[i] != call_names[i + 2]):
                    logger.error(f"Detected paired repetition pattern: {call_names[i]}-{call_names[i]} {call_names[i + 2]}-{call_names[i + 2]}. Truncating to first 2 calls.")
                    unique_tool_calls = unique_tool_calls[:2]
                    break
    
    if unique_tool_calls:
        logger.info(f"Detected {len(unique_tool_calls)} tool calls in LLM response (after filtering)")
    
    return unique_tool_calls


def _extract_tool_intentions_from_reasoning(text: str, known_tools: set) -> List[Dict[str, Any]]:
    """Extract tool intentions from model reasoning when explicit tool calls aren't present.
    
    This handles models like GLM and DeepSeek that use <think> tags for reasoning
    but don't output standard OpenAI tool call formats.
    """
    tool_calls = []
    
    # Extract content from <think> tags first
    think_pattern = r'<think>(.*?)</think>'
    think_matches = re.findall(think_pattern, text, re.IGNORECASE | re.DOTALL)
    
    reasoning_text = text
    if think_matches:
        # Combine all thinking content
        reasoning_text = ' '.join(think_matches) + ' ' + text
        logger.debug(f"Found {len(think_matches)} <think> sections to analyze")
    
    # Look for tool usage intentions in the reasoning
    tool_intention_patterns = [
        # "I need to use read_file to read main.dart"
        r'(?:need to|should|will|let me|going to)\s+(?:use\s+)?(?:the\s+)?(\w+)\s+(?:tool\s+)?(?:to\s+)?(?:read|list|search|write|edit|modify)\s+(?:the\s+)?([^\s\.\,]+)',
        # "Let me read/edit the main.dart file"
        r'let me\s+(read|list|search|edit|modify)\s+(?:the\s+)?([^\s\.\,]+)',
        # "I'll use read_file/edit_file with path main.dart"
        r"I\'ll\s+use\s+(\w+)(?:\s+with\s+path\s+|\s+to\s+(?:read|edit|modify)\s+|\s+for\s+)([^\s\.\,]+)",
        # "First, read/modify main.dart"
        r'(?:first|then|next),?\s+(read|list|search|edit|modify)\s+([^\s\.\,]+)',
        # "Use read_file/edit_file function to read/modify main.dart"
        r'use\s+(\w+)(?:\s+function)?\s+to\s+(?:read|list|search|edit|modify)\s+([^\s\.\,]+)',
        # "I must examine/modify the main.dart file"
        r'(?:must|need to|should)\s+(?:examine|check|read|modify|edit|update)\s+(?:the\s+)?([^\s\.\,]+)',
        # Direct mentions of modification without tool names
        r'(?:modify|edit|change|update)\s+(?:the\s+)?([^\s\.\,]+)\s+(?:file|code)'
    ]
    
    for pattern in tool_intention_patterns:
        matches = re.finditer(pattern, reasoning_text, re.IGNORECASE)
        
        for match in matches:
            groups = match.groups()
            if len(groups) >= 2:
                action_or_tool = groups[0].lower()
                target = groups[1].strip()
                
                # Map actions to tools
                tool_name = None
                arguments = {}
                
                if action_or_tool in known_tools:
                    tool_name = action_or_tool
                elif action_or_tool == 'read':
                    tool_name = 'read_file'
                    arguments = {'path': target}
                elif action_or_tool == 'list':
                    tool_name = 'list_directory'
                    arguments = {'path': target if target not in ['.', 'root', 'directory'] else '.'}
                elif action_or_tool == 'search':
                    tool_name = 'search_files'
                    arguments = {'query': target}
                elif action_or_tool in ['edit', 'modify']:
                    # User wants to modify - start by reading the file first
                    tool_name = 'read_file'
                    arguments = {'path': target}
                
                # Try to extract more specific arguments from context
                if tool_name and not arguments:
                    if tool_name == 'read_file':
                        arguments = {'path': target}
                    elif tool_name == 'list_directory':
                        arguments = {'path': target if target not in ['.', 'root', 'directory'] else '.'}
                    elif tool_name == 'search_files':
                        arguments = {'query': target, 'path': '.'}
                
                if tool_name and arguments:
                    tool_call = {
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(arguments)
                        }
                    }
                    tool_calls.append(tool_call)
                    logger.info(f"Inferred tool call from reasoning: {tool_name} with {arguments}")
    
    # Special case: if user asks to read specific file and model mentions it
    if not tool_calls:
        # Look for file names in the text (common patterns)
        file_patterns = [
            r'(\w+\.\w+)',  # filename.extension
            r'([\w\/\-\.]+\.(?:py|js|ts|dart|java|cpp|c|h|json|yaml|yml|xml|html|css|md|txt|cfg|conf))',  # various file extensions
        ]
        
        for pattern in file_patterns:
            file_matches = re.findall(pattern, text, re.IGNORECASE)
            for filename in file_matches[:1]:  # Only take first match to avoid spam
                if len(filename) > 2:  # Avoid very short matches
                    tool_call = {
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": json.dumps({"path": filename})
                        }
                    }
                    tool_calls.append(tool_call)
                    logger.info(f"Inferred file read from context: {filename}")
                    break
    
    return tool_calls


def _check_model_supports_tools(model_path: str, model_name: str = None) -> bool:
    """Check if a model supports function calling/tools by examining its configuration.
    
    Args:
        model_path: Path to the model (HuggingFace ID or local path)
        model_name: Optional model name for additional context
        
    Returns:
        True if the model likely supports function calling, False otherwise
    """
    try:
        # List of model patterns/names known to support function calling
        function_calling_models = {
            # OpenAI-style models
            'gpt-4', 'gpt-3.5', 'gpt-35',
            # Anthropic models
            'claude',
            # Google models
            'gemini', 'bard',
            # Meta models with function calling
            'llama-3.1', 'llama-3.2', 'code-llama',
            # Mistral models with function calling
            'mistral', 'mixtral',
            # Other function calling models
            'hermes', 'functionary', 'gorilla', 'nexusraven',
            # Specific MLX models known to support tools
            'qwen', 'yi-', 'deepseek',
            # Models with "chat" or "instruct" that often support tools
            'chat', 'instruct', 'assistant'
        }
        
        # Check model name/path against known patterns
        model_identifier = (model_name or model_path or "").lower()
        
        # Check for explicit function calling indicators
        for pattern in function_calling_models:
            if pattern in model_identifier:
                logger.debug(f"Model {model_identifier} matches function calling pattern: {pattern}")
                return True
        
        # Try to load tokenizer config to check for function calling support
        try:
            from huggingface_hub import hf_hub_download
            import json
            import os
            
            # Try to get tokenizer config
            config_path = None
            if os.path.exists(model_path):
                # Local model
                tokenizer_config_path = os.path.join(model_path, "tokenizer_config.json")
                if os.path.exists(tokenizer_config_path):
                    config_path = tokenizer_config_path
            else:
                # HuggingFace model - try to download config
                try:
                    config_path = hf_hub_download(
                        repo_id=model_path,
                        filename="tokenizer_config.json",
                        local_files_only=False
                    )
                except:
                    pass
            
            if config_path and os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # Check for function calling related tokens or settings
                chat_template = config.get('chat_template', '')
                if chat_template:
                    # Look for function calling indicators in chat template
                    function_indicators = [
                        'function', 'tool', 'call', 'available_tools',
                        'tools', 'function_call', 'tool_call'
                    ]
                    
                    template_lower = chat_template.lower()
                    for indicator in function_indicators:
                        if indicator in template_lower:
                            logger.debug(f"Found function calling indicator '{indicator}' in chat template for {model_identifier}")
                            return True
                
                # Check for special tokens related to function calling
                special_tokens = config.get('added_tokens_decoder', {})
                for token_info in special_tokens.values():
                    if isinstance(token_info, dict):
                        content = token_info.get('content', '').lower()
                        if any(indicator in content for indicator in ['function', 'tool', 'call']):
                            logger.debug(f"Found function calling token: {content} for {model_identifier}")
                            return True
        
        except Exception as e:
            logger.debug(f"Could not check tokenizer config for {model_identifier}: {e}")
        
        # Try to check model config as well
        try:
            from huggingface_hub import hf_hub_download
            import json
            import os
            
            config_path = None
            if os.path.exists(model_path):
                # Local model
                model_config_path = os.path.join(model_path, "config.json")
                if os.path.exists(model_config_path):
                    config_path = model_config_path
            else:
                # HuggingFace model
                try:
                    config_path = hf_hub_download(
                        repo_id=model_path,
                        filename="config.json",
                        local_files_only=False
                    )
                except:
                    pass
            
            if config_path and os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # Check model architecture - some architectures are more likely to support tools
                architectures = config.get('architectures', [])
                for arch in architectures:
                    arch_lower = arch.lower()
                    # Models with these architectures often support function calling
                    if any(pattern in arch_lower for pattern in ['llama', 'mistral', 'qwen', 'yi', 'gemma']):
                        # Check if it's a recent/instruct version
                        if any(indicator in model_identifier for indicator in ['instruct', 'chat', '3.1', '3.2', '2.1']):
                            logger.debug(f"Model {model_identifier} has architecture {arch} and appears to be an instruct variant")
                            return True
        
        except Exception as e:
            logger.debug(f"Could not check model config for {model_identifier}: {e}")
        
        # Default to False if we can't determine support
        logger.debug(f"Could not determine function calling support for {model_identifier}, defaulting to False")
        return False
        
    except Exception as e:
        logger.error(f"Error checking tool support for {model_path}: {e}")
        return False


def _check_model_supports_standard_tool_calling(model_name: str) -> bool:
    """Check if a model supports standard OpenAI-style tool calling.
    
    Args:
        model_name: Name of the model to check
        
    Returns:
        True if model supports standard OpenAI tool calling, False if it needs server-side execution
    """
    # Models that support standard OpenAI tool calling format
    openai_compatible_models = {
        'gpt-4', 'gpt-3.5', 'gpt-35',
        'claude', 'claude-3',
        'qwen2.5-coder',  # Qwen 2.5 supports OpenAI format
        'llama-3.1', 'llama-3.2',
        'mistral', 'mixtral'
    }
    
    model_lower = model_name.lower()
    
    # Check for explicit OpenAI compatibility
    for pattern in openai_compatible_models:
        if pattern in model_lower:
            logger.debug(f"Model {model_name} supports standard OpenAI tool calling")
            return True
    
    # Models that need server-side tool execution (like Qwen3)
    server_side_models = {
        'qwen3-coder',  # Qwen 3 needs server-side execution
        'deepseek',
        'glm',
    }
    
    for pattern in server_side_models:
        if pattern in model_lower:
            logger.debug(f"Model {model_name} needs server-side tool execution")
            return False
    
    # Default to server-side execution for unknown models
    logger.debug(f"Model {model_name} unknown, defaulting to server-side tool execution")
    return False


async def _execute_tools_and_continue_conversation(
    model: str,
    original_messages: List[Dict[str, Any]],
    detected_tool_calls: List[Dict[str, Any]],
    tool_executor,
    config: GenerationConfig,
    max_iterations: int = 5
) -> str:
    """Execute tools server-side and continue conversation until final answer.
    
    Args:
        model: Name of the model to use
        original_messages: Original conversation messages
        detected_tool_calls: Tool calls detected from model response
        tool_executor: Tool executor instance
        config: Generation configuration
        max_iterations: Maximum number of tool execution iterations
        
    Returns:
        Final response text from the model
    """
    from mlx_rag.queued_inference import queued_generate_text
    from mlx_rag.model_manager import get_model_manager
    import json
    
    model_manager = get_model_manager()
    loaded_model = model_manager.get_model_for_inference(model)
    
    if not loaded_model:
        logger.error(f"Model {model} not loaded for multi-turn tool execution")
        return "Error: Model not available for tool execution"
    
    # Start with original conversation
    conversation_history = original_messages.copy()
    
    # Track executed tool calls to detect loops
    executed_calls_history = []
    repeated_call_count = {}
    
    # Determine if this is a problematic model that needs more lenient loop detection
    is_problematic_model = any(pattern in model.lower() for pattern in [
        'qwen3', 'qwen-3', 'coder-30b', 'deepseek'
    ])
    
    # Adjust thresholds based on model
    if is_problematic_model:
        max_repeats = 5  # More lenient for problematic models
        min_iterations_for_pattern_check = 4  # Require more iterations before checking patterns
        logger.info(f"Using lenient loop detection for potentially problematic model: {model}")
    else:
        max_repeats = 3  # Standard threshold
        min_iterations_for_pattern_check = 3
    
    for iteration in range(max_iterations):
        logger.info(f"Tool execution iteration {iteration + 1}/{max_iterations}")
        
        # Check for repetitive tool calls before executing
        for tool_call in detected_tool_calls:
            function_name = tool_call["function"]["name"]
            arguments = tool_call["function"]["arguments"]
            call_signature = f"{function_name}({arguments})"
            
            # Track repeated calls
            if call_signature not in repeated_call_count:
                repeated_call_count[call_signature] = 0
            repeated_call_count[call_signature] += 1
            
            # If a specific call has been repeated too many times, stop
            if repeated_call_count[call_signature] > max_repeats:
                logger.error(f"Tool call '{call_signature}' has been repeated {repeated_call_count[call_signature]} times, breaking loop")
                return f"I've noticed that I'm repeatedly trying the same tool call: {function_name}. Let me provide a final answer based on what I've already learned."
        
        # Add current tool calls to history
        executed_calls_history.append(detected_tool_calls)
        
        # Check if we're in a pattern loop (same sequence of calls) - but only after enough iterations
        if len(executed_calls_history) >= min_iterations_for_pattern_check:
            # Check if last 2 iterations are identical - but be more lenient for problematic models
            recent_calls_1 = [call["function"]["name"] for call in executed_calls_history[-1]]
            recent_calls_2 = [call["function"]["name"] for call in executed_calls_history[-2]]
            
            # For problematic models, also check if the calls are actually making progress
            # (e.g., read_file -> edit_file is legitimate progression)
            if recent_calls_1 == recent_calls_2:
                # Check if this is a legitimate workflow progression
                legitimate_progressions = [
                    ['list_directory', 'read_file'],
                    ['read_file', 'edit_file'],
                    ['search_files', 'read_file'],
                    ['list_directory', 'search_files'],
                    ['read_file', 'write_file']
                ]
                
                is_legitimate_progression = any(
                    set(recent_calls_1).issubset(set(progression)) or set(progression).issubset(set(recent_calls_1))
                    for progression in legitimate_progressions
                )
                
                if not is_legitimate_progression or not is_problematic_model:
                    logger.error(f"Detected identical tool call patterns in last 2 iterations: {recent_calls_1}")
                    return "I notice I'm repeating the same sequence of tool calls. Let me provide a final answer based on the information I've already gathered."
                else:
                    logger.info(f"Allowing repeated tool calls for legitimate workflow progression: {recent_calls_1}")
            
            # Check for alternating patterns (A-B-A-B) - only for non-problematic models or after more iterations
            if len(executed_calls_history) >= 4 and (not is_problematic_model or len(executed_calls_history) >= 6):
                calls_1 = [call["function"]["name"] for call in executed_calls_history[-4]]
                calls_2 = [call["function"]["name"] for call in executed_calls_history[-3]]
                calls_3 = [call["function"]["name"] for call in executed_calls_history[-2]]
                calls_4 = [call["function"]["name"] for call in executed_calls_history[-1]]
                
                if calls_1 == calls_3 and calls_2 == calls_4:
                    logger.error(f"Detected alternating tool call pattern: {calls_1} <-> {calls_2}")
                    return "I notice I'm alternating between the same tool calls without making progress. Let me provide a final answer with the information I have."
        
        # Execute detected tool calls
        try:
            tool_results = await tool_executor.execute_tool_calls(detected_tool_calls)
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return f"I encountered an error while executing tools: {str(e)}"
        
        if not tool_results:
            logger.warning("No tool results returned")
            return "I couldn't execute the requested tools."
        
        # Add tool results to conversation
        for i, (tool_call, result) in enumerate(zip(detected_tool_calls, tool_results)):
            function_name = tool_call["function"]["name"]
            
            if result.success:
                tool_message = {
                    "role": "user",
                    "content": f"Tool '{function_name}' result: {result.result}"
                }
            else:
                tool_message = {
                    "role": "user",  
                    "content": f"Tool '{function_name}' failed: {result.error}"
                }
            
            conversation_history.append(tool_message)
            logger.debug(f"Added tool result {i+1} to conversation")
        
        # Generate model response with tool results
        try:
            # Add more specific instruction to encourage final answer
            if iteration >= 2:  # After 2 iterations, be more insistent
                instruction = "You have sufficient information from the tool results. Please provide your final answer now without making additional tool calls."
            else:
                instruction = "Based on the tool results above, please provide your final answer to the original question."
                
            conversation_history.append({
                "role": "user",
                "content": instruction
            })
            
            # Format conversation for the model
            prompt_string = await _apply_chat_template(
                loaded_model.mlx_wrapper.tokenizer, 
                conversation_history, 
                model
            )
            
            # Generate response
            result = await queued_generate_text(model, prompt_string, config)
            
            if not result or not result.text:
                logger.error("No response from model after tool execution")
                return "I was unable to generate a response after executing the tools."
            
            # Check if the model wants to make more tool calls
            new_tool_calls = _parse_tool_calls_from_text(result.text)
            
            if new_tool_calls:
                logger.info(f"Model wants to execute {len(new_tool_calls)} more tools in iteration {iteration + 1}")
                
                # Additional loop protection: limit total tool calls
                total_calls_so_far = sum(len(calls) for calls in executed_calls_history)
                if total_calls_so_far >= 15:  # Maximum 15 total tool calls
                    logger.error(f"Reached maximum total tool calls ({total_calls_so_far}), stopping")
                    return "I've executed many tool calls and should now provide a final answer based on the gathered information."
                
                detected_tool_calls = new_tool_calls
                
                # Add model's intermediate response to conversation
                conversation_history.append({
                    "role": "assistant",
                    "content": result.text
                })
                
                # Continue to next iteration
                continue
            else:
                # Model provided final answer
                logger.info(f"Multi-turn tool execution completed after {iteration + 1} iterations")
                return result.text
                
        except Exception as e:
            logger.error(f"Error in conversation iteration {iteration + 1}: {e}")
            return f"I encountered an error while processing the tool results: {str(e)}"
    
    # If we reach max iterations, return the last response
    logger.warning(f"Reached maximum iterations ({max_iterations}) for tool execution")
    return "I've executed the requested tools but reached the maximum number of iterations. Please try breaking down your request into smaller parts."


async def _generate_with_early_tool_stopping(
    model_name: str,
    prompt: str,
    config: GenerationConfig,
    loaded_model
) -> Any:
    """Generate text with ultra-early stopping for problematic tool-calling models.
    
    This function monitors generation token-by-token and stops immediately when
    it detects the start of repetitive tool calls, while preserving valid tool calls.
    """
    from mlx_rag.queued_inference import queued_generate_text
    
    logger.info(f"🛡️ Using early tool stopping for {model_name}")
    
    # Start generation
    accumulated_text = ""
    tool_call_count = 0
    consecutive_tool_calls = 0
    last_tool_call_pos = -1
    last_complete_tool_call = -1
    
    try:
        # Use streaming generation to monitor token-by-token
        async for chunk in loaded_model.mlx_wrapper.generate_stream(prompt, config):
            accumulated_text += chunk
            
            # Check for TOOL_CALL patterns as soon as they appear
            current_tool_calls = accumulated_text.count("TOOL_CALL:")
            
            if current_tool_calls > tool_call_count:
                # New tool call detected
                tool_call_count = current_tool_calls
                
                # Find position of latest tool call
                latest_pos = accumulated_text.rfind("TOOL_CALL:")
                
                # Check if this is close to the previous one (consecutive)
                if last_tool_call_pos != -1 and (latest_pos - last_tool_call_pos) < 300:
                    consecutive_tool_calls += 1
                else:
                    consecutive_tool_calls = 1
                
                last_tool_call_pos = latest_pos
                
                logger.info(f"🛡️ Tool call #{tool_call_count} detected at position {latest_pos}, consecutive: {consecutive_tool_calls}")
                
                # More lenient stopping: Allow up to 3 tool calls before stopping
                if tool_call_count >= 3 or consecutive_tool_calls >= 3:
                    logger.warning(f"🛡️ EARLY STOP: {tool_call_count} tool calls, {consecutive_tool_calls} consecutive - stopping generation")
                    break
            
            # Look for complete tool call patterns (TOOL_CALL: + PARAMETERS:)
            if "PARAMETERS:" in accumulated_text:
                # Count complete tool call pairs
                complete_calls = accumulated_text.count("TOOL_CALL:") if "PARAMETERS:" in accumulated_text else 0
                if complete_calls > last_complete_tool_call:
                    last_complete_tool_call = complete_calls
                    logger.info(f"🛡️ Complete tool call pattern #{complete_calls} detected")
            
            # Allow longer responses to capture complete tool calls
            if len(accumulated_text) > 3000:  # Increased limit to allow complete tool calls
                logger.warning(f"🛡️ EARLY STOP: Response length {len(accumulated_text)} exceeded limit - stopping generation")
                break
        
        # Create a mock result object similar to queued_generate_text
        class MockResult:
            def __init__(self, text: str):
                self.text = text
                self.prompt_tokens = len(prompt.split())
                self.completion_tokens = len(text.split())
                self.total_tokens = self.prompt_tokens + self.completion_tokens
                self.generation_time_seconds = 1.0
                self.tokens_per_second = self.completion_tokens / self.generation_time_seconds
        
        logger.info(f"🛡️ Early stopping generation completed: {len(accumulated_text)} chars, {tool_call_count} tool calls")
        return MockResult(accumulated_text)
        
    except Exception as e:
        logger.error(f"🛡️ Error in early stopping generation: {e}")
        # Fallback to normal generation
        return await queued_generate_text(model_name, prompt, config)


def _create_system_prompt_with_tools(
    tools: List[Dict[str, Any]], 
    context: Optional[str] = None,
    user_query: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    collection_path: Optional[str] = None
) -> str:
    """Create a comprehensive system prompt that instructs the LLM on how to use tools.
    
    Args:
        tools: List of available tools in OpenAI format
        context: Optional context about the current task or domain  
        user_query: Optional user query for contextual prompt generation
        conversation_history: Optional conversation history for context
        collection_path: Optional path to the RAG collection (prevents hallucination)
        
    Returns:
        Comprehensive system prompt with tool guidance
    """
    from mlx_rag.tool_prompts import generate_tool_system_prompt

    if not tools:
        return "You are a helpful AI assistant. Provide clear, accurate, and concise responses."

    # Build enhanced context with collection path information
    enhanced_context = context or ""
    
    if collection_path:
        path_context = f"\n\nIMPORTANT: You are working with a RAG collection located at: {collection_path}\n"
        path_context += f"When using tools, all paths should be relative to this collection directory.\n"
        path_context += f"Do not use made-up or hallucinated paths. Only use real paths that exist within: {collection_path}\n"
        enhanced_context += path_context

    # Use the enhanced context
    return generate_tool_system_prompt(tools, enhanced_context)






def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="MLX-RAG API",
        description="A lightweight RESTful wrapper around Apple's MLX engine",
        version=__version__
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Mount static files for admin interface
    from pathlib import Path
    static_path = Path(__file__).parent / "templates" / "static"
    if static_path.exists():
        app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

    # Exception handlers
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"}
        )

    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with basic server info."""
        return {
            "name": "MLX-RAG API",
            "version": __version__,
            "status": "running"
        }

    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy"}

    # API v1 routes
    @app.get("/v1/manager/models")
    async def list_models_internal(db: Session = Depends(get_db_session)):
        """List all models (internal format)."""
        models = db.query(Model).all()
        return {
            "models": [
                {
                    "id": model.id,
                    "name": model.name,
                    "type": model.model_type,
                    "status": model.status,
                    "memory_required_gb": model.memory_required_gb,
                    "use_count": model.use_count,
                    "last_used_at": model.last_used_at.isoformat() if model.last_used_at else None,
                    "created_at": model.created_at.isoformat() if model.created_at else None,
                    "huggingface_id": model.huggingface_id,
                    "author": model.huggingface_id.split("/")[0] if model.huggingface_id and "/" in model.huggingface_id else "unknown",
                    "supports_tools": _check_model_supports_tools(model.path, model.name),
                }
                for model in models
            ]
        }

    @app.post("/v1/manager/models/rescan")
    async def rescan_models():
        """Rescan the HuggingFace cache for new models."""
        try:
            model_manager = get_model_manager()
            model_manager.scan_and_register_cached_models()
            return {"message": "Model cache scan completed."}
        except Exception as e:
            logger.error(f"Error during model cache scan: {e}")
            raise HTTPException(status_code=500, detail="Error scanning model cache")

    # RAG endpoints
    @app.post("/v1/rag/collections")
    async def create_rag_collection(name: str, path: str, db: Session = Depends(get_db_session)):
        """Create a new RAG collection."""
        try:
            rag_manager = get_rag_manager()
            rag_manager.create_collection(name, path)
            return {"message": f"RAG collection '{name}' created successfully."}
        except Exception as e:
            logger.error(f"Error creating RAG collection: {e}")
            raise HTTPException(status_code=500, detail="Error creating RAG collection")

    @app.get("/v1/rag/collections")
    async def list_rag_collections(db: Session = Depends(get_db_session)):
        """List all RAG collections."""
        collections = db.query(RAGCollection).all()
        return {"collections": [
            {
                "id": col.id,
                "name": col.name,
                "path": col.path,
                "status": col.status,
                "is_active": col.is_active,
            }
            for col in collections
        ]}

    @app.post("/v1/rag/collections/{collection_name}/activate")
    async def activate_rag_collection(collection_name: str, db: Session = Depends(get_db_session)):
        """Activate a RAG collection."""
        collection = db.query(RAGCollection).filter(RAGCollection.name == collection_name).first()
        if not collection:
            raise HTTPException(status_code=404, detail="RAG collection not found")

        # Deactivate all other collections
        db.query(RAGCollection).update({RAGCollection.is_active: False})
        collection.is_active = True
        db.commit()
        return {"message": f"RAG collection '{collection_name}' activated."}

    @app.delete("/v1/rag/collections/{collection_name}")
    async def delete_rag_collection(collection_name: str, db: Session = Depends(get_db_session)):
        """Delete a RAG collection."""
        collection = db.query(RAGCollection).filter(RAGCollection.name == collection_name).first()
        if not collection:
            raise HTTPException(status_code=404, detail="RAG collection not found")

        rag_manager = get_rag_manager()
        rag_manager.delete_collection(collection_name)

        db.delete(collection)
        db.commit()
        return {"message": f"RAG collection '{collection_name}' deleted."}

    @app.get("/v1/rag/languages")
    async def get_supported_languages():
        """Get supported programming languages for RAG processing."""
        rag_manager = get_rag_manager()
        languages = []
        
        for ext, config in rag_manager.language_config.items():
            languages.append({
                "extension": ext,
                "language": config['name'],
                "available": config['parser'] is not None and config['language'] is not None,
                "node_types": config['node_types'],
                "package_name": f"tree-sitter-{config['name'].lower().replace(' ', '-').replace('+', 'p')}"
            })
        
        # Sort by language name
        languages.sort(key=lambda x: x['language'])
        
        available_count = sum(1 for lang in languages if lang['available'])
        total_count = len(languages)
        
        return {
            "languages": languages,
            "summary": {
                "total_languages": total_count,
                "available_languages": available_count,
                "missing_languages": total_count - available_count
            }
        }

    @app.post("/v1/rag/collections/{collection_name}/reprocess")
    async def reprocess_rag_collection(collection_name: str, db: Session = Depends(get_db_session)):
        """Reprocess a RAG collection."""
        collection = db.query(RAGCollection).filter(RAGCollection.name == collection_name).first()
        if not collection:
            raise HTTPException(status_code=404, detail="RAG collection not found")

        rag_manager = get_rag_manager()
        rag_manager.reprocess_collection(collection_name)

        return {"message": f"Reprocessing of RAG collection '{collection_name}' started."}

    @app.post("/v1/rag/query")
    async def query_rag(query: str, db: Session = Depends(get_db_session)):
        """Query the active RAG collection."""
        active_collection = db.query(RAGCollection).filter(RAGCollection.is_active == True).first()
        if not active_collection:
            raise HTTPException(status_code=400, detail="No active RAG collection")

        rag_manager = get_rag_manager()
        response = rag_manager.query(query, active_collection.name)
        return {"query": query, "response": response}

    # @app.get("/v1/tools")
    # async def get_available_tools(
    #     use_langchain: bool = False,
    #     db: Session = Depends(get_db_session)
    # ):
    #     """Get available tools for the active RAG collection."""
    #     try:
    #         # Get the active RAG collection
    #         active_collection = db.query(RAGCollection).filter(RAGCollection.is_active == True).first()
            
    #         if not active_collection:
    #             return {"tools": [], "system": "original"}
            
    #         if use_langchain:
    #             # Use LangChain tool executor
    #             langchain_executor = get_langchain_tool_executor(active_collection.path)
                
    #             if not langchain_executor.has_available_tools():
    #                 return {"tools": [], "system": "langchain"}
                
    #             # Get tools in OpenAI format from LangChain
    #             tools = langchain_executor.get_openai_tool_definitions()
                
    #             return {
    #                 "tools": tools, 
    #                 "system": "langchain",
    #                 "tool_count": len(tools),
    #                 "collection_path": active_collection.path
    #             }
    #         else:
    #             # Use original tool executor
    #             tool_executor = get_tool_executor(active_collection.path)
                
    #             if not tool_executor.has_available_tools():
    #                 return {"tools": [], "system": "original"}
                
    #             # Get tools in OpenAI format
    #             tools = tool_executor.get_tools_for_openai_request()
                
    #             return {
    #                 "tools": tools,
    #                 "system": "original", 
    #                 "tool_count": len(tools),
    #                 "collection_path": active_collection.path
    #             }
            
    #     except Exception as e:
    #         logger.error(f"Error getting available tools: {e}")
    #         return {"tools": [], "error": str(e)}

    # @app.post("/v1/tools/execute")
    # async def execute_tool(
    #     request: dict,
    #     use_langchain: bool = False,
    #     db: Session = Depends(get_db_session)
    # ):
    #     """Execute a tool call."""
    #     try:
    #         # Get the active RAG collection
    #         active_collection = db.query(RAGCollection).filter(RAGCollection.is_active == True).first()
            
    #         if not active_collection:
    #             raise HTTPException(
    #                 status_code=status.HTTP_400_BAD_REQUEST,
    #                 detail="No active RAG collection for tool execution"
    #             )
            
    #         # Extract tool call details
    #         function_name = request.get("function_name")
    #         arguments_str = request.get("arguments", "{}")
    #         tool_call_id = request.get("tool_call_id")
            
    #         if not function_name:
    #             raise HTTPException(
    #                 status_code=status.HTTP_400_BAD_REQUEST,
    #                 detail="function_name is required"
    #             )
            
    #         # Parse arguments
    #         try:
    #             if isinstance(arguments_str, str):
    #                 arguments = json.loads(arguments_str)
    #             else:
    #                 arguments = arguments_str
    #         except json.JSONDecodeError as e:
    #             raise HTTPException(
    #                 status_code=status.HTTP_400_BAD_REQUEST,
    #                 detail=f"Invalid JSON in arguments: {e}"
    #             )
            
    #         if use_langchain:
    #             # Use LangChain tool executor
    #             langchain_executor = get_langchain_tool_executor(active_collection.path)
                
    #             if not langchain_executor.has_available_tools():
    #                 raise HTTPException(
    #                     status_code=status.HTTP_400_BAD_REQUEST,
    #                     detail="No LangChain tools available for execution"
    #                 )
                
    #             # Create tool call in the expected format
    #             tool_call = {
    #                 "id": tool_call_id or f"call_{uuid.uuid4().hex[:8]}",
    #                 "type": "function",
    #                 "function": {
    #                     "name": function_name,
    #                     "arguments": json.dumps(arguments)
    #                 }
    #             }
                
    #             # Execute using LangChain
    #             results = await langchain_executor.execute_multiple_tool_calls_async([tool_call])
                
    #             if not results:
    #                 raise HTTPException(
    #                     status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    #                     detail="LangChain tool execution failed - no results returned"
    #                 )
                
    #             # Return the first result
    #             result = results[0]
                
    #             # Return in a format that's easy for the frontend to use
    #             return {
    #                 "success": result.success,
    #                 "result": result.result,
    #                 "error": result.error,
    #                 "tool_call_id": tool_call_id,
    #                 "function_name": function_name,
    #                 "system": "langchain"
    #             }
    #         else:
    #             # Use original tool executor
    #             tool_executor = get_tool_executor(active_collection.path)
                
    #             if not tool_executor.has_available_tools():
    #                 raise HTTPException(
    #                     status_code=status.HTTP_400_BAD_REQUEST,
    #                     detail="No tools available for execution"
    #                 )
                
    #             # Create tool call in the expected format
    #             tool_call = {
    #                 "id": tool_call_id or f"call_{uuid.uuid4().hex[:8]}",
    #                 "type": "function",
    #                 "function": {
    #                     "name": function_name,
    #                     "arguments": json.dumps(arguments)
    #                 }
    #             }
                
    #             # Execute the tool call
    #             results = await tool_executor.execute_tool_calls([tool_call])
                
    #             if not results:
    #                 raise HTTPException(
    #                     status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    #                     detail="Tool execution failed - no results returned"
    #                 )
                
    #             # Return the first result
    #             result = results[0]
                
    #             # Return in a format that's easy for the frontend to use
    #             return {
    #                 "success": result.success,
    #                 "result": result.result,
    #                 "error": result.error,
    #                 "tool_call_id": tool_call_id,
    #                 "function_name": function_name,
    #                 "system": "original"
    #             }
            
    #     except HTTPException:
    #         raise
    #     except Exception as e:
    #         logger.error(f"Error executing tool: {e}")
    #         raise HTTPException(
    #             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    #             detail=f"Tool execution failed: {str(e)}"
    #         )

    # Chat session management endpoints
    @app.get("/v1/chat/sessions")
    async def list_chat_sessions(db: Session = Depends(get_db_session)):
        """List all chat sessions ordered by last message time."""
        try:
            sessions = db.query(ChatSession).order_by(ChatSession.last_message_at.desc().nullslast(), ChatSession.updated_at.desc()).all()            
            sessions_data = [
                {
                    "session_id": session.session_id,
                    "title": session.get_display_title(),
                    "created_at": session.created_at.isoformat() if session.created_at else None,
                    "updated_at": session.updated_at.isoformat() if session.updated_at else None,
                    "last_message_at": session.last_message_at.isoformat() if session.last_message_at else None,
                    "message_count": session.message_count,
                    "model_name": session.model_name,
                    "rag_collection_name": session.rag_collection_name
                }
                for session in sessions
            ]
            return {"sessions": sessions_data}
        except Exception as e:
            print(f"[BACKEND] Error listing chat sessions: {e}")
            logger.error(f"Error listing chat sessions: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error listing chat sessions"
            )

    @app.post("/v1/chat/sessions")
    async def create_chat_session(
        request: dict,
        db: Session = Depends(get_db_session)
    ):
        """Create a new chat session."""
        try:
            import uuid
            
            session_id = str(uuid.uuid4())
            title = request.get("title", "New Chat")
            model_name = request.get("model_name")
            rag_collection_name = request.get("rag_collection_name")
            
            new_session = ChatSession(
                session_id=session_id,
                title=title,
                model_name=model_name,
                rag_collection_name=rag_collection_name
            )
            
            db.add(new_session)
            db.commit()
            
            return {
                "session_id": session_id,
                "title": title,
                "created_at": new_session.created_at.isoformat(),
                "model_name": model_name,
                "rag_collection_name": rag_collection_name
            }
        except Exception as e:
            logger.error(f"Error creating chat session: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error creating chat session"
            )

    @app.get("/v1/chat/sessions/{session_id}")
    async def get_chat_session(
        session_id: str,
        db: Session = Depends(get_db_session)
    ):
        """Get a specific chat session with its messages."""
        try:
            session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
            if not session:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Chat session '{session_id}' not found"
                )            
            messages = db.query(ChatMessage).filter(
                ChatMessage.session_id == session_id
            ).order_by(ChatMessage.created_at.asc()).all()
            
            session_data = {
                "session_id": session.session_id,
                "title": session.title,
                "created_at": session.created_at.isoformat() if session.created_at else None,
                "updated_at": session.updated_at.isoformat() if session.updated_at else None,
                "last_message_at": session.last_message_at.isoformat() if session.last_message_at else None,
                "message_count": session.message_count,
                "model_name": session.model_name,
                "rag_collection_name": session.rag_collection_name,
                "messages": [
                    {
                        "id": msg.id,
                        "role": msg.role,
                        "content": msg.content,
                        "created_at": msg.created_at.isoformat() if msg.created_at else None,
                        "model_name": msg.model_name,
                        "rag_collection_name": msg.rag_collection_name,
                        "metadata": msg.get_metadata()
                    }
                    for msg in messages
                ]
            }
            return session_data
        except HTTPException:
            print(f"[BACKEND] HTTPException in get_chat_session for {session_id}")
            raise
        except Exception as e:
            print(f"[BACKEND] Exception in get_chat_session for {session_id}: {e}")
            logger.error(f"Error getting chat session {session_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error getting chat session"
            )

    @app.put("/v1/chat/sessions/{session_id}/title")
    async def update_chat_session_title(
        session_id: str,
        request: dict,
        db: Session = Depends(get_db_session)
    ):
        """Update the title of a chat session."""
        try:
            session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
            if not session:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Chat session '{session_id}' not found"
                )
            
            new_title = request.get("title", "").strip()
            if not new_title:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Title cannot be empty"
                )
            
            session.title = new_title
            db.commit()
            
            return {
                "session_id": session_id,
                "title": new_title,
                "updated_at": session.updated_at.isoformat()
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error updating chat session title: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error updating chat session title"
            )

    @app.delete("/v1/chat/sessions/{session_id}")
    async def delete_chat_session(
        session_id: str,
        db: Session = Depends(get_db_session)
    ):
        """Delete a chat session and all its messages."""
        try:
            session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
            if not session:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Chat session '{session_id}' not found"
                )
            
            # Delete all messages (cascade should handle this, but being explicit)
            db.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete()
            
            # Delete the session
            db.delete(session)
            db.commit()
            
            return {
                "message": f"Chat session '{session_id}' deleted successfully"
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error deleting chat session {session_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error deleting chat session"
            )

    @app.post("/v1/chat/sessions/{session_id}/messages")
    async def add_chat_message(
        session_id: str,
        request: dict,
        db: Session = Depends(get_db_session)
    ):
        """Add a message to a chat session."""
        try:
            session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
            if not session:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Chat session '{session_id}' not found"
                )
            
            role = request.get("role")
            content = request.get("content")
            model_name = request.get("model_name")
            rag_collection_name = request.get("rag_collection_name")
            metadata = request.get("metadata", {})
            
            if not role or not content:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Role and content are required"
                )
            
            new_message = ChatMessage(
                session_id=session_id,
                role=role,
                content=content,
                model_name=model_name,
                rag_collection_name=rag_collection_name
            )
            
            if metadata:
                new_message.set_metadata(metadata)
            
            db.add(new_message)
            
            # Update session stats
            session.message_count += 1
            session.update_last_message()
            if model_name:
                session.model_name = model_name
            if rag_collection_name:
                session.rag_collection_name = rag_collection_name
            
            db.commit()
            
            return {
                "message_id": new_message.id,
                "session_id": session_id,
                "role": role,
                "content": content,
                "created_at": new_message.created_at.isoformat()
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error adding message to session {session_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error adding message to session"
            )

    @app.post("/v1/chat")
    async def chat(
        message: str = Form(...),
        model: str = Form(...),
        rag_collection: Optional[str] = Form(None),
        history: Optional[str] = Form(None),
        images: Optional[List[UploadFile]] = File(None),
        db: Session = Depends(get_db_session)
    ):
        """Handle a chat request with RAG support, auto-loading, and image uploads."""
        logger.info(f"Chat request received - model: {model}, rag_collection: {rag_collection}, images: {len(images) if images else 0}")
        
        if not message or not model:
            logger.error(f"Missing required fields - message: {bool(message)}, model: {bool(model)}")
            raise HTTPException(status_code=400, detail="Message and model are required.")
        
        # Parse history if provided
        parsed_history = []
        if history:
            try:
                parsed_history = json.loads(history)
            except json.JSONDecodeError:
                logger.warning(f"Invalid history JSON, using empty history: {history}")
                parsed_history = []
        
        # Process uploaded images if any
        image_urls = []
        if images:
            logger.info(f"Processing {len(images)} uploaded images")
            for i, image_file in enumerate(images):
                try:
                    # Validate image file
                    if not image_file.content_type or not image_file.content_type.startswith('image/'):
                        logger.warning(f"Skipping non-image file: {image_file.filename}")
                        continue
                    
                    # Read image content and convert to base64
                    image_content = await image_file.read()
                    import base64
                    
                    # Create data URL
                    mime_type = image_file.content_type
                    base64_data = base64.b64encode(image_content).decode('utf-8')
                    data_url = f"data:{mime_type};base64,{base64_data}"
                    
                    image_urls.append(data_url)
                    logger.debug(f"Processed uploaded image {i+1}: {image_file.filename} ({len(image_content)} bytes)")
                    
                except Exception as e:
                    logger.error(f"Error processing uploaded image {i+1}: {e}")
                    continue
        
        model_name = model
        rag_collection_name = rag_collection

        async def stream_response():
            chunk_count = 0
            last_chunk = ""
            duplicate_count = 0
            recent_chunks = []  # Keep track of recent chunks for pattern detection
            accumulated_content = ""  # Track total content length
            repetition_patterns = set()  # Track repeated patterns
            
            try:
                # Check if model exists in database
                model_record = db.query(Model).filter(Model.name == model_name).first()
                if not model_record:
                    yield f"Error: Model '{model_name}' not found. Please install it first."
                    return

                model_manager = get_model_manager()
                
                # Check if model is loaded, auto-load if not (like chat/completions endpoint)
                language_model = model_manager.get_model_for_inference(model_name)
                if not language_model:
                    logger.info(f"Model {model_name} not loaded, attempting to load...")
                    yield f"Loading model {model_name}..."
                    success = await model_manager.load_model_async(
                        model_name=model_name,
                        model_path=model_record.path,
                        priority=10  # High priority for chat requests
                    )
                    if not success:
                        yield f"Error: Failed to load model '{model_name}'"
                        return

                    # Re-fetch the loaded model after loading
                    language_model = model_manager.get_model_for_inference(model_name)
                    if not language_model:
                        yield f"Error: Model '{model_name}' failed to load properly"
                        return
                    
                    yield f"Model {model_name} loaded successfully.\n\n"

                # Get RAG context if a collection is selected
                rag_context = ""
                if rag_collection_name:
                    try:
                        rag_manager = get_rag_manager()
                        # Get the ChromaDB collection and query it directly
                        chroma_collection = rag_manager.chroma_client.get_collection(name=rag_collection_name)
                        results = chroma_collection.query(
                            query_texts=[message],
                            n_results=5
                        )
                        # Join the retrieved documents as context
                        if results["documents"][0]:
                            rag_context = "\n".join(results["documents"][0])
                            logger.info(f"RAG context retrieved: {len(rag_context)} characters from {len(results['documents'][0])} documents")
                    except Exception as e:
                        logger.error(f"Error retrieving RAG context: {e}")
                        yield f"Warning: Could not retrieve RAG context from '{rag_collection_name}': {e}\n\n"

                # Check if this is a vision model and we have images
                is_vision_model = False
                if language_model and hasattr(language_model.mlx_wrapper, 'model_type'):
                    is_vision_model = language_model.mlx_wrapper.model_type == "vision"
                
                logger.debug(f"Vision model: {is_vision_model}, Images: {len(image_urls)}")
                
                if is_vision_model and image_urls:
                    # For vision models with images, use vision processing
                    try:
                        # Create ChatCompletionMessage objects for vision processing
                        multimodal_content = []
                        multimodal_content.append(ChatMessageContent(type="text", text=message))
                        for image_url in image_urls:
                            multimodal_content.append(ChatMessageContent(
                                type="image_url", 
                                image_url={"url": image_url}
                            ))
                        
                        # Create structured messages
                        structured_messages = []
                        for msg in parsed_history:
                            structured_messages.append(ChatCompletionMessage(
                                role=msg.get("role", "user"),
                                content=msg.get("content", "")
                            ))
                        
                        # Add the current user message with images
                        structured_messages.append(ChatCompletionMessage(
                            role="user",
                            content=multimodal_content
                        ))
                        
                        # Extract chat messages and images for vision processing
                        chat_messages, processed_images = _format_chat_prompt(structured_messages)
                        
                        # Process image paths
                        processed_image_paths = await _process_image_urls(processed_images)
                        
                        # Generate using vision model
                        config = GenerationConfig(
                            max_tokens=2048,
                            temperature=0.7
                        )
                        
                        result = await queued_generate_vision(model_name, chat_messages, processed_image_paths, config)
                        
                        # Clean up temporary image files
                        for img_path in processed_image_paths:
                            try:
                                import os
                                os.unlink(img_path)
                            except Exception as e:
                                logger.warning(f"Failed to cleanup temporary image file {img_path}: {e}")
                        
                        # Stream the result
                        content = result.text if result else "Error: Vision processing failed"
                        
                        # Stream in chunks for better UX
                        chunk_size = 20
                        for i in range(0, len(content), chunk_size):
                            chunk = content[i:i+chunk_size]
                            yield chunk
                            # Small delay for streaming effect
                            import asyncio
                            await asyncio.sleep(0.02)
                        
                        logger.info(f"🚀 [VISION-STREAM] Vision processing completed, streamed {len(content)} characters")
                        return
                        
                    except Exception as vision_error:
                        logger.error(f"Vision processing failed: {vision_error}")
                        yield f"Error: Vision processing failed: {str(vision_error)}"
                        return
                else:
                    # For text-only models or no images, use standard text generation
                    # Construct the prompt with explicit termination guidance
                    prompt = ""
                    if rag_context:
                        prompt += f"Context (relevant code and documentation):\n{rag_context}\n\nPlease use the above context to answer the following question.\n\n"
                    
                    # Add history to prompt
                    for msg in parsed_history:
                        prompt += f"{msg['role']}: {msg['content']}\n"
                    
                    prompt += f"user: {message}\nassistant: I'll provide a clear, concise response to your question without unnecessary repetition.\n\n"

                    # Generate the response
                    from mlx_rag.mlx_integration import GenerationConfig
                    config = GenerationConfig(
                        max_tokens=2048,  # Limit max tokens to prevent infinite generation
                        temperature=0.7
                    )
                    
                    logger.info(f"🚀 [STREAM] Starting generation for model: {model_name}")
                    logger.info(f"🚀 [STREAM] Prompt length: {len(prompt)} characters")
                    logger.info(f"🚀 [STREAM] Config: max_tokens={config.max_tokens}, temp={config.temperature}")
                    
                    async for chunk in language_model.mlx_wrapper.generate_stream(prompt, config):
                        chunk_count += 1
                        
                        # Skip empty chunks
                        if not chunk or not chunk.strip():
                            logger.debug(f"🚀 [STREAM] Skipping empty chunk {chunk_count}")
                            continue
                            
                        # Enhanced repetition detection
                        
                        # Keep track of recent chunks for pattern detection
                        if len(recent_chunks) > 50:  # Keep last 50 chunks
                            recent_chunks.pop(0)
                        recent_chunks.append(chunk)
                        accumulated_content += chunk
                        
                        # 1. Exact duplicate detection (more aggressive)
                        if chunk == last_chunk:
                            duplicate_count += 1
                            if duplicate_count > 5:  # Reduced from 10 to 5
                                logger.error(f"🚀 [STREAM] Detected exact repetition ({duplicate_count} times) at chunk {chunk_count}, terminating")
                                break
                        else:
                            duplicate_count = 0
                            last_chunk = chunk
                        
                        # 2. Pattern repetition detection
                        if len(recent_chunks) >= 10:  # Check for patterns in recent chunks
                            # Look for repeated 3-chunk patterns
                            recent_str = ''.join(recent_chunks[-9:])  # Last 9 chunks
                            pattern_3 = ''.join(recent_chunks[-3:])  # Last 3 chunks as pattern
                            if len(pattern_3.strip()) > 5 and recent_str.count(pattern_3) >= 3:
                                logger.error(f"🚀 [STREAM] Detected 3-chunk pattern repetition at chunk {chunk_count}: '{pattern_3[:50]}...', terminating")
                                break
                            
                            # Look for repeated 2-chunk patterns
                            pattern_2 = ''.join(recent_chunks[-2:])  # Last 2 chunks as pattern
                            if len(pattern_2.strip()) > 3 and recent_str.count(pattern_2) >= 4:
                                logger.error(f"🚀 [STREAM] Detected 2-chunk pattern repetition at chunk {chunk_count}: '{pattern_2[:30]}...', terminating")
                                break
                        
                        # 3. Content length stagnation detection
                        if chunk_count % 20 == 0 and chunk_count > 100:  # Check every 20 chunks after 100
                            content_growth_rate = len(accumulated_content) / chunk_count
                            if content_growth_rate < 2.0:  # Less than 2 characters per chunk on average
                                logger.error(f"🚀 [STREAM] Detected content stagnation (growth rate: {content_growth_rate:.2f} chars/chunk) at chunk {chunk_count}, terminating")
                                break
                        
                        # 4. Fuzzy similarity detection for near-identical chunks
                        if len(recent_chunks) >= 3:
                            last_3_chunks = recent_chunks[-3:]
                            if len(set(last_3_chunks)) <= 1:  # All 3 chunks are identical
                                logger.error(f"🚀 [STREAM] Detected identical chunk sequence at chunk {chunk_count}, terminating")
                                break
                            
                            # Check if chunks are very similar (differ by only 1-2 characters)
                            similar_chunks = 0
                            for i in range(len(last_3_chunks)):
                                for j in range(i+1, len(last_3_chunks)):
                                    chunk_a, chunk_b = last_3_chunks[i], last_3_chunks[j]
                                    if len(chunk_a) > 0 and len(chunk_b) > 0:
                                        # Simple similarity: count character differences
                                        min_len = min(len(chunk_a), len(chunk_b))
                                        if min_len > 0:
                                            diff_count = sum(1 for k in range(min_len) if chunk_a[k] != chunk_b[k])
                                            diff_ratio = diff_count / min_len
                                            if diff_ratio < 0.3:  # Less than 30% different
                                                similar_chunks += 1
                            
                            if similar_chunks >= 2:  # At least 2 pairs are very similar
                                logger.error(f"🚀 [STREAM] Detected similar chunk pattern at chunk {chunk_count}, terminating")
                                break
                        
                        # Log every 100th chunk for debugging
                        if chunk_count % 100 == 0 or chunk_count <= 10:
                            logger.info(f"🚀 [STREAM] Chunk {chunk_count}: {repr(chunk[:50])}")
                        
                        # Safety limit to prevent runaway generation
                        if chunk_count > 5000:
                            logger.error(f"🚀 [STREAM] Hit safety limit at {chunk_count} chunks, terminating")
                            break
                        
                        yield chunk
                    
                    logger.info(f"🚀 [STREAM] Generation completed after {chunk_count} chunks")
                    
            except Exception as e:
                logger.error(f"🚀 [STREAM] Error in chat endpoint after {chunk_count} chunks: {e}", exc_info=True)
                yield f"Error: {str(e)}"

        return StreamingResponse(stream_response(), media_type="text/plain")

    @app.get("/v1/models/{model_name}")
    async def get_model(model_name: str, db: Session = Depends(get_db_session)):
        """Get specific model details."""
        model = db.query(Model).filter(Model.name == model_name).first()
        if not model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_name}' not found"
            )

        return {
            "id": model.id,
            "name": model.name,
            "path": model.path,
            "version": model.version,
            "type": model.model_type,
            "status": model.status,
            "memory_required_gb": model.memory_required_gb,
            "use_count": model.use_count,
            "last_used_at": model.last_used_at.isoformat() if model.last_used_at else None,
            "created_at": model.created_at.isoformat() if model.created_at else None,
            "updated_at": model.updated_at.isoformat() if model.updated_at else None,
            "error_message": model.error_message,
            "metadata": model.get_metadata(),
        }

    @app.post("/v1/models/{model_name}/load")
    async def load_model(
        model_name: str,
        priority: int = 0,
        db: Session = Depends(get_db_session)
    ):
        """Load a model."""
        model_record = db.query(Model).filter(Model.name == model_name).first()
        if not model_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_name}' not found"
            )

        try:
            model_manager = get_model_manager()

            # Check if already loaded
            if model_name in model_manager._loaded_models:
                return {
                    "message": f"Model '{model_name}' is already loaded",
                    "status": "loaded"
                }

            # Check system compatibility
            system_monitor = get_system_monitor()
            can_load, compatibility_message = system_monitor.check_model_compatibility(
                model_record.memory_required_gb
            )

            # Only block for hardware compatibility, not memory warnings
            if not can_load and "MLX requires Apple Silicon" in compatibility_message:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=compatibility_message
                )

            # Store any warnings to include in response
            memory_warning = None
            if "warning" in compatibility_message.lower():
                memory_warning = compatibility_message

            # Initiate loading
            success = await model_manager.load_model_async(
                model_name=model_name,
                model_path=model_record.path,
                priority=priority
            )

            if success:
                response = {
                    "message": f"Model '{model_name}' loaded successfully",
                    "status": "loaded"
                }
                # Include memory warning if present
                if memory_warning:
                    response["memory_warning"] = memory_warning
                return response
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to load model '{model_name}'"
                )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error loading model: {str(e)}"
            )

    @app.post("/v1/models/{model_name}/unload")
    async def unload_model(
        model_name: str,
        db: Session = Depends(get_db_session)
    ):
        """Unload a model."""
        model_record = db.query(Model).filter(Model.name == model_name).first()
        if not model_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_name}' not found"
            )

        try:
            model_manager = get_model_manager()
            success = model_manager.unload_model(model_name)

            if success:
                return {
                    "message": f"Model '{model_name}' unloaded successfully",
                    "status": "unloaded"
                }
            else:
                return {
                    "message": f"Model '{model_name}' was not loaded",
                    "status": "not_loaded"
                }

        except Exception as e:
            logger.error(f"Error unloading model {model_name}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error unloading model: {str(e)}"
            )

    @app.delete("/v1/models/{model_name}")
    async def delete_model(
        model_name: str,
        remove_files: bool = True,
        db: Session = Depends(get_db_session)
    ):
        """Delete a model from the database and optionally remove files."""
        model_record = db.query(Model).filter(Model.name == model_name).first()
        if not model_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_name}' not found"
            )

        try:
            # First unload if loaded
            model_manager = get_model_manager()
            model_manager.unload_model(model_name)

            # Remove from database
            db.delete(model_record)
            db.commit()

            # Optionally remove downloaded files
            if remove_files and model_record.path:
                try:
                    import shutil
                    import os
                    from pathlib import Path

                    # If it's a HuggingFace cache path, remove the entire model directory
                    if ".cache" in model_record.path and "models--" in model_record.path:
                        # Extract the model directory from the path
                        cache_path = Path(model_record.path)
                        if cache_path.exists():
                            # Find the models--* directory
                            for parent in cache_path.parents:
                                if parent.name.startswith("models--"):
                                    if parent.exists():
                                        shutil.rmtree(parent)
                                        logger.info(f"Removed model files at {parent}")
                                    break
                    elif os.path.exists(model_record.path):
                        # Remove local model directory
                        if os.path.isdir(model_record.path):
                            shutil.rmtree(model_record.path)
                        else:
                            os.remove(model_record.path)
                        logger.info(f"Removed model files at {model_record.path}")

                except Exception as file_error:
                    logger.warning(f"Could not remove model files: {file_error}")
                    # Don't fail the deletion if file removal fails

            return {
                "message": f"Model '{model_name}' deleted successfully",
                "removed_files": remove_files,
                "status": "deleted"
            }

        except Exception as e:
            logger.error(f"Error deleting model {model_name}: {e}")
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error deleting model: {str(e)}"
            )

    @app.get("/v1/models/{model_name}/health")
    async def model_health(
        model_name: str,
        db: Session = Depends(get_db_session)
    ):
        """Check model health status."""
        model = db.query(Model).filter(Model.name == model_name).first()
        if not model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_name}' not found"
            )

        return {
            "model": model_name,
            "status": model.status,
            "healthy": model.status == "loaded",
            "last_used": model.last_used_at.isoformat() if model.last_used_at else None,
        }

    @app.post("/v1/models/{model_name}/generate")
    async def generate_text(
        model_name: str,
        request_data: dict,
        db: Session = Depends(get_db_session)
    ):
        """Generate text using a model."""
        model_record = db.query(Model).filter(Model.name == model_name).first()
        if not model_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_name}' not found"
            )

        try:
            model_manager = get_model_manager()

            # Check if model is loaded
            loaded_model = model_manager.get_model_for_inference(model_name)
            if not loaded_model:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Model '{model_name}' is not loaded. Load it first with POST /v1/models/{model_name}/load"
                )

            # Extract generation parameters
            prompt = request_data.get("prompt", "")
            if not prompt:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Prompt is required"
                )

            # Create generation config
            config = GenerationConfig(
                max_tokens=request_data.get("max_tokens", 100),
                temperature=request_data.get("temperature", 0.0),
                top_p=request_data.get("top_p", 1.0),
                top_k=request_data.get("top_k", 0),
                repetition_penalty=request_data.get("repetition_penalty", 1.0),
                repetition_context_size=request_data.get("repetition_context_size", 20),
                seed=request_data.get("seed")
            )

            # Generate text with transparent queuing
            result = await queued_generate_text(model_name, prompt, config)

            return {
                "model": model_name,
                "prompt": result.prompt,
                "text": result.text,
                "usage": {
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "total_tokens": result.total_tokens
                },
                "timing": {
                    "generation_time_seconds": result.generation_time_seconds,
                    "tokens_per_second": result.tokens_per_second
                }
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error generating text with model {model_name}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error generating text: {str(e)}"
            )

    @app.get("/v1/system/status")
    async def system_status(db: Session = Depends(get_db_session)):
        """Get system status including memory usage."""
        system_monitor = get_system_monitor()
        system_summary = system_monitor.get_system_summary()

        model_manager = get_model_manager()
        manager_status = model_manager.get_system_status()

        return {
            "status": "running",
            "system": system_summary,
            "model_manager": manager_status,
            "mlx_compatible": system_summary["mlx_compatible"]
        }

    @app.get("/v1/system/version")
    async def get_version():
        """Get application version information."""
        from mlx_rag import __version__, __author__, __description__
        return {
            "version": __version__,
            "author": __author__,
            "description": __description__,
            "name": "MLX-RAG"
        }

    @app.get("/v1/settings")
    async def get_settings(db: Session = Depends(get_db_session)):
        """Get application settings."""
        settings = db.query(AppSettings).all()
        return {
            setting.key: setting.get_typed_value()
            for setting in settings
        }

    @app.put("/v1/settings/{key}")
    async def update_setting(
        key: str,
        value: dict,
        db: Session = Depends(get_db_session)
    ):
        """Update a setting value."""
        setting = db.query(AppSettings).filter(AppSettings.key == key).first()
        if not setting:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Setting '{key}' not found"
            )

        # Extract value from request body
        new_value = value.get("value")
        if new_value is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Value is required"
            )

        setting.set_typed_value(new_value)
        db.commit()

        return {
            "key": key,
            "value": setting.get_typed_value(),
            "updated": True
        }

    # HuggingFace model discovery endpoints
    @app.get("/v1/discover/models")
    async def discover_models(
        query: str = "",
        limit: int = 20,
        sort: str = "downloads"
    ):
        """Discover MLX-compatible models from HuggingFace."""
        try:
            hf_client = get_huggingface_client()
            models = hf_client.search_mlx_models(query=query, limit=limit, sort=sort)

            return {
                "models": [
                    {
                        "id": model.id,
                        "name": model.name,
                        "author": model.author,
                        "downloads": model.downloads,
                        "likes": model.likes,
                        "model_type": model.model_type,
                        "size_gb": model.size_gb,
                        "estimated_memory_gb": model.estimated_memory_gb,
                        "mlx_compatible": model.mlx_compatible,
                        "has_mlx_version": model.has_mlx_version,
                        "mlx_repo_id": model.mlx_repo_id,
                        "tags": model.tags,
                        "description": model.description,
                        "updated_at": model.updated_at
                    }
                    for model in models
                ],
                "total": len(models)
            }
        except Exception as e:
            logger.error(f"Error discovering models: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error discovering models from HuggingFace"
            )

    @app.get("/v1/discover/popular")
    async def discover_popular_models(limit: int = 20):
        """Get popular MLX models."""
        try:
            hf_client = get_huggingface_client()
            models = hf_client.get_popular_mlx_models(limit=limit)

            return {
                "models": [
                    {
                        "id": model.id,
                        "name": model.name,
                        "author": model.author,
                        "downloads": model.downloads,
                        "likes": model.likes,
                        "model_type": model.model_type,
                        "size_gb": model.size_gb,
                        "estimated_memory_gb": model.estimated_memory_gb,
                        "mlx_compatible": model.mlx_compatible,
                        "description": model.description
                    }
                    for model in models
                ],
                "total": len(models)
            }
        except Exception as e:
            logger.error(f"Error getting popular models: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error getting popular models"
            )

    @app.get("/v1/discover/trending")
    async def discover_trending_models(limit: int = 20):
        """Get trending MLX models from HuggingFace."""
        try:
            hf_client = get_huggingface_client()
            models = hf_client.search_trending_mlx_models(limit=limit)

            return {
                "models": [
                    {
                        "id": model.id,
                        "name": model.name,
                        "author": model.author,
                        "downloads": model.downloads,
                        "likes": model.likes,
                        "model_type": model.model_type,
                        "size_gb": model.size_gb,
                        "estimated_memory_gb": model.estimated_memory_gb,
                        "mlx_compatible": model.mlx_compatible,
                        "has_mlx_version": model.has_mlx_version,
                        "mlx_repo_id": model.mlx_repo_id,
                        "tags": model.tags,
                        "description": model.description,
                        "updated_at": model.updated_at
                    }
                    for model in models
                ],
                "total": len(models)
            }
        except Exception as e:
            logger.error(f"Error getting trending models: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error getting trending models"
            )

    @app.get("/v1/discover/categories")
    async def get_model_categories():
        """Get categorized model lists."""
        try:
            hf_client = get_huggingface_client()
            categories = hf_client.get_model_categories()
            return {"categories": categories}
        except Exception as e:
            logger.error(f"Error getting model categories: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error getting model categories"
            )

    @app.get("/v1/discover/vision")
    async def discover_vision_models(query: str = "", limit: int = 10):
        """Discover vision/multimodal models using HuggingFace pipeline filters."""
        try:
            hf_client = get_huggingface_client()
            models = hf_client.search_vision_models(query=query, limit=limit)

            return {
                "models": [
                    {
                        "id": model.id,
                        "name": model.name,
                        "author": model.author,
                        "downloads": model.downloads,
                        "likes": model.likes,
                        "model_type": model.model_type,
                        "size_gb": model.size_gb,
                        "estimated_memory_gb": model.estimated_memory_gb,
                        "mlx_compatible": model.mlx_compatible,
                        "has_mlx_version": model.has_mlx_version,
                        "mlx_repo_id": model.mlx_repo_id,
                        "tags": model.tags,
                        "description": model.description,
                        "updated_at": model.updated_at
                    }
                    for model in models
                ],
                "total": len(models)
            }
        except Exception as e:
            logger.error(f"Error discovering vision models: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error discovering vision models"
            )

    @app.get("/v1/discover/stt")
    async def discover_stt_models(query: str = "", limit: int = 10):
        """Discover STT/speech-to-text models using HuggingFace pipeline filters."""
        try:
            hf_client = get_huggingface_client()
            models = hf_client.search_stt_models(query=query, limit=limit)

            return {
                "models": [
                    {
                        "id": model.id,
                        "name": model.name,
                        "author": model.author,
                        "downloads": model.downloads,
                        "likes": model.likes,
                        "model_type": model.model_type,
                        "size_gb": model.size_gb,
                        "estimated_memory_gb": model.estimated_memory_gb,
                        "mlx_compatible": model.mlx_compatible,
                        "has_mlx_version": model.has_mlx_version,
                        "mlx_repo_id": model.mlx_repo_id,
                        "tags": model.tags,
                        "description": model.description,
                        "updated_at": model.updated_at
                    }
                    for model in models
                ],
                "total": len(models)
            }
        except Exception as e:
            logger.error(f"Error discovering STT models: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error discovering STT models"
            )

    @app.get("/v1/discover/embeddings")
    async def discover_embedding_models(query: str = "", limit: int = 20):
        """Discover embedding models using HuggingFace pipeline filters."""
        try:
            hf_client = get_huggingface_client()
            models = hf_client.search_embedding_models(query=query, limit=limit)

            return {
                "models": [
                    {
                        "id": model.id,
                        "name": model.name,
                        "author": model.author,
                        "downloads": model.downloads,
                        "likes": model.likes,
                        "model_type": model.model_type,
                        "size_gb": model.size_gb,
                        "estimated_memory_gb": model.estimated_memory_gb,
                        "mlx_compatible": model.mlx_compatible,
                        "has_mlx_version": model.has_mlx_version,
                        "mlx_repo_id": model.mlx_repo_id,
                        "tags": model.tags,
                        "description": model.description,
                        "updated_at": model.updated_at
                    }
                    for model in models
                ],
                "total": len(models)
            }
        except Exception as e:
            logger.error(f"Error discovering embedding models: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error discovering embedding models"
            )

    @app.get("/v1/discover/compatible")
    async def discover_compatible_models(
        query: str = "",
        max_memory_gb: Optional[float] = None
    ):
        """Discover models compatible with current system."""
        try:
            # Get system memory if not specified
            if max_memory_gb is None:
                system_monitor = get_system_monitor()
                memory_info = system_monitor.get_memory_info()
                max_memory_gb = memory_info.total_gb * 0.8  # Use 80% of total RAM

            hf_client = get_huggingface_client()
            models = hf_client.search_compatible_models(query, max_memory_gb)

            return {
                "models": [
                    {
                        "id": model.id,
                        "name": model.name,
                        "author": model.author,
                        "downloads": model.downloads,
                        "likes": model.likes,
                        "model_type": model.model_type,
                        "size_gb": model.size_gb,
                        "estimated_memory_gb": model.estimated_memory_gb,
                        "mlx_compatible": model.mlx_compatible,
                        "description": model.description,
                        "memory_fit": f"{model.estimated_memory_gb:.1f}GB required, {max_memory_gb:.1f}GB available"
                    }
                    for model in models
                ],
                "max_memory_gb": max_memory_gb,
                "total": len(models)
            }
        except Exception as e:
            logger.error(f"Error discovering compatible models: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error discovering compatible models"
            )

    @app.get("/v1/discover/models/{model_id:path}")
    async def get_model_details(model_id: str):
        """Get detailed information about a specific HuggingFace model."""
        try:
            hf_client = get_huggingface_client()
            model = hf_client.get_model_details(model_id)

            if not model:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Model '{model_id}' not found on HuggingFace"
                )

            # Check system compatibility
            system_monitor = get_system_monitor()
            can_load, compatibility_message = system_monitor.check_model_compatibility(
                model.estimated_memory_gb or 0
            )

            return {
                "id": model.id,
                "name": model.name,
                "author": model.author,
                "downloads": model.downloads,
                "likes": model.likes,
                "created_at": model.created_at,
                "updated_at": model.updated_at,
                "model_type": model.model_type,
                "library_name": model.library_name,
                "pipeline_tag": model.pipeline_tag,
                "tags": model.tags,
                "size_gb": model.size_gb,
                "estimated_memory_gb": model.estimated_memory_gb,
                "mlx_compatible": model.mlx_compatible,
                "has_mlx_version": model.has_mlx_version,
                "mlx_repo_id": model.mlx_repo_id,
                "description": model.description,
                "system_compatible": can_load,
                "compatibility_message": compatibility_message
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting model details for {model_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error getting model details"
            )

    # OpenAI-compatible endpoints
    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: ChatCompletionRequest,
        db: Session = Depends(get_db_session),
        api_key: Optional[str] = Depends(validate_api_key)
    ):
        # Print request for debug purposes
        print(f"[DEBUG] Chat request: {request}")
        
        """OpenAI-compatible chat completions endpoint."""
        try:
            # Check if model exists in database
            model_record = db.query(Model).filter(Model.name == request.model).first()
            if not model_record:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Model '{request.model}' not found. Install it first with POST /v1/models/install"
                )

            model_manager = get_model_manager()

            # Check if model is loaded, auto-load if not
            loaded_model = model_manager.get_model_for_inference(request.model)
            if not loaded_model:
                logger.info(f"Model {request.model} not loaded, attempting to load...")
                success = await model_manager.load_model_async(
                    model_name=request.model,
                    model_path=model_record.path,
                    priority=10  # High priority for chat requests
                )
                if not success:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail=f"Failed to load model '{request.model}'"
                    )

                # Re-fetch the loaded model after loading
                loaded_model = model_manager.get_model_for_inference(request.model)
                if not loaded_model:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail=f"Model '{request.model}' failed to load properly"
                    )

            # Get messages
            messages = request.messages

            # Add default system prompt if none provided
            has_system_message = any(msg.role == "system" for msg in messages)

            # Extract structured messages and image URLs first to check if we have images
            chat_messages, images = _format_chat_prompt(messages)

            # Only add system message if no system message exists AND no images (vision models don't handle system messages well)
            if not has_system_message and not images:
                # Add a helpful default system message for non-vision models
                default_system = ChatCompletionMessage(
                    role="system",
                    content="You are a helpful AI assistant. Provide clear, accurate, and concise responses."
                )
                messages = [default_system] + list(messages)
                # Re-extract after adding system message
                chat_messages, images = _format_chat_prompt(messages)
            logger.debug(f"🔧 After _format_chat_prompt: {len(images)} images extracted")

            # Enforce server-side maximum token limit
            MAX_TOKENS_LIMIT = 16384  # 16k max
            if request.max_tokens > MAX_TOKENS_LIMIT:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"max_tokens cannot exceed {MAX_TOKENS_LIMIT}, requested {request.max_tokens}"
                )

            # Create generation config with aggressive repetition prevention
            logger.debug(f"Creating config - Images: {len(images)}")
            
            max_tokens = request.max_tokens
            temperature = request.temperature
            repetition_penalty = request.repetition_penalty
            
            config = GenerationConfig(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repetition_penalty=repetition_penalty,
                repetition_context_size=100,  # Large context window for repetition detection
                seed=request.seed
            )
            logger.debug(f"Config created - Images: {len(images)}")

            import time
            import uuid

            completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
            created_time = int(time.time())

            # Check if this is a vision model
            is_vision_model = False
            if loaded_model and hasattr(loaded_model.mlx_wrapper, 'model_type') and loaded_model.mlx_wrapper.model_type == "vision":
                is_vision_model = True
            logger.debug(f"🔧 Vision model check: is_vision_model={is_vision_model}, model_type={getattr(loaded_model.mlx_wrapper, 'model_type', 'unknown') if loaded_model else 'no_model'}")

            # Always use non-streaming mode
            # logger.debug(f"🔧 Request stream setting: {request.stream} (forcing non-streaming)")

            # Always use non-streaming response (streaming disabled)
            
            if is_vision_model:
                # For vision models, process images and pass structured messages
                processed_image_paths = await _process_image_urls(images)
                result = await queued_generate_vision(request.model, chat_messages, processed_image_paths, config)

                # Clean up temporary image files
                for img_path in processed_image_paths:
                    try:
                        import os
                        os.unlink(img_path)
                    except Exception as e:
                        logger.warning(f"Failed to cleanup temporary image file {img_path}: {e}")
            else:
                # For text models, format to a string and generate
                prompt_string = await _apply_chat_template(loaded_model.mlx_wrapper.tokenizer, chat_messages, request.model)
                
                # Text generation
                generation_timeout = 30
                try:
                    import asyncio
                    result = await asyncio.wait_for(
                        queued_generate_text(request.model, prompt_string, config),
                        timeout=generation_timeout
                    )
                except asyncio.TimeoutError:
                    logger.error(f"Generation timeout ({generation_timeout}s) exceeded for {request.model}")
                    raise HTTPException(
                        status_code=status.HTTP_408_REQUEST_TIMEOUT,
                        detail=f"Model generation took too long (>{generation_timeout}s). This may indicate repetitive output."
                    )
                

                if not result:
                     raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Generation failed and returned no result."
                    )
                
                # Build a simple response message (no server-side tool handling)
                response_message = ChatCompletionMessage(
                    role="assistant",
                    content=result.text
                )

                response = ChatCompletionResponse(
                    id=completion_id,
                    created=created_time,
                    model=request.model,
                    choices=[
                        ChatCompletionChoice(
                            index=0,
                            message=response_message,
                            finish_reason="stop"
                        )
                    ],
                    usage=ChatCompletionUsage(
                        prompt_tokens=result.prompt_tokens,
                        completion_tokens=result.completion_tokens,
                        total_tokens=result.total_tokens
                    )
                )
                print(f"[DEBUG] Returning response: {response}")
                return response

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in chat completions: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Chat completion failed: {str(e)}"
            )

    # Audio endpoints
    @app.post("/v1/audio/transcriptions")
    async def create_transcription(
        file: UploadFile = File(...),
        model: str = Form("whisper-1"),
        language: Optional[str] = Form(None),
        prompt: Optional[str] = Form(None),
        response_format: str = Form("json"),
        temperature: float = Form(0.0),
        api_key: Optional[str] = Depends(validate_api_key),
        db: Session = Depends(get_db_session)
    ):
        """OpenAI-compatible audio transcription endpoint."""
        try:
            # Validate audio file
            if not file.content_type or not file.content_type.startswith('audio/'):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="File must be an audio file"
                )

            # Map OpenAI model names to our audio models
            model_mapping = {
                "whisper-1": "mlx-community/whisper-tiny",
                "whisper-large": "mlx-community/whisper-large-v3",
                "whisper-small": "mlx-community/whisper-small",
                "whisper-medium": "mlx-community/whisper-medium",
                "whisper-tiny": "mlx-community/whisper-tiny",
                "whisper-base": "mlx-community/whisper-base",
                "whisper-large-v2": "mlx-community/whisper-large-v2",
                "whisper-large-v3": "mlx-community/whisper-large-v3",
                "parakeet": "parakeet-tdt-0.6b-v2",
                # Legacy mappings for backwards compatibility
                "whisper-small-mlx": "mlx-community/whisper-small",
                "whisper-large-mlx": "mlx-community/whisper-large-v3",
                "whisper-medium-mlx": "mlx-community/whisper-medium",
                "whisper-tiny-mlx": "mlx-community/whisper-tiny"
            }

            # Get actual model name
            actual_model_name = model_mapping.get(model, model)

            # Check if audio model exists and is loaded
            model_record = db.query(Model).filter(Model.name == actual_model_name).first()
            if not model_record:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Audio model '{actual_model_name}' not found. Install it first with POST /v1/models/install"
                )

            model_manager = get_model_manager()

            # Check if model is loaded, auto-load if not
            loaded_model = model_manager.get_model_for_inference(actual_model_name)
            use_direct_whisper = False
            
            if not loaded_model:
                logger.info(f"Audio model {actual_model_name} not loaded, attempting to load...")
                success = await model_manager.load_model_async(
                    model_name=actual_model_name,
                    model_path=model_record.path,
                    priority=10  # High priority for audio requests
                )
                if success:
                    # Get the loaded model
                    loaded_model = model_manager.get_model_for_inference(actual_model_name)
                
                # If loading failed or model still not available, try direct MLX-Whisper for Whisper models
                if not loaded_model and model_record.model_type == 'whisper':
                    logger.info(f"Using direct MLX-Whisper API for {actual_model_name}")
                    use_direct_whisper = True
                elif not loaded_model:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail=f"Failed to load audio model '{actual_model_name}'"
                    )

            # Check if this is an audio model (skip check for direct whisper usage)
            if not use_direct_whisper and not hasattr(loaded_model.mlx_wrapper, 'transcribe_audio'):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Model '{actual_model_name}' is not an audio transcription model"
                )

            # Save uploaded file temporarily
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_file_path = temp_file.name

            try:
                if use_direct_whisper:
                    # Use MLX-Whisper directly for maximum compatibility
                    import mlx_whisper
                    logger.info(f"Direct MLX-Whisper transcription for {actual_model_name}")
                    
                    result = mlx_whisper.transcribe(
                        audio=temp_file_path,
                        path_or_hf_repo=model_record.path,
                        language=language,
                        initial_prompt=prompt,
                        temperature=temperature
                    )
                    
                    # Update usage count manually for direct usage
                    try:
                        model_record.increment_use_count()
                        db.commit()
                    except Exception as e:
                        logger.warning(f"Failed to update usage count for {actual_model_name}: {e}")
                    
                    text_content = result.get("text", "")
                else:
                    # Transcribe using queued audio processing
                    result = await queued_transcribe_audio(
                        model_name=actual_model_name,
                        file_path=temp_file_path,
                        language=language,
                        initial_prompt=prompt,
                        temperature=temperature
                    )
                    text_content = result.get("text", "")

                # Return response based on format
                if response_format == "json":
                    return {"text": text_content}
                elif response_format == "text":
                    return Response(content=text_content, media_type="text/plain")
                elif response_format == "verbose_json":
                    if use_direct_whisper:
                        # Direct whisper returns full result dict
                        return result
                    elif isinstance(result, dict):
                        return result
                    else:
                        return {"text": text_content}
                else:
                    # For srt, vtt formats - basic implementation
                    return {"text": text_content}

            finally:
                # Clean up temporary file
                os.unlink(temp_file_path)

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in audio transcription: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Transcription failed: {str(e)}"
            )

    @app.post("/v1/audio/speech")
    async def create_speech(
        request: AudioSpeechRequest,
        api_key: Optional[str] = Depends(validate_api_key)
    ):
        """OpenAI-compatible text-to-speech endpoint."""
        try:
            # Use MLX Audio for TTS
            try:
                import mlx_audio
            except ImportError:
                raise HTTPException(
                    status_code=status.HTTP_501_NOT_IMPLEMENTED,
                    detail="MLX Audio not installed. Install with: pip install mlx-audio"
                )

            # Generate speech
            # Map OpenAI voices to available models
            voice_mapping = {
                "alloy": "kokoro",
                "echo": "kokoro",
                "fable": "kokoro",
                "onyx": "kokoro",
                "nova": "kokoro",
                "shimmer": "kokoro"
            }

            model_name = voice_mapping.get(request.voice, "kokoro")

            # Generate audio using queued processing
            audio_content = await queued_generate_speech(
                text=request.input,
                voice=model_name,
                speed=request.speed
            )

            # Return audio response
            media_type_mapping = {
                "mp3": "audio/mpeg",
                "opus": "audio/opus",
                "aac": "audio/aac",
                "flac": "audio/flac",
                "wav": "audio/wav"
            }

            media_type = media_type_mapping.get(request.response_format, "audio/wav")

            return Response(
                content=audio_content,
                media_type=media_type,
                headers={
                    "Content-Disposition": f"attachment; filename=speech.{request.response_format}"
                }
            )

        except ImportError as e:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail=f"Text-to-speech not available: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Error in speech generation: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Speech generation failed: {str(e)}"
            )

    @app.post("/v1/embeddings")
    async def create_embeddings(
        request: EmbeddingRequest,
        db: Session = Depends(get_db_session),
        api_key: Optional[str] = Depends(validate_api_key)
    ):
        """OpenAI-compatible embeddings endpoint."""
        try:
            # Log API key usage (for debugging)
            if api_key:
                logger.debug(f"API key provided for embeddings: {api_key[:8]}...")
            else:
                logger.debug("No API key provided for embeddings")

            # Check if model exists in database
            model_record = db.query(Model).filter(Model.name == request.model).first()
            if not model_record:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Embedding model '{request.model}' not found. Install it first with POST /v1/models/install"
                )

            model_manager = get_model_manager()

            # Check if model is loaded, auto-load if not
            loaded_model = model_manager.get_model_for_inference(request.model)
            if not loaded_model:
                logger.info(f"Embedding model {request.model} not loaded, attempting to load...")
                success = await model_manager.load_model_async(
                    model_name=request.model,
                    model_path=model_record.path,
                    priority=10  # High priority for embedding requests
                )
                if not success:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail=f"Failed to load embedding model '{request.model}'"
                    )

            # Convert input to list of strings
            if isinstance(request.input, str):
                texts = [request.input]
            else:
                texts = request.input

            # Validate inputs
            if not texts:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Input texts cannot be empty"
                )

            # Check for text length limits (8192 tokens max per text)
            MAX_TEXT_LENGTH = 8192
            for i, text in enumerate(texts):
                if len(text.split()) > MAX_TEXT_LENGTH:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Text {i} exceeds maximum length of {MAX_TEXT_LENGTH} tokens"
                    )

            # Generate embeddings with transparent queuing
            result = await queued_generate_embeddings(request.model, texts)

            # Extract embeddings and usage info from result
            embeddings = result.get("embeddings", [])
            prompt_tokens = result.get("prompt_tokens", sum(len(text.split()) for text in texts))
            total_tokens = result.get("total_tokens", prompt_tokens)

            # Validate embeddings
            if not embeddings:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="No embeddings generated"
                )

            # Apply dimensions reduction if requested
            if request.dimensions and request.dimensions > 0:
                for i, embedding in enumerate(embeddings):
                    if len(embedding) > request.dimensions:
                        embeddings[i] = embedding[:request.dimensions]

            # Encode embeddings based on format
            if request.encoding_format == "base64":
                import base64
                import struct
                encoded_embeddings = []
                for embedding in embeddings:
                    # Convert float list to bytes then base64
                    bytes_data = b''.join(struct.pack('f', x) for x in embedding)
                    b64_data = base64.b64encode(bytes_data).decode('utf-8')
                    encoded_embeddings.append(b64_data)
                embeddings = encoded_embeddings

            # Create response data
            embedding_data = [
                EmbeddingData(
                    embedding=embeddings[i],
                    index=i
                )
                for i in range(len(embeddings))
            ]

            response = EmbeddingResponse(
                data=embedding_data,
                model=request.model,
                usage=EmbeddingUsage(
                    prompt_tokens=prompt_tokens,
                    total_tokens=total_tokens
                )
            )

            return response

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in embeddings endpoint: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Embeddings generation failed: {str(e)}"
            )

    @app.get("/v1/models")
    async def list_models_openai_format(
        db: Session = Depends(get_db_session),
        api_key: Optional[str] = Depends(validate_api_key)
    ):
        """OpenAI-compatible models list endpoint."""
        try:
            models = db.query(Model).all()

            import time

            return {
                "object": "list",
                "data": [
                    {
                        "id": model.name,
                        "object": "model",
                        "created": int(model.created_at.timestamp()) if model.created_at else int(time.time()),
                        "owned_by": "mlx-rag",
                        "permission": [],
                        "root": model.name,
                        "parent": None,
                        "supports_tools": _check_model_supports_tools(model.path, model.name)
                    }
                    for model in models
                ]
            }
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error listing models"
            )

    @app.post("/v1/models/install")
    async def install_model(
        request: ModelInstallRequest,
        db: Session = Depends(get_db_session)
    ):
        """Install a model from HuggingFace Hub."""
        try:
            # Get model details from HuggingFace
            hf_client = get_huggingface_client()
            model_info = hf_client.get_model_details(request.model_id)

            if not model_info:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Model '{request.model_id}' not found on HuggingFace"
                )

            if not model_info.mlx_compatible:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Model '{request.model_id}' is not MLX compatible"
                )

            # Check system compatibility
            system_monitor = get_system_monitor()
            estimated_memory = model_info.estimated_memory_gb or 4.0  # Default estimate
            can_load, compatibility_message = system_monitor.check_model_compatibility(estimated_memory)

            # Only block for hardware compatibility, not memory warnings
            if not can_load and "MLX requires Apple Silicon" in compatibility_message:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=compatibility_message
                )

            # Store any warnings to include in response
            memory_warning = None
            if "warning" in compatibility_message.lower():
                memory_warning = compatibility_message

            # Use provided name or default to model name
            model_name = request.name or model_info.name

            # Check if model already exists
            existing_model = db.query(Model).filter(Model.name == model_name).first()
            if existing_model:
                return {
                    "message": f"Model '{model_name}' already installed",
                    "model_name": model_name,
                    "model_id": request.model_id,
                    "status": "already_installed"
                }

            # Create model record in database
            new_model = Model(
                name=model_name,
                path=request.model_id,  # Store HF model ID as path
                version=None,
                model_type=model_info.model_type,
                huggingface_id=request.model_id,
                memory_required_gb=int(estimated_memory),
                status="unloaded"
            )

            # Set metadata
            metadata = {
                "author": model_info.author,
                "downloads": model_info.downloads,
                "likes": model_info.likes,
                "tags": model_info.tags,
                "description": model_info.description,
                "size_gb": model_info.size_gb,
                "estimated_memory_gb": model_info.estimated_memory_gb,
                "mlx_repo_id": model_info.mlx_repo_id
            }
            new_model.set_metadata(metadata)

            db.add(new_model)
            db.commit()

            response = {
                "message": f"Model '{model_name}' installed successfully",
                "model_name": model_name,
                "model_id": request.model_id,
                "estimated_memory_gb": estimated_memory,
                "status": "installed"
            }
            # Include memory warning if present
            if memory_warning:
                response["memory_warning"] = memory_warning
            return response

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error installing model {request.model_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Model installation failed: {str(e)}"
            )

    # Model management endpoints
    @app.get("/v1/manager/status")
    async def get_manager_status():
        """Get detailed model manager status."""
        try:
            model_manager = get_model_manager()
            return {
                "loaded_models": model_manager.get_loaded_models(),
                "system_status": model_manager.get_system_status(),
                "queue_status": model_manager._loading_queue.get_queue_status()
            }
        except Exception as e:
            logger.error(f"Error getting manager status: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error getting manager status"
            )

    @app.get("/v1/manager/models/{model_name}/status")
    async def get_model_status(model_name: str):
        """Get detailed status of a specific model."""
        try:
            model_manager = get_model_manager()
            model_status = model_manager.get_model_status(model_name)
            return model_status
        except Exception as e:
            logger.error(f"Error getting model status for {model_name}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error getting model status"
            )

    @app.post("/v1/manager/models/{model_name}/priority")
    async def update_model_priority(model_name: str, priority_data: dict):
        """Update model loading priority in queue."""
        try:
            new_priority = priority_data.get("priority", 0)
            # TODO: Implement priority update in queue
            return {
                "model": model_name,
                "priority": new_priority,
                "message": "Priority update requested"
            }
        except Exception as e:
            logger.error(f"Error updating priority for {model_name}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error updating model priority"
            )

    # Admin interface routes
    @app.get("/admin")
    async def admin_interface():
        """Serve the admin interface."""
        from fastapi.responses import HTMLResponse
        from pathlib import Path
        import sys

        # Read the admin template - handle both development and bundled app
        # Use modular template for better maintainability
        template_name = "admin_modular.html"
        
        if hasattr(sys, 'frozen') and sys.frozen:
            # Running as bundled app - use PyInstaller's _MEIPASS
            if hasattr(sys, '_MEIPASS'):
                template_path = Path(sys._MEIPASS) / "mlx_rag" / "templates" / template_name
                logger.info(f"PyInstaller bundled app: Looking for template at {template_path}")
                logger.info(f"_MEIPASS directory: {sys._MEIPASS}")
                # List contents of _MEIPASS for debugging
                meipass_path = Path(sys._MEIPASS)
                if meipass_path.exists():
                    logger.info(f"_MEIPASS contents: {list(meipass_path.iterdir())}")
                    mlx_rag_path = meipass_path / "mlx_rag"
                    if mlx_rag_path.exists():
                        logger.info(f"mlx_rag directory contents: {list(mlx_rag_path.iterdir())}")
            else:
                # Fallback for frozen apps without _MEIPASS
                template_path = Path(sys.executable).parent / "mlx_rag" / "templates" / template_name
                logger.info(f"Frozen app fallback: Looking for template at {template_path}")
        else:
            # Running in development
            template_path = Path(__file__).parent / "templates" / template_name
            logger.info(f"Development mode: Looking for template at {template_path}")

        if not template_path.exists():
            # Try to find template in alternate locations
            logger.error(f"Template not found at {template_path}")
            if hasattr(sys, 'frozen') and sys.frozen:
                # Try some alternate paths in bundled app
                alternate_paths = [
                    Path(sys.executable).parent / "templates" / template_name,
                    Path(sys.executable).parent / template_name,
                    Path(sys.executable).parent / "Contents" / "Resources" / "templates" / template_name,
                ]
                for alt_path in alternate_paths:
                    logger.info(f"Trying alternate path: {alt_path}")
                    if alt_path.exists():
                        template_path = alt_path
                        logger.info(f"Found template at alternate path: {template_path}")
                        break
                else:
                    # List actual directory contents for debugging
                    exec_dir = Path(sys.executable).parent
                    logger.error(f"Executable directory: {exec_dir}")
                    logger.error(f"Executable directory contents: {list(exec_dir.iterdir())}")

            if not template_path.exists():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Admin interface template not found at {template_path}"
                )

        with open(template_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        # Replace version placeholder with actual version
        from mlx_rag import __version__
        html_content = html_content.replace('{{ version }}', __version__)

        return HTMLResponse(content=html_content)

    @app.post("/v1/system/shutdown")
    async def shutdown_server():
        """Gracefully shutdown the server."""
        import asyncio
        import os

        def shutdown():
            """Shutdown the server."""
            logger.info("Shutdown requested via API")
            # This will trigger the lifespan cleanup
            os._exit(0)

        # Schedule shutdown after a short delay to allow response to be sent
        asyncio.get_event_loop().call_later(1.0, shutdown)

        return {"message": "Server shutting down"}

    @app.post("/v1/system/restart")
    async def restart_server():
        """Restart the server with updated settings."""
        import asyncio
        import os
        import sys
        import subprocess

        def restart():
            """Restart the server."""
            logger.info("Restart requested via API")

            # Try to restart using the same command line arguments
            try:
                # Get the current command line
                python_executable = sys.executable
                script_args = sys.argv

                # Start new process
                subprocess.Popen([python_executable] + script_args)

                # Exit current process
                os._exit(0)
            except Exception as e:
                logger.error(f"Failed to restart: {e}")
                # Fallback to shutdown
                os._exit(0)

        # Schedule restart after a short delay to allow response to be sent
        asyncio.get_event_loop().call_later(1.0, restart)

        return {"message": "Server restarting with updated settings"}

    @app.get("/v1/models/current/capabilities")
    async def get_current_model_capabilities():
        """Get capabilities of currently loaded models (vision support, etc.)."""
        try:
            model_manager = get_model_manager()
            loaded_models = model_manager.get_loaded_models()
            
            capabilities = {
                "models": [],
                "has_vision_model": False,
                "has_text_model": False,
                "has_audio_model": False,
                "has_embedding_model": False
            }
            
            for model_name in loaded_models:
                loaded_model = model_manager.get_model_for_inference(model_name)
                if loaded_model:
                    model_caps = {
                        "name": model_name,
                        "type": getattr(loaded_model.mlx_wrapper, 'model_type', 'text'),
                        "supports_vision": False,
                        "supports_text": True,
                        "supports_audio": False,
                        "supports_embeddings": False
                    }
                    
                    # Check model type and capabilities
                    model_type = getattr(loaded_model.mlx_wrapper, 'model_type', 'text')
                    
                    if model_type == 'vision':
                        model_caps["supports_vision"] = True
                        capabilities["has_vision_model"] = True
                    elif model_type == 'whisper' or model_type == 'audio':
                        model_caps["supports_audio"] = True
                        capabilities["has_audio_model"] = True
                    elif model_type == 'embedding':
                        model_caps["supports_embeddings"] = True
                        capabilities["has_embedding_model"] = True
                    else:
                        # Default to text model
                        capabilities["has_text_model"] = True
                    
                    capabilities["models"].append(model_caps)
            
            return capabilities
            
        except Exception as e:
            logger.error(f"Error getting model capabilities: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error getting model capabilities: {str(e)}"
            )
               
    @app.get("/v1/debug/model/{model_id:path}")
    async def debug_model_info(model_id: str):
        """Debug endpoint to inspect model card content for size estimation troubleshooting."""
        try:
            hf_client = get_huggingface_client()

            # Get raw model info
            from huggingface_hub import model_info
            model = model_info(model_id)

            # Extract all relevant fields for debugging
            debug_info = {
                "model_id": model_id,
                "description": getattr(model, 'description', None),
                "tags": getattr(model, 'tags', []),
                "library_name": getattr(model, 'library_name', None),
                "pipeline_tag": getattr(model, 'pipeline_tag', None),
                "card_data": getattr(model, 'card_data', None),
                "size_estimation": None,
                "debug_log": []
            }

            # Try our size estimation with debug logging
            try:
                model_name = model.id.lower()
                description = getattr(model, 'description', '') or ''

                debug_info["debug_log"].append(f"Model name (lower): {model_name}")
                debug_info["debug_log"].append(f"Description length: {len(description)}")
                debug_info["debug_log"].append(f"Description content: {repr(description[:500])}")

                # Test our regex patterns
                import re
                param_patterns = [
                    r'(\d+(?:\.\d+)?)\s*[Bb](?:illion)?\s+param',
                    r'(\d+(?:\.\d+)?)\s*[Bb](?:illion)?\s+parameter',
                    r'(\d+(?:\.\d+)?)\s*[Bb](?:illion)?\s+model',
                    r'(\d+(?:\.\d+)?)\s*[Bb](?:illion)?\s+weights',
                    r'(\d+(?:\.\d+)?)\s*[Bb](?:illion)?\s*-?\s*param',
                    r'Parameters?:\s*(\d+(?:\.\d+)?)\s*[Bb]',
                    r'Model size:\s*(\d+(?:\.\d+)?)\s*[Bb]',
                    r'(\d+(?:\.\d+)?)\s*[Bb](?:illion)?\s*parameter',
                ]

                param_count_billions = None
                for i, pattern in enumerate(param_patterns):
                    matches = re.findall(pattern, description, re.IGNORECASE)
                    debug_info["debug_log"].append(f"Pattern {i+1} '{pattern}': {matches}")
                    if matches:
                        param_count_billions = float(matches[0])
                        debug_info["debug_log"].append(f"Found parameter count: {param_count_billions}B")
                        break

                # Test quantization detection
                quantization_info = {
                    "detected_bits": 16,  # default
                    "indicators": []
                }

                if "4bit" in model_name or "4-bit" in model_name:
                    quantization_info["detected_bits"] = 4
                    quantization_info["indicators"].append("4bit/4-bit")
                elif "8bit" in model_name or "8-bit" in model_name:
                    quantization_info["detected_bits"] = 8
                    quantization_info["indicators"].append("8bit/8-bit")

                debug_info["quantization"] = quantization_info

                if param_count_billions:
                    bits_per_param = quantization_info["detected_bits"]
                    base_memory_gb = (param_count_billions * 1e9 * bits_per_param) / (8 * 1024**3)
                    total_memory_gb = base_memory_gb * 1.25

                    debug_info["size_estimation"] = {
                        "param_count_billions": param_count_billions,
                        "bits_per_param": bits_per_param,
                        "base_memory_gb": base_memory_gb,
                        "total_memory_gb": total_memory_gb,
                        "calculation": f"{param_count_billions}B × {bits_per_param}bit ÷ 8 ÷ 1024³ × 1.25 = {total_memory_gb:.2f}GB"
                    }
                else:
                    debug_info["debug_log"].append("No parameter count found in description")

                # Also check safetensors metadata
                debug_info["debug_log"].append("Checking safetensors metadata...")
                try:
                    if hasattr(model, 'safetensors') and model.safetensors:
                        debug_info["debug_log"].append(f"Found safetensors metadata: {model.safetensors}")
                        for file_name, metadata in model.safetensors.items():
                            debug_info["debug_log"].append(f"File {file_name}: {metadata}")
                            if isinstance(metadata, dict):
                                if 'total' in metadata:
                                    total_params = metadata.get('total', 0)
                                    debug_info["debug_log"].append(f"Found 'total' in metadata: {total_params}")
                                    if total_params > 1000000:
                                        params_in_billions = total_params / 1e9
                                        debug_info["debug_log"].append(f"Calculated parameter count: {params_in_billions}B")
                    else:
                        debug_info["debug_log"].append("No safetensors metadata found")
                except Exception as e:
                    debug_info["debug_log"].append(f"Error checking safetensors: {str(e)}")

            except Exception as e:
                debug_info["debug_log"].append(f"Error in size estimation: {str(e)}")

            return debug_info

        except Exception as e:
            logger.error(f"Error debugging model {model_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error debugging model: {str(e)}"
            )

    return app
