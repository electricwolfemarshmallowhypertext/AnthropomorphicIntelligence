import os
import time
import io
import wave
import json
import base64
import logging

from sympy.sets.sets import true
import whisperx
import soundfile as sf
import numpy as np

from openai import OpenAI
from dashscope import MultiModalConversation

import dashscope
from typing import Optional

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="modelscope")

from utils.logger import Logger
from dotenv import load_dotenv
load_dotenv('client.env')

logger = Logger(__name__, level=logging.INFO, msg_color=True).get_logger()

class ASRModel:
    def __init__(
        self, 
        model_name: str, 
        device: str = 'cuda',
        compute_type: str = 'float16',
        language: str = 'en'
    ) -> None:
        self.model = whisperx.load_model(model_name, device, compute_type=compute_type)
        self.device = device
        self.language = language
        self.hf_token = os.getenv("HF_TOKEN", '')
        if self.hf_token == '':
            logger.warning("HF_TOKEN is not set, please set it in the environment variables, otherwise will throw error loading diarization model...")

    def inference(
        self,
        audio_path: str,
        batch_size: int = 16,
        min_speakers: int = 2,
        max_speakers: int = 3,
        chunk_size: int = 6
    ) -> dict:
        audio = whisperx.load_audio(audio_path)
        result = self.model.transcribe(audio, chunk_size=chunk_size, batch_size=batch_size, language=self.language)
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=self.device)
        result_aligned = whisperx.align(result["segments"], model_a, metadata, audio, self.device, return_char_alignments=False)

        diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=self.hf_token, device=self.device)
        diarize_segments = diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
        logger.info(f"Diarization Model: min_speakers: {min_speakers}, max_speakers: {max_speakers}")
        result_final = whisperx.assign_word_speakers(diarize_segments, result_aligned)

        return result_final

class FineGrainedASRModel:
    # use API Endpoint
    def __init__(self, model_name: str, system_prompt: str = ""):
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model_name = model_name
        self.system_prompt = system_prompt
    def __call__(self, audio_path: str, prompt: str = "", output_audio: bool = False):
        base64_audio = self.encode_audio(audio_path)
        # Use a smaller chunk size or handle full file if possible. 
        # For Qwen-Omni, 10MB base64 limit might be too large or just right, 
        # but splitting audio might contextually break the ASR/Understanding.
        # Let's assume the clips are short enough (< 30s usually) and send as one request if possible.
        
        MAX_BASE64_LENGTH = 10000000
        if len(base64_audio) > MAX_BASE64_LENGTH:
             parts = [base64_audio[i:i + MAX_BASE64_LENGTH] for i in range(0, len(base64_audio), MAX_BASE64_LENGTH)]
        else:
             parts = [base64_audio]

        full_text_result = ""
        full_audio_result = b"" # To store binary audio data if we want to save it later

        for idx, part in enumerate(parts):
            logger.info(f"Processing part {idx + 1}/{len(parts)}...")
            max_retries = 10
            for attempt in range(max_retries):
                try:
                    completion = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_audio",
                                        "input_audio": {
                                            "data": f"data:audio/mp3;base64,{part}", 
                                            "format": "mp3",
                                        },
                                    },
                                    {"type": "text", "text": prompt}
                                ],
                            },
                        ],
                        modalities=["text", "audio"] if output_audio else ["text"],
                        audio={"voice": "Cherry", "format": "wav"} if output_audio else None,
                        stream=True,
                        stream_options={"include_usage": True}
                    )

                    for chunk in completion:
                        if chunk.choices:
                            delta = chunk.choices[0].delta
                            if hasattr(delta, 'content') and delta.content:
                                full_text_result += delta.content
                            if hasattr(delta, 'audio') and delta.audio and 'data' in delta.audio:
                                 # The example shows printing delta, let's assume we might want to collect audio data too
                                 # But user query mainly asks to "take out the response result", which implies text mostly?
                                 # "modalities=['text', 'audio']" was requested in query example.
                                 pass 
                        else:
                             pass
                             # usage info
                    break # Success, exit retry loop
                
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Error processing part {idx + 1} (attempt {attempt + 1}/{max_retries}): {e}. Retrying...")
                        time.sleep(2)
                    else:
                        logger.error(f"Error processing part {idx + 1}: {e}")
                        # We raise the exception here so that the caller knows it failed
                        raise e
        
        return full_text_result.strip()

    
    def encode_audio(
        self, 
        audio_path: str
    ) -> str:
        with open(audio_path, "rb") as audio_file:
            return base64.b64encode(audio_file.read()).decode("utf-8")

class ChatModelClient:
    """Client for OpenAI-compatible chat, vision, and multimodal model endpoints."""
    def __init__(
            self, model_name: str, 
            base_url: str = "https://openrouter.ai/api/v1",
            system_prompt: Optional[str] = None, 
            verbose: bool = False, 
            **kwargs):
        """
        Args:
            model_name (str): The name of the model.
            system_prompt (Optional[str]): The system prompt to use (default: STANDARD_GAME_PROMPT)
            verbose (bool): If True, additional debug info will be printed.
            **kwargs: Additional keyword arguments to pass to the OpenAI API call.
        """
        self.model_name = model_name 
        self.verbose = verbose 
        self.system_prompt = system_prompt
        logger.info(f"Chat model system prompt: {self.system_prompt}")

        try:
            from openai import OpenAI
            from openai._exceptions import OpenAIError
        except ImportError:
            raise ImportError("OpenAI package is required for ChatModelClient. Install it with: pip install openai")
        self.api_key = None
        if "openrouter" in base_url:
            self.api_key = os.getenv("OPENROUTER_API_KEY")
            # self.api_key = os.getenv("OPENROUTER_API_KEY") # Set the open router api key from an environment variable
        elif "dashscope" in base_url:
            self.api_key = os.getenv("DASHSCOPE_API_KEY")
            # self.api_key = os.getenv("DASHSCOPE_API_KEY") # Set the dashscope api key from an environment variable
        elif "openai" in base_url or "api.openai.com" in base_url:
            self.api_key = os.getenv("OPENAI_API_KEY")
            # self.api_key = os.getenv("OPENAI_API_KEY") # Set the OpenAI api key from an environment variable
        elif "api.uniapi.io" in base_url:
            self.api_key = os.getenv("UNIAPI_API_KEY")
            # self.api_key = os.getenv("UNIAPI_API_KEY") # Set the UniAPI api key from an environment variable
        elif "openai.azure" in base_url:
            self.api_key = os.getenv("AZURE_API_KEY")
        else:
            raise ValueError(f"Unsupported base URL: {base_url}")
        
        if not self.api_key:
            raise ValueError(f"API key not found for {base_url}. Please set the appropriate environment variable (OPENROUTER_API_KEY, DASHSCOPE_API_KEY, or OPENAI_API_KEY).")
        
        self.client = OpenAI(base_url=base_url, api_key=self.api_key, timeout=60)

    def _make_request(self, content: str, temperature: float = 0.9) -> str:
        """Make a single text request and return the generated message."""
        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": content}]
        response = self.client.chat.completions.create(model=self.model_name, messages=messages, n=1, stop=None, temperature=temperature)#**self.kwargs)
        return response.choices[0].message.content.strip()

    def _make_video_request(
        self, 
        video_url: str, 
        text_prompt: str = None,
    ) -> str:
        logger.info(f"Video URL: {video_url}")
        messages = [
            {
                'role': 'user',
                'content': [
                    {'video': video_url, "fps": 2},
                    {'text': text_prompt}
                ]
            }
        ]

        response = MultiModalConversation.call(
            api_key=self.api_key,
            model=self.model_name,
            messages=messages,
            stream=True,
            enable_thinking=False
        )

        if response is None:
            # Raise immediately so the caller can retry, instead of failing later with a NoneType error.
            raise RuntimeError("DashScope MultiModalConversation.call returned an empty response (response is None)")

        # Track the full reasoning stream.
        reasoning_content = ""
        # Track the full answer stream.
        answer_content = ""
        # Track whether the model has moved from reasoning to the final answer.
        is_answering = False

        print("=" * 20 + "Reasoning" + "=" * 20)
        
        for chunk in response:
            # Some streaming chunks may be None or missing output / choices.
            if chunk is None:
                logger.warning("Received an empty streaming chunk (chunk is None); retrying the full video inference request")
                # Raise so the caller can retry the full MultiModalConversation.call.
                raise RuntimeError("Empty streaming chunk from DashScope (chunk is None). Trigger retry.")

            output = getattr(chunk, "output", None)
            if output is None:
                logger.warning("The streaming chunk output is None; this request will be retried")
                raise RuntimeError("Empty streaming chunk from DashScope (chunk is None). Trigger retry.")


            choices = getattr(output, "choices", None)
            if not choices:
                # This can be a progress / heartbeat packet without real content.
                logger.debug("The streaming chunk choices are empty or missing; skipping this chunk")
                continue

            message = choices[0].message

            # message is usually dict-like and follows the DashScope structure.
            reasoning_content_chunk = None
            try:
                reasoning_content_chunk = message.get("reasoning_content", None)
            except Exception:
                # If it is not a dict (very unlikely), ignore the reasoning field.
                reasoning_content_chunk = None

            try:
                msg_content = message.content
            except Exception:
                # Skip malformed message objects.
                logger.warning("The streaming chunk message structure is invalid and content is unavailable; skipping this chunk")
                continue

            # Ignore chunks with neither reasoning nor answer content.
            if (msg_content == [] and (reasoning_content_chunk == "" or reasoning_content_chunk is None)):
                continue

            # Handle reasoning chunks.
            if reasoning_content_chunk is not None and msg_content == []:
                try:
                    print(message.reasoning_content, end="")
                    reasoning_content += message.reasoning_content
                except Exception:
                    # Fallback for alternate structures that still expose reasoning_content_chunk.
                    print(reasoning_content_chunk, end="")
                    reasoning_content += str(reasoning_content_chunk)
            # Handle answer chunks.
            elif msg_content != []:
                if not is_answering:
                    print("\n" + "=" * 20 + "Final Answer" + "=" * 20)
                    is_answering = true

                try:
                    text_piece = msg_content[0]["text"]
                except Exception:
                    # Last-resort fallback for unexpected message structures.
                    text_piece = str(msg_content)

                print(text_piece, end="", flush=True)
                answer_content += text_piece

        return answer_content
    
    def _make_image_request(
        self,
        image_list: list,
        text_prompt: str = None,
        max_tokens: int = 4096,
    ) -> str:
        """
        Process a request containing a list of images.
        :param image_list: List of base64-encoded images.
        :param text_prompt: Text prompt.
        :param max_tokens: Maximum token count.
        :return: Model response.
        """
        logger.info(f"Processing {len(image_list)} images")
        
        # Build the message payload.
        content = [{"type": "text", "text": text_prompt if text_prompt else "Please analyze these images."}]
        
        # Append all images.
        for img_base64 in image_list:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_base64}"
                }
            })
        
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        
        # Call the API.
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content.strip()

    def _retry_request(
            self, 
            content: str, 
            retries: int = 10, 
            delay: int = 5,
            type: str = "video",
            video_url: str = None,
            image_list: list = None,
            max_tokens: int = 4096,
            temperature: float = 0.9,
        ) -> str:
        """
        Attempt to make an API request with retries.

        Args:
            content (str): The input to process.
            retries (int): The number of attempts to try.
            delay (int): Seconds to wait between attempts.
            type (str): The type of request (text, video, image)
            video_url (str): Video URL for video requests
            image_list (list): List of base64 encoded images for image requests
            max_tokens (int): Maximum tokens for the response

        Raises:
            Exception: The last exception caught if all retries fail.
        """
        last_exception = None
        for attempt in range(1, retries + 1):
            try:
                if type == "text":
                    response = self._make_request(content, temperature=temperature)
                elif type == "video":
                    response = self._make_video_request(video_url=video_url, text_prompt=content)
                elif type == "image":
                    response = self._make_image_request(image_list=image_list, text_prompt=content, max_tokens=max_tokens)
                else:
                    raise ValueError(f"Unsupported type: {type}")
                if self.verbose:
                    print(f"\nObservation: {content}\nResponse: {response}")
                return response

            except Exception as e:
                last_exception = e
                print(f"Attempt {attempt} failed with error: {e}")
                if attempt < retries:
                    time.sleep(delay)
        raise last_exception

    def __call__(
            self, 
            content: str,
            type: str = "text",
            video_url: str = None,
            image_list: list = None,
            max_tokens: int = 4096,
            temperature: float = 0.9,
        ) -> str:
        """
        Process the input using the configured chat-model endpoint and return the response.

        Args:
            content (str): The input string to process (prompt text).
            type (str): The type of request - "text", "video", or "image"
            video_url (str): Video URL for video requests
            image_list (list): List of base64 encoded images for image requests
            max_tokens (int): Maximum tokens for the response

        Returns:
            str: The generated response.
        """
        if type == "text":
            if not isinstance(content, str):
                raise ValueError(f"Content must be a string. Received type: {type(content)}")
            return self._retry_request(content=content, type="text", temperature=temperature)
        elif type == "video":
            if not isinstance(video_url, str):
                raise ValueError(f"Video URL must be a string. Received type: {type(video_url)}")
            return self._retry_request(content=content, type="video", video_url=video_url)
        elif type == "image":
            if not isinstance(image_list, list):
                raise ValueError(f"Image list must be a list. Received type: {type(image_list)}")
            return self._retry_request(content=content, type="image", image_list=image_list, max_tokens=max_tokens)
        else:
            raise ValueError(f"Unsupported type: {type}")
