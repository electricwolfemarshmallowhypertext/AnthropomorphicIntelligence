import base64
from io import BytesIO

import audioread
import av
import librosa
import numpy as np
from PIL import Image
import torch
from typing import Optional
from qwen_omni_utils.v2_5.vision_process import extract_vision_info, fetch_image, fetch_video


SAMPLE_RATE=16000
def _check_if_video_has_audio(video_path):
    container = av.open(video_path)
    audio_streams = [stream for stream in container.streams if stream.type == "audio"]
    if not audio_streams:
        return False
    return True


def process_audio_info(conversations: list[dict] | list[list[dict]], use_audio_in_video: bool):
    """
    Read and process audio info

    Support dict keys:

    type = audio
    - audio
    - audio_start
    - audio_end

    type = video
    - video
    - video_start
    - video_end
    """
    audios = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if not isinstance(message["content"], list):
                continue
            for ele in message["content"]:
                if ele["type"] == "audio":
                    if "audio" in ele or "audio_url" in ele:
                        path = ele.get("audio", ele.get("audio_url"))
                        audio_start = ele.get("audio_start", 0.0)
                        audio_end = ele.get("audio_end", None)
                        if isinstance(path, np.ndarray):
                            if path.ndim > 1:
                                raise ValueError("Support only mono audio")
                            audios.append(
                                path[int(SAMPLE_RATE * audio_start) : None if audio_end is None else int(SAMPLE_RATE * audio_end)]
                            )
                            continue
                        elif path.startswith("data:audio"):
                            _, base64_data = path.split("base64,", 1)
                            data = BytesIO(base64.b64decode(base64_data))
                        elif path.startswith("http://") or path.startswith("https://"):
                            data = audioread.ffdec.FFmpegAudioFile(path)
                        elif path.startswith("file://"):
                            data = path[len("file://") :]
                        else:
                            data = path
                    else:
                        raise ValueError("Unknown audio {}".format(ele))
                elif use_audio_in_video and ele["type"] == "video":
                    if "video" in ele or "video_url" in ele:
                        path = ele.get("video", ele.get("video_url"))
                        audio_start = ele.get("video_start", 0.0)
                        audio_end = ele.get("video_end", None)
                        assert _check_if_video_has_audio(
                            path
                        ), "Video must has audio track when use_audio_in_video=True"
                        if path.startswith("http://") or path.startswith("https://"):
                            data = audioread.ffdec.FFmpegAudioFile(path)
                        elif path.startswith("file://"):
                            data = path[len("file://") :]
                        else:
                            data = path
                    else:
                        raise ValueError("Unknown video {}".format(ele))
                else:
                    continue
                audios.append(
                    librosa.load(
                        data,
                        sr=SAMPLE_RATE,
                        offset=audio_start,
                        duration=(audio_end - audio_start) if audio_end is not None else None,
                    )[0]
                )
    if len(audios) == 0:
        audios = None
    return audios

# read audio chunks from one video, reading the whole video and spliting the audio chunks
def get_audio_chunks_from_one_video(conversations: list[dict] | list[list[dict]], use_audio_in_video: bool):
    audios = []
    audios_data = []
    audio_starts = []
    audio_ends = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if not isinstance(message["content"], list):
                continue
            for ele in message["content"]:
                if ele["type"] == "audio":
                    # TODO: read raw audio file
                    pass
                elif use_audio_in_video and ele["type"] == "video":
                    if "video" in ele or "video_url" in ele:
                        path = ele.get("video", ele.get("video_url"))
                        # audio_start = ele.get("video_start", 0.0)
                        # audio_end = ele.get("video_end", None)
                        audio_start = ele.get("chunk_start", 0.0)
                        audio_end = ele.get("chunk_end", None)
                        audio_starts.append(audio_start)
                        audio_ends.append(audio_end)
                        assert _check_if_video_has_audio(
                            path
                        ), "Video must has audio track when use_audio_in_video=True"
                        if path.startswith("http://") or path.startswith("https://"):
                            data = audioread.ffdec.FFmpegAudioFile(path)
                        elif path.startswith("file://"):
                            data = path[len("file://") :]
                        else:
                            data = path
                    else:
                        raise ValueError("Unknown video {}".format(ele))
                    audios_data.append(data)
                else:
                    continue
    check_same_source = all(x == audios_data[0] for x in audios_data)
    if not check_same_source:
        raise ValueError("Currently only support audio chunks from the same video source")
    audio_in_one = librosa.load(
        audios_data[0],
        sr=SAMPLE_RATE,
        offset=audio_starts[0],
        duration=(audio_ends[-1] - audio_starts[0]) if audio_ends[-1] is not None else None,
    )[0]
    for i in range(len(audio_starts)):
        cur_audio = audio_in_one[int(SAMPLE_RATE * audio_starts[i]): int(SAMPLE_RATE * audio_ends[i])]
        audios.append(cur_audio)
    if len(audios) == 0:
        audios = None
    return audios

def check_source(vision_infos):
    source = vision_infos[0].get('video', None)
    if source is None:
        return False
    for info in vision_infos[1:]:
        if info['video'] != source:
            return False
    return True
    
def process_interleave_mm_info(conversations, use_audio_in_video, return_video_kwargs=False):
    if use_audio_in_video:
        # read audio from video files
        audios = get_audio_chunks_from_one_video(conversations, use_audio_in_video)
    else:
        # no audio input
        audios = None 
    vision = process_interleave_vision_info(conversations, return_video_kwargs=return_video_kwargs)
    return (audios,) + vision

def process_interleave_vision_info(
    conversations: list[dict] | list[list[dict]],
    return_video_kwargs: bool = False,
) -> tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None, Optional[dict]]:

    vision_infos = extract_vision_info(conversations)
    # if all the clip belongs to the same video, only read the video one time and segment
    flag_video_source = check_source(vision_infos)
    image_inputs = []
    video_inputs = []
    video_sample_fps_list = []

    if flag_video_source:
        vision_infos[0]['nframes'] = (vision_infos[0]['video_end'] - vision_infos[0]['video_start']) * 2
        video_input, video_sample_fps = fetch_video(vision_infos[0], return_video_sample_fps=True)
        video_inputs = [video_input[i:i+2] for i in range(0, len(video_input), 2)]
        video_sample_fps_list = [video_sample_fps] * len(video_inputs)
        
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info))
        elif "video" in vision_info and not flag_video_source:
            video_input, video_sample_fps = fetch_video(vision_info, return_video_sample_fps=True)
            video_sample_fps_list.append(video_sample_fps)
            video_inputs.append(video_input)
        elif not flag_video_source:
            raise ValueError("image, image_url or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    if return_video_kwargs:
        return image_inputs, video_inputs, {'fps': video_sample_fps_list}
    return image_inputs, video_inputs