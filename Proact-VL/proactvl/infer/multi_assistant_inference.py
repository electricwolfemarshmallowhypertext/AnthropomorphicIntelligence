import torch
import os
from datetime import datetime
import logging
from dataclasses import dataclass
from typing import Dict, Optional
from collections import OrderedDict
import numpy as np
import json

from proactvl.model.modeling_proact import ProAct_OmniModel, ProActConfig
from proactvl.utils.utils import _split_words
from proactvl.infer.assistant import Assistant
from proactvl.infer.talker import Talker
from proactvl.infer.video_reader import VideoReader

# SYSTEM_INPUT = '''You are a live video commentator.
# Watch the video and provide commentary only when significant events or visual changes occur, such as key actions, transitions, or highlights.
# Stay completely silent during calm or uneventful moments.
# If the user input includes other commentators’ lines in the format “(SPEAKER_X): ...”, treat them as co-commentators and decide on your own whether to respond or stay silent.
# Your goal is to produce realistic, context-aware, event-driven commentary that focuses on meaningful visual moments rather than continuous narration.'''
# import random
# sys_prompt_idx = random.randint(0, 4)
# SYSTEM_INPUT = SOCCERNET_SYSTEM_PROMPTS[sys_prompt_idx]
# SYSTEM_PROMPT = f'<|im_start|>system\n{SYSTEM_INPUT}<|im_end|>\n'
# ASSISTANT_PROMPT = '<|im_start|>assistant\n'

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Logging setup: suppress noisy logs during data reading
logging.getLogger("qwen_omni_utils.v2_5.vision_processor").setLevel(logging.ERROR)
logging.getLogger("root").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)


PENALTY_FACTOR = 0
ACCUMULATE_COUNTER = 100000
import time

# During user forward, fill assistant id/name/active score/begin_second; during assistant forward, generate response
@dataclass
class AssistantResponse:
    assistant_id: int = None
    assistant_name: str = None
    commentary: str = None
    active: bool = False
    score: float = 0.0
    begin_second: int = -1

    # def __bool__(self):
    #     return self.active and self.commentary is not None and self.commentary != '' and self.commentary != '<|SILENCE|>'
    
    def short_repr(self) -> str:
        return f"[{self.assistant_name}]({self.score:.2f}) {'✓' if self.active else '×'}: {self.commentary or 'No commentary'}"

class MultiAssistantStreamInference:
    def __init__(self, model_config, ckpt_path, infer_config, generate_config, talker_config, device='cuda'):
        if ckpt_path is not None:
            # self.model = ProAct_OmniModel.from_pretrained(model_config, ckpt_path, weight_dir_prefix=weight_dir_prefix).to(device)
            print(f'Loading model from checkpoint: {ckpt_path}')
            self.model = ProAct_OmniModel.from_pretrained(ckpt_path).to(device)
        else:
            print(f'Initializing model from scratch with config: {model_config}')
            self.model = ProAct_OmniModel(model_config).to(device)
        self.model.eval()
        self.device = device

        self.use_audio_in_video = infer_config.get('use_audio_in_video', False)
        self.max_kv_tokens = infer_config.get('max_kv_tokens', 4096)
        self.assistant_num = infer_config.get('assistant_num', 2)

        self.generate_config = generate_config

        self.tokenizer = self.model.processor.tokenizer
        self.generate_config['eos_token_id'] = self.tokenizer.eos_token_id
        self.generate_config['pad_token_id'] = self.tokenizer.pad_token_id
        
        # Store KV cache and states in assistants to avoid initializing multiple models; assistant ids start from 0 by default
        self.assistants = [Assistant(i, self.model, self.max_kv_tokens, self.use_audio_in_video, self.generate_config, device=device) for i in range(self.assistant_num)]
        self.id2assistant = {assistant.assistant_id: assistant for assistant in self.assistants}
        # set threshold
        state_threshold = infer_config.get('state_threshold', 0.5)
        for a in self.assistants:
            a.set_threshold(state_threshold)

        self.enable_tts = infer_config.get('enable_tts', False)
        self.talker = Talker(self.assistant_num, talker_config) if self.enable_tts else None

        self.session_id = 0
        self.commentary_history = []
        self.video_reader = None

    def register_video_reader(self, video_path, video_begin, video_end):
        self.video_reader = VideoReader(video_path, video_begin, video_end, self.model.processor)

    def set_assistant_count(self, count: int) -> None:
        print(f"[Assistant] Setting assistant count to: {count}")
        self.assistant_num = count
        self.assistants = [Assistant(i, self.model, self.max_kv_tokens, self.use_audio_in_video, self.generate_config, device=self.device) for i in range(self.assistant_num)]
        self.id2assistant = {assistant.assistant_id: assistant for assistant in self.assistants}
        for a in self.assistants:
            a.prime_system_prompt()



    def new_session(self, task: str = 'all'):
        # Initialize session id from current time
        self.session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.session_output_dir = os.path.join('./infer_output', f'session_{self.session_id}')
        for assistant in self.assistants:
            assistant.clear_session()
            assistant.set_session(self.session_id, self.session_output_dir, task=task)
        if self.enable_tts and self.talker is not None:
            self.talker.clear_session()
            self.talker.set_session(self.session_id, self.session_output_dir)

        print("New session started for multi-assistant inference.")

    def prime_system_prompts(self):
        for assistant in self.assistants:
            assistant.clear_session()
            assistant.prime_system_prompt()
        for assistant in self.assistants:
            assistant.monitor()

    def _build_last_comments_excluding(
        self,
        exclude_assistant_id: int,
        assistant_responses: Dict[int, AssistantResponse],
    ) -> str:
        if assistant_responses is None:
            return ""
        if type(assistant_responses) is str:
            return assistant_responses
        
        """Build previous-turn context excluding the current assistant."""
        parts = []
        for rid, resp in assistant_responses.items():
            if rid == exclude_assistant_id:
                continue
            print(f'{rid}, {resp}')
            if resp.commentary and resp.commentary!='' and resp.commentary!=self.model.silence_eos_token and resp.active:  # Effective and has valid text
                parts.append(f"[SPEAKER_{rid}]: {resp.commentary}")
        return "".join(parts)

    def _make_user_prompt(self, last_comments_str, user_query):
        messages = [
            {
                "role": "user",
                "content": []
            }
        ]
        if last_comments_str and last_comments_str != '':
            messages[0]["content"].append({
                "type": "text",
                "text": f'<|history_start|>{last_comments_str}<|history_end|>'
            })
        messages[0]["content"].append({
            "type": "video",
            "video": "<|VIDEO|>"
        })
        if user_query and user_query != '':
            messages[0]["content"].append({
                "type": "text",
                "text": f'<|query_start|>{user_query}<|query_end|>'
            })
        messages[0]["content"].append({
            "type": "text",
            "text": '<|FLAG|>'
        })
        text = self.model.processor.apply_chat_template(messages, tokenize=False, use_default_system_prompt=False, add_generation_tokens=False)
        user_idx = text.index('<|im_start|>user\n')
        text = text[user_idx:]
        return text

    def _select_speaker(self, assistant_responses: Dict[int, AssistantResponse]) -> Optional[int]:
        # Only the active assistant with the highest score generates a response
        active_assistants = [resp for resp in assistant_responses.values() if resp.active]
        if not active_assistants:
            return None
        # Sort by score and choose the highest-scoring assistant
        active_assistants.sort(key=lambda x: x.score, reverse=True)
        selected_assistant = active_assistants[0]
        return selected_assistant.assistant_id

    # Read video chunks through video reader
    def infer_one_chunk(self, begin_second, history=None, user_query=None, previous_responses=None):
        # only use the fisrt assistant for inference
        # 0. Initialize return object
        time1 = time.time()
        next_responses: Dict[int, AssistantResponse] = OrderedDict()
        mm_inputs = self.video_reader.get_inputs(begin_second)
        for a in self.assistants:
            next_responses[a.assistant_id] = AssistantResponse(
                assistant_id=a.assistant_id,
                assistant_name=f"SPEAKER_{a.assistant_id}",
                commentary=None,
                active=False,
                score=0.0,
                begin_second=begin_second,
            )  
            last_comments = ''
            if history is not None:
                last_comments += history
                print(f'Assistant {a.assistant_id} history prompt: {history}')
            background_comments = self._build_last_comments_excluding(a.assistant_id, previous_responses)
            last_comments += background_comments
            cur_user_prompt = self._make_user_prompt(last_comments, user_query)

            tmp_inputs = mm_inputs.copy()
            tmp_inputs['text'] = cur_user_prompt
            inputs = self.model.processor(**tmp_inputs).to(self.model.device).to(self.model.llm.dtype)
            # 2. forward each assistant with the current chunk, update kv cache and detect active status
            flag, score = a.forward_user(inputs, generate_flag=True)
            next_responses[a.assistant_id].active = flag
            next_responses[a.assistant_id].score = score
        time2 = time.time()
        token_cnt = 0
        speaker_id = self._select_speaker(next_responses)
        if speaker_id is None:
            # All silence
            for a in self.assistants:
                a.forward_custom_assistant(self.model.silence_eos_token)
                next_responses[a.assistant_id].commentary = self.model.silence_eos_token
        else:
            speaker = self.id2assistant[speaker_id]
            for a in self.assistants:
                if a.assistant_id != speaker_id:
                    a.forward_custom_assistant(self.model.silence_eos_token)
                    next_responses[a.assistant_id].commentary = self.model.silence_eos_token
                    # Model may classify as active, but only one speaker is allowed per second; force others inactive
                    next_responses[a.assistant_id].active = False
                else:
                    resp, token_cnt = speaker.forward_assistant()
                    next_responses[speaker_id].commentary = resp

        time3 = time.time()
        extra_info = {
            'cache': {
                'time': (time2 - time1)/len(self.assistants)
            },
            'forward': {
                'time': time3 - time2,
                'token_cnt': token_cnt
            }
        }       
        return next_responses, extra_info

    
    # Interface for web demo: consume frames directly and return audio as well
    def infer_one_chunk_backend(self, audios, images, videos, user_query, previous_assistant_responses, begin_second, history=None):
        # 1. Initialize return object
        next_responses: Dict[int, AssistantResponse] = OrderedDict()
        # 2. forward each assistant with the current chunk, update kv cache and detect active status
        for a in self.assistants:
            next_responses[a.assistant_id] = AssistantResponse(
                assistant_id=a.assistant_id,
                assistant_name=f"SPEAKER_{a.assistant_id}",
                commentary=None,
                active=False,
                score=0.0,
                begin_second=begin_second,
            )

            last_comments = ''
            if history is not None:
                last_comments += history
            background_comments = self._build_last_comments_excluding(a.assistant_id, previous_assistant_responses)
            last_comments += background_comments
            if history is not None:
                print(f'Assistant {a.assistant_id} history prompt: {history}')
            cur_user_prompt = self._make_user_prompt(last_comments, user_query)
            # print(f'Assistant {a.assistant_id} user prompt: {cur_user_prompt}')
            flag, score = a.forward_user_with_chunk(cur_user_prompt, audios, images, videos, generate_flag=True)

            next_responses[a.assistant_id].active = flag
            next_responses[a.assistant_id].score = score
        # ================================= Only one speaker is allowed in the same second =================================
        speaker_id = self._select_speaker(next_responses)
        if speaker_id is None:
            # All silence
            for a in self.assistants:
                a.forward_custom_assistant('<|SILENCE|>')
                next_responses[a.assistant_id].commentary = '<|SILENCE|>'
            audio = self.talker.register_text(None, None) if self.enable_tts and self.talker is not None else None
            return next_responses, audio

        speaker = self.id2assistant[speaker_id]
        audio = None
        for a in self.assistants:
            if a.assistant_id != speaker_id:
                a.forward_custom_assistant('<|SILENCE|>')
                next_responses[a.assistant_id].commentary = '<|SILENCE|>'
                # Model may classify as active, but only one speaker is allowed per second; force others inactive
                next_responses[a.assistant_id].active = False
            else:
                resp, _ = speaker.forward_assistant()
                next_responses[speaker_id].commentary = resp
                resp = resp.replace(' ...', '')
                if self.talker is not None:
                    audio = self.talker.register_text(speaker_id, resp.strip())
        return next_responses, audio

    # audio generation
    def register_commentary(self, text, assistant_id, begin_second):
        self.commentary_history.append({
            'text': [text],
            'word_list': _split_words(text),
            'assistant_id': assistant_id,
            'begin_second': begin_second,
            'end_second': begin_second + 1,
        })

    def post_audio_generation(self):
        with open(os.path.join(self.session_output_dir, f'commentary_history.json'), 'w', encoding='utf-8') as f:
            json.dump(self.commentary_history, f, ensure_ascii=False, indent=4)

        # First merge adjacent segments from the same assistant; if next second has no commentary, end time can be extended
        new_history = []
        pre = 0
        while pre < len(self.commentary_history):
            logger.debug(f'Merging commentary segments, current index: {pre}')
            pre_commentary = self.commentary_history[pre]
            post = pre + 1
            while post < len(self.commentary_history):
                post_commentary = self.commentary_history[post]
                if pre_commentary['word_list'][-1].endswith(('.', '!', '?')) and post-pre > 1:
                    break
                elif post-pre > 5:
                    break
                elif post_commentary['assistant_id'] == pre_commentary['assistant_id']:
                    pre_commentary['text'].extend(post_commentary['text'])
                    pre_commentary['word_list'].extend(post_commentary['word_list'])
                    pre_commentary['end_second'] = post_commentary['end_second']
                    post += 1
                elif post_commentary['assistant_id'] is None:
                    pre_commentary['end_second'] = post_commentary['end_second']
                    post += 1
                else:
                    break
            # Whether broken by sentence end or by long interval, scan again to absorb trailing silence segments
            while post < len(self.commentary_history):
                post_commentary = self.commentary_history[post]
                if post_commentary['assistant_id'] is None:
                    pre_commentary['end_second'] = post_commentary['end_second']
                    post += 1
                else:
                    break
            new_history.append(pre_commentary)
            pre = post

        if self.enable_tts and self.talker is not None:
            self.talker.post_audio_generation(new_history)