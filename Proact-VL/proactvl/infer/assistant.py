import torch
import os
from collections import deque
from typing import Deque
import logging
from proactvl.utils.utils import prune_cache_span
from proactvl.utils.conversations import BASE_PROMPT
import random

MIN_PIXELS = 128*28*28
MAX_PIXELS = 540*28*28

# sys_prompt_idx = random.randint(0, 4)
# SYSTEM_INPUT = SOCCERNET_SYSTEM_PROMPTS[sys_prompt_idx]
# SYSTEM_PROMPT = f'<|im_start|>system\n{SYSTEM_INPUT}<|im_end|>\n'
ASSISTANT_PROMPT = '<|im_start|>assistant\n'

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

YELLOW = '\033[93m'
GREEN = '\033[92m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'

class Assistant:
    def __init__(self, assistant_id, model, max_kv_tokens, use_audio_in_video, generate_config, device='cuda'):
        self.assistant_id = assistant_id
        self.assistant_name = f'SPEAKER_{assistant_id}'
        self.model = model
        self.device = device

        self.max_kv_tokens = max_kv_tokens
        self.evict_percent = 0.2
        self.use_audio_in_video = use_audio_in_video
        self.generate_config = generate_config

        self.max_position_id = -1
        self.kv = None
        self.system_token_cnt = 0
        self.token_cnts: Deque[int] = deque()
        self.dynamic_token_cnt = 0
        self.window_position_ids_for_stream = torch.empty((3, 1, 0), dtype=torch.long).to(device)
        # Track accumulated utterance text; once it grows long enough, start truncating at sentence boundaries
        self.accumulate_counter = 0
        # Whether the next step should stop
        self.to_be_stopped = False

        self.session_id = None
        self.session_output_dir = None

        self.task = None

        # self.high_threshold = generate_config.
        # self.low_threshold = 0.8
        # self.is_talking = False
        # self.state_threshold = self.model.state_threshold
        self.state_threshold = 0.5
        
    # This function has two purposes: set the system prompt (main task) and mitigate attention sink issues;
    # it requires special handling and is ideally called once at session start.
    @torch.no_grad()
    def prime_system_prompt(self, system_prompt=None):
        if system_prompt is None:
            system_prompt = BASE_PROMPT
        messages = [
            {
                'role': 'system',
                'content': system_prompt,
            }
        ]
        text = self.model.processor.apply_chat_template(messages)
        print(f'[Assistant] setting system prompt of assistant {self.assistant_id} to:\n{YELLOW}{system_prompt}{RESET}')
        sys_inputs = self.model.processor(
            text=text,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=self.use_audio_in_video,
        ).to(self.model.device).to(self.model.llm.dtype)

        position_ids = self.model.get_position_ids(max_position_id=self.max_position_id, **sys_inputs)

        out = self.model(
            **sys_inputs,
            position_ids=position_ids,
            output_hidden_states=False,
            return_dict=True,
            use_cache=True,
            output_active_logits=False,
        )

        self.max_position_id = position_ids.max().item()
        self.kv = out.past_key_values
        self.system_token_cnt = sys_inputs['input_ids'].shape[-1]

        
    def set_threshold(self, threshold: float):
        print(f'[Assistant] setting state threshold of assistant {self.assistant_id} to: {threshold}')
        self.state_threshold = threshold

    # Cache multimodal input; if generate_flag is enabled, return the flag used to determine whether a response is needed
    @torch.no_grad()
    def forward_cache(self, inputs, generate_flag):
        position_ids = self.model.get_position_ids(max_position_id=self.max_position_id, **inputs)

        outputs = self.model(
            **inputs,
            position_ids=position_ids,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True,
            use_cache=True,
            past_key_values=self.kv,
            output_active_logits=generate_flag,
        )

        self.max_position_id = position_ids[0][0][-1].item()
        self.kv = outputs.past_key_values
        self.window_position_ids_for_stream = torch.cat(
            [self.window_position_ids_for_stream, position_ids], dim=-1
        )
        cur_token_cnt = inputs['input_ids'].shape[-1]
        self.token_cnts.append(cur_token_cnt)
        self.dynamic_token_cnt += cur_token_cnt
        if generate_flag:
            assert self.model.processor.tokenizer.convert_ids_to_tokens([inputs['input_ids'][0][-3]])[0] =='<|FLAG|>'
            score = torch.sigmoid(outputs.active_logits[0][-3]).item()
            flag = score > self.state_threshold
            return flag, score
        return None, None

    # Cache multimodal input; if generate_flag is enabled, return the flag used to determine whether a response is needed
    @torch.no_grad()
    def forward_cache_with_chunk(self, text, audios, images, videos, generate_flag):
        # print(f'{text}')
        min_pixels = MIN_PIXELS
        max_pixels = MAX_PIXELS
        # print(min_pixels, max_pixels)
        size = {
            'shortest_edge': min_pixels,
            'longest_edge': max_pixels,
        }
        inputs = self.model.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=self.use_audio_in_video,
            size=size,
        ).to(self.model.device).to(self.model.llm.dtype)
        position_ids = self.model.get_position_ids(max_position_id=self.max_position_id, **inputs)

        outputs = self.model(
            **inputs,
            position_ids=position_ids,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True,
            use_cache=True,
            past_key_values=self.kv,
            output_active_logits=generate_flag,
        )

        self.max_position_id = position_ids[0][0][-1].item()
        self.kv = outputs.past_key_values
        self.window_position_ids_for_stream = torch.cat(
            [self.window_position_ids_for_stream, position_ids], dim=-1
        )
        cur_token_cnt = inputs['input_ids'].shape[-1]
        self.token_cnts.append(cur_token_cnt)
        self.dynamic_token_cnt += cur_token_cnt
        if generate_flag:
            # user template: <|im_start|>user\n<|vision_bos|><|VIDEO|><|vision_eos|>(SPEAKER_0)***<|im_end|>\n
            assert self.model.processor.tokenizer.convert_ids_to_tokens([inputs['input_ids'][0][-3]])[0] =='<|FLAG|>'
            score = torch.sigmoid(outputs.active_logits[0][-3]).item()

            # flag = score > self.model.state_threshold
            # if self.is_talking:
            #     flag = score > self.model.state_threshold
            #     if not flag:
            #         self.is_talking = False
            # else:
            #     flag = score > self.model.state_threshold
            #     if flag:
            #         self.is_talking = True
            flag = score > self.state_threshold
            # print(f'Assistant {self.assistant_id} active score: {score:.4f}, threshold: {self.model.state_threshold}， active flag: {flag}')
            # print(f'{flag}, {score}')
            return flag, score
        return None, None

    @torch.no_grad()
    def forward_generate(self, text):
        generate_inputs = self.model.processor(
            text=text,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=self.use_audio_in_video,
        ).to(self.model.device).to(self.model.llm.dtype)
        # input_ids length must be 1
        assert generate_inputs['input_ids'].shape[-1] == 1

        # In overridden prepare_inputs_for_generation, position ids are incremented by 1 first,
        # so use max_position_id directly as input position id here.
        position_ids = self.model.get_position_ids(max_position_id=self.max_position_id, **generate_inputs).add_(-1)
        # print(position_ids)

        init_max_pos_id = position_ids[0][0][-1].item()+1
        outputs = self.model.generate(
            input_ids=generate_inputs['input_ids'],
            attention_mask=generate_inputs['attention_mask'],
            position_ids=position_ids,
            cache_position=torch.tensor([[self.kv.get_seq_length()]], device=generate_inputs['input_ids'].device),
            past_key_values=self.kv,
            use_cache=True,
            output_active_logits=False,
            **self.generate_config,
        )
        # Compute newly introduced tokens; input token length is 1, and the last output token does not update KV cache, so exclude it
        new_tokens = outputs[:, :-1]
        new_token_cnt = new_tokens.shape[-1]
        response = self.model.processor.batch_decode(outputs[:, 1:-1], skip_special_tokens=False)[0]
        self.token_cnts.append(new_token_cnt)
        self.dynamic_token_cnt += new_token_cnt

        # Position ids are modified in-place during generation
        self.max_position_id = position_ids[0][0][-1].item()
        window_position_ids_cat = torch.arange(init_max_pos_id, self.max_position_id+1, device=position_ids.device).unsqueeze(0).unsqueeze(0).expand(3, 1, -1)
        self.window_position_ids_for_stream = torch.cat(
            [self.window_position_ids_for_stream, window_position_ids_cat], dim=-1
        )
        assert self.window_position_ids_for_stream.shape[-1] + self.system_token_cnt == self.kv.layers[0].keys.shape[2], \
            f"KV length mismatch, expected {self.window_position_ids_for_stream.shape[-1] + self.system_token_cnt}, got {self.kv.layers[0].keys.shape[2]}"
        
        return response, new_token_cnt

    @torch.no_grad()
    def forward_system(self):
        pass

    @torch.no_grad()
    def forward_user_with_chunk(self, text,audios, images, videos, generate_flag):
        result = self.forward_cache_with_chunk(text, audios, images, videos, generate_flag)
        self._ensure_budget()
        self.monitor()
        return result

    @torch.no_grad()
    def forward_user(self, inputs, generate_flag):
        result = self.forward_cache(inputs, generate_flag)
        self._ensure_budget()
        self.monitor()
        return result

    @torch.no_grad()
    def forward_custom_assistant(self, text):
        # Prefill assistant prompt without generating a response
        prefill_text = f'<|im_start|>assistant\n{text}<|im_end|>\n'
        inputs = self.model.processor(
            text=prefill_text,
            audio=None,
            images=None,
            videos=None,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=self.use_audio_in_video,
        ).to(self.model.device).to(self.model.llm.dtype)

        self.forward_cache(inputs, generate_flag=False)
        self._ensure_budget()
        self.monitor()
        
    @torch.no_grad()
    def forward_assistant(self):
        # Three steps: prefill assistant prompt, generate response, then append trailing <|im_end|>\n
        # ASSISTANT_PROMPT: ASSISTANT_PROMPT = '<|im_start|>assistant\n', <|im_start|>assistant is used to prefill cache, \n is used for generation
        prefill_text = '<|im_start|>assistant'
        inputs = self.model.processor(
            text=prefill_text,
            audio=None,
            images=None,
            videos=None,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device).to(self.model.llm.dtype)
        self.forward_cache(inputs, generate_flag=False)

        generate_text = '\n'
        response, token_cnt = self.forward_generate(generate_text)

        postfix_text = '<|im_end|>\n'
        inputs = self.model.processor(
            text=postfix_text,
            audio=None,
            images=None,
            videos=None,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device).to(self.model.llm.dtype)
        self.forward_cache(inputs, generate_flag=False)

        # Merge the last three entries in self.token_cnts into one
        len1 = self.token_cnts.pop()
        len2 = self.token_cnts.pop()
        len3 = self.token_cnts.pop()
        self.token_cnts.append(len1 + len2 + len3)
        self.monitor()
        return response, token_cnt

    @torch.no_grad()
    def _ensure_budget(self):
        if self.dynamic_token_cnt > self.max_kv_tokens - self.system_token_cnt:

            cur_pop_num = 0
            while cur_pop_num < int(self.max_kv_tokens*self.evict_percent) and len(self.token_cnts) > 0:
                cur_pop_num += self.token_cnts.popleft()
            logger.debug(f'Current KV length {self.kv.layers[0].keys.shape[2]} exceeds budget by {self.dynamic_token_cnt + self.system_token_cnt - self.max_kv_tokens}, evicting {cur_pop_num} tokens')

            window_position_ids_for_stream = self.window_position_ids_for_stream
            self.window_position_ids_for_stream = window_position_ids_for_stream[:, :, cur_pop_num:]

            shift_size = self.window_position_ids_for_stream[0][0][0].item() - (self.system_token_cnt)  # Use first token position id as shift size
            self.window_position_ids_for_stream -= shift_size
            self.max_position_id = self.window_position_ids_for_stream[0][0][-1].item()

            logger.debug(f'Updating KV cache, evicting {cur_pop_num} tokens')
            # Update KV cache
            self.dynamic_token_cnt -= cur_pop_num
            if True:
                self.kv = prune_cache_span(self.kv, self.system_token_cnt, self.system_token_cnt + cur_pop_num)

                assert self.kv.layers[0].keys.shape[2] == self.dynamic_token_cnt + self.system_token_cnt, f"KV length mismatch, expected {self.dynamic_token_cnt + self.system_token_cnt}, got {self.kv.layers[0].keys.shape[-1]}"
                assert self.dynamic_token_cnt == self.window_position_ids_for_stream.shape[-1], f'Window position ids length mismatch, expected {self.dynamic_token_cnt}, got {self.window_position_ids_for_stream.shape[-1]}'
                for i in range(len(self.kv.layers)):
                    self.kv.layers[i].keys = self.model.shift_position_ids(-shift_size, self.kv.layers[i].keys, self.system_token_cnt)

                assert self.kv.layers[0].keys.shape[2] == self.dynamic_token_cnt + self.system_token_cnt, \
                    f"KV length mismatch, expected {self.dynamic_token_cnt + self.system_token_cnt}, got {self.kv.layers[0].keys.shape[2]}"
            elif False:
                # Clear cache and refill
                logger.debug(f'Refilling KV cache')
                self.max_position_id = -1
                self.kv = None
                self.system_token_cnt = 0
                self.token_cnts: Deque[int] = deque()
                self.dynamic_token_cnt = 0
                self.window_position_ids_for_stream = torch.empty((3, 1, 0), dtype=torch.long).to(self.device)
                # Refill system prompt
                self.prime_system_prompt()
            else:
                self.max_position_id = self.system_token_cnt - 1
                self.kv = prune_cache_span(self.kv, self.system_token_cnt, self.kv.layers[0].keys.shape[2])
                print(f'KV cache length updated to {self.kv.layers[0].keys.shape}')
                self.token_cnts: Deque[int] = deque()
                self.dynamic_token_cnt = 0
                self.window_position_ids_for_stream = torch.empty((3, 1, 0), dtype=torch.long).to(self.device)

    def monitor(self):
        # brief introduction
        logger.debug(f'Assistant ID: {self.assistant_id}, Name: {self.assistant_name}')
        # Window-related parameters
        logger.debug(f" Max Position ID: {self.max_position_id}, System Token Count: {self.system_token_cnt}, dynamic Token Count: {self.dynamic_token_cnt}, token counts queue: {list(self.token_cnts)}")
        logger.debug(f'Window Position IDs shape: {self.window_position_ids_for_stream.shape}')
        # state of kv cache
        if self.kv is not None:
            # logger.debug(f'KV Cache shape: {self.kv.key_cache[0].shape}')
            logger.debug(f'KV Cache shape: {self.kv.layers[0].keys.shape}')

        # brief introduction
        # print(f'Assistant ID: {self.assistant_id}, Name: {self.assistant_name}')
        # # Window-related parameters
        # print(f" Max Position ID: {self.max_position_id}, System Token Count: {self.system_token_cnt}, dynamic Token Count: {self.dynamic_token_cnt}, token counts queue: {list(self.token_cnts)}")
        # print(f'Window Position IDs shape: {self.window_position_ids_for_stream.shape}')
        # # state of kv cache
        # if self.kv is not None:
        #     print(f'KV Cache shape: {self.kv.key_cache[0].shape}')

    def clear_session(self):
        self.max_position_id = -1
        self.kv = None
        self.system_token_cnt = 0
        self.token_cnts: Deque[int] = deque()
        self.dynamic_token_cnt = 0
        self.window_position_ids_for_stream = torch.empty((3, 1, 0), dtype=torch.long).to(self.device)

    def set_session(self, session_id, session_output_dir, task: str = 'all'):
        self.session_id = session_id
        self.session_output_dir = session_output_dir
        self.task = task