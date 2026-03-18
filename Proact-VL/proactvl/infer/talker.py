import torch
import os
import time
from kokoro import KPipeline
import soundfile as sf
import numpy as np
import librosa
import logging
from proactvl.utils.utils import _split_words

TARGET_SR = 24000
TARGET_SECONDS = 1
TARGET_SAMPLES = int(TARGET_SR * TARGET_SECONDS)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Talker:
    def __init__(self, assistant_num, talker_config):
        self.assistant_num = assistant_num
        # more options is available at https://huggingface.co/hexgrad/Kokoro-82M/tree/main/voices
        self.voices = ['af_heart', 'af_alloy', 'af_aoede', 'af_jessica']
        assert assistant_num <= len(self.voices), f'Currently supports at most {len(self.voices)} assistant TTS voices'
        
        # tts
        self.assistant_voices = {
            i: self.voices[i] for i in range(assistant_num)
        }
        print(f'Initialized Talker with {assistant_num} assistants. Assigned voices: {self.assistant_voices}')
        self.tts_pipeline = KPipeline(lang_code='a')
        print(f'Initialized KPipeline with {self.tts_pipeline}')
        # self.tts_min_words = talker_config.get('tts_min_words', 8)
        # self.tts_max_words = talker_config.get('tts_max_words', 30)
        # self.tts_wait_sec = talker_config.get('tts_wait_sec', 2)

        # self.word_buffers = []
        # self.last_speak_time = 0
        # self.history = []
        # config
        self.config = talker_config
        # self.output_dir = talker_config.get('save_dir', './infer_output')
        self.all_audio = np.array([], dtype=np.float32)
        self.seconds = 0
        self.previous_text = {'speaker_id': None, 'text': ''}  # list of dict with keys: speaker_id, text


        # session info
        self.session_id = None
        self.session_output_dir = None

    def get_audio(self, speaker_id, text):
        if text is None or text.strip() == '':
            return None
        generator = self.tts_pipeline(text, voice=self.assistant_voices.get(speaker_id, 'af_heart'), speed=1.3)
        audio_chunks = []
        for _, _, audio in generator:
            audio_chunks.append(audio)
        if len(audio_chunks) > 0:
            audio = np.concatenate(audio_chunks)
            return audio
        else:
            return None

    def register_text(self, speaker_id, text):
        if text is None or text.strip() == '':
            return None
        if text.endswith(' ...'):
            text = text[:-4].strip()
        if  self.previous_text['speaker_id'] is None or self.previous_text['speaker_id'] == speaker_id:
            self.previous_text['speaker_id'] = speaker_id
            self.previous_text['text'] += f' {text}'
            # If self.previous_text['text'] contains complete sentences, extract and run TTS on them
            if any(punct in self.previous_text['text'] for punct in ['.', '!', '?']):
                sentences = []
                temp_sentence = ''
                for char in self.previous_text['text']:
                    temp_sentence += char
                    if char in ['.', '!', '?']:
                        sentences.append(temp_sentence.strip())
                        temp_sentence = ''
                # Keep incomplete sentence fragments
                self.previous_text['text'] = temp_sentence.strip()
                audio_to_return = np.array([], dtype=np.float32)
                for sentence in sentences:
                    audio = self.get_audio(self.previous_text['speaker_id'], sentence)
                    if audio is not None:
                        audio_to_return = np.concatenate((audio_to_return, audio))
                return audio_to_return
            elif len(self.previous_text['text'].strip().split(' ')) >= 10:
                # Also trigger TTS when word count reaches a threshold
                audio_to_return = self.get_audio(self.previous_text['speaker_id'], self.previous_text['text'])
                self.previous_text['speaker_id'] = None
                self.previous_text['text'] = ''
                return audio_to_return

        elif self.previous_text['speaker_id'] != speaker_id:
            # Speaker switched: first run TTS on previous text
            audio_to_return = np.array([], dtype=np.float32)
            if self.previous_text['text'].strip() != '':
                audio_to_return = self.get_audio(self.previous_text['speaker_id'], self.previous_text['text'])
            # Then update previous_text
            self.previous_text['speaker_id'] = speaker_id
            self.previous_text['text'] = text
            return audio_to_return

    def post_audio_generation(self, commentary_history):
        audios = []
        for commentary in commentary_history:
            text = ' '.join(commentary['word_list'])
            begin_second = commentary['begin_second']
            assistant_id = commentary['assistant_id']
            end_second = commentary['end_second']
            gen = self.tts_pipeline(text, voice=self.assistant_voices[assistant_id], speed=1.6)
            audio_chunks = []
            for _, _, audio in gen:
                audio_chunks.append(audio)
            if len(audio_chunks) > 0:
                audio = np.concatenate(audio_chunks)
                # audio_1s = enforce_exact_duration(audio, sr=24000)
                target_length = TARGET_SAMPLES * (end_second - begin_second)
                cur_len = len(audio)
                if cur_len > target_length:
                    audio = audio[:target_length]
                    logger.info(f'TTS audio for segment {begin_second}-{end_second}s is longer than target length, truncating.')
                elif cur_len < target_length:
                    pad = target_length - cur_len
                    audio = np.pad(audio, (0, pad), mode="constant")
                    logger.info(f'TTS audio for segment {begin_second}-{end_second}s is shorter than target length, padding.')
                audios.append(audio)
                # sf.write(os.path.join(self.session_output_dir, f'{begin_second}_{end_second}.wav'), audio, samplerate=TARGET_SR)
        # merge all audio segments
        if len(audios) > 0:
            final_audio = np.concatenate(audios)
            sf.write(os.path.join(self.session_output_dir, f'final_commentary.wav'), final_audio, samplerate=TARGET_SR)


    # # TODO
    # def forward_tts(self, text, assistant_id, begin_second):
    #     # convert text to speech using kokoro, and save to wav file under session output dir with name format '{begin_second}.wav'
    #     self.word_buffers.extend(_split_words(text))
    #     say_text = ' '.join(self.word_buffers)
    #     gen = self.tts_pipeline(say_text, voice=self.assistant_voices[assistant_id], speed=1.5)
    #     audio_chunks = []
    #     for _, _, audio in gen:
    #         audio_chunks.append(audio)
    #     if len(audio_chunks) > 0:
    #         audio = np.concatenate(audio_chunks)
    #         # audio_1s = enforce_exact_duration(audio, sr=24000)
    #         sf.write(os.path.join(self.session_output_dir, f'{begin_second}.wav'), audio, samplerate=TARGET_SR)

    #     self.word_buffers = []

    def merge_tts_segments(self):
        pass

    def clear_session(self):
        pass

    def set_session(self, session_id, session_output_dir):
        self.session_id = session_id
        self.session_output_dir = session_output_dir
        os.makedirs(self.session_output_dir, exist_ok=True)

def enforce_exact_duration(wav: np.ndarray, sr: int) -> np.ndarray:
    """
    Force input audio to exactly 1 second (24kHz), i.e., 24000 samples.
    """
    # Resample to 24kHz
    if sr != TARGET_SR:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    cur_len = len(wav)
    if cur_len == TARGET_SAMPLES:
        return wav.astype(np.float32)

    # Compute stretch ratio: rate>1 => faster playback => shorter audio
    rate = cur_len / float(TARGET_SAMPLES)
    if not np.isclose(rate, 1.0, atol=1e-3):
        wav = librosa.effects.time_stretch(wav.astype(np.float32), rate=rate)

    # Exact trim/pad
    if len(wav) > TARGET_SAMPLES:
        wav = wav[:TARGET_SAMPLES]
    elif len(wav) < TARGET_SAMPLES:
        pad = TARGET_SAMPLES - len(wav)
        wav = np.pad(wav, (0, pad), mode="constant")

    return wav[:TARGET_SAMPLES].astype(np.float32)
