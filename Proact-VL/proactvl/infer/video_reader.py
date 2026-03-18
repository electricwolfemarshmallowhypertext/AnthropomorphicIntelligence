from proactvl.utils.proact_process import process_interleave_mm_info
from qwen_vl_utils import process_vision_info as process_vision_info_vl

MIN_PIXELS = 128*28*28
MAX_PIXELS = 540 * 28 * 28

class VideoReader:
    def __init__(self, video_path, video_begin, video_end, processor, read_duration=60):
        self.video_path = video_path
        self.video_begin = video_begin
        self.video_end = video_end
        self.read_duration = read_duration
        self.processor = processor

        self.vision_info = None
        self.processor_inputs = None
        self.videos = None
        self.video_metadatas = None

        self.process_video_begin = video_begin
        self.process_video_end = video_begin
        self.process_vision_info(self.process_video_begin)

    # Read [video_begin, min(video_begin + read_duration, self.video_end))
    def process_vision_info(self, video_begin):
        self.process_video_begin = video_begin
        self.process_video_end = min(video_begin + self.read_duration, self.video_end)
        print(f'Reading video {self.video_path} from {self.process_video_begin} to {self.process_video_end}')

        # ✅ 2 frames per second
        chunk_seconds = self.process_video_end - video_begin
        nframes = chunk_seconds * 2

        self.vision_info = [
            {
                "role": "user",
                "content": [{
                    "type": "video",
                    "video": self.video_path,
                    "video_start": video_begin,
                    "video_end": self.process_video_end,
                    "nframes": nframes,
                    "min_pixels": MIN_PIXELS,
                    "max_pixels": MAX_PIXELS,
                }]
            }
        ]
        print(self.processor.__class__.__name__)
        if self.processor.__class__.__name__ == 'Qwen2_5OmniProcessor':
            audios, images, videos = process_interleave_mm_info(self.vision_info, False, return_video_kwargs=False)
            self.videos = videos
            self.processor_inputs = {
                'audios': audios,
                'images': images,
                'videos': videos,
            }

        elif self.processor.__class__.__name__ == 'Qwen3VLProcessor':
            images, videos, video_kwargs = process_vision_info_vl(
                self.vision_info,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True
            )

            if videos is not None:
                videos, video_metadatas = zip(*videos)
                videos, video_metadatas = list(videos), list(video_metadatas)
            else:
                video_metadatas = None

            # ✅ Split into chunks of 2 frames
            chunk_length = len(videos[0]) // 2
            videos = [videos[0][i:i+2] for i in range(0, len(videos[0]), 2)]
            video_metadatas = [{
                'fps': video_metadatas[0]['fps'],
                'frames_indices': video_metadatas[0]['frames_indices'][i*2:i*2+2],
                'total_num_frames': video_metadatas[0]['total_num_frames'],
                'video_backend': video_metadatas[0]['video_backend'],
            } for i in range(chunk_length)]

            self.videos = videos
            self.video_metadatas = video_metadatas
            self.processor_inputs = {
                'images': None,
                'videos': videos,
                'video_metadata': video_metadatas,
                'return_tensors': 'pt',
                'do_resize': False,
                **video_kwargs,
            }

        elif self.processor.__class__.__name__ in ['Qwen2VLProcessor', 'Qwen2_5_VLProcessor']:
            images, videos, video_kwargs = process_vision_info_vl(self.vision_info, return_video_kwargs=True)
            videos = videos[0]
            videos = [videos[i:i+2] for i in range(0, len(videos), 2)]

            size = {'shortest_edge': MIN_PIXELS, 'longest_edge': MAX_PIXELS}
            self.videos = videos
            self.processor_inputs = {
                'images': None,
                'videos': videos,
                'return_tensors': "pt",
                'padding': True,
                'size': size,
            }
        else:
            raise NotImplementedError

    def get_inputs(self, sec_idx):
        # ✅ If beyond current chunk, roll to the next segment
        if sec_idx >= self.process_video_end:
            self.process_vision_info(self.process_video_end)

        # ✅ Offset of current second in chunk (equivalent to i in the second code version)
        i = sec_idx - self.process_video_begin

        # ✅ Prevent out-of-bounds when the last segment is shorter than read_duration
        if i < 0 or i >= len(self.videos):
            raise IndexError(f"sec_idx={sec_idx} out of current chunk range: "
                            f"[{self.process_video_begin}, {self.process_video_end}), "
                            f"i={i}, len(videos)={len(self.videos)}")

        processor_to_return = self.processor_inputs.copy()

        if self.processor.__class__.__name__ == 'Qwen2_5OmniProcessor':
            processor_to_return['videos'] = self.videos[i:i+1]
        elif self.processor.__class__.__name__ == 'Qwen3VLProcessor':
            processor_to_return['videos'] = self.videos[i:i+1]
            processor_to_return['video_metadata'] = self.video_metadatas[i:i+1]
        elif self.processor.__class__.__name__ in ['Qwen2VLProcessor', 'Qwen2_5VLProcessor', 'Qwen2_5_VLProcessor']:
            processor_to_return['videos'] = self.videos[i:i+1]
        else:
            raise NotImplementedError

        return processor_to_return
