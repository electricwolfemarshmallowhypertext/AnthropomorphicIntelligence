import json
import os
import time
import cv2
import base64
import subprocess
import argparse
import re
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from utils.tools import AudioTools
from core.model import ASRModel, ChatModelClient, FineGrainedASRModel
from dataclasses import dataclass
from abc import ABC, abstractmethod
from utils.logger import Logger

logger = Logger("dataset").get_logger()

#----------------------Dataset Class-------------------------#
class BaseVideoDataset(ABC):
    @abstractmethod
    def load_dataset(self, data_path: str, *args, **kwargs) -> list[dict]:
        pass
    
class LOLVideoDataset(BaseVideoDataset):
    def __init__(self, data_path: str) -> None:
        self.data = None
        self.data_path = data_path

    def load_dataset(self, data_path: str) -> list[dict]:
        with open(data_path, "r") as f:
            self.data = json.load(f)

        return self.data
    
#------------------------------------------------------------#
class DataWithNPCSpeaker(BaseVideoDataset):
    def __init__(self, data_path: str) -> None:
        self.data = None
        self.data_path = data_path

    def load_dataset(self) -> None:
        pass

    def clean_speaker(self, output_dir: str = None):
        """
        Find the most frequent speaker in each file, remove segments from other speakers, and save to output_dir
        :param output_dir: Output directory, if None, use the original file directory
        """
        from collections import Counter
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        for json_path in self.data:
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                segments = data[0].get('segments', [])
                speaker_counter = Counter(item.get('speaker') for item in segments if item.get('speaker'))
                
                if speaker_counter:
                    most_frequent = speaker_counter.most_common(1)[0][0]
                    filtered_segments = [item for item in segments if item.get('speaker') == most_frequent]
                    
                    # Create new data structure, preserving original structure
                    new_data = [{
                        **data[0],
                        'segments': filtered_segments
                    }]
                    
                    # Determine output path
                    if output_dir:
                        output_path = os.path.join(output_dir, os.path.basename(json_path))
                    else:
                        output_path = json_path
                    
                    # Save new file
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(new_data, f, ensure_ascii=False, indent=2)
                    
                    results[json_path] = {
                        'most_frequent_speaker': most_frequent,
                        'original_segments': len(segments),
                        'filtered_segments': len(filtered_segments),
                        'output_path': output_path
                    }
                    logger.info(f"{os.path.basename(json_path)}: Keep {most_frequent}, "
                              f"{len(segments)} segments -> {len(filtered_segments)} segments")
                else:
                    logger.warning(f"No speaker information found in {json_path}")
                    
            except Exception as e:
                logger.error(f"Failed to process {json_path}: {e}")
        
        return results

class Cyberpunk(DataWithNPCSpeaker):
    def __init__(self, data_path: str) -> None:
        self.data = None
        self.data_path = data_path
        self.load_dataset()

    def load_dataset(self) -> None:
        if os.path.isdir(self.data_path):
            self.data = [os.path.join(self.data_path, file) for file in os.listdir(self.data_path)]
        else:
            self.data = [self.data_path]

class BlackMythWukong(DataWithNPCSpeaker):
    def __init__(self, data_path: str) -> None:
        self.data = None
        self.data_path = data_path
        self.load_dataset()

    def load_dataset(self) -> None:
        if os.path.isdir(self.data_path):
            self.data = [os.path.join(self.data_path, file) for file in os.listdir(self.data_path)]
        else:
            self.data = [self.data_path]

class EldenRing(DataWithNPCSpeaker):
    def __init__(self, data_path: str) -> None:
        self.data = None
        self.data_path = data_path
        self.load_dataset()

    def load_dataset(self) -> None:
        if os.path.isdir(self.data_path):
            self.data = [os.path.join(self.data_path, file) for file in os.listdir(self.data_path)]
        else:
            self.data = [self.data_path]

class TearsOfTheKingdom(DataWithNPCSpeaker):
    def __init__(self, data_path: str) -> None:
        self.data = None
        self.data_path = data_path
        self.load_dataset()

    def load_dataset(self) -> None:
        if os.path.isdir(self.data_path):
            self.data = [os.path.join(self.data_path, file) for file in os.listdir(self.data_path)]
        else:
            self.data = [self.data_path]

class Starcraft2(DataWithNPCSpeaker):
    def __init__(self, data_path: str) -> None:
        self.data = None
        self.data_path = data_path
        self.load_dataset()

    def load_dataset(self) -> None:
        if os.path.isdir(self.data_path):
            self.data = [os.path.join(self.data_path, file) for file in os.listdir(self.data_path)]
        else:
            self.data = [self.data_path]

class Minecraft(BaseVideoDataset):
    def __init__(
        self, 
        video_path: str, 
        vision_model: str = "qwen3-vl-plus",
        segment_minutes: int = 5,
        system_prompt: str = None
    ) -> None:
        super().__init__()
        self.video_path = video_path
        self.data = []
        self.data_type = [".webm", ".mkv", ".mp4"]
        self.segment_minutes = segment_minutes
        self.system_prompt = None

        if system_prompt and os.path.isfile(system_prompt):
            with open(system_prompt, "r", encoding="utf-8") as f:
                self.system_prompt = f.read()
        else:
            self.system_prompt = system_prompt

        self.client = ChatModelClient(
            base_url="https://dashscope.aliyuncs.com/api/v1",
            model_name=vision_model, 
            system_prompt=self.system_prompt
        )

        self.results = []

    # -------- Utilities for time and JSON sanitization -------- #
    def _format_seconds_to_mmss(self, seconds: float) -> str:
        try:
            total = int(round(float(seconds)))
            m = total // 60
            s = total % 60
            return f"{m}:{s:02d}"
        except Exception:
            return str(seconds)

    def _strip_markdown_code_fences(self, text: str) -> str:
        """
        Remove fences like ```json ... ``` or ``` ... ```.
        Only process strings; return non-strings unchanged.
        """
        if not isinstance(text, str):
            return text
        s = text.strip()
        if s.startswith("```"):
            # Remove the opening ```json or ``` marker.
            s = s[3:]
            if s.lower().startswith("json"):
                s = s[4:]
            # Remove the leading newline.
            s = s.lstrip("\n\r ")
            # Remove the trailing ```.
            if s.endswith("```"):
                s = s[:-3]
            s = s.strip()
        return s

    def _coerce_json(self, payload):
        """
        Best-effort parse a model response into a Python object: list/dict.
        - For strings, remove markdown fences first, then call json.loads.
        - If parsing fails, return the original value.
        - If the top level is a dict containing actions, return that field.
        """
        try:
            obj = payload
            if isinstance(payload, str):
                stripped = self._strip_markdown_code_fences(payload)
                try:
                    obj = json.loads(stripped)
                except Exception:
                    obj = payload  # Keep the original value.
            # If the result is a dict with actions, return that field.
            if isinstance(obj, dict) and "actions" in obj:
                return obj.get("actions")
            return obj
        except Exception:
            return payload

    def _offset_time_string(self, time_str, offset_seconds: float):
        sec = self._parse_time_to_seconds(time_str)
        if sec is None:
            return time_str
        return self._format_seconds_to_mmss(sec + (offset_seconds or 0))

    def _offset_atom_actions_times(self, atom_actions, action_begin_seconds: float):
        """
        Add the top-level action_begin_time offset to all time fields in atom_actions.
        Only common fields are handled: action_begin_time, action_end_time,
        start_time, end_time, time, and timestamp.
        """
        if not isinstance(atom_actions, list):
            return atom_actions
        updated = []
        for item in atom_actions:
            if not isinstance(item, dict):
                updated.append(item)
                continue
            new_item = dict(item)
            for key in ["action_begin_time", "action_end_time", "start_time", "end_time", "time", "timestamp"]:
                if key in new_item:
                    new_item[key] = self._offset_time_string(new_item[key], action_begin_seconds)
            updated.append(new_item)
        return updated

    def _parse_time_to_seconds(self, t):
        """
        Convert a time string such as "MM:SS", "M:SS", or "SS" to seconds (float).
        Return numeric inputs directly.
        """
        if t is None:
            return None
        if isinstance(t, (int, float)):
            return float(t)
        if isinstance(t, str):
            s = t.strip()
            if ':' in s:
                parts = s.split(':')
                if len(parts) == 2:
                    try:
                        minutes = int(parts[0])
                        seconds = int(parts[1])
                        return float(minutes * 60 + seconds)
                    except Exception:
                        pass
                    
                if len(parts) == 3:
                    try:
                        hours = int(parts[0])
                        minutes = int(parts[1])
                        seconds = int(parts[2])
                        return float(hours * 3600 + minutes * 60 + seconds)
                    except Exception:
                        pass
            try:
                return float(s)
            except Exception:
                return None
        return None

    def load_dataset(self):
        if os.path.isdir(self.video_path):
            self.data = [
                os.path.join(self.video_path, f)
                for f in os.listdir(self.video_path)
                if any(f.endswith(ext) for ext in self.data_type)
            ]
        elif os.path.isfile(self.video_path):
            self.data = [self.video_path]
        else:
            raise ValueError(f"Invalid data path: {self.video_path}")
    
    def get_video_duration(self, video_path: str):
        """
        Get video duration (seconds)
        :param video_path: Video file path
        :return: video duration (seconds)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video file: {video_path}")
            return 0
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        return duration
    
    def cut_video_segment(self, video_path: str, start_time: float, end_time: float, output_path: str):
        """
        Cut a video segment with ffmpeg.
        :param video_path: Source video path.
        :param start_time: Start time in seconds.
        :param end_time: End time in seconds.
        :param output_path: Output video path.
        :return: The output path on success, or None on failure.
        """
        try:
            # Ensure the output directory exists.
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Compute the segment duration.
            duration = end_time - start_time
            
            # Cut the segment with ffmpeg without re-encoding for speed.
            command = [
                'ffmpeg',
                '-i', video_path,
                '-ss', str(start_time),
                '-t', str(duration),
                '-c', 'copy',  # Copy streams directly without re-encoding.
                '-strict', '-2',  # Allow experimental codecs such as Opus in MP4.
                '-y',  # Overwrite the output file.
                output_path
            ]
            
            print(f"Cutting video: {start_time:.2f}s - {end_time:.2f}s -> {output_path}")
            
            # Run the command.
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if result.returncode == 0 and os.path.exists(output_path):
                print(f"✓ Video segment created successfully: {output_path}")
                return output_path
            else:
                print(f"✗ Video segment creation failed: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"Error while cutting video: {e}")
            return None
    
    def extract_atom_action(
        self, 
        json_path: str,
        save_dir: str = None
    ) -> list[dict]:
        """
        Extract atom action from the JSON file.
        :param json_path: The path to the JSON file.
        :return: A list of atom actions.
        """
        # Accept either a file path or a directory path.
        json_files = []
        if os.path.isdir(json_path):
            json_files = [os.path.join(json_path, f) for f in os.listdir(json_path) if f.endswith('.json')]
        elif os.path.isfile(json_path) and json_path.endswith('.json'):
            json_files = [json_path]
        else:
            raise ValueError(f"Invalid json_path: {json_path}")

        results = []
        all_file_summaries = []
        global_out_dir = None
        # Default output directory: action_infer next to the JSON file.
        def ensure_save_dir(base_json_file: str):
            nonlocal save_dir
            if save_dir is None:
                base_dir = os.path.dirname(base_json_file)
                out_dir = os.path.join(base_dir, "action_infer")
            else:
                out_dir = save_dir
            os.makedirs(out_dir, exist_ok=True)
            return out_dir
        for file_path in json_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                logger.error(f"Failed to read JSON: {file_path}, {e}")
                continue

            video_path = data.get("video_path")
            if not video_path or not os.path.exists(video_path):
                logger.error(f"Invalid video path: {video_path}")
                continue

            actions = data.get("actions", [])
            logger.info(f"Processing file: {os.path.basename(file_path)}, action count: {len(actions)}")

            per_file_actions = []
            out_dir = ensure_save_dir(file_path)

            for idx, action in enumerate(actions, start=1):
                begin_raw = action.get("action_begin_time")
                end_raw = action.get("action_end_time")
                seg_num = action.get("segment_number")
                seg_start = action.get("segment_start_time", 0)
                seg_end = action.get("segment_end_time")
                question = action.get("player_question")
                guidance = action.get("assistant_guidance")
                
                begin_sec = self._parse_time_to_seconds(begin_raw)
                end_sec = self._parse_time_to_seconds(end_raw)

                if begin_sec is None or end_sec is None:
                    logger.warning(f"Action {idx}: failed to parse time begin={begin_raw}, end={end_raw}; skipping")
                    continue
                if end_sec <= begin_sec:
                    logger.warning(f"Action {idx}: end time is not greater than start time begin={begin_sec}, end={end_sec}; skipping")
                    continue
                patch_prompt = ""
                input_prompt = self.system_prompt + patch_prompt.format(begin_time=begin_raw, end_time=end_raw, question=question, guidance=guidance)
                try:
                    response = self.process_with_vision_model(
                        video_path=video_path,
                        prompt=input_prompt,
                        segment_number=seg_num,
                        start_time=begin_sec,
                        end_time=end_sec,
                        adjust_timestamps=False
                    )
                except Exception as e:
                    logger.error(f"Action {idx}: processing failed {e}")
                    continue

                if response is None:
                    logger.warning(f"Action {idx}: empty response; skipping")
                    continue

                # Parse the model response into atom actions, removing ```json fences when present.
                resp_content = response.get("response")
                loaded = self._coerce_json(resp_content)
                if isinstance(loaded, list):
                    atom_actions = loaded
                elif isinstance(loaded, dict) and "actions" in loaded:
                    atom_actions = loaded.get("actions")
                else:
                    atom_actions = loaded

                # Ensure the final value is a list to avoid upstream type errors.
                if not isinstance(atom_actions, list):
                    atom_actions = [atom_actions] if atom_actions is not None else []

                # Apply the top-level action_begin_time offset to atom action timestamps.
                atom_actions = self._offset_atom_actions_times(atom_actions, begin_sec)

                # Build the expected action structure: original action info plus atom actions.
                aug_action = {
                    "action_begin_time": begin_raw,
                    "action_end_time": end_raw,
                    "player_question": question,
                    "assistant_guidance": guidance,
                    "segment_number": seg_num,
                    "segment_start_time": seg_start,
                    "segment_end_time": seg_end,
                    "atom_actions": atom_actions
                }

                results.append(aug_action)
                per_file_actions.append(aug_action)

                # Save the result for each action.
                try:
                    base_name = os.path.basename(file_path).rsplit('.', 1)[0]
                    action_fname = f"{base_name}_seg{seg_num}_act{idx}_{int(begin_sec)}_{int(end_sec)}.json"
                    action_fpath = os.path.join(out_dir, action_fname)
                    with open(action_fpath, "w", encoding="utf-8") as wf:
                        json.dump(aug_action, wf, ensure_ascii=False, indent=2)
                except Exception as e:
                    logger.error(f"Failed to save action result: {e}")

            # Save the summary for the current JSON file.
            try:
                base_name = os.path.basename(file_path).rsplit('.', 1)[0]
                summary_fname = f"{base_name}__actions_infer_summary.json"
                summary_fpath = os.path.join(out_dir, summary_fname)

                # Copy the original JSON, replace actions, and refresh total_actions.
                top_output = data.copy()
                top_output["actions"] = per_file_actions
                try:
                    top_output["total_actions"] = len(per_file_actions)
                except Exception:
                    pass

                with open(summary_fpath, "w", encoding="utf-8") as wf:
                    json.dump(top_output, wf, ensure_ascii=False, indent=2)
                all_file_summaries.append(top_output)
                if global_out_dir is None:
                    global_out_dir = out_dir
            except Exception as e:
                logger.error(f"Failed to save summary result: {e}")

        # Save the merged summary across all source files.
        try:
            if all_file_summaries:
                total_actions = 0
                for item in all_file_summaries:
                    try:
                        total_actions += int(item.get("total_actions", 0))
                    except Exception:
                        pass
                merged_payload = {
                    "source_root": json_path,
                    "total_files": len(all_file_summaries),
                    "total_actions": total_actions,
                    "files": all_file_summaries,
                }
                merged_path = os.path.join(global_out_dir or (save_dir or os.path.dirname(json_files[0])), "all_jsons_all_actions_results.json")
                with open(merged_path, "w", encoding="utf-8") as wf:
                    json.dump(merged_payload, wf, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save merged summary file: {e}")

        return results
    
    def refine_atom_action(self, json_path: str, save_dir: str = None):
        """
        Refine atom action from the JSON file.
        Supports both single video JSON and merged multi-video JSON (all_jsons_all_actions_results.json).
        Supports incremental processing: checks existing results and only processes missing actions.
        
        :param json_path: The path to the JSON file (single video or all_jsons_all_actions_results.json)
        :param save_dir: The output directory (default: same as extract_atom_action output structure)
        :return: A list of refined atom actions.
        """
        refine_client = ChatModelClient(
            model_name="openai/gpt-4o-mini", 
            system_prompt=self.system_prompt
        )
        
        # Load the input file.
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read JSON: {json_path}, {e}")
            return []
        
        # Detect the JSON structure: single-video object vs. multi-video summary.
        if isinstance(data, dict) and "files" in data:
            # Summary file structure (all_jsons_all_actions_results.json).
            logger.info(f"Detected a summary JSON file containing {len(data.get('files', []))} videos")
            videos_to_process = data.get("files", [])
            is_summary = True
        elif isinstance(data, dict) and "actions" in data:
            # Single-video object structure.
            videos_to_process = [data]
            is_summary = False
        else:
            logger.error(f"Unrecognized JSON structure: {json_path}")
            return []
        
        # Resolve the output directory.
        if save_dir is None:
            base_dir = os.path.dirname(json_path)
            out_dir = os.path.join(base_dir if base_dir else ".", "refine_atom")
        else:
            out_dir = save_dir
        os.makedirs(out_dir, exist_ok=True)
        
        # Incremental mode: inspect existing results first.
        def load_existing_results(out_dir: str):
            """Load an existing all_jsons_all_actions_refined_results.json file."""
            existing_file = os.path.join(out_dir, "all_jsons_all_actions_refined_results.json")
            if os.path.exists(existing_file):
                try:
                    with open(existing_file, "r", encoding="utf-8") as f:
                        existing_data = json.load(f)
                    logger.info(f"Found existing refined result file: {existing_file}")
                    return existing_data
                except Exception as e:
                    logger.warning(f"Failed to read existing result file: {e}")
            return None
        
        def get_processed_actions(existing_data, video_path: str):
            """Get the set of already processed actions for one video."""
            if not existing_data or "files" not in existing_data:
                return set()
            
            for video in existing_data.get("files", []):
                if video.get("video_path") == video_path:
                    processed = set()
                    for action in video.get("actions", []):
                        key = (
                            action.get("action_begin_time"),
                            action.get("action_end_time"),
                            action.get("segment_number")
                        )
                        processed.add(key)
                    return processed
            return set()
        
        # Load existing results.
        existing_data = load_existing_results(out_dir)
        if existing_data:
            logger.info("Incremental processing mode enabled")
            logger.info(f"Existing results: {existing_data.get('total_files')} videos, {existing_data.get('total_actions')} actions")
            existing_video_map = {}
            for video in existing_data.get("files", []):
                vpath = video.get("video_path")
                existing_video_map[vpath] = video
        else:
            logger.info("Full processing mode enabled")
            existing_video_map = {}
        
        # Process all videos.
        all_file_summaries = []
        global_out_dir = out_dir
        
        for video_idx, video_data in enumerate(videos_to_process, start=1):
            video_path = video_data.get("video_path")
            if not video_path:
                logger.error("Missing video path; skipping")
                continue
            
            video_name = os.path.basename(video_path).rsplit('.', 1)[0]
            actions = video_data.get("actions", [])
            
            # Incremental mode: check which actions for this video are already done.
            processed_action_keys = get_processed_actions(existing_data, video_path)
            existing_video_actions = existing_video_map.get(video_path, {}).get("actions", [])
            
            # Collect missing actions.
            total_actions_count = len(actions)
            missing_actions = []
            for idx, action in enumerate(actions, start=1):
                key = (
                    action.get("action_begin_time"),
                    action.get("action_end_time"),
                    action.get("segment_number")
                )
                if key not in processed_action_keys:
                    missing_actions.append((idx, action))
            
            if processed_action_keys:
                logger.info(f"Processing video [{video_idx}/{len(videos_to_process)}]: {video_name}")
                logger.info(f"  Total actions: {total_actions_count}, completed: {len(processed_action_keys)}, missing: {len(missing_actions)}")
            else:
                logger.info(f"Processing video [{video_idx}/{len(videos_to_process)}]: {video_name}, action count: {len(actions)}")
            
            # Reuse existing results directly if nothing is missing.
            if not missing_actions and existing_video_actions:
                logger.info("  Skipping; using existing results")
                all_file_summaries.append(existing_video_map[video_path])
                continue
            
            # Create a subdirectory for the current video.
            video_subdir = os.path.join(out_dir, video_name)
            os.makedirs(video_subdir, exist_ok=True)
            
            # Process missing actions.
            per_video_actions = list(existing_video_actions)  # Start from the existing results.
            actions_to_process = missing_actions if missing_actions else [(idx+1, action) for idx, action in enumerate(actions)]
            
            for idx, action in actions_to_process:
                try:
                    # Call the refinement model.
                    content = json.dumps(action, ensure_ascii=False)
                    response = refine_client(
                        content=content,
                        type="text"
                    )
                    
                    # Parse the response.
                    parsed = self._coerce_json(response)
                    
                    # Build the refined action object.
                    new_action = dict(action)
                    new_action["atom_actions"] = parsed
                    if "refined_response" in new_action:
                        del new_action["refined_response"]
                    
                    per_video_actions.append(new_action)
                    
                    # Save each action result under the video subdirectory.
                    seg_num = action.get("segment_number")
                    begin_sec = self._parse_time_to_seconds(action.get("action_begin_time"))
                    end_sec = self._parse_time_to_seconds(action.get("action_end_time"))
                    
                    action_fname = f"{video_name}_seg{seg_num}_act{idx}_{int(begin_sec) if begin_sec else 0}_{int(end_sec) if end_sec else 0}_refined.json"
                    action_fpath = os.path.join(video_subdir, action_fname)
                    with open(action_fpath, "w", encoding="utf-8") as wf:
                        json.dump(new_action, wf, ensure_ascii=False, indent=2)
                    
                    logger.info(f"  ✓ Action {idx} refined successfully")
                    
                except Exception as e:
                    logger.error(f"Video {video_name} action {idx}: refinement failed {e}")
                    # Preserve the original action on failure.
                    per_video_actions.append(action)
            
            # Save the per-video summary into the video subdirectory.
            try:
                summary_fname = f"{video_name}__actions_refined_summary.json"
                summary_fpath = os.path.join(video_subdir, summary_fname)
                
                video_output = video_data.copy()
                video_output["actions"] = per_video_actions
                video_output["total_actions"] = len(per_video_actions)
                
                with open(summary_fpath, "w", encoding="utf-8") as wf:
                    json.dump(video_output, wf, ensure_ascii=False, indent=2)
                all_file_summaries.append(video_output)
                logger.info(f"Saved refined summary for video {video_name}: {summary_fpath}")
            except Exception as e:
                logger.error(f"Failed to save refined video summary: {e}")
        
        # Save the merged summary across all videos.
        try:
            if all_file_summaries:
                total_actions = sum(int(item.get("total_actions", 0)) for item in all_file_summaries)
                merged_payload = {
                    "source_root": json_path,
                    "total_files": len(all_file_summaries),
                    "total_actions": total_actions,
                    "files": all_file_summaries,
                }
                merged_path = os.path.join(global_out_dir, "all_jsons_all_actions_refined_results.json")
                
                # Back up the previous file.
                if os.path.exists(merged_path):
                    import shutil
                    import datetime
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_path = merged_path.replace(".json", f"_backup_{timestamp}.json")
                    shutil.copy(merged_path, backup_path)
                    logger.info(f"Backed up the old file to: {backup_path}")
                
                # Save the new result.
                with open(merged_path, "w", encoding="utf-8") as wf:
                    json.dump(merged_payload, wf, ensure_ascii=False, indent=2)
                logger.info(f"Saved full refined result to: {merged_path}")
                logger.info(f"Final statistics: {len(all_file_summaries)} videos, {total_actions} actions")
        except Exception as e:
            logger.error(f"Failed to save the full refined summary file: {e}")
        
        # Return all refined actions.
        all_refined_actions = []
        for video in all_file_summaries:
            all_refined_actions.extend(video.get("actions", []))
        return all_refined_actions
        
    
    def adjust_timestamps_in_response(self, response: str, time_offset: float):
        """
        Adjust timestamps in an API response by applying the segment time offset.
        :param response: Raw API response, possibly a string or JSON-like object.
        :param time_offset: Time offset in seconds.
        :return: The adjusted response.
        """
        import re
        
        if time_offset == 0:
            return response
        
        print(f"  - Adjusting timestamps: adding offset {time_offset:.2f}s ({time_offset/60:.2f} min)")
        
        try:
            # Try to parse the response as JSON.
            if isinstance(response, str):
                try:
                    response_json = json.loads(response)
                except:
                    response_json = response
            else:
                response_json = response
            
            # If the response is a string, try to rewrite timestamp patterns inside it.
            if isinstance(response_json, str):
                # Find timestamps in MM:SS format.
                def adjust_time_match(match):
                    time_str = match.group(0)
                    # Parse MM:SS or M:SS timestamps.
                    parts = time_str.split(':')
                    if len(parts) == 2:
                        minutes = int(parts[0])
                        seconds = int(parts[1])
                        total_seconds = minutes * 60 + seconds + time_offset
                        new_minutes = int(total_seconds // 60)
                        new_seconds = int(total_seconds % 60)
                        return f"{new_minutes}:{new_seconds:02d}"
                    return time_str
                
                # Match timestamps in MM:SS format.
                adjusted_response = re.sub(r'\b\d{1,2}:\d{2}\b', adjust_time_match, response_json)
                return adjusted_response
            
            # Recurse into lists and dicts.
            elif isinstance(response_json, list):
                adjusted_list = []
                for item in response_json:
                    adjusted_item = self._adjust_item_timestamps(item, time_offset)
                    adjusted_list.append(adjusted_item)
                return adjusted_list
            
            elif isinstance(response_json, dict):
                return self._adjust_item_timestamps(response_json, time_offset)
            
            else:
                return response
                
        except Exception as e:
            print(f"  ⚠️ Failed to adjust timestamps: {e}; returning the original response")
            return response
    
    def _adjust_item_timestamps(self, item: dict, time_offset: float):
        """
        Recursively adjust timestamp fields in a dictionary.
        :param item: Dictionary item.
        :param time_offset: Time offset in seconds.
        :return: Adjusted dictionary.
        """
        if not isinstance(item, dict):
            return item
        
        adjusted_item = item.copy()
        
        # Common timestamp field names.
        time_fields = [
            'action_begin_time', 'action_end_time',
            'start_time', 'end_time',
            'begin_time', 'timestamp', 'time'
        ]
        
        for field in time_fields:
            if field in adjusted_item:
                value = adjusted_item[field]
                # Handle MM:SS strings.
                if isinstance(value, str) and ':' in value:
                    parts = value.split(':')
                    if len(parts) == 2:
                        try:
                            minutes = int(parts[0])
                            seconds = int(parts[1])
                            total_seconds = minutes * 60 + seconds + time_offset
                            new_minutes = int(total_seconds // 60)
                            new_seconds = int(total_seconds % 60)
                            adjusted_item[field] = f"{new_minutes}:{new_seconds:02d}"
                        except:
                            pass
                # Handle numeric values in seconds.
                elif isinstance(value, (int, float)):
                    adjusted_item[field] = value + time_offset
        
        return adjusted_item
    
    def extract_frames_1fps(self, video_path: str, sec_interval: int = 1, start_time: float = 0, end_time: float = None):
        """
        Extract frames from a video at 1 fps within an optional time range.
        :param video_path: Video file path.
        :param sec_interval: Frame extraction interval in seconds.
        :param start_time: Start time in seconds.
        :param end_time: End time in seconds, or None for the end of the video.
        :return: A list of base64-encoded images.
        """
        image_list = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Failed to open video file: {video_path}")
            return image_list
        
        # Read the FPS and total duration.
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_duration = total_frames / fps if fps > 0 else 0
        
        # Use the full duration when no end time is provided.
        if end_time is None:
            end_time = total_duration
        
        print(f"Video FPS: {fps}, extraction range: {start_time:.2f}s - {end_time:.2f}s")
        
        # Determine the frame interval.
        frame_interval = int(sec_interval * fps)
        
        # Seek to the starting frame.
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frame_count = start_frame
        extracted_count = 0
        
        while frame_count < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract frames at the requested interval.
            if (frame_count - start_frame) % frame_interval == 0:
                # Encode the frame as JPEG.
                _, buffer = cv2.imencode('.jpg', frame)
                # Convert the JPEG bytes to base64.
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                image_list.append(img_base64)
                extracted_count += 1
                
            frame_count += 1
        
        cap.release()
        print(f"Extracted {extracted_count} frames ({start_time:.2f}s to {end_time:.2f}s)")
        return image_list
    
    def process_with_vision_model(
        self, video_path: str, 
        prompt: str = None, 
        segment_number: int = None, 
        start_time: float = None, 
        end_time: float = None,
        adjust_timestamps: bool = True
    ):
        """
        Process Minecraft video: first extract frames at 1 fps, then use a multimodal vision model to process all frames.
        Supports processing by video segments.

        Args:
            video_path (str): Path to the video file.
            prompt (str, optional): Prompt for the model. Defaults to None.
            segment_number (int, optional): Segment index for identification. Defaults to None.
            start_time (float, optional): Start time of the segment in seconds. Defaults to None.
            end_time (float, optional): End time of the segment in seconds. Defaults to None.

        Returns:
            Model response result.
        """

        print(f"Prompt: {prompt}")
        segment_info = f" (segment {segment_number})" if segment_number is not None else ""
        print(f"Starting video processing{segment_info}: {video_path}")
        
        # Step1: cut segment
        cut_video_path = video_path
        if start_time is not None and end_time is not None:
            # Build the output path for the cut video segment.
            video_dir = os.path.dirname(video_path)
            video_basename = os.path.basename(video_path).split('.')[0]
            video_ext = os.path.splitext(video_path)[1]
            
            # Create the directory for video segments.
            segments_dir = os.path.join(video_dir, "video_segments")
            cut_video_filename = f"{video_basename}_segment_{segment_number}_{int(start_time)}_{int(end_time)}{video_ext}"
            cut_video_path = os.path.join(segments_dir, cut_video_filename)
            
            print(f"Step 1: cutting the video segment ({start_time:.2f}s - {end_time:.2f}s)...")
            cut_video_result = self.cut_video_segment(video_path, start_time, end_time, cut_video_path)
            
            if cut_video_result is None:
                print("Video segment creation failed; falling back to the original video")
                cut_video_path = video_path
            else:
                print(f"✓ Video segment saved: {cut_video_path}")
        
        # Step 1.5: Compress the cut video and remove audio
        compressed_path = cut_video_path.replace(".mp4", "_compressed.mp4")
        command = [
            'ffmpeg',
            '-i', cut_video_path,
            '-vcodec', 'libx264',
            '-crf', '28',
            '-an',  # Remove audio
            '-y',  # Overwrite
            compressed_path
        ]
        try:
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"✓ Video compression succeeded: {compressed_path}")
            cut_video_path = compressed_path  # Use compressed video for further processing
        except subprocess.CalledProcessError as e:
            print(f"✗ Video compression failed: {e.stderr.decode()}")
            # Continue with original cut video if compression fails

        # Step 2: extract frames at 1 fps for fallback or image-based mode.
        print("Step 2: extracting frames...")
        image_list = self.extract_frames_1fps(video_path, start_time=start_time or 0, end_time=end_time)
        
        if not image_list:
            print("Frame extraction failed; no images were extracted")
            # return None  # Disabled because video mode does not always need images.
        
        # Step 3: call the multimodal vision model API.
        print("Step 3: preparing to call the vision model")
        print(f"  - Model: {self.client.model_name}")
        print(f"  - Video path: {cut_video_path}")
        
        try:
            # Use DashScope's video mode.
            result = self.client(
                content=prompt,
                type="video",
                video_url=cut_video_path
            )
            print("✓ Successfully received the model response")
            
            # Adjust timestamps if requested and this is a segmented video.
            time_offset = start_time or 0
            if adjust_timestamps and time_offset > 0:
                print("Step 4: adjusting timestamps in the response")
                adjusted_result = self.adjust_timestamps_in_response(result, time_offset)
            else:
                adjusted_result = result
            
            response_data = {
                "video_path": video_path,
                "cut_video_path": cut_video_path if cut_video_path != video_path else None,
                "frames_count": len(image_list),
                "response": adjusted_result,
                "original_response": result if time_offset > 0 else None,
                "time_offset_seconds": time_offset,
                "model": self.client.model_name
            }
            
            # Add segment metadata.
            if segment_number is not None:
                response_data["segment_number"] = segment_number
                response_data["segment_start_time"] = start_time or 0
                response_data["segment_end_time"] = end_time
            
            return response_data
            
        except Exception as e:
            print(f"✗ Error while calling the vision model API: {e}")
            return None
    
    def batch_process_with_vision_model(self, prompt: str = None, save_dir: str = None):
        """
        Batch process all videos in the dataset, with support for automatic segmentation.
        
        Args:
            prompt (str, optional): The prompt or instruction to be used for vision model inference.
            save_dir (str, optional): The directory to save processing results. If None, uses the video's location.
        """
        if save_dir is None:
            save_dir = os.path.dirname(self.video_path) if os.path.isfile(self.video_path) else self.video_path
        
        os.makedirs(save_dir, exist_ok=True)
        
        for video_file in tqdm(self.data, desc="Processing videos", colour="cyan"):
            print(f"\nProcessing Video: {video_file}")
            
            # Get video duration
            duration = self.get_video_duration(video_file)
            print(f"Total video duration: {duration:.2f} s ({duration/60:.2f} min)")
            
            # Calculate how many segments to split
            segment_duration = self.segment_minutes * 60  # Convert minutes to seconds.
            num_segments = int(duration / segment_duration) + (1 if duration % segment_duration > 0 else 0)
            
            print(f"Splitting the video into {num_segments} segments, {self.segment_minutes} minutes each")
            
            video_results = []  # Store the results of all segments of the video
            
            # Process each segment one by one.
            for segment_idx in range(num_segments):
                start_time = segment_idx * segment_duration
                end_time = min((segment_idx + 1) * segment_duration, duration)
                
                print(f"\n--- Processing segment {segment_idx + 1}/{num_segments} ---")
                print(f"Time range: {start_time:.2f}s - {end_time:.2f}s ({start_time/60:.2f}min - {end_time/60:.2f}min)")
                
                result = self.process_with_vision_model(
                    video_file, 
                    prompt, 
                    segment_number=segment_idx + 1,
                    start_time=start_time,
                    end_time=end_time
                )
                
                if result:
                    video_results.append(result)
                    # Save the result of each segment.
                    segment_output_file = os.path.join(
                        save_dir, 
                        f"{os.path.basename(video_file).split('.')[0]}_segment_{segment_idx + 1}_results.json"
                    )
                    with open(segment_output_file, "w", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    print(f"Segment {segment_idx + 1} result saved to: {segment_output_file}")
                else:
                    print(f"Segment {segment_idx + 1} processing failed")
            
            # Summarize all segments of the video
            if video_results:
                video_summary = {
                    "video_path": video_file,
                    "total_duration": duration,
                    "segment_minutes": self.segment_minutes,
                    "total_segments": num_segments,
                    "segments": video_results
                }
                
                self.results.append(video_summary)
                
                # Save the summary of the video
                summary_output_file = os.path.join(
                    save_dir, 
                    f"{os.path.basename(video_file).split('.')[0]}_all_segments_summary.json"
                )
                with open(summary_output_file, "w", encoding="utf-8") as f:
                    json.dump(video_summary, f, ensure_ascii=False, indent=2)
                print(f"\nVideo summary saved to: {summary_output_file}")
        
        # Save all videos results
        if self.results:
            all_results_file = os.path.join(save_dir, "all_videos_all_segments_results.json")
            with open(all_results_file, "w", encoding="utf-8") as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2)
            print(f"\n-------------------------------- All Videos Results Saved to: {all_results_file} --------------------------------")
    
    def post_process(self, json_path: str, save_dir: str = None, token_per_second: int = 4):
        """
        Adjust sub_annotation end_time based on commentary length.
        end_time = begin_time + (word_count / token_per_second)
        Also check for overlap with the next sub_annotation begin_time.
        If there is a conflict, call gpt-4o-mini to shorten the commentary.
        """
        if save_dir is None:
            # By default, save under a refined_results folder next to json_path.
            base_dir = os.path.dirname(json_path) if os.path.isfile(json_path) else json_path
            save_dir = os.path.join(base_dir, "refined_results")
        
        os.makedirs(save_dir, exist_ok=True)

        # Initialize Repolish Client
        repolish_client = None
        prompt_path = "prompt/minecraft/repolish_system_prompt.txt"
        if os.path.exists(prompt_path):
            try:
                with open(prompt_path, "r", encoding="utf-8") as f:
                    sys_prompt = f.read()
                # Initialize client with gpt-4o-mini
                repolish_client = ChatModelClient(
                    model_name="gpt-4o-mini",
                    base_url="https://api.uniapi.io/v1", 
                    system_prompt=sys_prompt
                )
                print("Repolish client initialized with gpt-4o-mini using prompt:", prompt_path)
            except Exception as e:
                print(f"Failed to initialize repolish client: {e}")
        else:
            print(f"Repolish prompt not found at {prompt_path}")

        # Build the input file list.
        if os.path.isfile(json_path):
            json_files = [json_path]
        elif os.path.isdir(json_path):
            json_files = [os.path.join(json_path, f) for f in os.listdir(json_path) if f.endswith('.json')]
        else:
            print(f"Invalid path: {json_path}")
            return

        issues = []

        for file_path in tqdm(json_files, desc="Post-processing Files"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Support either a single object or a list.
                data_list = data if isinstance(data, list) else [data]
                
                for video_idx, video_item in enumerate(data_list):
                    annotations = video_item.get("annotations", [])
                    for anno_idx, annotation in enumerate(annotations):
                        sub_annotations = annotation.get("sub_annotations", [])
                        if not sub_annotations:
                            continue
                        
                        # Always use tqdm so the progress bar remains visible.
                        desc_str = f"Processing Video {video_idx} Anno {anno_idx} Subs"
                        iterator = tqdm(sub_annotations, desc=desc_str, leave=False, mininterval=1.0)

                        for i, sub in enumerate(iterator):
                            commentary = sub.get("commentary", "")
                            begin_time_str = sub.get("begin_time")
                            
                            # Compute the required duration.
                            words = commentary.strip().split()
                            duration = len(words) / token_per_second
                            
                            # Parse the start time.
                            begin_sec = self._parse_time_to_seconds(begin_time_str)
                            if begin_sec is None:
                                continue
                                
                            new_end_sec = begin_sec + duration
                            
                            # Check for overlap with the next item.
                            limit_sec = None
                            
                            if i < len(sub_annotations) - 1:
                                next_sub = sub_annotations[i+1]
                                next_begin_str = next_sub.get("begin_time")
                                next_begin_sec = self._parse_time_to_seconds(next_begin_str)
                                
                                if next_begin_sec is not None:
                                    limit_sec = next_begin_sec
                                    if new_end_sec > next_begin_sec:
                                        # On conflict, try to repolish the commentary.
                                        available_duration = max(0.1, next_begin_sec - begin_sec)
                                        max_words = max(1, int(available_duration * token_per_second))
                                        
                                        repolished = False
                                        if repolish_client:
                                            try:
                                                # Construct input
                                                prompt_input = json.dumps({
                                                    "commentary": commentary,
                                                    "max_words": max_words
                                                }, ensure_ascii=False)
                                                
                                                # Call model
                                                refined_resp = repolish_client(content=prompt_input, type="text")
                                                
                                                # Parse output
                                                refined_json = self._coerce_json(refined_resp)
                                                
                                                if isinstance(refined_json, dict) and "refined_commentary" in refined_json:
                                                    refined_commentary = refined_json["refined_commentary"]
                                                    
                                                    # Log success
                                                    issues.append({
                                                        "file": os.path.basename(file_path),
                                                        "video_idx": video_idx,
                                                        "annotation_idx": anno_idx,
                                                        "sub_idx": i,
                                                        "type": "repolished",
                                                        "original": commentary,
                                                        "refined": refined_commentary,
                                                        "original_end": new_end_sec,
                                                        "limit_end": next_begin_sec
                                                    })
                                                    
                                                    # Update commentary
                                                    sub["commentary"] = refined_commentary
                                                    repolished = True
                                                else:
                                                    issues.append({
                                                        "file": os.path.basename(file_path),
                                                        "type": "repolish_failed_format",
                                                        "response": refined_resp
                                                    })
                                            except Exception as e:
                                                print(f"Repolish error: {e}")
                                                issues.append({
                                                    "file": os.path.basename(file_path),
                                                    "type": "repolish_error",
                                                    "error": str(e)
                                                })

                                        if not repolished:
                                            # Record unresolved conflicts.
                                            issues.append({
                                                "file": os.path.basename(file_path),
                                                "video_idx": video_idx,
                                                "annotation_idx": anno_idx,
                                                "sub_idx": i,
                                                "commentary": commentary,
                                                "original_begin": begin_time_str,
                                                "calculated_end": new_end_sec,
                                                "limit_end": next_begin_sec,
                                                "overflow": new_end_sec - next_begin_sec
                                            })
                                        
                                        # Truncate the end time regardless, to prevent overlap.
                                        new_end_sec = next_begin_sec

                            # Update end_time.
                            # Preserve the original format if it was an MM:SS string.
                            if isinstance(begin_time_str, str) and ":" in begin_time_str:
                                sub["end_time"] = self._format_seconds_to_mmss(new_end_sec)
                            else:
                                sub["end_time"] = new_end_sec
                
                # Save the result.
                out_path = os.path.join(save_dir, os.path.basename(file_path))
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                import traceback
                traceback.print_exc()
                
        # Save conflict records.
        if issues:
            issue_path = os.path.join(save_dir, "issues.json")
            with open(issue_path, "w", encoding="utf-8") as f:
                json.dump(issues, f, ensure_ascii=False, indent=2)
            print(f"Recorded {len(issues)} issues to {issue_path}")
        
class Ego4dGoalstep(BaseVideoDataset):
    def __init__(self, data_path: str) -> None:
        self.data = None
        self.data_path = data_path
        self.train_data = False
        self.load_dataset()

        self.system_prompt = None
        self.client = None

    def load_dataset(self) -> None:
        if "train" in self.data_path:
            self.train_data = True
        else:
            self.train_data = False
        with open(self.data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
            # print(self.data.items())
        return self.data
    
    def process_data(self, detail_guidance: bool = False) -> None:
        data_list = []
        for data in self.data['videos']:
            cur_video_item_data = {
                "video_path": data['video_uid'] + ".mp4",
                "video_begin": data['start_time'],
                "video_end": data['end_time'],
                "duration": data['end_time'] - data['start_time'],
                "speakers": "SPEAKER_00",
                "history": "",
                "query": data['goal_description'],
                "annotations": []
            }
            # segments = data['segments']
            # print(segment)
            # print(data.items())
            segment_time_anno_list = []
            segment_list = data['segments']
            for segment in segment_list:
                if segment['segments']:
                    for segment_item in segment['segments']:
                        if segment_item['is_relevant'] == "irrelevant":
                            continue
                        if segment_item['is_relevant'] == "optional":
                            if detail_guidance:
                                segment_time_anno_list.append({
                                    "begin_time": segment_item['start_time'],
                                    "end_time": segment_item['end_time'],
                                    "step_description": segment_item['step_description']
                                })
                        else:
                            segment_time_anno_list.append({
                                "begin_time": segment_item['start_time'],
                                "end_time": segment_item['end_time'],
                                "step_description": segment_item['step_description']
                            })
                else:
                    if segment['is_relevant'] == "irrelevant":
                            continue
                    if segment['is_relevant'] == "optional":
                        if detail_guidance:
                            segment_time_anno_list.append({
                                "begin_time": segment['start_time'],
                                "end_time": segment['end_time'],
                                "step_description": segment['step_description']
                            })
                    else:
                        segment_time_anno_list.append({
                            "begin_time": segment['start_time'],
                            "end_time": segment['end_time'],
                            "step_description": segment['step_description']
                        })

            cur_video_item_data['annotations'] = segment_time_anno_list
            # with open(os.path.join(os.path.dirname(self.data_path), "ego4d_goalstep_data.json"), "w", encoding="utf-8") as f:
            #     json.dump(cur_video_item_data, f, ensure_ascii=False, indent=2)
            data_list.append(cur_video_item_data)
        with open(os.path.join(os.path.dirname(self.data_path), f"ego4d_goalstep_{'train' if self.train_data else 'val'}_data.json"), "w", encoding="utf-8") as f:
            json.dump(data_list, f, ensure_ascii=False, indent=2)
            # if segments['segments']:
                # print("has segments")
            # break
    
    def _extract_json_from_response(self, response: str):
        """
        Extract a JSON object from a response string.
        Supports removing markdown code fences such as ```json ... ``` before parsing.
        """
        if not isinstance(response, str):
            return response
        
        # Remove markdown code fences.
        text = response.strip()
        if text.startswith("```"):
            # Remove the opening ```json or ``` marker.
            text = text[3:]
            if text.lower().startswith("json"):
                text = text[4:]
            # Remove the leading newline.
            text = text.lstrip("\n\r ")
            # Remove the trailing ```.
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        
        # Try to parse JSON directly.
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # If direct parsing fails, try to extract JSON fragments.
            # First try the array form: [...]
            array_match = re.search(r'\[.*\]', text, re.DOTALL)
            if array_match:
                try:
                    return json.loads(array_match.group(0))
                except json.JSONDecodeError:
                    pass
            # Then try the object form: {...}
            object_match = re.search(r'\{.*\}', text, re.DOTALL)
            if object_match:
                try:
                    return json.loads(object_match.group(0))
                except json.JSONDecodeError:
                    pass
            # Fall back to the original string if all parsing attempts fail.
            logger.warning("Failed to extract JSON from the response; returning the raw response")
            return response
    
    def polish_data(
        self, 
        system_prompt: str = None, 
        base_url: str = "https://openrouter.ai/api/v1", 
        model_name: str = "openai/gpt-4.1",
        max_workers: int = 5
    ) -> None:
        data_list = []
        with open(os.path.join(os.path.dirname(self.data_path), f"ego4d_goalstep_{'train' if self.train_data else 'val'}_data.json"), "r", encoding="utf-8") as f:
            data_list = json.load(f)
        
        if system_prompt and os.path.isfile(system_prompt):
            with open(system_prompt, "r", encoding="utf-8") as f:
                self.system_prompt = f.read()
        else:
            self.system_prompt = system_prompt

        self.client = ChatModelClient(
            base_url=base_url,
            model_name=model_name, 
            system_prompt=self.system_prompt
        )
        
        def process_single_data(data):
            """Process a single data item."""
            try:
                # Use json.dumps to guarantee valid JSON with double quotes.
                user_prompt = f"""```json
{{
    "query": {json.dumps(data['query'], ensure_ascii=False)},
    "annotations": {json.dumps(data['annotations'], ensure_ascii=False)}
}}
```"""
                annotations_list = []
                # random_num = random.randint(1, 3)
                random_num = 1
                for i in range(random_num):
                    response = self.client(content=user_prompt, type="text", temperature=0.9)
                    # Extract JSON from the response.
                    extracted_json = self._extract_json_from_response(response)
                    
                    # Check whether the response uses the new array format.
                    if isinstance(extracted_json, list):
                        # Array format: iterate over each segment.
                        for segment in extracted_json:
                            annotations_list.append(
                                {
                                    "query": segment.get('generated_question', ''),
                                    "sub_annotations": [
                                        {
                                            # Use `is not None` instead of `or` so 0 is not treated as falsy.
                                            "begin_time": item.get("begin_time") if item.get("begin_time") is not None else item.get("action_begin_time"),
                                            "end_time": item.get("end_time") if item.get("end_time") is not None else item.get("action_end_time"),
                                            "commentary": item.get("commentary") or item.get("refined_description") or item.get("step_description"),
                                            "speaker": "SPEAKER_00", 
                                            "active": True
                                        } for item in segment.get('refined_steps', [])
                                    ]
                                }
                            )
                    else:
                        # Single-object format: keep backward compatibility.
                        annotations_list.append(
                            {
                                "query": extracted_json.get('generated_question', ''),
                                "sub_annotations": [
                                    {
                                        # Use `is not None` instead of `or` so 0 is not treated as falsy.
                                        "begin_time": item.get("begin_time") if item.get("begin_time") is not None else item.get("action_begin_time"),
                                        "end_time": item.get("end_time") if item.get("end_time") is not None else item.get("action_end_time"),
                                        "commentary": item.get("commentary") or item.get("refined_description") or item.get("step_description"),
                                        "speaker": "SPEAKER_00", 
                                        "active": True
                                    } for item in extracted_json.get('refined_steps', [])
                                ]
                            }
                        )
                
                return {
                    "video_path": data['video_path'],
                    "video_begin": data['video_begin'],
                    "video_end": data['video_end'],
                    "duration": data['duration'],
                    "speakers": data['speakers'],
                    "history": data['history'],
                    "annotations": annotations_list
                }
            except Exception as e:
                logger.error(f"Error processing data item (video_path: {data.get('video_path', 'unknown')}): {e}")
                import traceback
                traceback.print_exc()
                return None
        
        # Process items concurrently with a thread pool.
        result_list = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks and keep their original indexes.
            future_to_index = {executor.submit(process_single_data, data): idx for idx, data in enumerate(data_list)}
            
            # Store results in a dict keyed by index.
            results_dict = {}
            
            # Show progress with tqdm.
            with tqdm(total=len(data_list), desc="Processing data") as pbar:
                for future in as_completed(future_to_index):
                    idx = future_to_index[future]
                    try:
                        result = future.result()
                        if result is not None:
                            results_dict[idx] = result
                    except Exception as e:
                        logger.error(f"Error retrieving task result (index: {idx}): {e}")
                    pbar.update(1)
        
        # Restore the original order, keeping only successful results.
        result_list = [results_dict[idx] for idx in sorted(results_dict.keys())]
        
        # Write all results in one pass.
        output_path = os.path.join(os.path.dirname(self.data_path), f"ego4d_goalstep_{'train' if self.train_data else 'val'}_data_polished.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_list, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Successfully processed and saved {len(result_list)}/{len(data_list)} data items to {output_path}")

    def post_process_data(
        self, 
        system_prompt: str = None, 
        base_url: str = "https://openrouter.ai/api/v1", 
        model_name: str = None,
        max_workers: int = 5,
        token_per_second: int = 4
    ) -> None:
        """
        Standalone post-processing function:
        1. Read the existing polished data
        2. Check whether end_time matches text length
        3. If end_time is too long, shorten it
        4. If the text is too long, repolish it concurrently with the LLM
        """
        import os
        input_path = os.path.join(os.path.dirname(self.data_path), f"ego4d_goalstep_{'train' if self.train_data else 'val'}_data_polished.json")
        if not os.path.exists(input_path):
            logger.error(f"Generated polished data not found: {input_path}")
            return

        with open(input_path, "r", encoding="utf-8") as f:
            result_list = json.load(f)

        if system_prompt and os.path.isfile(system_prompt):
            with open(system_prompt, "r", encoding="utf-8") as f:
                repolish_system_prompt = f.read()
        else:
            repolish_system_prompt = system_prompt or "You are a helpful assistant."

        # Initialize the client used for repolishing.
        self.client = ChatModelClient(
            base_url=base_url,
            model_name=model_name, 
            system_prompt=repolish_system_prompt
        )

        repolish_tasks = [] # Tasks that require repolishing.

        # Step 1: scan all data and decide which items need repolishing.
        for video_idx, video_item in enumerate(result_list):
            annotations = video_item.get("annotations", [])
            for anno_idx, annotation in enumerate(annotations):
                sub_annotations = annotation.get("sub_annotations", [])
                for i, sub in enumerate(sub_annotations):
                    # Normalize multiple possible field names.
                    # Use `is not None` instead of `or` so 0 is not treated as falsy.
                    desc = sub.get("commentary") or sub.get("refined_description") or sub.get("step_description") or ""
                    begin_time = sub.get("begin_time") if sub.get("begin_time") is not None else sub.get("action_begin_time")
                    end_time = sub.get("end_time") if sub.get("end_time") is not None else sub.get("action_end_time")
                    
                    # Normalize field names for later processing.
                    sub["commentary"] = desc
                    sub["begin_time"] = begin_time
                    sub["end_time"] = end_time
                    
                    # Remove legacy field names to keep the data tidy.
                    for old_key in ["refined_description", "step_description", "action_begin_time", "action_end_time"]:
                        if old_key in sub:
                            del sub[old_key]

                    begin_sec = self._parse_time_to_seconds(begin_time)
                    original_end_sec = self._parse_time_to_seconds(end_time)
                    
                    if begin_sec is None or original_end_sec is None:
                        continue
                        
                    words = desc.strip().split()
                    required_duration = len(words) / token_per_second
                    
                    # Determine the hard upper bound for end_time.
                    limit_sec = original_end_sec
                    if i < len(sub_annotations) - 1:
                        next_sub = sub_annotations[i+1]
                        next_begin_time = next_sub.get("begin_time") if next_sub.get("begin_time") is not None else next_sub.get("action_begin_time")
                        next_begin_sec = self._parse_time_to_seconds(next_begin_time)
                        if next_begin_sec is not None:
                            limit_sec = min(limit_sec, next_begin_sec)

                    actual_available_duration = limit_sec - begin_sec

                    if required_duration > actual_available_duration and actual_available_duration > 0:
                        # This item needs repolishing.
                        max_words = max(1, int(actual_available_duration * token_per_second))
                        repolish_tasks.append({
                            "video_idx": video_idx,
                            "anno_idx": anno_idx,
                            "sub_idx": i,
                            "desc": desc,
                            "max_words": max_words,
                            "limit_sec": limit_sec,
                            "begin_sec": begin_sec
                        })
                    else:
                        # No repolish needed; just shorten the end timestamp.
                        final_end_sec = min(begin_sec + required_duration, limit_sec)
                        if isinstance(end_time, str) and ":" in end_time:
                            sub["end_time"] = self._format_seconds_to_mmss(final_end_sec)
                        else:
                            sub["end_time"] = final_end_sec

        # Step 2: run repolish tasks concurrently.
        if repolish_tasks:
            logger.info(f"Found {len(repolish_tasks)} overly long text entries; starting concurrent repolishing...")
            
            def do_repolish(task):
                try:
                    prompt_input = json.dumps({
                        "original_instruction": task["desc"],
                        "max_words": task["max_words"]
                    }, ensure_ascii=False)
                    
                    response = self.client(content=prompt_input, type="text", temperature=0.9)
                    refined_json = self._extract_json_from_response(response)
                    
                    if isinstance(refined_json, dict) and "refined_commentary" in refined_json:
                        new_desc = refined_json["refined_commentary"]
                        new_words = new_desc.strip().split()
                        new_duration = len(new_words) / token_per_second
                        new_end_sec = min(task["begin_sec"] + new_duration, task["limit_sec"])
                        return {**task, "new_desc": new_desc, "new_end_sec": new_end_sec}
                except Exception as e:
                    logger.error(f"Repolishing failed: {e}")
                return {**task, "new_desc": task["desc"], "new_end_sec": task["limit_sec"]}

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(do_repolish, t) for t in repolish_tasks]
                for f in tqdm(as_completed(futures), total=len(futures), desc="Concurrent repolishing"):
                    res = f.result()
                    sub = result_list[res["video_idx"]]["annotations"][res["anno_idx"]]["sub_annotations"][res["sub_idx"]]
                    sub["commentary"] = res["new_desc"]
                    original_end = sub.get("end_time")
                    if isinstance(original_end, str) and ":" in original_end:
                        sub["end_time"] = self._format_seconds_to_mmss(res["new_end_sec"])
                    else:
                        sub["end_time"] = res["new_end_sec"]

        # Step 3: save to a new file instead of overwriting the original.
        import os
        base, ext = os.path.splitext(input_path)
        output_path = base + "_aligned" + ext
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_list, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Post-processing complete; results have been updated at: {input_path}")

    def _parse_time_to_seconds(self, t):
        """Convert multiple time formats to seconds."""
        if t is None:
            return None
        if isinstance(t, (int, float)):
            return float(t)
        if isinstance(t, str):
            s = t.strip()
            if ":" in s:
                parts = s.split(":")
                if len(parts) == 2:
                    return float(int(parts[0]) * 60 + int(parts[1]))
                if len(parts) == 3:
                    return float(int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2]))
            try:
                return float(s)
            except:
                pass
        return None

    def _format_seconds_to_mmss(self, seconds: float) -> str:
        """Convert seconds to M:SS format."""
        try:
            total = int(round(float(seconds)))
            m = total // 60
            s = total % 60
            return f"{m}:{s:02d}"
        except:
            return str(seconds)
