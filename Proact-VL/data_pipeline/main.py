import json
import os
from pickle import FALSE
import warnings

from tqdm import tqdm

from utils.dataset import Cyberpunk, Minecraft, BlackMythWukong, EldenRing, TearsOfTheKingdom, Starcraft2, Ego4dGoalstep
from utils.tools import AudioTools
from concurrent.futures import ThreadPoolExecutor, as_completed
from core.model import ASRModel, ChatModelClient, FineGrainedASRModel


from dotenv import load_dotenv
load_dotenv('client.env')

warnings.filterwarnings("ignore")

def strip_markdown_code_fences(text: str) -> str:
    """
    Remove markdown code fences like ```json ... ``` or ``` ... ```
    """
    if not isinstance(text, str):
        return text
    s = text.strip()
    if s.startswith("```"):
        # Remove starting ```json or ```
        s = s[3:]
        if s.lower().startswith("json"):
            s = s[4:]
        # Remove leading newlines
        s = s.lstrip("\n\r ")
        # Remove ending ```
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()
    return s

def extract_audio(args):
    audio_tools = AudioTools()
    audio_tools.extract_audio_all(
        video_path=args.video_path,
        output_dir=args.output_dir,
        output_format=args.output_format,
        num_workers=args.workers,
        overwrite_output=args.overwrite_output,
    )

def tone_analysis(args):
    if os.path.isdir(args.asr_path):
        asr_files = [os.path.join(args.asr_path, f) for f in os.listdir(args.asr_path) if f.endswith(".json")]
    else:
        asr_files = [args.asr_path]

    clips_dir = ""
    if args.save_asr_dir:
        os.makedirs(args.save_asr_dir, exist_ok=True)
        clips_dir = os.path.join(args.save_asr_dir, "clips")
        os.makedirs(clips_dir, exist_ok=True)
    
    # Initialize AudioTools
    audio_tools = AudioTools()
    print(f"System prompt: {args.system_prompt}")
    # Initialize FineGrainedASRModel (Qwen-Omni)
    qwen_omni = FineGrainedASRModel(model_name="qwen3-omni-flash", system_prompt=args.system_prompt)

    for asr_file in tqdm(asr_files, desc="Tone Analysis", colour="magenta"):
        try:
            with open(asr_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if not data or 'segments' not in data[0]:
                continue
            
            # Find corresponding video file
            video_file = None
            base_name = os.path.splitext(os.path.basename(asr_file))[0]
            
            if os.path.isfile(args.video_path):
                 if os.path.splitext(os.path.basename(args.video_path))[0] == base_name:
                     video_file = args.video_path
                 elif len(asr_files) == 1:
                     video_file = args.video_path
            elif os.path.isdir(args.video_path):
                for ext in audio_tools.supported_format:
                    p = os.path.join(args.video_path, base_name + ext)
                    if os.path.exists(p):
                        video_file = p
                        break
            
            segments = data[0]['segments']
            
            # 1. Identify and collect all segments needing processing
            tasks_to_process = [] # List of tuples: (segment_index, clip_path, start, end)
            
            for i, item in enumerate(segments):
                start = item.get('start', 0)
                end = item.get('end', 0)
                words = item.get('words', [])
                
                # Calculate duration and word count
                duration = end - start
                word_count = len(words)
                
                # Calculate speed (words per second)
                if duration > 0:
                    speed = word_count / duration
                else:
                    speed = 0
                    
                item['speed'] = speed
                
                # Filter by threshold
                if speed < args.speed_threshold:
                    if video_file and clips_dir:
                        clip_name = f"{base_name}_{start:.2f}_{end:.2f}.mp3"
                        clip_path = os.path.join(clips_dir, clip_name)
                        tasks_to_process.append({
                            "index": i,
                            "clip_path": clip_path,
                            "start": start,
                            "end": end
                        })
            
            # 2. Cut all audio clips first (Parallel if possible or sequential but optimized)
            # We use ThreadPool for cutting audio clips as IO bound
            if tasks_to_process:
                 with ThreadPoolExecutor(max_workers=args.workers) as executor:
                    # Submit cut tasks
                    future_to_task = {}
                    for task in tasks_to_process:
                        if not os.path.exists(task['clip_path']): # Avoid re-cutting if exists
                             future = executor.submit(audio_tools.cut_audio, video_file, task['clip_path'], task['start'], task['end'])
                             future_to_task[future] = task
                    
                    # Wait for cutting to complete
                    for future in as_completed(future_to_task):
                        try:
                            future.result()
                        except Exception as e:
                            print(f"Error cutting audio: {e}")

            # 3. Process with Qwen-Omni model in parallel
            if tasks_to_process:
                # Limit concurrency to avoid rate limits (Qwen-Omni might have stricter limits)
                # Default workers might be too high (e.g. 16), let's cap it or use a specific arg.
                # If rate limit error occurs, reducing workers is a good first step.
                concurrency = min(args.workers, 2) # Conservative default for API calls to avoid 429
                
                with ThreadPoolExecutor(max_workers=concurrency) as executor:
                    future_to_index = {}
                    for task in tasks_to_process:
                        if os.path.exists(task['clip_path']):
                            # Construct prompt with transcription context
                            segment_text = segments[task['index']].get('text', '')
                            full_prompt = f"{args.system_prompt}\n\nTranscription: {segment_text}"
                            
                            future = executor.submit(qwen_omni, task['clip_path'], prompt=full_prompt, output_audio=False)
                            future_to_index[future] = task['index']
                    
                    for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc=f"Analyzing {base_name}", leave=False):
                        idx = future_to_index[future]
                        try:
                            omni_response = future.result()
                            # Extract JSON content if present
                            try:
                                cleaned_response = strip_markdown_code_fences(omni_response)
                                response_json = json.loads(cleaned_response)
                                if "text_with_tone" in response_json:
                                     omni_response = response_json["text_with_tone"]
                            except json.JSONDecodeError:
                                # Not a valid JSON, keep original response
                                pass
                            except Exception:
                                pass
                            
                            segments[idx]['omni_analysis'] = omni_response
                        except Exception as e:
                            print(f"Error analyzing segment {idx}: {e}")
                            segments[idx]['omni_analysis'] = f"Error: {e}"
            
            # Save results if output directory is specified
            if args.save_asr_dir:
                data[0]['segments'] = segments
                save_path = os.path.join(args.save_asr_dir, os.path.basename(asr_file))
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            else:
                 # Just for feedback if no save dir
                 pass
                 
        except Exception as e:
            print(f"Error processing {asr_file}: {e}")

def openai_polish(args):
    os.makedirs(args.save_polish_dir, exist_ok=True)
    if os.path.isdir(args.asr_path):
        asr_files = [os.path.join(args.asr_path, f) for f in os.listdir(args.asr_path) if f.endswith(".json")]
    else:
        asr_files = [args.asr_path]
    
    client = ChatModelClient(model_name=os.getenv("POLISH_MODEL_NAME", "deepseek/deepseek-v3.2-exp"), system_prompt=args.system_prompt, base_url=os.getenv("POLISH_BASE_URL", "https://openrouter.ai/api/v1"))
    for ars_file in asr_files:
        output = []
        speaker_map = {}
        with open(ars_file, "r", encoding="utf-8") as f:
            datasets = json.load(f)
            
        batch_data = []
        for item in datasets[0]['segments']:
            text = item['text']
            # If omni_analysis exists, keep it and prefer it as model input later.
            omni_text = item.get('omni_analysis', text)
            # print(f"item:{item}")
            try:
                batch_data.append({
                    "text": text,
                    "omni_analysis": omni_text,
                    "start": item['start'],
                    "end": item['end'],
                    "speaker": item['speaker']
                })
            except Exception as e:
                print(f"Error: {e} for {ars_file}, text: {text}")
                continue
            # print(f"Before text: {text}")
        
        # Concurrent request client
        results_map = {}
        pairs = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_idx = {executor.submit(client, batch_data[i].get('omni_analysis', batch_data[i]['text'])): i for i in range(len(batch_data))}
            for future in tqdm(as_completed(future_to_idx), total=len(batch_data), desc="Polishing", colour="green"):
                idx = future_to_idx[future]
                try:
                    results_map[idx] = future.result()
                except Exception as e:
                    results_map[idx] = f"[ERROR] {e}"
                    
        for i in range(len(batch_data)):
            if batch_data[i]['speaker'] not in speaker_map:
                speaker_map[batch_data[i]['speaker']] = []
            polish_text = results_map.get(i, "")
            try:
                # Strip markdown code fences if present
                cleaned_text = strip_markdown_code_fences(polish_text)
                polish_text = json.loads(cleaned_text)['polished_text']
                speaker_map[batch_data[i]['speaker']].append({"text": polish_text, "start": batch_data[i]['start'], "end": batch_data[i]['end']})
                pairs.append({
                    "original": batch_data[i].get('omni_analysis', batch_data[i]['text']),
                    "polished": polish_text,
                    "start": batch_data[i]['start'],
                    "end": batch_data[i]['end'],
                    "speaker": batch_data[i]['speaker']
                })
            except Exception as e:
                print(f"Error: {e} for {ars_file}, polish_text: {polish_text}")
                continue
            
            
        for item in speaker_map:
            output.append({"Speaker": item, "Conversation": speaker_map[item]})
        with open(os.path.join(args.save_polish_dir, f"{os.path.basename(ars_file).split('.')[0]}.json"), "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        with open(os.path.join(args.save_polish_dir, f"{os.path.basename(ars_file).split('.')[0]}.pairs.json"), "w", encoding="utf-8") as f:
            json.dump(pairs, f, ensure_ascii=False, indent=2)

def asr(args):
    asr_model = ASRModel("large-v3", args.device, args.compute_type, args.language)

    if "," in args.audio_path:
        audio_inputs = [p.strip() for p in args.audio_path.split(",") if p.strip()]
    elif os.path.isdir(args.audio_path):
        audio_inputs = [
            os.path.join(args.audio_path, f)
            for f in os.listdir(args.audio_path)
        ]
    else:
        audio_inputs = [args.audio_path]

    os.makedirs(args.save_asr_dir, exist_ok=True)
    for audio_input in tqdm(audio_inputs, desc="Processing audio files", colour="blue"):
        results = []
        result = asr_model.inference(
            audio_path=audio_input,
            batch_size=args.batch_size,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers
        )
        results.append(result)
        with open(os.path.join(args.save_asr_dir, f"{os.path.basename(audio_input).split('.')[0]}.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

def merge_asr_punctuation(args):
    """
    Merge segments based on punctuation.
    If the current segment does not end with ".", "!", or "?", merge it with the next segment until the ending punctuation is ".", "!", or "?".
    """
    if not os.path.exists(args.save_asr_dir):
        os.makedirs(args.save_asr_dir)

    # Read all json files in args.asr_path
    if os.path.isdir(args.asr_path):
        all_file = os.listdir(args.asr_path)
    else:
        if os.path.isfile(args.asr_path):
            all_file = [os.path.basename(args.asr_path)]
            args.asr_path = os.path.dirname(args.asr_path)
        else:
            all_file = []

    # Define ending punctuation
    ending_punctuation = ('.', '!', '?', '。', '！', '？')
    
    for file in tqdm(all_file, desc="Merging ASR by Punctuation", colour="cyan"):
        if file.endswith(".json") or file.endswith(".jsonl"):
            try:
                with open(os.path.join(args.asr_path, file), "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                if not data or 'segments' not in data[0]:
                    continue

                segments = data[0]['segments']
                if not segments:
                    continue
                
                merged_segments = []
                if segments:
                    # Initialize current merged segment
                    current_segment = {
                        'text': segments[0]['text'],
                        'start': segments[0]['start'],
                        'end': segments[0]['end']
                    }
                    if 'words' in segments[0]:
                        current_segment['words'] = segments[0]['words']
                    if 'speaker' in segments[0]:
                        current_segment['speaker'] = segments[0]['speaker']
                    
                    for next_segment in segments[1:]:
                        # Check if the end of the current segment ends with a sentence-ending punctuation mark
                        current_text = current_segment['text'].rstrip()

                        # Only keep merging when the current sentence lacks terminal punctuation
                        # and the speaker stays the same.
                        current_speaker = current_segment.get('speaker', None)
                        next_speaker = next_segment.get('speaker', None)
                        same_speaker = (
                            current_speaker is not None
                            and next_speaker is not None
                            and current_speaker == next_speaker
                        )

                        if current_text and current_text[-1] in ending_punctuation or not same_speaker:
                            # Save the current segment and start a new one if the sentence has ended
                            # or the speaker has changed.
                            merged_segments.append(current_segment)
                            current_segment = {
                                'text': next_segment['text'],
                                'start': next_segment['start'],
                                'end': next_segment['end']
                            }
                            if 'words' in next_segment:
                                current_segment['words'] = next_segment['words']
                            if 'speaker' in next_segment:
                                current_segment['speaker'] = next_segment['speaker']
                        else:
                            # Otherwise continue merging into the current segment.
                            current_segment['text'] += next_segment['text']
                            # Extend the end time to the next segment's end time.
                            current_segment['end'] = next_segment['end']
                            if 'words' in current_segment and 'words' in next_segment:
                                current_segment['words'].extend(next_segment['words'])
                    
                    # Add the last merged segment
                    merged_segments.append(current_segment)

                data[0]['segments'] = merged_segments
                
                with open(os.path.join(args.save_asr_dir, file), "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

            except Exception as e:
                print(f"Error merging {file}: {e}")

def extract_role(args):
    client = ChatModelClient(model_name="deepseek/deepseek-v3.2-exp", system_prompt=args.system_prompt)
    
    role_metadata = []
    for file in os.listdir(args.polish_dir):
        if not os.path.isdir(os.path.join(args.polish_dir, file)) and file != "role_metadata.json":
            try:
                with open(os.path.join(args.polish_dir, file), "r", encoding="utf-8") as f:
                    data = json.load(f)
                sub_video_metadata = []
                speaker_content = ""
                for speaker in data:
                    speaker_content += f"Transcripts of live commentary:\n"
                    for conversation in speaker["Conversation"]:
                        speaker_content += f"{conversation['text']} "
                    
                    speaker_persona = client(speaker_content)
                    role_data = {"json_path": file, "speaker": speaker["Speaker"], "persona": speaker_persona, "content": speaker_content}
                    sub_video_metadata.append(role_data)
                role_metadata.append(sub_video_metadata)
            except Exception as e:
                print(file)
                print(f"Error: {e}")
        with open(os.path.join(args.polish_dir, "role_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(role_metadata, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="lol")
    parser.add_argument("--func", type=str, default="extract_audio")
    parser.add_argument("--video_path", type=str, default="data/video.mp4")
    parser.add_argument("--output_dir", type=str, default="data/extracted_audio")
    parser.add_argument("--output_format", type=str, default="mp3")
    parser.add_argument("--audio_path", type=str, default="data/extracted_audio")
    parser.add_argument("--model_name", type=str, default="large-v3")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--compute_type", type=str, default="float16")
    parser.add_argument("--language", type=str, default="zh")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--min_speakers", type=int, default=2)
    parser.add_argument("--max_speakers", type=int, default=2)
    parser.add_argument("--save_asr_dir", type=str, default="")
    parser.add_argument("--finegrain_asr_model_name", type=str, default="qwen3-omni-flash")
    parser.add_argument("--save_finegrain_asr_dir", type=str, default="")
    parser.add_argument("--asr_path", type=str, default="data/asr_batch_results.json")
    parser.add_argument("--save_polish_dir", type=str, default="")
    parser.add_argument("--polish_dir", type=str, default="")
    parser.add_argument("--system_prompt", type=str, default="")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--vision_model", type=str, default="google/gemini-2.5-flash-lite", help="Vision model name (supports multimodal models like GPT-4o, Gemini, Qwen-VL)")
    parser.add_argument("--segment_minutes", type=int, default=3, help="Video segment duration (minutes), for processing long videos")
    parser.add_argument("--json_path", type=str, default="data/polish")
    parser.add_argument("--speed_threshold", type=float, default=1.5, help="Speed threshold (words/second)")
    parser.add_argument("--overwrite_output", action="store_true", help="Overwrite non-empty output directories for extraction steps")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    if args.debug:
        import debugpy
        debugpy.listen("localhost", 9501)
        print("Waiting for debugger to attach...")
        debugpy.wait_for_client()
        print("Debugger attached")

    # Read system prompt from file or text
    if args.system_prompt and os.path.isfile(args.system_prompt):
        with open(args.system_prompt, "r", encoding="utf-8") as f:
            args.system_prompt = f.read()
    else:
        args.system_prompt = args.system_prompt
    
    shared_audio_datasets = {"lol", "csgo", "streetfighter6", "yu_gi_oh"}
    shared_audio_handlers = {
        "extract_audio": extract_audio,
        "asr": asr,
        "merge_asr_punctuation": merge_asr_punctuation,
        "polish": openai_polish,
        "extract_role": extract_role,
    }
    clean_speaker_dataset_classes = {
        "cyberpunk": Cyberpunk,
        "black_myth_wukong": BlackMythWukong,
        "elden_ring": EldenRing,
        "tears_of_the_kingdom": TearsOfTheKingdom,
        "starcraft2": Starcraft2,
    }
    shared_clean_speaker_handlers = {
        "polish": openai_polish,
        "merge_asr_punctuation": merge_asr_punctuation,
    }
    tone_analysis_datasets = {"cyberpunk", "elden_ring", "tears_of_the_kingdom", "starcraft2"}

    if args.data_name in shared_audio_datasets:
        handler = shared_audio_handlers.get(args.func)
        if handler:
            handler(args)
    elif args.data_name in clean_speaker_dataset_classes:
        dataset = clean_speaker_dataset_classes[args.data_name](data_path=args.json_path)
        if args.func == "clean_speaker":
            dataset.clean_speaker(args.output_dir)
        elif args.func == "tone_analysis" and args.data_name in tone_analysis_datasets:
            tone_analysis(args)
        else:
            handler = shared_clean_speaker_handlers.get(args.func)
            if handler:
                handler(args)
    elif args.data_name in {"minecraft", "genshin_impact"}:
        segment_minutes = getattr(args, "segment_minutes", 5)
        vision_dataset = Minecraft(
            video_path=args.video_path,
            vision_model=args.vision_model,
            segment_minutes=segment_minutes,
            system_prompt=args.system_prompt
        )

        if args.func == "process_vision":
            vision_dataset.load_dataset()
            vision_dataset.batch_process_with_vision_model(
                prompt=args.system_prompt,
                save_dir=getattr(args, "output_dir", None)
            )
        elif args.func in {"extract_atom_action", "extrac_atom_action"}:
            vision_dataset.extract_atom_action(
                json_path=args.json_path,
                save_dir=args.output_dir or None
            )
        elif args.func == "refine_atom_action":
            vision_dataset.refine_atom_action(
                json_path=args.json_path,
                save_dir=args.output_dir or None
            )
    elif args.data_name == "ego4d_goalstep":
        ego4d_goalstep = Ego4dGoalstep(data_path=args.json_path)
        polish_base_url = os.getenv("POLISH_BASE_URL", "https://openrouter.ai/api/v1")
        polish_model_name = os.getenv("POLISH_MODEL_NAME", "deepseek/deepseek-v3.2-exp")
        if args.func == "load_dataset":
            ego4d_goalstep.process_data(detail_guidance=False)
            ego4d_goalstep.polish_data(
                system_prompt=args.system_prompt,
                base_url=polish_base_url,
                model_name=polish_model_name,
                max_workers=args.workers
            )
        elif args.func == "post_process":
            ego4d_goalstep.post_process_data(
                system_prompt=args.system_prompt,
                base_url=polish_base_url,
                model_name=polish_model_name,
                max_workers=args.workers
            )
