import json
import logging
import subprocess
from dataclasses import dataclass
from utils.logger import Logger
from abc import ABC, abstractmethod
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
logger = Logger(__name__, level=logging.INFO, msg_color=True).get_logger()


"""
Tools for audio extraction
"""    
class AudioTools:
    def __init__(self,) -> None:
        if not self.check_ffmpeg():
            raise RuntimeError("FFmpeg is not installed. Please install it (e.g., 'brew install ffmpeg' on macOS or 'sudo apt install ffmpeg' on Linux).")

    @property
    def supported_format(self) -> list[str]:
        return ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.3gp', '.mpg', '.mpeg']
       
    def check_ffmpeg(self) -> bool:
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def has_audio_stream(self, file_path: str) -> bool:
        """Check if the file has an audio stream."""
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'a',
                '-show_entries', 'stream=index',
                '-of', 'csv=p=0',
                str(file_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return len(result.stdout.strip()) > 0
        except Exception as e:
            logger.warning(f"Error checking audio stream for {file_path}: {e}")
            return False

    def extract_audio_all(
        self,
        video_path: str,
        output_dir: str,
        num_workers: int = 4,
        output_format: str = "mp3",
    ) -> None:
    
        logger.info(f"Extracting audio from {video_path} to {output_dir}")
        video_path = Path(video_path)
        output_path = Path(output_dir)
        
        
        if output_path.exists():
            if output_path.glob('*') and any(output_path.iterdir()):
                logger.warning(f"Output directory {output_dir} already exists and contains files. Please delete or rename them.")
                choice = input("Do you want to overwrite the directory? (Y/N): ")
                if choice.lower() == 'n':
                    return
                else:
                    pass
        else:
            output_path.mkdir(parents=True, exist_ok=True)
        
        file_list = video_path.glob('*')
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_file = {executor.submit(self.extract_audio, f, output_path, output_format): f for f in file_list if f.suffix.lower() in self.supported_format}
            for future in as_completed(future_to_file):
                video_file = future_to_file[future]
                try:
                    future.result()
                    logger.info(f"Successfully extracted audio from {video_file} to {output_path}")
                except Exception as e:
                    logger.error(f"Error extracting audio from {video_file} to {output_path}: {e}")
        logger.info("✅ All audio extracted successfully")

    def extract_audio(
        self, 
        video_file: str, 
        output_dir:str,
        output_format: str = "mp3",
    ) -> None:
        logger.info(f"Extracting audio from {video_file} to {output_dir}")
        try:
            video_name = video_file.stem
            output_file = output_dir / f"{video_name}.{output_format}"

            cmd = [
                'ffmpeg',
                '-i', str(video_file),
                '-vn',  
                '-acodec', 'libmp3lame' if output_format == 'mp3' else 'aac',
                '-ab', '192k',
                '-ar', '16000',
                '-y',
                str(output_file)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"Successfully extracted audio from {video_file} to {output_file}")
            else:
                logger.error(f"Failed to extract audio from {video_file} to {output_file}: {result.stderr}")

        except Exception as e:
            logger.error(f"Error extracting audio from {video_file} to {output_dir}: {e}")
 
    def cut_audio(
        self,
        input_file: str,
        output_file: str,
        start_time: float,
        end_time: float,
        output_format: str = "mp3"
    ) -> None:
        """
        Cut a segment of audio from an input file.
        """
        duration = end_time - start_time
        try:
            cmd = [
                'ffmpeg',
                '-ss', str(start_time),
                '-i', str(input_file),
                '-t', str(duration),
                '-vn',
                '-acodec', 'libmp3lame' if output_format == 'mp3' else 'aac',
                '-y',
                str(output_file)
            ]
            # Quiet mode to reduce log noise
            cmd.extend(['-loglevel', 'error'])
            
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error cutting audio from {input_file}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error cutting audio: {e}")
            raise 
