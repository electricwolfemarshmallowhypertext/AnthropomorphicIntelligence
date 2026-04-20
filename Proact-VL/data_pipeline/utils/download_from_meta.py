from __future__ import annotations
import argparse
import json
import shlex
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import os

FORMAT_RE = re.compile(r"format\(s\):\s*([0-9A-Za-z+]+)")

# ------------------------------
# Keep the original helper functions intact.
# ------------------------------

def get_video_duration(video_path: Path) -> str | None:
    try:
        cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", str(video_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration_seconds = float(result.stdout.strip())
        hours = int(duration_seconds // 3600)
        minutes = int((duration_seconds % 3600) // 60)
        seconds = int(duration_seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
        return None


def is_clip_fields_empty(game: dict) -> bool:
    clip_start = game.get("clip_start", "")
    clip_end = game.get("clip_end", "")
    clip_duration = game.get("clip_duration", "")
    return not clip_start and not clip_end and not clip_duration


def update_video_durations(games: list[dict], output_dir: Path) -> bool:
    changed = False
    for game in games:
        if is_clip_fields_empty(game):
            video_file = game.get("video_file")
            if video_file:
                video_path = output_dir / video_file
                if video_path.exists():
                    duration = get_video_duration(video_path)
                    if duration:
                        game["clip_duration"] = duration
                        changed = True
                        print(f"Updated duration for {video_file}: {duration}")
    return changed


def clean_video_files(output_dir: Path):
    video_patterns = ["*.mp4*", "*.mkv*", "*.webm*", "*.avi*", "*.mov*", "*.flv*"]
    for pattern in video_patterns:
        for file_path in output_dir.glob(pattern):
            if file_path.is_file():
                filename = file_path.name
                base_name = filename.split('.')[0]
                new_filename = f"{base_name}.mp4"
                if new_filename != filename:
                    new_path = file_path.parent / new_filename
                    if new_path.exists():
                        new_path.unlink()
                    file_path.rename(new_path)
                    print(f"Renamed: {filename} -> {new_filename}")
    print("File cleanup completed; all outputs are normalized to .mp4")


def build_command(game: dict, output_dir: Path, cli_height: int | None = None) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / game["video_file"]

    parts = ["yt-dlp", "--ignore-errors", "--no-warnings"]
    
    height_limit = cli_height
    if height_limit is None:
        res = game.get("resolution")
        if res and "x" in res:
            try:
                height_limit = int(res.split("x")[1])
            except ValueError:
                height_limit = None

    if height_limit:
        parts += ["-f", f"bestvideo[height<={height_limit}]+bestaudio/best[height<={height_limit}]/best"]

    is_playlist = game.get("is_playlist", False)
    if is_playlist:
        base_name = Path(game["video_file"]).stem
        ext = Path(game["video_file"]).suffix or ".mp4"
        parts += ["-o", str(output_dir / f"{base_name}_%(playlist_index)s{ext}")]
        if game.get("playlist_start"):
            parts += ["--playlist-start", str(game["playlist_start"])]
        if game.get("playlist_end"):
            parts += ["--playlist-end", str(game["playlist_end"])]
    else:
        parts += ["--no-playlist", "-o", str(out_path)]

    if game.get("clip_start") and game.get("clip_end"):
        parts += ["--download-sections", f"*{game['clip_start']}-{game['clip_end']}"]

    parts.append(game["source_url"])
    return " ".join(shlex.quote(p) for p in parts)


def check_yt_dlp_available() -> bool:
    """Check whether yt-dlp is available."""
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True, timeout=5)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False

def run(cmd: str, dry: bool) -> str | None:
    print("Executing:", cmd)
    if dry:
        return None
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    output_lines = []
    for line in process.stdout:
        print(line, end='')
        output_lines.append(line)
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)
    return ''.join(output_lines)

def get_output_prefix(game: dict) -> str:
    base_name = Path(game["video_file"]).stem
    return f"{base_name}_" if game.get("is_playlist", False) else base_name


def exists_finished_file_with_prefix(output_dir: Path, prefix: str) -> bool:
    video_exts = {".mp4", ".mkv", ".webm", ".avi", ".mov", ".flv"}
    for p in output_dir.glob(f"{prefix}*"):
        if p.is_file() and p.suffix.lower() in video_exts:
            return True
    return False


def download_game(g: dict, output_dir: Path, args) -> bool:
    # Skip the download if a finished video with the same prefix already exists.
    if not args.dry:
        prefix = get_output_prefix(g)
        # if exists_finished_file_with_prefix(output_dir, prefix):
        #     print(f"Detected an existing finished file with prefix '{prefix}', skipping download: {g.get('video_file')}")
        #     return False

    cmd = build_command(g, output_dir, args.max_height)
    try:
        output = run(cmd, args.dry)
        if output:
            m = FORMAT_RE.search(output)
            if m and g.get("format_id") != m.group(1):
                g["format_id"] = m.group(1)
                return True
    except subprocess.CalledProcessError:
        pass
    return False

# ------------------------------
# Added: recursively process meta.json files.
# ------------------------------

def process_meta(meta_path: Path, args):
    print(f"\n==== Processing {meta_path} ====")

    with meta_path.open() as f:
        meta = json.load(f)

    games = meta["games"]
    output_dir = meta_path.parent
    changed = False

    if args.sequential:
        for g in games:
            if download_game(g, output_dir, args):
                changed = True
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = [executor.submit(download_game, g, output_dir, args) for g in games]
            for fut in as_completed(futures):
                if fut.result():
                    changed = True

    if not args.dry:
        clean_video_files(output_dir)
        if update_video_durations(games, output_dir):
            changed = True

    if changed:
        with meta_path.open("w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print("Updated meta.json")


def main():
    p = argparse.ArgumentParser(description="Recursively find meta.json files and download them")
    p.add_argument("root", type=Path, help="Root directory used to recursively search for meta.json")
    p.add_argument("--max-height", type=int)
    p.add_argument("--jobs", "-j", type=int, default=4)
    p.add_argument("--dry", action="store_true")
    p.add_argument("--sequential", "-s", action="store_true")
    args = p.parse_args()

    root = args.root.expanduser().resolve()
    meta_files = list(root.rglob("meta.json"))
    if not meta_files:
        print("No meta.json files found")
        return

    print(f"Found {len(meta_files)} meta.json files")
    for meta in meta_files:
        process_meta(meta, args)


if __name__ == "__main__":
    main()
