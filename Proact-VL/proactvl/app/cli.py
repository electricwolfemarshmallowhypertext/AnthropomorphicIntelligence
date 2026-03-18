#!/usr/bin/env python
"""
Proact-VL Demo CLI launcher
Supports configuration via command line arguments or environment variables
"""
import argparse
import os
import sys
import uvicorn


def parse_args():
    parser = argparse.ArgumentParser(
        description="Proact-VL Demo Server - Real-time Video Stream AI Assistant",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Server configuration
    server_group = parser.add_argument_group('Server Configuration')
    server_group.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help='Server listening address'
    )
    server_group.add_argument(
        '--port',
        type=int,
        default=int(os.getenv('PORT', '8001')),
        help='Server listening port'
    )
    server_group.add_argument(
        '--reload',
        action='store_true',
        default=os.getenv('RELOAD', '0') != '0',
        help='Enable auto-reload (development mode)'
    )
    server_group.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default=os.getenv('LOG_LEVEL', 'INFO'),
        help='Log level'
    )
    
    # Basic configuration
    basic_group = parser.add_argument_group('Basic Configuration')
    basic_group.add_argument(
        '--use-dummy',
        action='store_true',
        default=os.getenv('USE_DUMMY', '0') != '0',
        help='Use dummy model (for testing, does not load real model)'
    )
    basic_group.add_argument(
        '--allowed-origins',
        type=str,
        default=os.getenv('ALLOWED_ORIGINS', '*'),
        help='CORS allowed origins, comma-separated'
    )
    
    # Model configuration
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument(
        '--ckpt-path',
        type=str,
        required=True,
        help='Model checkpoint path'
    )
    model_group.add_argument(
        '--device-id',
        type=int,
        default=int(os.getenv('DEVICE_ID', '0')),
        help='GPU device ID'
    )
    
    # Inference configuration
    infer_group = parser.add_argument_group('Inference Configuration')
    infer_group.add_argument(
        '--use-audio-in-video',
        action='store_true',
        default=os.getenv('USE_AUDIO_IN_VIDEO', '0') != '0',
        help='Use audio in video'
    )
    infer_group.add_argument(
        '--max-kv-tokens',
        type=int,
        default=int(os.getenv('MAX_KV_TOKENS', '16384')),
        help='Maximum KV cache tokens'
    )
    infer_group.add_argument(
        '--assistant-num',
        type=int,
        default=int(os.getenv('ASSISTANT_NUM', '1')),
        help='Number of assistants'
    )
    infer_group.add_argument(
        '--enable-tts',
        action='store_true',
        default=os.getenv('ENABLE_TTS', '1') != '0',
        help='Enable text-to-speech'
    )
    infer_group.add_argument(
        '--no-tts',
        dest='enable_tts',
        action='store_false',
        help='Disable text-to-speech'
    )
    infer_group.add_argument(
        '--save-dir',
        type=str,
        default=os.getenv('SAVE_DIR', './infer_output'),
        help='Inference output save directory'
    )
    infer_group.add_argument(
        '--threshold',
        type=float,
        default=float(os.getenv('THRESHOLD', '0.5')),
        help='Gate threshold (0.0-1.0)'
    )
    
    # Generation configuration
    gen_group = parser.add_argument_group('Generation Configuration')
    gen_group.add_argument(
        '--do-sample',
        action='store_true',
        default=os.getenv('DO_SAMPLE', '1') != '0',
        help='Enable sampling'
    )
    gen_group.add_argument(
        '--no-sample',
        dest='do_sample',
        action='store_false',
        help='Disable sampling (greedy decoding)'
    )
    gen_group.add_argument(
        '--max-new-tokens',
        type=int,
        default=int(os.getenv('MAX_NEW_TOKENS', '24')),
        help='Maximum number of tokens to generate'
    )
    gen_group.add_argument(
        '--temperature',
        type=float,
        default=float(os.getenv('TEMPERATURE', '0.9')),
        help='Sampling temperature (0.0-2.0)'
    )
    gen_group.add_argument(
        '--top-p',
        type=float,
        default=float(os.getenv('TOP_P', '0.9')),
        help='Nucleus sampling parameter (0.0-1.0)'
    )
    gen_group.add_argument(
        '--repetition-penalty',
        type=float,
        default=float(os.getenv('REPETITION_PENALTY', '1.25')),
        help='Repetition penalty coefficient (1.0 = no penalty)'
    )
    
    return parser.parse_args()


def set_env_from_args(args):
    """Set command line arguments as environment variables for Settings class to read"""
    # Basic configuration
    os.environ['USE_DUMMY'] = '1' if args.use_dummy else '0'
    os.environ['ALLOWED_ORIGINS'] = args.allowed_origins
    os.environ['LOG_LEVEL'] = args.log_level
    
    # Model configuration
    os.environ['CKPT_PATH'] = args.ckpt_path
    os.environ['DEVICE_ID'] = str(args.device_id)
    
    # Inference configuration
    os.environ['USE_AUDIO_IN_VIDEO'] = '1' if args.use_audio_in_video else '0'
    os.environ['MAX_KV_TOKENS'] = str(args.max_kv_tokens)
    os.environ['ASSISTANT_NUM'] = str(args.assistant_num)
    os.environ['ENABLE_TTS'] = '1' if args.enable_tts else '0'
    os.environ['SAVE_DIR'] = args.save_dir
    os.environ['THRESHOLD'] = str(args.threshold)
    
    # Generation configuration
    os.environ['DO_SAMPLE'] = '1' if args.do_sample else '0'
    os.environ['MAX_NEW_TOKENS'] = str(args.max_new_tokens)
    os.environ['TEMPERATURE'] = str(args.temperature)
    os.environ['TOP_P'] = str(args.top_p)
    os.environ['REPETITION_PENALTY'] = str(args.repetition_penalty)


def print_config(args):
    """Print startup configuration"""
    print("=" * 60)
    print("Proact-VL Demo Server - Configuration")
    print("=" * 60)
    print(f"\n[Server Configuration]")
    print(f"  Address: {args.host}:{args.port}")
    print(f"  Auto-reload: {'Yes' if args.reload else 'No'}")
    print(f"  Log level: {args.log_level}")
    
    print(f"\n[Model Configuration]")
    print(f"  Use dummy model: {'Yes' if args.use_dummy else 'No'}")
    if not args.use_dummy:
        print(f"  Checkpoint path: {args.ckpt_path}")
        print(f"  GPU device: cuda:{args.device_id}")
    
    print(f"\n[Inference Configuration]")
    print(f"  Number of assistants: {args.assistant_num}")
    print(f"  Gate threshold: {args.threshold}")
    print(f"  Max KV tokens: {args.max_kv_tokens}")
    print(f"  TTS: {'Enabled' if args.enable_tts else 'Disabled'}")
    
    print(f"\n[Generation Configuration]")
    print(f"  Sampling: {'Enabled' if args.do_sample else 'Greedy'}")
    if args.do_sample:
        print(f"  Max tokens: {args.max_new_tokens}")
        print(f"  Temperature: {args.temperature}")
        print(f"  Top-p: {args.top_p}")
        print(f"  Repetition penalty: {args.repetition_penalty}")
    
    print("=" * 60)
    print()


def main():
    # Parse command line arguments
    args = parse_args()
    
    # Print configuration
    print_config(args)
    
    # Set arguments as environment variables
    set_env_from_args(args)
    
    # Start uvicorn server
    try:
        uvicorn.run(
            "proactvl.app.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level.lower(),
            ws_max_size=20000000  # 20MB
        )
    except KeyboardInterrupt:
        print("\n\nServer stopped")
        sys.exit(0)
    except Exception as e:
        print(f"\nStartup failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
