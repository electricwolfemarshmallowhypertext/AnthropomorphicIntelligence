from pathlib import Path
# import textarena
from textarena.envs.registration import make
import wrappers
import json
import os
import argparse
import logging
import time
from typing import Dict, List, Tuple, Optional
from utils.utils import (
    create_agent,
    get_game_summary,
    start_vllm_server,
    start_vllm_server_with_gpus,
    stop_vllm_server,
    allocate_gpus,
)

if not os.path.exists('logs'):
    os.makedirs('logs')
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=f'logs/model_scale_experiment_{time.strftime("%Y%m%d_%H%M%S")}.log'
)
logger = logging.getLogger(__name__)

def run_game(game: str, player0_model: str, player1_model: str, with_concept: bool, num_rounds: int = 20,
             mode: str = "vllm", player0_api_base: str = None, player0_api_key: str = None,
             player1_api_base: str = None, player1_api_key: str = None) -> Dict:
    """Run a set of games between Player-0 and Player-1 with or without concept guidance"""
    logger.info(f"Starting games for {game} with Player-0: {player0_model}, Player-1: {player1_model} (with_concept={with_concept})")
    
    # Initialize agents
    player0 = create_agent(player0_model, 8000, mode, player0_api_base, player0_api_key, is_player0=True)
    player1 = create_agent(player1_model, 8001, mode, player1_api_base, player1_api_key, is_player0=False)
    
    # Get game summary if using concept guidance
    if with_concept:
        env = make(env_id=game)
        env = wrappers.LLMObservationWrapper(env=env)
        game_summary = get_game_summary(game, player0, env)
        player1.system_prompt = f"GAME ANALYSIS AND WINNING STRATEGIES:\n{game_summary}\n\nREMEMBER: Apply these strategic principles consistently to maximize your chances of winning.\n\n{player1.system_prompt}"
    
    # Track results
    results = {
        "game": game,
        "player0_model": player0_model,
        "player1_model": player1_model,
        "with_concept": with_concept,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "total_games": num_rounds
    }
    
    # Play games
    for game_num in range(num_rounds):
        try:
            env = make(env_id=game)
            env = wrappers.LLMObservationWrapper(env=env)
            env = wrappers.SimpleRenderWrapper(
                env=env,
                player_names={0: "Player0", 1: "Player1"},
            )
            
            env.reset(num_players=2)
            done = False
            
            while not done:
                player_id, observation = env.get_observation()
                action = player0(observation) if player_id == 0 else player1(observation)
                done, _ = env.step(action=action)
            
            rewards = env.close()
            
            # Record outcome
            if rewards[1] > rewards[0]:
                results["wins"] += 1
            elif rewards[1] < rewards[0]:
                results["losses"] += 1
            else:
                results["draws"] += 1
                
            logger.info(f"Game {game_num + 1}/{num_rounds} completed. Rewards: Player0={rewards[0]}, Player1={rewards[1]}")
            
        except Exception as e:
            logger.error(f"Error in game {game_num + 1}: {str(e)}")
            continue
    
    # Calculate win rate
    results["win_rate"] = results["wins"] / num_rounds if num_rounds > 0 else 0
    return results

def run_experiment(games: List[str], player0_model: str, player1_model: str, output_file: str, num_rounds: int = 20,
                   mode: str = "vllm", player0_api_base: str = None, player0_api_key: str = None,
                   player1_api_base: str = None, player1_api_key: str = None):
    """Run the full experiment across all games"""
    results = []
    
    for game in games:
        # Run without concept
        logger.info(f"Running {game} without concept")
        results_without = run_game(game, player0_model, player1_model, with_concept=False, num_rounds=num_rounds,
                                   mode=mode, player0_api_base=player0_api_base, player0_api_key=player0_api_key,
                                   player1_api_base=player1_api_base, player1_api_key=player1_api_key)
        results.append(results_without)
        
        # Run with concept
        logger.info(f"Running {game} with concept")
        results_with = run_game(game, player0_model, player1_model, with_concept=True, num_rounds=num_rounds,
                               mode=mode, player0_api_base=player0_api_base, player0_api_key=player0_api_key,
                               player1_api_base=player1_api_base, player1_api_key=player1_api_key)
        results.append(results_with)
        
        # Save results after each game
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Run model scale experiment with concept guidance")
    parser.add_argument("--games", type=str, required=True, help="Comma-separated list of games to evaluate")
    parser.add_argument("--output-file", type=str, required=True, help="Output JSON file path")
    parser.add_argument("--num-rounds", type=int, default=20, help="Number of rounds per game")
    parser.add_argument("--player0-model", type=str, required=True, help="Model name for Player-0")
    parser.add_argument("--player0-path", type=str, required=False, help="Path to Player-0 model (for vLLM mode)")
    parser.add_argument("--player1-model", type=str, required=True, help="Model name for Player-1")
    parser.add_argument("--player1-path", type=str, required=False, help="Path to Player-1 model (for vLLM mode)")
    parser.add_argument("--gpu", type=int, default=4, help="Number of GPUs to use (vLLM mode)")
    
    # API mode arguments
    parser.add_argument("--mode", type=str, default="vllm", choices=["vllm", "api"], 
                       help="Mode: 'vllm' for local vLLM servers, 'api' for external API endpoints")
    parser.add_argument("--player0-api-base", type=str, help="API base URL for Player-0 (API mode)")
    parser.add_argument("--player0-api-key", type=str, help="API key for Player-0 (API mode, or set API_KEY_0 env var)")
    parser.add_argument("--player1-api-base", type=str, help="API base URL for Player-1 (API mode)")
    parser.add_argument("--player1-api-key", type=str, help="API key for Player-1 (API mode, or set API_KEY_1 env var)")
    args = parser.parse_args()
    
    # Validate arguments based on mode
    if args.mode == "vllm":
        if not args.player0_path or not args.player1_path:
            parser.error("--player0-path and --player1-path are required for vLLM mode")
    elif args.mode == "api":
        if not args.player0_api_base or not args.player1_api_base:
            parser.error("--player0-api-base and --player1-api-base are required for API mode")
    
    # Parse games
    games = [game.strip() for game in args.games.split(",")]
    
    server_processes = []
    
    if args.mode == "vllm":
        # Start vLLM servers for both models
        gpu_allocations = []
        if args.gpu >= 2:
            try:
                gpu_allocations = allocate_gpus(args.gpu, 2)
                logger.info(f"Allocated GPUs for vLLM servers: {gpu_allocations}")
            except ValueError as e:
                logger.warning(f"GPU allocation failed ({e}); falling back to shared configuration.")
                gpu_allocations = []

        # Start Player-0 server
        logger.info(f"Starting vLLM server for Player-0 ({args.player0_model}) at {args.player0_path}...")
        if gpu_allocations:
            proc0 = start_vllm_server_with_gpus(
                model_path=args.player0_path,
                model_name=args.player0_model,
                port=8000,
                gpus=gpu_allocations[0]
            )
        else:
            proc0 = start_vllm_server(
                model_path=args.player0_path,
                model_name=args.player0_model,
                port=8000,
                gpu=args.gpu
            )
        server_processes.append(proc0)
        
        # Start Player-1 server
        logger.info(f"Starting vLLM server for Player-1 ({args.player1_model}) at {args.player1_path}...")
        if gpu_allocations:
            proc1 = start_vllm_server_with_gpus(
                model_path=args.player1_path,
                model_name=args.player1_model,
                port=8001,
                gpus=gpu_allocations[1]
            )
        else:
            proc1 = start_vllm_server(
                model_path=args.player1_path,
                model_name=args.player1_model,
                port=8001,
                gpu=args.gpu
            )
        server_processes.append(proc1)
    else:
        logger.info(f"Using API mode with Player-0: {args.player0_api_base}, Player-1: {args.player1_api_base}")
    
    try:
        # Run experiment
        results = run_experiment(
            games, 
            args.player0_model, 
            args.player1_model, 
            args.output_file, 
            args.num_rounds,
            mode=args.mode,
            player0_api_base=args.player0_api_base,
            player0_api_key=args.player0_api_key,
            player1_api_base=args.player1_api_base,
            player1_api_key=args.player1_api_key
        )
        
        # Print summary
        print("\nExperiment Results Summary:")
        print("=" * 80)
        for result in results:
            print(f"\nGame: {result['game']}")
            print(f"Player-0 Model: {result['player0_model']}")
            print(f"Player-1 Model: {result['player1_model']}")
            print(f"With Concept: {result['with_concept']}")
            print(f"Win Rate: {result['win_rate']:.2%}")
            print(f"Wins: {result['wins']}, Losses: {result['losses']}, Draws: {result['draws']}")
            print("-" * 40)
            
    finally:
        # Stop all vLLM servers
        if args.mode == "vllm":
            for proc in server_processes:
                stop_vllm_server(proc)

if __name__ == "__main__":
    main() 