from pathlib import Path
import sys
project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

import custom_environment.TextArena.textarena as ta
import json
import os
import argparse
import logging
import time
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Lock
from utils.utils import start_vllm_server, stop_vllm_server

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=f'logs/model_scale_experiment_{time.strftime("%Y%m%d_%H%M%S")}.log'
)
logger = logging.getLogger(__name__)

def get_agent(model_name: str, port: int):
    """Create an agent with specified model and port"""
    agent = ta.agents.OpenRouterAgent(
        model_name=model_name,
        api_base=f"http://localhost:{port}/v1",
        api_key="your_api_key_here",
        timeout=120
    )
    return agent

def get_game_summary(game: str, agent, env) -> str:
    """Generate or load a summary of the game rules with strategic advice"""
    summary_dir = "environment_summary"
    summary_file = os.path.join(summary_dir, f"{game}.jsonl")
    
    if os.path.exists(summary_file):
        logger.info(f"Loading existing game summary for {game}")
        try:
            with open(summary_file, 'r') as f:
                summary_data = json.load(f)
                return summary_data.get("summary", "")
        except Exception as e:
            logger.error(f"Error loading game summary: {str(e)}")
    
    logger.info(f"Generating new game summary for {game}")
    
    env.reset(num_players=2)
    player_id, observation = env.get_observation()
    
    original_system_prompt = agent.system_prompt
    summary_prompt = "You are an expert game strategist with deep knowledge of game theory and optimal play. Your task is to provide concise, actionable strategic advice that will help a player win. Focus on identifying winning patterns, key decision points, and optimal strategies."
    agent.system_prompt = summary_prompt
    
    prompt = (
        f"For this game '{game}', provide brief winning strategies based on this initial observation.\n\n"
        f"WINNING STRATEGIES:\n\n"
        f"Top 3-5 strategic principles that lead to victory\n\n"
        f"Best opening moves or early game tactics\n\n"
        f"Key patterns to recognize during gameplay\n\n"
        f"Critical mistakes to avoid\n\n"
        f"Rules of the Game: {observation}\n\n"
        f"Keep your response concise and focused on practical advice that will maximize winning chances."
    )
    
    try:
        summary = agent(prompt)
        os.makedirs(summary_dir, exist_ok=True)
        with open(summary_file, 'w') as f:
            json.dump({"game": game, "summary": summary}, f)
        logger.info(f"Saved game summary for {game}")
    except Exception as e:
        logger.error(f"Error generating game summary: {str(e)}")
        summary = f"Error generating summary: {str(e)}"
    
    agent.system_prompt = original_system_prompt
    env.close()
    return summary

def run_game(game: str, player0_model: str, player1_model: str, with_concept: bool, num_rounds: int = 20) -> Dict:
    """Run a set of games between Player-0 and Player-1 with or without concept guidance"""
    logger.info(f"Starting games for {game} with Player-0: {player0_model}, Player-1: {player1_model} (with_concept={with_concept})")
    
    # Initialize agents
    player0 = get_agent(player0_model, 8000)  # Player-0 always uses port 8000
    player1 = get_agent(player1_model, 8001)  # Player-1 always uses port 8001
    
    # Get game summary if using concept guidance
    if with_concept:
        env = ta.make(env_id=game)
        env = ta.wrappers.LLMObservationWrapper(env=env)
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
            env = ta.make(env_id=game)
            env = ta.wrappers.LLMObservationWrapper(env=env)
            env = ta.wrappers.SimpleRenderWrapper(
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

def run_experiment(games: List[str], player0_model: str, player1_model: str, output_file: str, num_rounds: int = 20):
    """Run the full experiment across all games"""
    results = []
    
    for game in games:
        # Run without concept
        logger.info(f"Running {game} without concept")
        results_without = run_game(game, player0_model, player1_model, with_concept=False, num_rounds=num_rounds)
        results.append(results_without)
        
        # Run with concept
        logger.info(f"Running {game} with concept")
        results_with = run_game(game, player0_model, player1_model, with_concept=True, num_rounds=num_rounds)
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
    parser.add_argument("--player0-path", type=str, required=True, help="Path to Player-0 model")
    parser.add_argument("--player1-model", type=str, required=True, help="Model name for Player-1")
    parser.add_argument("--player1-path", type=str, required=True, help="Path to Player-1 model")
    parser.add_argument("--gpu", type=int, default=4, help="Number of GPUs to use")
    args = parser.parse_args()
    
    # Parse games
    games = [game.strip() for game in args.games.split(",")]
    
    # Start vLLM servers for both models
    server_processes = []
    
    # Start Player-0 server
    logger.info(f"Starting vLLM server for Player-0 ({args.player0_model}) at {args.player0_path}...")
    proc0 = start_vllm_server(
        model_path=args.player0_path,
        model_name=args.player0_model,
        port=8000,
        gpu=args.gpu
    )
    server_processes.append(proc0)
    
    # Start Player-1 server
    logger.info(f"Starting vLLM server for Player-1 ({args.player1_model}) at {args.player1_path}...")
    proc1 = start_vllm_server(
        model_path=args.player1_path,
        model_name=args.player1_model,
        port=8001,
        gpu=args.gpu
    )
    server_processes.append(proc1)
    
    try:
        # Run experiment
        results = run_experiment(
            games, 
            args.player0_model, 
            args.player1_model, 
            args.output_file, 
            args.num_rounds
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
        for proc in server_processes:
            stop_vllm_server(proc)

if __name__ == "__main__":
    main() 