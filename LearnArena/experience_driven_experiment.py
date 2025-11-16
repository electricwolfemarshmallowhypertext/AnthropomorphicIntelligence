from pathlib import Path
import sys
from envs.registration import make
import agents
import wrappers
import json
import os
import argparse
import logging
import time
from typing import Dict, List, Tuple, Optional
import re
from learnarena_utils.utils import (
    create_agent,
    score_game_quality,
    generate_experience_analysis,
    start_vllm_server,
    stop_vllm_server
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=f'logs/experience_driven_{time.strftime("%Y%m%d_%H%M%S")}.log'
)
logger = logging.getLogger(__name__)

class ExperienceDrivenExperiment:
    """
    Experience-Driven Adaptation: Simple score-based history selection and direct usage without summarization.
    Key features:
    - Score-based selection of top 3 histories (win=10, draw=5, loss=0)
    - Direct history usage without any analysis or summarization
    - No game summaries or external guidance
    """
    
    def __init__(self, player0_model: str = "qwen2.5-32b-chat", player0_port: int = 8020, 
                 player1_port: int = 8010, mode: str = "vllm",
                 player0_api_base: str = None, player0_api_key: str = None,
                 player1_api_base: str = None, player1_api_key: str = None):
        self.player0_model = player0_model
        self.player0_port = player0_port
        self.player1_port = player1_port
        self.mode = mode
        self.player0_api_base = player0_api_base
        self.player0_api_key = player0_api_key
        self.player1_api_base = player1_api_base
        self.player1_api_key = player1_api_key
        
    def get_agent(self, model_name: str, port: int, is_player0: bool = True):
        """Create an agent with specified model and port"""
        api_base = self.player0_api_base if is_player0 else self.player1_api_base
        api_key = self.player0_api_key if is_player0 else self.player1_api_key
        
        return create_agent(
            model_name=model_name,
            port=port,
            mode=self.mode,
            api_base=api_base,
            api_key=api_key,
            is_player0=is_player0,
            timeout=120
        )
    
    def score_game_quality(self, player0_agent, game_history: Dict, game_outcome: str) -> int:
        """Score the game quality on a scale of 0-10 using Player-0 as judge"""
        return score_game_quality(player0_agent, game_history, game_outcome)
    def generate_experience_analysis(self, player1_agent, selected_experiences: List[Dict]) -> str:
        """
        Learning from Experience: Player-1 analyzes selected past experiences and draws conclusions.
        """
        return generate_experience_analysis(player1_agent, selected_experiences)    
    def select_top_histories_by_score(self, game_experiences: List[Dict], k: int = 3) -> List[Dict]:
        """
        Select top k histories from previous games based on simple scores.
        """
        if len(game_experiences) < k:
            return game_experiences
        
        # Sort by score (higher is better) and return top k
        sorted_experiences = sorted(game_experiences, key=lambda x: x.get('score', 0), reverse=True)
        selected = sorted_experiences[:k]
        
        logger.info(f"Selected top {k} games by score: scores {[exp.get('score', 0) for exp in selected]}")
        return selected
    
    def create_direct_history_prompt(self, selected_histories: List[Dict], game: str) -> str:
        """
        Create prompt with selected histories used directly without any summarization or limitations.
        """
        if not selected_histories:
            return ""
        
        prompt = f"Here are your top {len(selected_histories)} previous games in {game} to learn from:\n\n"
        
        for i, history in enumerate(selected_histories, 1):
            prompt += f"=== GAME #{i} ===\n"
            prompt += f"Outcome: {history.get('outcome', 'Unknown')}\n"
            prompt += f"Score: {history.get('score', 0)}\n"
            prompt += f"Number of moves: {len(history.get('moves', []))}\n\n"
            
            # Include all moves from the game without any truncation
            if history.get('moves'):
                prompt += "Game progression:\n"
                for j, move in enumerate(history['moves']):
                    player = move.get('player', 'Unknown')
                    action = move.get('action', 'N/A')
                    # Include full observation without truncation
                    observation = move.get('observation', '')
                    
                    prompt += f"  Move {j+1} (Player {player}):\n"
                    prompt += f"    Observation: {observation}\n"
                    prompt += f"    Action: {action}\n"
                
                prompt += "\n"
            
            prompt += f"Final outcome: {history.get('outcome', 'Unknown')}\n"
            prompt += "=" * 30 + "\n\n"
        
        prompt += "Use these previous games to inform your strategy in the upcoming game.\n\n"
        
        return prompt
    
    def run_single_game(self, game: str, player1_model: str, game_round: int, 
                       game_experiences: List[Dict], history_limit: Optional[int] = None) -> Dict:
        """
        Run a single game between Player-0 and Player-1.
        
        Args:
            game: Game environment ID
            player1_model: Model name for Player-1
            game_round: Current round number
            game_experiences: Previous game experiences for learning
            history_limit: Limit on history size for selection
        """
        logger.info(f"Starting game round {game_round} for {game} with Player-1: {player1_model}")
        
        # Initialize agents
        player0_agent = self.get_agent(self.player0_model, self.player0_port, is_player0=True)
        player1_agent = self.get_agent(player1_model, self.player1_port, is_player0=False)
        
        # Initialize environment
        env = make(env_id=game)
        env = wrappers.LLMObservationWrapper(env=env)
        env = wrappers.SimpleRenderWrapper(
            env=env,
            player_names={0: "Player0", 1: "Player1"},
        )
        
        # Apply learning if we have previous experiences
        history_prompt = ""
        experience_analysis = ""
        selected_histories = []
        
        if game_round > 1 and game_experiences:
            logger.info("Selecting top histories by score")
            
            # Only consider games before history limit
            available_experiences = game_experiences
            if history_limit is not None:
                available_experiences = game_experiences[-history_limit:]
            
            selected_histories = self.select_top_histories_by_score(available_experiences)
            if selected_histories:
                # Get direct history prompt
                history_prompt = self.create_direct_history_prompt(selected_histories, game)
                
                # Get experience analysis from Player-1
                experience_analysis = self.generate_experience_analysis(player1_agent, selected_histories)
                
                # Store original prompt before adding to system prompt
                original_prompt = player1_agent.system_prompt
                
                # Combine learning sources
                learning_prompt = ""
                
                # Add experience analysis if available
                if experience_analysis and not "Error" in experience_analysis:
                    learning_prompt += f"EXPERIENCE ANALYSIS:\n{experience_analysis}\n\n"
                
                # Add direct history
                if history_prompt:
                    learning_prompt += f"GAME HISTORY:\n{history_prompt}\n"
                
                if learning_prompt:
                    learning_prompt += "Apply these insights and historical patterns to maximize your performance.\n\n"
                    combined_prompt = f"{learning_prompt}{original_prompt}"
                    player1_agent.system_prompt = combined_prompt
                
                logger.debug(f"Added learning prompt with {len(selected_histories)} selected games and experience analysis to Player1's system prompt")
        
        # Play the game
        agents = {0: player0_agent, 1: player1_agent}
        
        env.reset(num_players=len(agents))
        game_history = {
            "moves": [],
            "outcome": None
        }
        
        done = False
        move_count = 0
        
        try:
            while not done:
                player_id, observation = env.get_observation()
                logger.debug(f"Player {player_id}'s turn - Move {move_count + 1}")
                
                action = agents[player_id](observation)
                logger.debug(f"Player {player_id} action: {action}")
                
                done, info = env.step(action=action)
                
                game_history["moves"].append({
                    "player": player_id,
                    "observation": observation,
                    "action": action
                })
                move_count += 1
            
            rewards = env.close()
            logger.info(f"Game {game_round} completed. Rewards: Player0={rewards[0]}, Player1={rewards[1]}")
            
            # Determine game outcome
            if rewards[0] > rewards[1]:
                game_history["outcome"] = "Player 0 won"
                player1_won = False
            elif rewards[1] > rewards[0]:
                game_history["outcome"] = "Player 1 won"
                player1_won = True
            else:
                game_history["outcome"] = "Draw"
                player1_won = False
            
            # Score the game simply
            game_score = self.score_game_quality(player0_agent, game_history, game_history["outcome"])
            
            result = {
                "game": game,
                "round": game_round,
                "player0_model": self.player0_model,
                "player1_model": player1_model,
                "rewards": rewards,
                "outcome": game_history["outcome"],
                "player1_won": player1_won,
                "moves": game_history["moves"],
                "score": game_score,
                "history_prompt": history_prompt,
                "experience_analysis": experience_analysis,
                "selected_histories_info": [
                    {
                        "round": exp.get("round", "Unknown"),
                        "outcome": exp.get("outcome", "Unknown"),
                        "score": exp.get("score", 0)
                    }
                    for exp in selected_histories
                ],
                "adaptation_method": "experience_driven_score_based",
                "error": None
            }
            
            logger.info(f"Game {game_round} outcome: {game_history['outcome']}, Score: {game_score}")
            return result
            
        except Exception as e:
            logger.error(f"Error during game {game_round}: {str(e)}")
            return {
                "game": game,
                "round": game_round,
                "player0_model": self.player0_model,
                "player1_model": player1_model,
                "rewards": [0, 0],
                "outcome": "Error",
                "player1_won": False,
                "moves": game_history["moves"],
                "score": 0,
                "history_prompt": "",
                "experience_analysis": "",
                "selected_histories_info": [],
                "adaptation_method": "experience_driven_score_based",
                "error": str(e)
            }
    
    def run_experiment(self, games: List[str], player1_model: str, output_file: str, 
                      num_rounds: int = 20, history_limit: Optional[int] = None) -> List[Dict]:
        """
        Run the complete experience-driven experiment for a given model.
        
        Args:
            games: List of game environment IDs
            player1_model: Model name for Player-1
            output_file: Path to save results
            num_rounds: Number of rounds per game
            history_limit: Limit on history size for selection
        """
        logger.info(f"Starting Experience-Driven Experiment for {player1_model}")
        logger.info(f"Games: {games}")
        logger.info(f"Rounds per game: {num_rounds}")
        logger.info(f"History limit: {history_limit}")
        
        all_results = []
        
        for game in games:
            logger.info(f"Starting experiment for game: {game}")
            game_experiences = []  # Store experiences for this game
            game_results = []
            
            for round_num in range(1, num_rounds + 1):
                # Run single game
                result = self.run_single_game(
                    game=game,
                    player1_model=player1_model,
                    game_round=round_num,
                    game_experiences=game_experiences,
                    history_limit=history_limit
                )
                
                game_results.append(result)
                all_results.append(result)
                
                # Add to experiences for learning (exclude current round from future learning)
                if result["error"] is None:
                    experience = {
                        "round": round_num,
                        "outcome": result["outcome"],
                        "score": result["score"],
                        "moves": result["moves"]
                    }
                    game_experiences.append(experience)
                
                # Save results after each game
                with open(output_file, 'w') as f:
                    json.dump(all_results, f, indent=2)
                
                logger.info(f"Completed round {round_num}/{num_rounds} for {game}")
            
            # Calculate game statistics
            wins = sum(1 for r in game_results if r["player1_won"])
            win_rate = wins / num_rounds if num_rounds > 0 else 0
            avg_score = sum(r["score"] for r in game_results) / num_rounds if num_rounds > 0 else 0
            
            logger.info(f"Game {game} completed:")
            logger.info(f"  Win rate: {win_rate:.2%} ({wins}/{num_rounds})")
            logger.info(f"  Average score: {avg_score:.1f}/10")
        
        logger.info("Experience-Driven Experiment completed")
        return all_results

def parse_games_input(games_str: str) -> List[str]:
    """Parse comma-separated games string into a list of games"""
    if not games_str:
        raise ValueError("Games input cannot be empty")
    
    games = [game.strip() for game in games_str.split(',')]
    games = [game for game in games if game]
    
    if not games:
        raise ValueError("No valid games found in input")
        
    return games

def main():
    parser = argparse.ArgumentParser(description="Experience-Driven Adaptation Experiment")
    parser.add_argument("--player1-model", type=str, required=True, help="Model name for Player-1 (evaluated model)")
    parser.add_argument("--player1-path", type=str, required=False, help="Path to Player-1 model (for vLLM mode)")
    parser.add_argument("--player0-path", type=str, required=False, help="Path to Player-0 model (for vLLM mode)")
    parser.add_argument("--games", type=str, required=True, help="Comma-separated list of games to evaluate")
    parser.add_argument("--output-file", type=str, required=True, help="Output JSON file path")
    parser.add_argument("--num-rounds", type=int, default=20, help="Number of rounds per game")
    parser.add_argument("--history-limit", type=int, default=None, help="Limit on history size for selection")
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
    try:
        games = parse_games_input(args.games)
        logger.info(f"Parsed games: {games}")
    except ValueError as e:
        logger.error(f"Error parsing games input: {str(e)}")
        sys.exit(1)
    
    # Create output directory
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    server_processes = []
    
    if args.mode == "vllm":
        # Import and start vLLM servers
        try:
            # Start Player-0 server
            logger.info(f"Starting vLLM server for Player-0 at {args.player0_path}...")
            proc0 = start_vllm_server(
                model_path=args.player0_path,
                model_name="qwen2.5-32b-chat",
                port=8020,
                gpu=args.gpu
            )
            server_processes.append(proc0)
            
            # Start Player-1 server
            logger.info(f"Starting vLLM server for Player-1 ({args.player1_model}) at {args.player1_path}...")
            proc1 = start_vllm_server(
                model_path=args.player1_path,
                model_name=args.player1_model,
                port=8010,
                gpu=args.gpu
            )
            server_processes.append(proc1)
            
        except ImportError:
            logger.error("Failed to import vLLM utilities")
            sys.exit(1)
    else:
        logger.info(f"Using API mode with Player-0: {args.player0_api_base}, Player-1: {args.player1_api_base}")
    
    try:
        # Run experiment
        experiment = ExperienceDrivenExperiment(
            player0_model="qwen2.5-32b-chat",
            player0_port=8020,
            player1_port=8010,
            mode=args.mode,
            player0_api_base=args.player0_api_base,
            player0_api_key=args.player0_api_key,
            player1_api_base=args.player1_api_base,
            player1_api_key=args.player1_api_key
        )
        results = experiment.run_experiment(
            games=games,
            player1_model=args.player1_model,
            output_file=args.output_file,
            num_rounds=args.num_rounds,
            history_limit=args.history_limit
        )
        
        # Print summary
        print("\nExperience-Driven Experiment Results Summary:")
        print("=" * 80)
        
        # Group results by game
        games_results = {}
        for result in results:
            game = result['game']
            if game not in games_results:
                games_results[game] = []
            games_results[game].append(result)
        
        total_wins = 0
        total_games = 0
        total_score = 0
        
        for game, game_results in games_results.items():
            wins = sum(1 for r in game_results if r["player1_won"])
            win_rate = wins / len(game_results) if game_results else 0
            avg_score = sum(r["score"] for r in game_results) / len(game_results) if game_results else 0
            
            print(f"\nGame: {game}")
            print(f"Player-0 Model: qwen2.5-32b-chat")
            print(f"Player-1 Model: {args.player1_model}")
            print(f"Win Rate: {win_rate:.2%} ({wins}/{len(game_results)})")
            print(f"Average Score: {avg_score:.1f}/10")
            print("-" * 40)
            
            total_wins += wins
            total_games += len(game_results)
            total_score += avg_score * len(game_results)
        
        # Overall statistics
        overall_win_rate = total_wins / total_games if total_games > 0 else 0
        overall_avg_score = total_score / total_games if total_games > 0 else 0
        
        print(f"\nOVERALL PERFORMANCE:")
        print(f"Total Win Rate: {overall_win_rate:.2%} ({total_wins}/{total_games})")
        print(f"Overall Average Score: {overall_avg_score:.1f}/10")
        print(f"Results saved to: {args.output_file}")
        
    finally:
        # Stop all vLLM servers
        if args.mode == "vllm":
            for proc in server_processes:
                stop_vllm_server(proc)

if __name__ == "__main__":
    main()
