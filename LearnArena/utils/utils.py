# utils.py
import json
import os
import time
import requests
from typing import Dict, Any, List, Optional
import subprocess
from openai import OpenAI

STANDARD_GAME_PROMPT = "You are a competitive game player. Make sure you read the game instructions carefully, and always follow the required format."
class LLMAgent():
    """ Agent class using the OpenAI API to generate responses. """
    def __init__(self, model_name: str, system_prompt: Optional[str] = STANDARD_GAME_PROMPT, verbose: bool = False, api_base: Optional[str] = None, api_key: Optional[str] = None, timeout: int = 30, **kwargs):
        """
        Initialize the LLM agent.

        Args:
            model_name (str): The name of the model.
            system_prompt (Optional[str]): The system prompt to use (default: STANDARD_GAME_PROMPT)
            verbose (bool): If True, additional debug info will be printed.
            api_base (Optional[str]): The base URL for the OpenAI API.
            api_key (Optional[str]): The API key for the OpenAI API.
            timeout (int): Timeout in seconds for each request (default: 30)
            **kwargs: Additional keyword arguments to pass to the OpenAI API call.
        """
        self.model_name = model_name 
        self.verbose = verbose 
        self.system_prompt = system_prompt
        self.kwargs = kwargs
        self.timeout = timeout
        self._current_request = None
        # Use provided API key or get from environment variable
        if api_key is None:
            api_key = os.getenv("OPEN_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not found. Please set the OPEN_API_KEY environment variable or provide it directly.")
        
        # Use provided API base or default to localhost
        if api_base is None:
            api_base = "http://localhost:8010/v1"
        
        self.client = OpenAI(base_url=api_base, api_key=api_key)
        

    def _make_request(self, observation: str) -> str:
        """ Make a single API request and return the generated message. """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": observation}
        ]

        # Cancel any existing request
        if self._current_request is not None:
            try:
                self._current_request.cancel()
            except:
                pass

        try:
            # Create a new request with timeout
            self._current_request = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                n=1,
                timeout=self.timeout,
                **self.kwargs
            )
            response = self._current_request
            self._current_request = None
            return response.choices[0].message.content.strip()
        except Exception as e:
            self._current_request = None
            if "timeout" in str(e).lower():
                raise TimeoutError(f"Request timed out after {self.timeout} seconds")
            raise e

    def _retry_request(self, observation: str, retries: int = 5, delay: int = 5) -> str:
        """
        Attempt to make an API request with retries.

        Args:
            observation (str): The input to process.
            retries (int): The number of attempts to try (default: 5).
            delay (int): Seconds to wait between attempts.

        Raises:
            Exception: The last exception caught if all retries fail.
        """
        last_exception = None
        for attempt in range(1, retries + 1):
            try:
                response = self._make_request(observation)
                if self.verbose:
                    print(f"\nObservation: {observation}\nResponse: {response}")
                return response

            except Exception as e:
                last_exception = e
                print(f"Attempt {attempt} failed with error: {e}")
                if attempt < retries:
                    time.sleep(delay)
        raise last_exception

    def __call__(self, observation: str) -> str:
        """
        Process the observation using the OpenRouter API and return the action.

        Args:
            observation (str): The input string to process.

        Returns:
            str: The generated response.
        """
        if not isinstance(observation, str):
            raise ValueError(f"Observation must be a string. Received type: {type(observation)}")
        return self._retry_request(observation)
    

def filter_and_fix_file(file_path):
    """
    Reads a JSONL file, removes invalid lines, and overwrites the original file with only valid lines.
    """
    valid_lines = []
    
    with open(file_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            if line.strip():  # Check if the line is not empty
                try:
                    json.loads(line)  # Attempt to load the line as JSON
                    valid_lines.append(line)  # Store valid lines
                except json.JSONDecodeError:
                    print(f"Invalid JSON line removed: {line.strip()}")  # Log invalid line
    
    # Overwrite the original file with valid lines
    with open(file_path, 'w', encoding='utf-8') as outfile:
        outfile.writelines(valid_lines)

def chat_completion(api_base: str, model_name: str, messages: list, max_tokens: int = 256, temperature: float = 0.7, api_key: str = "xxx") -> str:
    """Generic helper for chat completion supporting both vLLM and API modes."""
    if '/v1' not in api_base:
        api_base = api_base + '/v1'
    
    client = OpenAI(base_url=api_base, api_key=api_key)
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature
    )
    return completion.choices[0].message.content

def read_jsonl(file_path):
    filter_and_fix_file(file_path)
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def write_jsonl(file_path, data_list, append=False):
    """
    Writes a list of dictionaries to a JSONL file.
    If append is True, appends the data to the file instead of overwriting it.
    """
    mode = 'a' if append else 'w'
    
    # check if file exists
    if not os.path.exists(file_path):
        # check the parent directory and create if it doesn't exist
        parent_dir = os.path.dirname(file_path)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        
        # create the file if it doesn't exist
        with open(file_path, 'w', encoding='utf-8') as f:
            pass
        
        # update mode to write
        mode = 'w'


    with open(file_path, mode, encoding='utf-8') as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def start_vllm_server(model_path: str, model_name: str, port: int, gpu: int = 1):
    """
    Launches a vLLM OpenAI API server via subprocess.
    model_path: The path or name of the model you want to host
    port: Which port to host on
    gpu: The tensor-parallel-size (number of GPUs)
    """
    # Command to activate conda environment and start the server
    command = [
        'python', '-m', 'vllm.entrypoints.openai.api_server',
        f'--model={model_path}',
        f"--served-model-name={model_name}",
        f'--tensor-parallel-size={gpu}',
        f"--gpu-memory-utilization=0.85",
        f'--port={port}',
        '--trust-remote-code'
    ]

    process = subprocess.Popen(command, shell=False)
    
    wait_for_server(f"http://localhost:{port}", 600)
    
    print(f"[INFO] Started vLLM server for model '{model_path}' on port {port} (GPU={gpu}).")

    return process


def start_vllm_server_with_gpus(model_path: str, model_name: str, port: int, gpus: List[int]):
    """
    Launches a vLLM OpenAI API server via subprocess with specific GPUs assigned.

    Parameters:
    model_path: str - The path or name of the model you want to host.
    model_name: str - The name of the model to be served.
    port: int - The port to host the server on.
    gpus: List[int] - List of GPU indices to be assigned for this server.

    Returns:
    process: subprocess.Popen - The process running the vLLM server.
    """
    gpu_list = ",".join(map(str, gpus))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list

    command = [
        'python', '-m', 'vllm.entrypoints.openai.api_server',
        f'--model={model_path}',
        f'--served-model-name={model_name}',
        f'--tensor-parallel-size={len(gpus)}',
        '--gpu-memory-utilization=0.85',
        f'--port={port}',
        '--trust-remote-code'
    ]

    process = subprocess.Popen(command, shell=False, env=os.environ.copy())
    
    wait_for_server(f"http://localhost:{port}", 600)

    print(f"[INFO] Started vLLM server for model '{model_name}' on port {port} with GPUs {gpu_list}.")

    return process

def allocate_gpus(total_gpus: int, processes: int) -> List[List[int]]:
    """
    Allocate GPUs for multiple processes.

    Parameters:
    total_gpus: int - Total number of GPUs available.
    processes: int - Number of processes to allocate GPUs for.

    Returns:
    List[List[int]] - A list where each sublist contains the GPUs assigned to a process.
    """
    if total_gpus < processes:
        raise ValueError("Not enough GPUs available for the number of processes.")

    gpus_per_process = total_gpus // processes
    extra_gpus = total_gpus % processes

    allocation = []
    start = 0

    for i in range(processes):
        end = start + gpus_per_process + (1 if i < extra_gpus else 0)
        allocation.append(list(range(start, end)))
        start = end

    return allocation



def wait_for_server(url: str, timeout: int = 600):
    """
    Polls the server's /models endpoint until it responds with HTTP 200 or times out.
    """
    start_time = time.time()
    while True:
        try:
            r = requests.get(url + "/v1/models", timeout=3)
            if r.status_code == 200:
                print("[INFO] vLLM server is up and running.")
                return
        except Exception:
            pass
        if time.time() - start_time > timeout:
            raise RuntimeError(f"[ERROR] Server did not start at {url} within {timeout} seconds.")
        time.sleep(2)
        
def stop_vllm_server(process):
    process.terminate()
    process.wait()
    print("[INFO] Stopped vLLM server.")



def create_output_directory(model_name: str):
    """
    Creates the output directory named after the LLM model, if it doesn't exist.
    Returns the path to that directory.
    """
    output_dir = os.path.join("outputs", model_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


# ============================
# Common Agent Functions
# ============================

def create_agent(model_name: str, port: int, mode: str = "vllm", 
                 api_base: str = None, api_key: str = None, 
                 is_player0: bool = True, timeout: int = 120):
    """
    Create an agent with specified model and port.
    
    Args:
        model_name: Name of the model
        port: Port number for vLLM mode
        mode: Either "vllm" for local server or "api" for external API
        api_base: API base URL (for API mode)
        api_key: API key (for API mode)
        is_player0: Whether this is player 0 (affects environment variable lookup)
        timeout: Request timeout in seconds
        
    Returns:
        Configured agent with error handling wrapper
    """
    import logging
    
    logger = logging.getLogger(__name__)
    
    if mode == "api":
        # Try to get from environment if not provided
        if not api_key:
            env_var = "API_KEY_0" if is_player0 else "API_KEY_1"
            api_key = os.getenv(env_var, "your_api_key_here")
            
        agent = LLMAgent(
            model_name=model_name,
            api_base=api_base,
            api_key=api_key,
            timeout=timeout
        )
    else:
        # vLLM mode
        agent = LLMAgent(
            model_name=model_name,
            api_base=f"http://localhost:{port}/v1",
            api_key="your_api_key_here",
            timeout=timeout
        )
    
    # Wrap agent call for better error handling
    original_call = agent.__call__
    def safe_call(observation: str) -> str:
        try:
            response = original_call(observation)
            if response is None:
                logger.error(f"Agent {model_name} returned None response")
                return "Error: No response generated"
            return response
        except Exception as e:
            logger.error(f"Error in {model_name} agent call: {type(e).__name__}: {str(e)}")
            return f"Error: {type(e).__name__}: {str(e)}"
    
    agent.__call__ = safe_call
    return agent


# ============================
# Game Summary Functions
# ============================

def get_game_summary(game: str, agent, env, summary_dir: str = "environment_summary") -> str:
    """
    Generate or load a summary of the game rules with strategic advice for winning.
    
    Args:
        game: Game name/ID
        agent: Agent to use for summary generation
        env: Environment instance
        summary_dir: Directory to store summaries
        
    Returns:
        Game summary string with strategic advice
    """
    import json
    import logging
    
    logger = logging.getLogger(__name__)
    summary_file = os.path.join(summary_dir, f"{game}.jsonl")
    
    # Check if summary already exists
    if os.path.exists(summary_file):
        logger.info(f"Loading existing game summary for {game}")
        try:
            with open(summary_file, 'r') as f:
                summary_data = json.load(f)
                return summary_data.get("summary", "")
        except Exception as e:
            logger.error(f"Error loading game summary: {str(e)}")
    
    # Generate new summary
    logger.info(f"Generating new game summary for {game}")
    
    # Reset environment to get initial observation
    env.reset(num_players=2)
    player_id, observation = env.get_observation()
    
    # Save original system prompt
    original_system_prompt = agent.system_prompt
    
    # Set system prompt for summary generation
    summary_prompt = ("You are an expert game strategist with deep knowledge of game theory and optimal play. "
                     "Your task is to provide concise, actionable strategic advice that will help a player win. "
                     "Focus on identifying winning patterns, key decision points, and optimal strategies.")
    agent.system_prompt = summary_prompt
    
    # Generate the summary
    prompt = (
        f"For this game '{game}', provide very brief winning strategies based on this initial observation. "
        f"Don't explain rules in detail.\n\n"
        f"WINNING STRATEGIES:\n"
        f"- Top 3-5 strategic principles that lead to victory\n"
        f"- Best opening moves or early game tactics\n"
        f"- Key patterns to recognize during gameplay\n" 
        f"- Critical mistakes to avoid\n\n"
        f"Game observation: {observation}\n\n"
        f"Keep your response concise and focused on practical advice that will maximize winning chances."
    )
    
    try:
        summary = agent(prompt)
        
        # Create directory and save summary
        os.makedirs(summary_dir, exist_ok=True)
        with open(summary_file, 'w') as f:
            json.dump({"game": game, "summary": summary}, f)
            
        logger.info(f"Saved game summary for {game}")
    except Exception as e:
        logger.error(f"Error generating game summary: {str(e)}")
        summary = f"Error generating summary: {str(e)}"
    
    # Restore original system prompt
    agent.system_prompt = original_system_prompt
    env.close()
    
    return summary


# ============================
# Game Analysis Functions
# ============================

def score_game_quality(agent, game_history: Dict, game_outcome: str) -> int:
    """
    Score the game quality on a scale of 0-10 using an agent as judge.
    
    Args:
        agent: Agent to use for scoring (typically player 0)
        game_history: Dictionary containing game moves and outcome
        game_outcome: String description of game outcome
        
    Returns:
        Score from 0-10
    """
    import re
    import logging
    
    logger = logging.getLogger(__name__)
    
    if not game_history.get("moves"):
        return 0
    
    last_move = game_history["moves"][-1]
    
    # Save original system prompt
    original_prompt = agent.system_prompt
    
    try:
        # Set system prompt for scoring
        agent.system_prompt = ("You are an expert game judge. Evaluate game quality objectively "
                              "on a scale of 0-10 based on strategic depth, tactical execution, "
                              "and overall gameplay quality.")
        
        scoring_prompt = (
            f"Score this game on a scale of 0-10 based on overall gameplay quality.\n"
            f"Game outcome: {game_outcome}\n"
            f"Final observation: {last_move['observation']}\n"
            f"Final action: {last_move['action']}\n\n"
            f"Rating scale:\n"
            f"0-2: Poor - Basic mistakes, no clear strategy\n"
            f"3-4: Fair - Some good moves but inconsistent play\n"
            f"5-6: Good - Solid strategic play with clear planning\n"
            f"7-8: Very Good - Strong tactical execution and strategy\n"
            f"9-10: Excellent - Masterful play with optimal decisions\n\n"
            f"End your response with 'Score: X/10' where X is your rating."
        )
        
        score_response = agent(scoring_prompt)
        if score_response is None or "Error:" in score_response:
            logger.error(f"Error getting game score: {score_response}")
            return 0
        
        # Extract score using regex
        match = re.search(r'(\d+)/10', score_response)
        if match:
            return int(match.group(1))
        else:
            logger.warning(f"Could not extract score from response: {score_response}")
            return 0
            
    except Exception as e:
        logger.error(f"Exception getting game score: {type(e).__name__}: {str(e)}")
        return 0
    finally:
        # Restore original system prompt
        agent.system_prompt = original_prompt


def generate_instructor_feedback(agent, game_history: Dict, game_outcome: str) -> str:
    """
    Generate instructor feedback for a game (Learning from Instructor).
    
    Args:
        agent: Agent to use for feedback generation (typically player 0)
        game_history: Dictionary containing game moves and outcome
        game_outcome: String description of game outcome
        
    Returns:
        Feedback string with strategic advice
    """
    import logging
    
    logger = logging.getLogger(__name__)
    
    if not game_history.get("moves"):
        return "No game moves to analyze"
    
    last_move = game_history["moves"][-1]
    
    # Save original system prompt
    original_prompt = agent.system_prompt
    
    try:
        # Set system prompt for instructor role
        agent.system_prompt = ("You are an expert game instructor. Analyze the game and provide "
                              "constructive feedback and strategic advice to help the player improve. "
                              "Keep your advice concise and actionable, under 300 words.")
        
        feedback_prompt = (
            f"As an expert instructor, analyze this game and provide strategic advice for improvement.\n"
            f"Game outcome: {game_outcome}\n"
            f"Final observation: {last_move['observation']}\n"
            f"Final action: {last_move['action']}\n\n"
            f"Provide feedback in this format:\n"
            f"1. Key strengths in the gameplay\n"
            f"2. Areas for improvement\n"
            f"3. Specific strategic advice\n"
            f"4. Tactical recommendations for future games\n\n"
            f"Focus on actionable insights that will lead to better performance."
        )
        
        feedback = agent(feedback_prompt)
        if feedback is None or "Error:" in feedback:
            logger.error(f"Error getting instructor feedback: {feedback}")
            return "Error generating instructor feedback"
        
        return feedback
        
    except Exception as e:
        logger.error(f"Exception getting instructor feedback: {type(e).__name__}: {str(e)}")
        return f"Error: {type(e).__name__}: {str(e)}"
    finally:
        # Restore original system prompt
        agent.system_prompt = original_prompt


def generate_experience_analysis(agent, selected_experiences: List[Dict]) -> str:
    """
    Generate experience analysis from past games (Learning from Experience).
    
    Args:
        agent: Agent to use for analysis (typically player 1)
        selected_experiences: List of past game experience dictionaries
        
    Returns:
        Analysis string with insights and recommendations
    """
    import logging
    
    logger = logging.getLogger(__name__)
    
    if not selected_experiences:
        return ""
    
    # Save original system prompt
    original_prompt = agent.system_prompt
    
    try:
        # Set system prompt for self-analysis
        agent.system_prompt = ("You are analyzing your own gameplay across multiple matches. "
                              "Synthesize your experiences into actionable insights. "
                              "Focus on patterns, strategies, and lessons learned.")
        
        analysis_prompt = "Analyze your past gameplay experiences and provide key insights:\n\n"
        
        for i, exp in enumerate(selected_experiences, 1):
            outcome = exp.get('outcome', 'Unknown')
            score = exp.get('score', 0)
            analysis_prompt += f"Experience #{i} (Score: {score}/10, Outcome: {outcome}):\n"
            
            if 'moves' in exp and exp['moves']:
                last_move = exp['moves'][-1]
                analysis_prompt += f"Key move: {last_move.get('action', 'N/A')}\n"
            
            analysis_prompt += "\n"
        
        analysis_prompt += ("Based on these experiences, provide:\n"
                          "1. Key patterns you've identified\n"
                          "2. Successful strategies to continue\n"
                          "3. Mistakes to avoid\n"
                          "4. Strategic insights for future games\n\n"
                          "Synthesize into actionable guidelines for improved performance.")
        
        analysis = agent(analysis_prompt)
        if analysis is None or "Error:" in analysis:
            logger.error(f"Error getting experience analysis: {analysis}")
            return "Error generating experience analysis"
        
        return analysis
        
    except Exception as e:
        logger.error(f"Exception getting experience analysis: {type(e).__name__}: {str(e)}")
        return f"Error: {type(e).__name__}: {str(e)}"
    finally:
        # Restore original system prompt
        agent.system_prompt = original_prompt
