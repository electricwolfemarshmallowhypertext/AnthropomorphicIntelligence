import json, logging, requests, websockets
import ssl
import asyncio
from typing import List, Optional, Tuple, Dict, Any, Union
from urllib.parse import urlencode
import warnings
from urllib3.exceptions import InsecureRequestWarning
import traceback

# Suppress SSL warnings
warnings.filterwarnings('ignore', category=InsecureRequestWarning)

# Server URLs - Change these to match your server
MATCHMAKING_WS_URI = "wss://matchmaking.textarena.ai/ws"
MATCHMAKING_HTTP_URI = "https://matchmaking.textarena.ai"

# Environment ID mapping
NAME_TO_ID_DICT = {
    "Chess-v0": 0,
    "ConnectFour-v0": 1,
    "DontSayIt-v0": 3,
    "Battleship-v0": 5,
    "LiarsDice-v0": 6,
    "SimpleNegotiation-v0": 8,
    "Poker-v0": 9,
    "SpellingBee-v0": 10,
    "SpiteAndMalice-v0": 11,
    "Stratego-v0": 12,
    "Tak-v0": 13,
    "TruthAndDeception-v0": 14,
    "UltimateTicTacToe-v0": 15,
    "WordChains-v0": 16,
    "TicTacToe-v0": 35,
    "Breakthrough-v0": 37,
    "Checkers-v0": 38,
    "KuhnPoker-v0": 46,
    "LetterAuction-v0": 47,
    "MemoryGame-v0": 48,
    "Nim-v0": 50,
    "Othello-v0": 51,
    "PigDice-v0": 52,
    "SimpleBlindAuction-v0": 56,
    "Snake-v0": 69,
    "SecretMafia-v0": 75,
    "WildTicTacToe-v0": 77,
    "ReverseTicTacToe-v0": 78,
    "RandomizedTicTacToe-v0": 79,
    "QuantumTicTacToe-v0": 80,
    "IteratedRockPaperScissors-v0": 81,
    "Subset-v0": [0, 3, 35, 51, 52, 63, 66, 67, 68, 70, 75, 81]
}

class OnlineEnvWrapper:
    """
    A wrapper class to interact with online game environments via matchmaking and game servers.
    
    This class handles:
    - Registering a model with the matchmaking server
    - Connect to matchmaking queues for all types of games
    - Establishing a Websocket connection to the game server
    - Managing game loop, observations, actions and game state
    
    Intended for use with agents via async.
    """

    def __init__(self, env_ids: List[int], model_name: str, model_token: str):
        """
        Initialize the online environment wrapper.
        
        Args:
            env_ids: List of environment IDs to connect to
            model_name: Name of the model
            model_token: Token for the model
        """
        self.env_ids = env_ids
        self.model_name = model_name
        self.model_token = model_token
        
        # Connection variables
        self.websocket = None
        self.matchmaking_websocket = None
        self.game_url = None
        self.environment_id = None
        self.env_id = None
        
        # The full observations are stored as a dictionary mapping player id -> list of (sender_id, message) tuples
        self.full_observations = {}
        
        # Game state tracking
        self.current_player_id = None
        self.current_observation = None
        self.game_over = False
        self.server_shutdown = False 
        self.game_over_timeout = 30.0  # Time to wait for additional messages after game_over
        self.rewards = {}
        self.info = {}
        
        # Timeouts
        self.matchmaking_timeout = 1800  # Timeout for matchmaking (30 minutes)
        
        # Async queues for incoming/outgoing messages
        self.message_queue = asyncio.Queue()
        self.action_queue = asyncio.Queue()
        self.matchmaking_queue = asyncio.Queue()
        
        # State tracking
        self.in_game = False
        self.pending_action = False
        self.update_task = None  # Reference to the main update loop task
        self.matchmaking_complete = False
        
        # For compatibility
        DummyState = type("DummyState", (), {})
        self.state = DummyState()
        self.state.role_mapping = {0: "Player 0", 1: "Player 1", -1: "GAME"}

    async def _message_receiver(self):
        """
        Background task that listens to messages from the game server websocket
        and places them into the internal message queue for processing.
        
        Also performs a quick check for 'server shutdown' messages to gracefully exit early.
        """
        try:
            while True:
                try:
                    message = await self.websocket.recv()
                    print(f"Received: {message}")

                    # put the raw message into the queue for processing
                    await self.message_queue.put(message)
                    
                    # Proactively check for 'server_shutdown' command to allow early exit
                    try:
                        msg_data = json.loads(message)
                        if msg_data.get("command") == "server_shutdown":
                            print("Server shutdown message detected in receiver")
                            self.server_shutdown = True
                    except:
                        pass
                        
                except websockets.exceptions.ConnectionClosed:
                    print("WebSocket connection closed by server")
                    self.server_shutdown = True
                    break

                except Exception as e:
                    print(f"Error receiving message: {e}")
                    break

        except Exception as e:
            print(f"Message receiver error: {e}")
            self.server_shutdown = True  # Set server_shutdown to ensure all loops terminate

    async def _matchmaking_receiver(self):
        """
        Background task that listens to the matchmaking websocket.
        
        It reads and queues all messages until a match is found or the connection is closed.
        """
        try:
            while not self.matchmaking_complete:
                try:
                    message = await self.matchmaking_websocket.recv()
                    print(f"Received from matchmaking: {message}")

                    # pass the raw message to the matchmaking queue for processing
                    await self.matchmaking_queue.put(message)

                except websockets.exceptions.ConnectionClosed:
                    print("Matchmaking WebSocket connection closed")
                    break

                except Exception as e:
                    print(f"Error receiving matchmaking message: {e}")
                    break

        except Exception as e:
            print(f"Matchmaking receiver error: {e}")

    async def _action_sender(self):
        """
        Background task that listens for actions from the action_queue and sends them to the game server.

        Waits for actions like `"play x y"` or `"bet 3"`, and handles graceful shutdown when it receives `"CLOSE"`.
        """
        try:
            while True:
                # Wait for the next action to send
                action = await self.action_queue.get()

                # Special signal to close teh sender task
                if action == "CLOSE":
                    break
                
                try:
                    # Format and send the action
                    action_msg = {"command": "action", "action": action}
                    await self.websocket.send(json.dumps(action_msg))
                    print(f"Sent action: {action[:100]}...")

                    # Mark that we've sent an action and are waiting for a response
                    self.pending_action = True

                except Exception as e:
                    print(f"Error sending action: {e}")
                
                # Mark the task as done so that other coroutines waiting on .join() can proceed
                self.action_queue.task_done()

        except Exception as e:
            print(f"Action sender error: {e}")
            self.server_shutdown = True  # Trigger cleanup if error occurs

    async def _ping_sender(self):
        """
        Background task to send periodic pings to the game server.

        This helps to keep the connection alive and detect if the server is still responsive.
        """
        try:
            while not self.server_shutdown:
                try:
                    # Send a ping message to the server
                    await self.websocket.send(json.dumps({"command": "ping"}))
                    await asyncio.sleep(25)  # Send ping every 25 seconds

                except Exception as e:
                    print(f"Ping error: {e}")
                    break

        except Exception as e:
            print(f"Ping sender error: {e}")
            self.server_shutdown = True  # Trigger cleanup if error occurs


    async def _process_matchmaking_message(self, message_str: str):
        """
        Handle a single message received from the matchmaking server.
        
        Depending on the 'command', this would update the queue status,
        complete the matchmaking, or handle errors. This is called by the matchmaking loop.
        """
        try:
            message = json.loads(message_str)
            command = message.get("command")
            
            if command == "queued":
                # Status: In queue
                avg_queue_time = message.get("avg_queue_time", 0)
                num_players = message.get("num_players_in_queue", 0)
                print(f"In queue. Average wait time: {avg_queue_time:.1f}s. Players in queue: {num_players}")
                
            elif command == "match_found":
                # Status: Match found - capture the game server details and environment ID
                self.game_url = message.get("game_url")
                self.env_id = message.get("env_id")
                self.environment_id = message.get("environment_id")
                print(f"Match found! Connecting to game server: {self.game_url}")
                self.matchmaking_complete = True
                
            elif command == "error":
                # Status: Server-side error
                error_msg = message.get("message", "Unknown error")
                print(f"Matchmaking error: {error_msg}")
                
            elif command == "left":
                # Status: Client leaving the matchmaking queue
                print("Left matchmaking queue")
                
            else:
                print(f"Unknown matchmaking command: {command}")
                
        except json.JSONDecodeError:
            print(f"Invalid JSON received from matchmaking: {message_str}")

        except Exception as e:
            print(f"Error processing matchmaking message: {e}")

    async def connect_to_matchmaking(self):
        """
        Establish a WebSocket connection to the matchmaking server and queue for a game.

        This function:
        - Connects using the model's name and token (for identification/auth)
        - Sends a matchmaking 'queue' command with the desired environment(s)
        - Listens for queue updates and 'match_found'
        - Returns True if match was successful, False if timed out or errored
        """
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        # Connect with model info for models
        query_params = {
            "model_name": self.model_name,
            "model_token": self.model_token,
        }
        query_string = urlencode(query_params)
        ws_uri = f"{MATCHMAKING_WS_URI}?{query_string}"
        
        print(f"Connecting to matchmaking server: {ws_uri}")
        
        try:
            # Create WebSocket connection
            self.matchmaking_websocket = await websockets.connect(
                ws_uri,
                # ssl=ssl_context,  # Uncomment for HTTPS
                ping_interval=20,
                ping_timeout=60
            )
            
            # Start background tasks for matchmaking
            asyncio.create_task(self._matchmaking_receiver())
            
            # Queue for a game
            queue_message = {
                "command": "queue",
                "environments": self.env_ids
            }
            await self.matchmaking_websocket.send(json.dumps(queue_message))
            print(f"Sent queue request for environments: {self.env_ids}")
            
            # Wait for match to be found or timeout
            start_time = asyncio.get_event_loop().time()
            while not self.matchmaking_complete:
                try:
                    # check for a new matchmaking message every 1 second
                    message = await asyncio.wait_for(
                        self.matchmaking_queue.get(),
                        timeout=1.0
                    )
                    await self._process_matchmaking_message(message)

                except asyncio.TimeoutError:
                    # Check if we should timeout the matchmaking
                    elapsed = asyncio.get_event_loop().time() - start_time
                    if elapsed > self.matchmaking_timeout:
                        print("Timeout waiting for match")
                        await self.matchmaking_websocket.close()
                        return False
                    continue
            
            # Match found - closing matchmaking websocket cleanly
            try:
                await self.matchmaking_websocket.close()
            except:
                pass
                
            return self.game_url is not None
            
        except Exception as e:
            print(f"Matchmaking connection error: {e}")
            return False

    async def connect_to_game_server(self):
        """
        Connect to the matched game server after matchmaking is complete. 

        Establishes a WebSocket connection and starts the background tasks for message handling.
        - _message_receiver: Receives messages from the game server
        - _action_sender: Sends actions to the game server
        - _ping_sender: Sends periodic pings to keep the connection alive

        Returns:
            bool: True if connected successfully, False otherwise
        """
        if not self.game_url:
            print("No game server IP available")
            return False
                
        # Properly configure SSL context
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Short delay to allow server initialization
        await asyncio.sleep(1)

        # Connect to the game server with model token
        ws_uri = f"wss://{self.game_url}/ws?token={self.model_token}"
        # ws_uri = f"ws://localhost:8000/ws?token={self.model_token}"
        print(f"Connecting to game server: {ws_uri}")
        
        # Try multiple connection attempts
        max_attempts = 10
        for attempt in range(1, max_attempts + 1):
            try:
                # Create WebSocket connection with compatible parameters
                self.websocket = await websockets.connect(
                    ws_uri,
                    ssl=ssl_context,
                    ping_interval=30,
                    ping_timeout=90
                )
                
                # Start background tasks
                asyncio.create_task(self._message_receiver())
                asyncio.create_task(self._action_sender())
                asyncio.create_task(self._ping_sender())
                
                print("Connected to game server")
                return True
                
            except Exception as e:
                print(f"Connection error (attempt {attempt}/{max_attempts}): {e}")
                if attempt < max_attempts:
                    await asyncio.sleep(2)
                else:
                    print("All connection attempts failed")
                    return False

    async def connect(self):
        """
        Connect to the matchmaking server and then to the game server.
        
        This function handles the entire connection process:
        - Connect to matchmaking server
        - Queue for a game
        - Connect to the game server once a match is found
        - Start background tasks for message handling and action sending
        
        Returns:
            bool: True if connected successfully, False otherwise
        """
        # First connect to matchmaking
        matchmaking_success = await self.connect_to_matchmaking()
        if not matchmaking_success:
            print("Failed to get a match")
            return False
            
        # Then connect to the game server
        return await self.connect_to_game_server()

    async def _process_message(self, message_str: str):
        """
        Handle a single message received from the game server websocket.

        Recognized commands include:
        - 'observation': Game state update (usually your turn)
        - 'game_over': End of the game with outcome and reward
        - 'timed_out': Someone failed to act in time
        - 'error': Server-side error
        - 'action_ack': Acknowledgement that action was received
        - 'ping': Server heartbeat request (respond with pong)
        - 'server_shutdown': Server has ended the session

        This is the central router for all game server-driven events.

        """
        try:
            message = json.loads(message_str)
            command = message.get("command")
            
            if command == "observation":
                # Received game state - this player's turn to act
                observation = message.get("observation")
                player_id = message.get("player_id")
                
                print(f"Received observation for player {player_id}")
                self.current_player_id = player_id
                self.current_observation = observation
                self.full_observations[player_id] = observation
                self.pending_action = False
                self.in_game = True
                
            elif command == "game_over":
                # Game has completed - extract reason and any reward
                print("Game over received")
                self.game_over = True
                outcome = message.get("outcome", "unknown")
                reason = message.get("reason", "No reason provided")
                
                # Extract reward info if available
                trueskill_change = message.get("trueskill_change", 0)
                if self.current_player_id is not None:
                    self.rewards[self.current_player_id] = trueskill_change
                
                self.info = {"reason": reason, "outcome": outcome}
                print(f"Game over: {outcome}, reason: {reason}")
                
            elif command == "timed_out":
                # Someone timed out
                timeout_msg = message.get("message", "Unknown timeout")
                print(f"Game timeout: {timeout_msg}")
                self.game_over = True
                self.info = {"reason": "timeout", "message": timeout_msg}
                
            elif command == "error":
                # Server error
                error_msg = message.get("message", "Unknown error")
                print(f"Server error: {error_msg}")
                self.info = {"error": error_msg}
                
            elif command == "action_ack":
                # Server has received and acknowledged this player's action
                print("Action acknowledged by server")
                
            elif command == "pong":
                # Optional ping response - no action needed
                pass
                
            elif command == "ping":
                # Server ping - this client reponds with a pong
                try:
                    await self.websocket.send(json.dumps({"command": "pong"}))
                except Exception as e:
                    print(f"Error sending pong: {e}")
                    
            elif command == "server_shutdown":
                # Server indicates the session is over and shutting down
                print("Server shutdown message received")
                self.server_shutdown = True 
                
            else:
                print(f"Unknown command received: {command}")
                
        except json.JSONDecodeError:
            print(f"Invalid JSON received: {message_str}")

        except Exception as e:
            print(f"Error processing message: {e}")
            
    async def update_loop(self):
        """
        Main async loop that continuously processes messages from the game server.

        This function:
        - Waits for messages from the server via self.message_queue
        - Routes each message through _process_message()
        - Tracks when the game ends, and continues listening for a short time
        - Gracefully shuts down once server is confirmed inactive

        This loop stops when self.server_shutdown is set to True.
        """
        game_over_time = None  # Track when game_over was received
        
        while not self.server_shutdown:
            try:
                # Use a timeout only after game_over to allow final messages to arrive.
                timeout = 5.0 if self.game_over else None
                
                try:
                    # Wait for a message from the queue
                    # If game_over is set, wait for a message with a timeout
                    message = await asyncio.wait_for(self.message_queue.get(), timeout=timeout)
                    await self._process_message(message)
                    
                    # If this is the first game over, then start the timer
                    if self.game_over and game_over_time is None:
                        game_over_time = asyncio.get_event_loop().time()
                        print("Game over received, waiting for additional messages...")
                    
                except asyncio.TimeoutError:
                    # If we're in the post-game phase, then we track how long we've been waiting.
                    if self.game_over:
                        elapsed = asyncio.get_event_loop().time() - game_over_time
                        print(f"Timeout after {elapsed:.1f}s while waiting for additional messages after game over")
                        
                        if elapsed > self.game_over_timeout:
                            print(f"No more messages after {self.game_over_timeout}s of game over, exiting loop")
                            self.server_shutdown = True

                    else:
                        # Unexpected timeout, treating as a forced shutdown
                        print(f"Timeout while waiting for messages")
                        self.game_over = True
                        self.server_shutdown = True
                    
                # Check if we've waited long enough after game_over
                if self.game_over and game_over_time is not None:
                    elapsed = asyncio.get_event_loop().time() - game_over_time
                    if elapsed > self.game_over_timeout:
                        print(f"No more messages after {self.game_over_timeout}s of game over, exiting loop")
                        self.server_shutdown = True
                    
            except websockets.exceptions.ConnectionClosed:
                print("WebSocket connection closed by server")
                self.server_shutdown = True  # Set server_shutdown when connection is closed
                break
                
            except Exception as e:
                print(f"Error in update loop: {e}")
                await asyncio.sleep(0.1)
                
        print("Update loop exiting")

    async def async_get_observation(self) -> Tuple[Optional[int], List]:
        """
        Wait for and returns the current player's observation from the game server.

        If an observation is already available, it returns that immediately.
        Otherwise, it waits for an observation to be received, until either:
        - A valid observation is received
        - The server shuts down

        Returns:
            Tuple[player_id, observation], or (None, []) if timed out or invalid.
        """
        # If we already have an observation, return it
        if self.current_player_id is not None and self.current_observation:
            observation = self.current_observation
            player_id = self.current_player_id
            return player_id, observation
        
        # Start update loop if not already running
        if not self.server_shutdown:
            if self.update_task is None or self.update_task.done():
                self.update_task = asyncio.create_task(self.update_loop())
            
            try:
                # Wait for an observation to be received
                while not self.server_shutdown:
                    if self.current_player_id is not None and self.current_observation:
                        return self.current_player_id, self.current_observation
                    await asyncio.sleep(0.1)
                        
            except Exception as e:
                print(f"Error waiting for observation: {e}")
                    
        # If server is shutting down or we timed out, mark observation as invalid
        self.observation_valid = False
        return None, []

    def get_observation(self) -> Tuple[Optional[int], List]:
        """
        Synchronous wrapper for async_get_observation, so non-async agents can call this.
        
        Handles asyncio event loop setup internally.
        Raises a RuntimeError if observation retrieval failed.
        """
        try:
            # get the current event loop (or create one)
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                new_loop = True
            else:
                new_loop = False
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            new_loop = True

        try:
            # run the async observation retrieval
            player_id, obs = loop.run_until_complete(self.async_get_observation())

            # Raise if invalid observation
            if getattr(self, "observation_valid", True) is False:
                raise RuntimeError(f"No valid observation — reason: {self.info.get('reason', 'unknown')}")

            return player_id, obs
        
        finally:
            # Close the loop if we created a new one
            if new_loop:
                loop.close()



    async def async_step(self, action: str):
        """
        Asynchronously submit an action to the game server and wait for the result.

        This function:
        - Puts the action into the action queue
        - Waits for the action to be sent and acknowledged
        - If the server is shutting down, it exits early

        Args:
            action: The action to be performed (e.g., "play x y", "bet 3")

        Returns:
            Tuple[bool, dict]: A tuple indicating if the game is over and any additional info
        """
        if self.server_shutdown:
            return True, self.info # server already ended
        
        # Queue action to be sent
        await self.action_queue.put(action)
        
        # Block until the action is sent
        await self.action_queue.join()  

        # Clear current observation - expecting a new one
        self.current_observation = None
        
        # Wait for response (new observation or game over)
        start_time = asyncio.get_event_loop().time()
        while not self.server_shutdown and self.pending_action:
            await asyncio.sleep(0.1)
                
        return self.game_over, self.info


    def step(self, action: str):
        """
        Synchronous wrapper for async_step, so non-async agents can call this.

        Args:
            action: The action to be performed (e.g., "play x y", "bet 3")

        Returns:
            Tuple[bool, dict]: A tuple indicating if the game is over and any additional info
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                new_loop = True
            else:
                new_loop = False
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            new_loop = True
            
        try:
            return loop.run_until_complete(self.async_step(action))
        finally:
            if new_loop:
                loop.close()

    async def async_reset(self, num_players=None, seed=None):
        """
        Resets the environment and starts a new game session.

        This:
        - Clears all previous state (observations, rewards, flags)
        - Connects to matchmaking and game server (if not connected)
        - Starts the update loop
        - Waits until the first observation arrives

        Returns:
            The first observation, or an empty list if connection or observation fails.
        """
        # Reset state
        self.current_player_id = None
        self.current_observation = None
        self.game_over = False
        self.server_shutdown = False
        self.rewards = {}
        self.info = {}
        self.full_observations = {}
        self.in_game = False
        self.update_task = None
        self.matchmaking_complete = False
        
        # Connect to matchmaking server and game server if not already connected
        if not self.websocket:
            connected = await self.connect()
            if not connected:
                print("Failed to connect to server")
                await self.async_close()  # Clean up tasks if connection fails
                return []
                
        # Start the main message update loop
        self.update_task = asyncio.create_task(self.update_loop())
        
        try:
            # Wait until we either get an observation or the server shuts down
            while not self.server_shutdown and not self.in_game:
                await asyncio.sleep(0.1)
                
                if self.current_player_id is not None and self.current_observation:
                    self.in_game = True
                    return self.current_observation

        except Exception as e:
            print(f"Error waiting for game start: {e}")
                
        # Return current observation or empty list
        return self.current_observation if self.current_observation else []


    def reset(self, num_players=None, seed=None):
        """
        Synchronous wrapper for async_reset.

        Returns:
            The initial observation for the agent.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                new_loop = True
            else:
                new_loop = False
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            new_loop = True
            
        try:
            return loop.run_until_complete(self.async_reset(num_players))
        finally:
            if new_loop:
                loop.close()

    async def async_close(self):
        """
        Asynchronously close the environment and clean up resources.
        
        This function:
        - Signals the action sender to stop
        - Closes the game server websocket
        - Closes the matchmaking websocket if still open
        - Cancels the update loop task if running
        - Returns the rewards dictionary
        """
        # Set server_shutdown flag to ensure all loops terminate
        self.server_shutdown = True
        
        # Signal action sender to stop
        try:
            await self.action_queue.put("CLOSE")
        except:
            pass
            
        # Close game server websocket
        if self.websocket and not getattr(self.websocket, 'closed', True):
            try:
                await self.websocket.close()
            except:
                pass
            
        # Close matchmaking websocket if still open
        if self.matchmaking_websocket and not getattr(self.matchmaking_websocket, 'closed', True):
            try:
                await self.matchmaking_websocket.close()
            except:
                pass
                
        # Cancel update task if running
        if self.update_task and not self.update_task.done():
            try:
                self.update_task.cancel()
            except:
                pass
                
        return self.rewards

    def close(self):
        """
        Synchronous wrapper for async_close.

        This function handles the event loop setup and cleanup.

        Returns:
            The rewards dictionary from the last game.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                new_loop = True
            else:
                new_loop = False

        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            new_loop = True
            
        try:
            return loop.run_until_complete(self.async_close())
        
        finally:
            if new_loop:
                loop.close()


def register_model(model_name: str, description: str, email: str) -> str:
    """
    Registers a model with the matchmaking server and retrieves an authentication token.

    Args:
        model_name (str): The name to identify the model (e.g., "gpt4-mini-bot").
        description (str): Description of the model (useful for leaderboard or logs).
        email (str): Contact email for the model owner.

    Returns:
        str: The model token (used for future authenticated connections), or None on failure.
    """
    try:
        response = requests.post(
            f"{MATCHMAKING_HTTP_URI}/register_model",
            json={
                "model_name": model_name,
                "description": description,
                "email": email
            }
        )
        response.raise_for_status()
        data = response.json()
        return data.get("model_token")
    
    except Exception as e:
        print(f"Error registering model: {e}")
        return None


def make_online(
    env_id: Union[str, List[str]],
    model_name: str,
    model_token: Optional[str] = None,
    model_description: Optional[str] = None,
    email: Optional[str] = None,
) -> OnlineEnvWrapper:
    """
    Creates and returns an OnlineEnvWrapper instance for the selected environment(s).

    This is the main setup function for developers.

    Args:
        env_id (str or List[str]): Environment name(s), e.g., "SpellingBee-v0" or ["Chess-v0", "ConnectFour-v0"]
        model_name (str): Name of the model (used for identification).
        model_token (str, optional): Token received from prior registration. If not provided, registration will occur.
        model_description (str, optional): Description of the model (required if registering).
        email (str, optional): Email for registration (required if registering).

    Returns:
        OnlineEnvWrapper: An instance ready to be used with .reset(), .step(), etc.
    """
    # Convert env_id to a list if it's a single string
    env_ids = [env_id] if isinstance(env_id, str) else env_id
    
    # Convert environment names to IDs
    env_ids_int = []
    if env_ids[0] == "all":
        env_ids_int = list(id for id in NAME_TO_ID_DICT.values() if isinstance(id, int)) 
    elif env_ids[0] == "Subset-v0":
        env_ids_int = list(NAME_TO_ID_DICT["Subset-v0"])
    else:
        for env_name in env_ids:
            if env_name in NAME_TO_ID_DICT:
                env_ids_int.append(NAME_TO_ID_DICT[env_name])
            else:
                raise ValueError(f"Environment {env_name} not recognized")
    
    # If no model token is provided, register the model
    if model_token is None:
        if model_description is None or email is None:
            raise ValueError("Provide model_description and email if model_token is not given")
        
        model_token = register_model(model_name, model_description, email)
        if not model_token:
            raise ValueError("Failed to register model with server")
            
        print(f"Registered model and received token: {model_token}")
    
    return OnlineEnvWrapper(env_ids_int, model_name, model_token)