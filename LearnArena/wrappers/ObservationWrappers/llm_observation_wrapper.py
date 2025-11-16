"""
Portions of this file are adapted from TextArena:
https://github.com/LeonGuertler/TextArena

Original work:
Copyright (c) 2025 Leon Guertler and contributors
Licensed under the MIT License.
"""

import core
from core import ObservationWrapper, Env, Observations, Info, GAME_ID
from typing import Any, Dict, Optional, Tuple, List

__all__ = ["LLMObservationWrapper", "DiplomacyObservationWrapper"]


def _normalize_observation_entries(observation_data: Optional[Any]) -> List[Tuple[int, str]]:
    """Convert various observation payloads into (sender_id, message) tuples."""
    if observation_data is None:
        return []

    if isinstance(observation_data, str):
        return [(GAME_ID, observation_data)]

    normalized: List[Tuple[int, str]] = []
    try:
        iterator = iter(observation_data)
    except TypeError:
        return [(GAME_ID, str(observation_data))]

    for entry in iterator:
        sender_id = GAME_ID
        message: Any = ""

        if isinstance(entry, (tuple, list)):
            if len(entry) >= 2:
                sender_id = entry[0]
                message = entry[1]
            elif len(entry) == 1:
                message = entry[0]
            else:
                continue
        else:
            message = entry

        try:
            sender_id = int(sender_id)
        except (TypeError, ValueError):
            sender_id = GAME_ID

        normalized.append((sender_id, str(message)))

    return normalized


class LLMObservationWrapper(ObservationWrapper):
    """
    A wrapper for converting environment observations into formatted strings suitable
    for large language models (LLMs). It ensures that duplicate observations are not
    added to the full observations.
    """

    def __init__(self, env: Env):
        """
        Initializes the LLMObservationWrapper.

        Args:
            env (Env): The environment to wrap.
        """
        super().__init__(env)
        self.full_observations: Dict[int, List[Tuple[int, str]]] = {}

    def _convert_obs_to_str(self, player_id: int) -> Observations:
        """
        Converts the full observations into formatted strings for each recipient.

        Returns:
            Observations: A dictionary mapping recipient IDs to their formatted observation strings.
        """
        str_observation = ""
        
        if player_id in self.full_observations:
            for sender_id, message in self.full_observations[player_id]:
                if sender_id == GAME_ID:
                    sender_name = "GAME"
                else:
                    sender_name = self.env.state.role_mapping.get(sender_id, f"Player {sender_id}")
                str_observation += f"\n[{sender_name}] {message}"

        return str_observation

    def observation(self, player_id: int, observation: Optional[Observations]):
        if observation is None:
            return self._convert_obs_to_str(player_id=player_id)

        if player_id not in self.full_observations:
            self.full_observations[player_id] = []

        normalized = _normalize_observation_entries(observation)
        if normalized:
            self.full_observations[player_id].extend(normalized)

        return self._convert_obs_to_str(player_id=player_id)


    
class DiplomacyObservationWrapper(LLMObservationWrapper):
    def __init__(self, env: Env):
        super().__init__(env)

    def _get_history_conversation(self, player_id: int) -> str:
        """
        Get the history conversation for the given player.
        """
        history = []
        for sender_id, message in self.full_observations[player_id][1:]:
            if sender_id == GAME_ID:
                sender_name = "GAME"
            else:
                sender_name = self.env.state.role_mapping.get(sender_id, f"Player {sender_id}")
            history.append(f"[{sender_name}] {message}")
        return "\n".join(history)

    def observation(self, player_id: int, observation: Optional[Observations]):
        if observation is None:
            return self.env.get_prompt(player_id, self._get_history_conversation(player_id))

        if player_id not in self.full_observations:
            self.full_observations[player_id] = []

        normalized = _normalize_observation_entries(observation)
        if normalized:
            self.full_observations[player_id].extend(normalized)

        return self.env.get_prompt(player_id, self._get_history_conversation(player_id))


class FirstLastObservationWrapper(ObservationWrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        self.full_observations: Dict[int, List[Tuple[int, str]]] = {}

    def _convert_obs_to_str(self, player_id: int) -> Observations:
        return_str = self.full_observations[player_id][0][1]
        if len(self.full_observations[player_id]) > 1:
            return_str += "\n\n" + self.full_observations[player_id][-1][1]

        return return_str + "\n\n" + "Next Action:"

    def observation(self, player_id: int, observation: Optional[Observations]):
        if observation is None:
            return self._convert_obs_to_str(player_id=player_id)

        if player_id not in self.full_observations:
            self.full_observations[player_id] = []

        normalized = _normalize_observation_entries(observation)
        if normalized:
            self.full_observations[player_id].extend(normalized)

        return self._convert_obs_to_str(player_id=player_id)