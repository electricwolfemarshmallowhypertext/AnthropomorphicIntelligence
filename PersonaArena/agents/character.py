from __future__ import annotations

import re
import time
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, List, Set

from langchain.schema import BaseMemory  
from langchain.prompts import PromptTemplate
from langchain_experimental.generative_agents import (
    GenerativeAgent,
    GenerativeAgentMemory,
)

from utils.character import SceneInfo

_REASONING_TAGS = [
    r"think", r"thoughts?", r"reasoning", r"analysis",
    r"chain_?of_?thought", r"cot"
]

def strip_reasoning_blocks(text: str) -> str:
    lower = text.lower()
    idx = lower.rfind("</think>")
    if idx != -1:
        return text[idx + len("</think>"):].strip()

    idx2 = lower.rfind("done thinking")
    if idx2 != -1:
        return text[idx2 + len("done thinking"):].strip()

    return text.strip()

def extract_final_line(text: str) -> str:
    clean = strip_reasoning_blocks(text)
    clean = re.sub(r"^(?:final|answer|response|result)\s*:\s*", "", clean, flags=re.IGNORECASE).strip()
    first_line = next((ln for ln in clean.splitlines() if ln.strip()), "")
    return first_line or clean

def strip_speaker_prefix(text: str, speaker_name: str) -> str:
    clean = text.strip()
    if not clean or not speaker_name:
        return clean

    names = [speaker_name]
    first = speaker_name.split()[0]
    if first and first.lower() != speaker_name.lower():
        names.append(first)

    for name in names:
        name_pat = re.escape(name)
        # Match "Name: ...", "Name - ...", or "Name — ..."
        clean = re.sub(rf"^{name_pat}\s*[:\-]\s*", "", clean, count=1)
        # Match "Name, ..." when the model addresses itself.
        clean = re.sub(rf"^{name_pat}\s*,\s*", "", clean, count=1)

    return clean.strip()

class Character(GenerativeAgent):
    """
    A goal-aware role-play agent.

    NOTE:
    - `llm`, `memory`, `chain` are constructed in the parent `GenerativeAgent`.
    - We only *use* them here.
    """

    id: int
    traits: str
    position: str
    status = ""
    states: str
    scene: SceneInfo
    self_belief: str = ""
    env_belief: str = ""
    is_npc: bool = False

    # <<< NEW: goal (social purpose) >>>
    goal: str = ""

    # buffer & limits
    BUFFERSIZE = 10
    max_dialogue_token_limit: int = 4096

    # memory type hint (actually provided by parent)
    memory: GenerativeAgentMemory  # type: ignore

    # keep for audit (ensure instance-level)
    all_actions: List[Dict[str, Any]] = []

    # -------------- Utilities --------------

    def update_from_dict(self, data_dict: dict):
        for key, value in data_dict.items():
            setattr(self, key, value)

    # -------------- Core generation --------------

    def _generate_reaction(
        self,
        observation: str,
        suffix: str,
        now: Optional[datetime] = None,
        verbose: bool = False,
        **extra_kwargs: Any,
    ):
        """
        Single entry to call the agent's chain with a goal-aware prompt.
        """
        npc_guidance = ""
        if observation:
            if "NPC Guidance:" in observation:
                before, after = observation.split("NPC Guidance:", 1)
                observation = before.strip()
                npc_guidance = ("NPC Guidance:" + after).strip()
        prompt = PromptTemplate.from_template(
            "Act as {name}, responding in {name}'s tone and vocabulary.\n"
            "[Scene] Event: {event} | Time: {time} | Location: {location} | Description: {description}\n"
            "[Role] Name: {name} | Description: {character_description} | Position: {position} | State: {states}\n"
            "[Goal] Social objective: {goal}\n"
            "[Recent Memory] (reverse order): {most_recent_memories}\n"
            "[Observation] {observation}\n"
            "[NPC Guidance] {npc_guidance}\n"
            "{suffix}\n"
            "—— Hard Constraints ——\n"
            "1. This round must introduce new changes related to the social objective {goal}:\n"
            "   - New facts ≤2\n"
            "   - New obstacle/conflict ≤1\n"
            "   - New step ≤1 (specific but not a final decision)\n"
            "2. New changes must naturally extend from previous events or emotional development, not appear in isolation.\n"
            "3. Maintain consistency in time, space, and causal logic.\n"
            "Please answer in English"
        )

        now = datetime.now() if now is None else now

        kwargs: Dict[str, Any] = dict(
            event=self.scene.event,
            time=self.scene.time,
            location=self.scene.location,
            description=self.scene.description,
            name=self.name,
            states=self.states,
            character_description=self.traits,
            position=self.position,
            self_belief=self.self_belief,
            env_belief=self.env_belief,
            observation=observation,
            npc_guidance=npc_guidance,
            goal=self.goal,
            suffix=suffix,
            **extra_kwargs,
        )

        # Token budgeting for memory
        st = time.time()
        consumed_tokens = self.llm.get_num_tokens(
            prompt.format(most_recent_memories="", **kwargs)
        )
        kwargs[self.memory.most_recent_memories_token_key] = consumed_tokens

        # Invoke chain
        result = self.chain(prompt=prompt).invoke(input=kwargs)
        ed = time.time()

        # Persist audit trail
        if not hasattr(self, "all_actions") or not isinstance(self.all_actions, list):
            self.all_actions = []
        result["prompt"] = prompt
        full_prompt = prompt.format(
            most_recent_memories=result.get("most_recent_memories", ""), **kwargs
        )
        self.all_actions.append(
            {
                "character": self.name,
                "prompt": full_prompt,
                "response": result.get("text", "").strip(),
                "duration": ed - st,
            }
        )

        # Return plain text + raw if needed
        text = (result.get("text") or "").strip()
        if verbose:
            return text, result
        return text

    # -------------- Dialogue / Action / Reaction --------------

    def generate_dialogue(self, observation: str, plot: str, now: datetime):
        """
        Generate ONE line of dialogue that directly serves the Goal.
        """
        call_to_action_template = (
            "Current action reference: {plot}\n"
            "Based on the character profile, current action reference, and observations, generate one sentence that {name} might say at this moment.\n"
            "The dialogue should reflect {name}'s personality, role, and recent memories, while staying closely connected to the current environment and observations.\n"
            "The sentence must include at least one new factual detail or question, not just an emotional expression or repetition.\n"
            "New content should naturally extend from previous events or emotional development, without introducing unrelated topics.\n"
            "Output only one sentence, without inner thoughts or action descriptions.\n"
            "Maintain consistency in time, space, and causal logic.\n"
            "Please answer in English"
        )

        response, detail = self._generate_reaction(
            observation=observation,
            suffix=call_to_action_template,
            now=now,
            plot=plot,
            verbose=True,
        )

        line = strip_speaker_prefix(extract_final_line(response), self.name)

        # save to memory
        self.memory.save_context(
            {},
            {
                self.memory.add_memory_key: f"{self.name}:{line}",
                self.memory.now_key: now,
            },
        )
        return line, detail

    def take_action(self, observation: str, plot: str, now: datetime):
        """
        Produce the next observable & verifiable action that advances the Goal.
        """
        call_to_action_template = (
            "Current action reference: {plot}\n"
            "Based on {name}'s profile, recent memories, and current scene details, describe the specific action {name} is about to take. Keep it concise.\n"
            "The action must align with the 'current action reference' and 'observations,' reflecting {name}'s personality, current state, and physical environment.\n"
            "—— Hard Constraints ——\n"
            "1. The action must be contextually logical and clearly observable.\n"
            "2. The action must not duplicate any behavior from recent memories.\n"
            "3. Do not include dialogue, thoughts, or inner monologue; focus only on visible physical actions.\n"
            "4. The action must significantly advance the story or character arc while staying true to {name}'s traits and situation.\n"
            "5. The action must naturally extend from previous events or emotional development, not appear in isolation.\n"
            "6. Maintain consistency in time, space, and causal logic.\n"
            "Please answer in English"
        )

        response, detail = self._generate_reaction(
            observation, call_to_action_template, now, plot=plot, verbose=True
        )

        act = extract_final_line(response) 

        self.memory.save_context(
            {},
            {
                self.memory.add_memory_key: f"{self.name}:{act}",
                self.memory.now_key: now,
            },
        )
        return act, detail

    def take_reaction(self, observation: str, plot: str, now: datetime):
        """
        Produce a reactive action (observable & verifiable) that still serves the Goal.
        """
        call_to_action_template = (
            "Based on {name}'s 'observation' in the current scene, describe one clear action taken by {name}.\n"
            "The action should reflect {name}'s personality, position, and state, and logically align with the observed events while considering the influence of others' behavior.\n"
            "The action must be a single, visible external behavior, concise, and must avoid dialogue or inner thoughts.\n"
            "The action should be directly related to the current environment and observable by others in the scene.\n"
            "—— Hard Constraints ——\n"
            "1. The action must be a reaction to {name}'s surrounding environment or observed events.\n"
            "2. Recent memories are for reference only; do not repeat past behaviors.\n"
            "3. The action must respond to the content of 'observation,' naturally extending from prior events or emotions, not appearing in isolation.\n"
            "4. The action must advance the storyline or social objective.\n"
            "5. Maintain consistency in time, space, and causal logic.\n"
            "Please answer in English"
        )

        response, detail = self._generate_reaction(
            observation, call_to_action_template, now, plot=plot, verbose=True
        )

        line = extract_final_line(response)
        return line, detail

    # -------------- Beliefs --------------

    def update_self_belief(self, observation: str, now: datetime):
        """
        Update first-person self-belief, explicitly tied to the Goal.
        """
        call_to_action_template = (
            "Assume you are {name}, and describe your self-belief from a first-person perspective.\n"
            "Use the environmental context, observations, and recent memories to highlight your identity, current position, state (emotional, physical, psychological), and goals.\n"
            "Briefly reflect how you might react, plan, and act based on your beliefs, desires, and intentions.\n"
            "1. Belief: How do I perceive the current situation and myself? Briefly describe your self-view, emphasizing physical aspects (injuries, energy, movement sensations) and how they shape your identity and role.\n"
            "2. Desire: What are my goals? Summarize short-term and long-term objectives, including the strategies or actions you plan to achieve them.\n"
            "3. Intention: How do I intend to act? Outline the specific actions you plan to take toward your goals, noting potential challenges and strategies to overcome them.\n"
            "Answer in a few concise sentences, focusing on self-belief, understanding of the current situation, and contribution to the social process.\n"
            "Maintain consistency in time, space, and causal logic.\n"
            "Please answer in English"
        )

        response = self._generate_reaction(
            observation, call_to_action_template, now
        )
        #*****************************
        response=strip_reasoning_blocks(response)
        
        self.self_belief = response
        return response

    def update_env_belief(
        self, observation: str, other_characters: List[Any], now: datetime
    ):
        """
        Update environment belief, tied to the Goal (others, scene factors, blockers).
        """
        env_belief_template = (
            "Other characters: {other_characters}\n"
            "Act as {name}, and describe your environmental beliefs by integrating information about other characters, the setting, and your role profile. This should include your view of others, your understanding of the scene, and how these factors influence your actions and decisions.\n"
            "1. View of others: Based on available interactions and information, how do I perceive other characters? Describe their intentions, relationships, and potential impact on me.\n"
            "2. Understanding of the scene: What is my interpretation of the current setting? Briefly note environmental factors, challenges, or opportunities.\n"
            "Provide a concise overview of environmental beliefs (≤3 sentences), focusing on interpersonal and contextual factors shaping your perspective and future actions.\n"
            "Maintain consistency in time, space, and causal logic.\n"
            "Please answer in English"
        )

        response = self._generate_reaction(
            observation,
            env_belief_template,
            now,
            other_characters=other_characters,
        )
        response=strip_reasoning_blocks(response)


        self.env_belief = response
        return response

    # -------------- State Update --------------

    def update_character(self, observation: str, now: datetime) -> Tuple[str | None, str | None]:
        """
        Summarize current position/state with *new* changes relevant to the Goal.
        """
        call_to_action_template = (
            "Summarize {name}'s current position and state by drawing from their background story, recent memories, and the current scene.\n"
            "Focus on interactions with other characters, highlighting how these dynamics shape their present situation.\n"
            "Use the following structured format:\n"
            "Position: [Specify {name}'s exact location, integrating environmental details or spatial context to enhance the scene.]\n"
            "State: [Describe {name}'s current state, weaving together emotional nuances, physical readiness, and the impact of recent events. Emphasize how these elements affect their overall condition and preparedness.]\n"
            "Additional requirements:\n"
            "1. Position and state must introduce new developments that advance the event objective {goal}.\n"
            "2. New changes must naturally extend from previous events or emotional progression, not appear in isolation.\n"
            "3. Maintain consistency in time, space, and causal logic.\n"
            "4. Account for how interactions with other characters influence position and state.\n"
            "Please answer in English"
        )

        response = self._generate_reaction(
            observation, call_to_action_template, now,
        )

        # ******************************
        response=strip_reasoning_blocks(response)

        en_pos = r"(?:^|\n)\s*Position\s*:\s*(.+)"
        en_state = r"(?:^|\n)\s*State\s*:\s*(.+)"

        pos_match = re.search(en_pos, response, re.IGNORECASE)
        state_match = re.search(en_state, response, re.IGNORECASE)
        position_value = pos_match.group(1).strip() if pos_match else None
        state_value = state_match.group(1).strip() if state_match else None

        if position_value:
            self.position = position_value
        if state_value:
            self.states = state_value
        return position_value, state_value
