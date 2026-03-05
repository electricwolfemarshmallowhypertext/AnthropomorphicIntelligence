from __future__ import annotations
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from langchain.schema import BaseMemory  # noqa: F401  (kept for compatibility)
from langchain.prompts import PromptTemplate
from langchain_experimental.generative_agents import GenerativeAgent, GenerativeAgentMemory

from utils.character import CharacterInfo, SceneInfo


class Narrator(GenerativeAgent):
    id: int
    BUFFERSIZE = 10
    max_dialogue_token_limit: int = 600

    synopsis: List[str] = []
    plots: List[str] = []
    history: List[str] = []

    # Characters in scene
    characters: List[CharacterInfo]

    # Scene + Goal (social purpose)
    scene: SceneInfo
    goal: str = ""   

    # Memory & audit
    memory: GenerativeAgentMemory
    all_actions: List[Dict] = []
    round_plot_actions: List[Dict] = []

    def update_from_dict(self, data_dict: dict):
        for key, value in data_dict.items():
            setattr(self, key, value)

    # ---------------- Core Invoke ----------------
    def _generate_reaction(
        self,
        suffix: str,
        now: Optional[datetime] = None,
        verbose: bool = False,
        **extra_kwargs
    ) -> str:

        goal = getattr(self, "goal", "") or getattr(self.scene, "social_purpose", "") or ""

        prompt = PromptTemplate.from_template(
            "Please act as the screenwriter of a realistic social scene, highlighting each character’s unique traits and driving the plot forward.\n"
            "Event: {event}\n"
            "Goal: {goal}\n"
            "Time: {time}\n"
            "Location: {location}\n"
            "Description: {description}\n"
            "Characters: {characters}\n"
            "No other people are present in the scene besides the listed characters.\n"
            "—— Constraints ——\n"
            "1. Scene details must naturally extend from past events or emotional developments, not appear in isolation.\n"
            "2. Maintain consistency in time, space, and causal logic.\n"
            "Please answer in English"
            + suffix
        )

        now = datetime.now() if now is None else now
        kwargs: Dict[str, Any] = dict(
            event=self.scene.event,
            time=self.scene.time,
            location=self.scene.location,
            description=self.scene.description,
            characters=self.characters,
            goal=self.goal,
            **extra_kwargs
        )

        st = time.time()
        result = self.chain(prompt=prompt).invoke(input=kwargs)
        ed = time.time()

        result["prompt"] = prompt
        full_prompt = prompt.format(**kwargs)
        self.all_actions.append(
            {
                "character": self.name,
                "prompt": full_prompt,
                "response": (result.get("text") or "").strip(),
                "duration": ed - st,
            }
        )

        text = (result.get("text") or "").strip()
        if verbose:
            return text, result
        return text

    # ---------------- Character State Update ----------------
    def update_character(self, name: str, observation: str, now: datetime, verbose: bool = False):

        call_to_action_template = (
            "Observation: {observation}\n"
            "Character: {name}\n"
            "Based on the character’s background and scene observations, summarize {name}'s current position and state."
            "Focus on their interactions with others, and how these dynamics shape their situation and drive social progress.\n"
            "Use the following structured format:\n\n"
            "Position: [Specify {name}'s exact location, integrating environmental or spatial details to enhance scene visualization.]\n"
            "State: [Describe {name}'s current state, combining emotional nuances, physical readiness, and recent events, highlighting how interactions influence position and state.]\n"
            "—— Constraints ——\n"
            "1. Position and state must naturally extend from prior events or emotional developments, not appear in isolation.\n"
            "2. Maintain consistency in time, space, and causal logic.\n"
            "3. Reflect how interactions with other characters affect position and state.\n"
            "Please answer in English"
        )

        response, detail = self._generate_reaction(
            call_to_action_template, now, observation=observation, name=name, verbose=True
        )

        en_pos = r"Position:\s*(.+)"
        en_state = r"State:\s*(.+)"

        pos_match = re.search(en_pos, response, re.IGNORECASE)
        state_match = re.search(en_state, response, re.IGNORECASE)
        position_value = pos_match.group(1).strip() if pos_match else None
        state_value = state_match.group(1).strip() if state_match else None

        if verbose:
            return position_value, state_value, detail
        return position_value, state_value, None

    # ---------------- Action Influence ----------------
    def analyze_action(self, actor: str, action: str, now: datetime) -> Tuple[int, str]:

        call_to_action_template = (
            "Action: {action}\n"
            "Actor: {actor}\n"
            "Analyze the action and actor in context to determine which character in the list is most likely to react."
            "Describe the possible reaction of that character, with a brief explanation."
            "If the action directly causes a physical impact on a target, identify the specific effect, its cause, and outcome."
            "If no character is likely to react, return the actor’s name.\n"
            "—— Output Format ——\n"
            "[Reacting Character]|[Affected Action]\n"
            "—— Constraints ——\n"
            "1. The response must be concise and clear.\n"
            "2. Maintain consistency in time, space, and causal logic.\n"
            "Please answer in English"
        )

        response = self._generate_reaction(call_to_action_template, now, action=action, actor=actor)
        pattern = re.compile(r"\s*(.*)\|\s*(.*)")
        match = re.match(pattern, response)
        target_id = None
        affected = ""
        if match:
            target_name, affected = match.groups()
            for c in self.characters:
                if c.name == target_name:
                    target_id = c.id
                    break
        return target_id, affected

    def analyze_action_influence(
        self,
        actor: str,
        action: str,
        now: datetime,
        dialogue: str = "",
        source_type: str = "auto",
        verbose: bool = False,
    ):

        source_type = (source_type or "auto").strip().lower()
        focus_en = ""
        if source_type == "action":
            focus_en = "Focus ONLY on the action's impact; ignore dialogue content.\n"
        elif source_type == "dialogue":
            focus_en = "Focus ONLY on the dialogue's impact; ignore action content.\n"

        call_to_action_template = (
            "Action: {action}\n"
            "Dialogue: {dialogue}\n"
            "Actor: {actor}\n"
            "Analyze the action or dialogue and its impact, focusing on which character in the 'Characters' list is affected.\n"
            + focus_en
            + "—— Analysis Tasks ——\n"
            "1. Select one target character from the 'Characters' list.\n"
            "2. Describe the specific action or dialogue initiated by the actor.\n"
            "3. Explain the concrete impact of this action or dialogue on the target’s state or interaction.\n"
            "4. If no listed character is affected, return the actor’s name as the target and mark the type as none.\n"
            "5. Only perceivable actions count: if the character is not in the same space or cannot perceive the action/dialogue, no effect occurs.\n"
            "—— Output Format ——\n"
            "[Actor];;[Target];;[Influence type (action/dialogue/none)];;[Detailed effect of actor on target]\n"
            "—— Constraints ——\n"
            "1. The response must be concise, accurate, and follow the specified format.\n"
            "2. Maintain consistency in time, space, and causal logic.\n"
            "Please answer in English"
        )

        response, detail = self._generate_reaction(
            call_to_action_template, now, action=action, dialogue=dialogue, actor=actor, verbose=True
        )
        pattern = r"\[?([^;\[\]]+)\]?\s*;;\s*\[?([^;\[\]]+)\]?\s*;;\s*\[?([^;\[\]]+)\]?\s*;;\s*\[?([^;\[\]]+)\]?"
        matches = re.search(pattern, response)
        target_id = None
        impact = ""
        influence_type = "none"
        if matches:
            _actor = matches.group(1)
            target_name = matches.group(2)
            influence_type = matches.group(3).strip().lower() or "none"
            impact = matches.group(4)
            for c in self.characters:
                if c.name == target_name:
                    target_id = c.id
                    break
            if target_id is None and target_name:
                target_norm = target_name.strip().lower()
                for c in self.characters:
                    name_norm = c.name.strip().lower()
                    if target_norm and (target_norm in name_norm or name_norm in target_norm):
                        target_id = c.id
                        break
        else:
            for c in self.characters:
                if c.name == actor:
                    target_id = c.id
                    impact = ""
                    break

        if verbose:
            return target_id, impact, influence_type, detail
        return target_id, impact, influence_type

    # ---------------- Result & Scene ----------------
    def analyze_result(self, actions: str, now: datetime, verbose: bool = False):

        call_to_action_template = (
            "Action: {actions}\n"
            "Instruction: Act as an immediate event referee, quickly analyzing and judging the outcome and impact of interactions between the specified characters."
            "Speak in a concise, omniscient observer tone, narrating only the direct results of these actions, highlighting causal links and their effect on the social process.\n"
            "—— Guidelines ——\n"
            "1. Narrate only immediate and direct results, focusing on the consequences of interactions.\n"
            "2. Maintain a concise, straightforward omniscient tone.\n"
            "3. Base narration strictly on the provided character descriptions and actions; avoid speculation or redundancy.\n"
            "4. Do not repeat the input actions in the result; describe only outcomes.\n"
            "5. Maintain consistency in time, space, and causal logic.\n"
            "Please answer in English"
        )

        response, detail = self._generate_reaction(call_to_action_template, now, actions=actions, verbose=True)
        if verbose:
            return response, detail
        return response

    def update_scene(self, obs: str, verbose: bool = False):

        prompt = PromptTemplate.from_template(
            "Given an initial scene description and an observation, update the scene to reflect any direct and significant physical environmental changes.\n"
            "If the observation does not indicate major physical changes, keep the original scene description unchanged.\n"
            "Preserve the original scene structure and avoid introducing attributes not present in the initial description.\n"
            "—— Notes ——\n"
            "1. Update only physical environment changes; do not include any character actions, dialogue, or inner thoughts.\n"
            "2. Output must be strictly structured as 'Time', 'Location', and 'Environment Description' without extra text or prefixes.\n"
            "3. 'Environment Description' should describe only the physical state of the environment, excluding characters or lyrical content.\n"
            "4. Maintain consistency in time, space, and causal logic.\n"
            "Input:\n"
            "- Time: {time}\n"
            "- Location: {location}\n"
            "- Environment Description: {description}\n"
            "Observation: {observation}\n"
            "Output:\n"
            "- Time: {time}\n"
            "- Location: {location}\n"
            "- Environment Description: (updated physical environment description based on the observation)\n"
            "Please answer in English"
        )

        st = time.time()
        result = self.chain(prompt=prompt).invoke(
            input={
                "time": self.scene.time,
                "location": self.scene.location,
                "description": self.scene.description,
                "observation": obs,
            }
        )
        result["prompt"] = prompt
        response = (result.get("text") or "").strip()
        ed = time.time()
        full_prompt = prompt.format(
            **{
                "time": self.scene.time,
                "location": self.scene.location,
                "description": self.scene.description,
                "observation": obs,
            }
        )
        self.all_actions.append(
            {"character": self.name, "prompt": full_prompt, "response": response, "duration": ed - st}
        )

        p_time = re.compile(r"-\s*Time:\s*(.+)\n", re.IGNORECASE)
        p_loc = re.compile(r"-\s*Location:\s*(.+)\n", re.IGNORECASE)
        p_desc = re.compile(r"-\s*Environment Description:\s*([\s\S]+)", re.IGNORECASE)

        m_time = p_time.search(response)
        m_loc = p_loc.search(response)
        m_desc = p_desc.search(response)

        new_time = m_time.group(1).strip() if m_time else self.scene.time
        new_loc = m_loc.group(1).strip() if m_loc else self.scene.location
        new_desc = m_desc.group(1).strip() if m_desc else self.scene.description

        self.scene.time = new_time or self.scene.time
        self.scene.location = new_loc or self.scene.location
        self.scene.description = new_desc or self.scene.description

        if verbose:
            return self.scene, result
        return self.scene

    def summary_plot(self, actions: str) -> str:

        prompt = PromptTemplate.from_template(
            """
            "Actions: {actions}\n"
            "Based on the actions provided above, summarize the key plot of this segment.\n"
            "Focus on the main actions of the characters, the major changes or events they cause, and the direct results or implications.\n"
            "The summary must be concise and capture the essence of the narrative.\n"
            "Maintain consistency in time, space, and causal logic.\n"
            "Please answer in English"
            """
        )

        st = time.time()
        result = self.chain(prompt=prompt).invoke(input={"actions": actions, "goal": self.goal or ""})
        result["prompt"] = prompt
        response = (result.get("text") or "").strip()
        ed = time.time()
        full_prompt = prompt.format(**{"actions": actions, "goal": self.goal or ""})
        self.all_actions.append(
            {"character": self.name, "prompt": full_prompt, "response": response, "duration": ed - st}
        )

        self.plots.append(response)
        return response

    def generate_synopsis(
        self, actions: str, sequence: List[str], history: List[Dict[str, Any]], now: datetime
    ) -> Dict[str, str]:
        """
        Produce per-character current beats *directly tied to the Goal*.
        Output format per line: [Name]: [beat & next step]
        """
        prompt = PromptTemplate.from_template(
            "Act as the screenwriter of a realistic drama, designing the current plot for each character to drive the story forward.\n"
            "Story history: {history}\n"
            "Past actions: {actions}\n"
            "Character action order: {sequence}\n"
            "At this critical moment, each character is about to act based on their unique motives and the scene context. Using the story history and past actions, and following the character action order, generate the current plot and next move for each character. These actions should directly advance the narrative, reflecting their motives and the dynamics of the scene.\n"
            "—— Hard Constraints ——\n"
            "1. Actions must build on previous storylines and actions, naturally extending and advancing the plot.\n"
            "2. Each character’s action and intention should only describe themselves, not their effect on others.\n"
            "3. All characters in {sequence} must have a corresponding summary.\n"
            "4. Output must strictly follow the format: [Character Name]: [Concise description of plot and next action].\n"
            "5. Output only in the specified format, one line per character, no extra text.\n"
            "6. Actions must be based on prior events or emotional developments, not appear in isolation.\n"
            "7. Do not include dialogue or inner thoughts—only visible plot and actions.\n"
            "8. Each character summary is limited to 3–5 concise points; long narratives are forbidden.\n"
            "9. Maintain consistency in time, space, and causal logic.\n"
            "Please answer in English\n"
            "Current plot:"
        )

        result = self.chain(prompt=prompt).invoke(
            input={"actions": actions, "sequence": sequence, "history": history, "goal": self.goal}
        )
        result["prompt"] = prompt
        raw = (result.get("text") or "").strip().split("\n\n")[0]

        pattern = r"([^:\n]+):\s*([\s\S]+?)(?=\n|$)"
        matches = re.findall(pattern, raw)
        synopsis: Dict[str, str] = {}
        for name, plot in matches:
            synopsis[name.strip(" []'\"")] = plot.strip(" []'\"")
        return synopsis
