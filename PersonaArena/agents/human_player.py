from datetime import datetime

from agents.character import Character


class HumanPlayer(Character):
    """Human-in-the-loop character with manual action/dialogue input."""

    def _prompt_user(self, label: str, observation: str, plot: str) -> str:
        print("\n" + "=" * 60)
        print(f"[HumanPlayer] {self.name} | {label}")
        if self.scene:
            print(f"Scene: {self.scene.event} | {self.scene.time} | {self.scene.location}")
            print(f"Scene desc: {self.scene.description}")
        if getattr(self, "goal", ""):
            print(f"Goal: {self.goal}")
        if self.traits:
            print(f"Profile: {self.traits}")
        print(f"Position: {self.position} | States: {self.states}")
        if plot:
            ref_label = "Dialogue reference" if label == "dialogue" else "Action reference"
            print(f"{ref_label}: {plot}")
        if observation:
            print(f"Observation: {observation}")
        if label == "dialogue":
            print(" ")
            print("Expected: one spoken sentence only (no actions).")
        else:
            print(" ")
            print("Expected: one visible action only (no dialogue or inner thoughts).")
        print("=" * 60)
        return input("Please enter the content：").strip()

    def generate_dialogue(self, observation: str, plot: str, now: datetime):
        response = self._prompt_user("dialogue", observation, plot)
        detail = {
            "source": "human",
            "type": "dialogue",
            "time": now.isoformat(),
            "text": response,
        }
        self.memory.save_context(
            {},
            {
                self.memory.add_memory_key: f"{self.name}:{response}",
                self.memory.now_key: now,
            },
        )
        return response, detail

    def take_action(self, observation: str, plot: str, now: datetime):
        action = self._prompt_user("action", observation, plot)
        detail = {
            "source": "human",
            "type": "action",
            "time": now.isoformat(),
            "text": action,
        }
        self.memory.save_context(
            {},
            {
                self.memory.add_memory_key: f"{self.name}:{action}",
                self.memory.now_key: now,
            },
        )
        return action, detail

    def take_reaction(self, observation: str, plot: str, now: datetime):
        reaction = self._prompt_user("reaction", observation, plot)
        detail = {
            "source": "human",
            "type": "reaction",
            "time": now.isoformat(),
            "text": reaction,
        }
        self.memory.save_context(
            {},
            {
                self.memory.add_memory_key: f"{self.name}:{reaction}",
                self.memory.now_key: now,
            },
        )
        return reaction, detail

    def update_self_belief(self, observation: str, now: datetime):
        return self.self_belief

    def update_env_belief(self, observation: str, other_character, now: datetime):
        return self.env_belief
