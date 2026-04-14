import os
import time
import json
import math
import copy
import logging
import re
from datetime import datetime
from collections import OrderedDict

import faiss
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from yacs.config import CfgNode
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain_experimental.generative_agents import GenerativeAgentMemory
from langchain_community.embeddings import HuggingFaceEmbeddings

from dataloader.data import Data
from utils import utils
from utils.message import Message
from utils.character import SceneInfo
from agents import Character, EnvironmentAgent, HumanPlayer
from langchain.globals import set_debug

# set_debug(True)

os.environ["OPENAI_API_VERSION"] = "2024-12-01-preview"

logging.basicConfig(level=logging.ERROR)
# logging.basicConfig(level=logging.INFO)
lock = threading.Lock()


class Simulator:
    """
    Simulator class for running the simulation.
    """
    def __init__(self, config: CfgNode, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.round_cnt = 0
        self.file_name_path: list[str] = []
        self.play_event = threading.Event()
        self.now = datetime.now().replace(hour=8, minute=0, second=0)
        self.round_message: list[Message] = []
        self.round_obs = {}
        self.acting = OrderedDict()
        self.acted = OrderedDict()
        self.last_round = ""
        self.character_record = {}
        self.round_record = []
        self.plot_record = {}
        self.narrator_record = []
        self.records = []
        self.duration = 0
        self.terminated = False
        self.termination = {"decision": "continue", "reason": ""}

        self.checkpoint_round_interval = max(
            1, int(config.get("checkpoint_round_interval", config.get("judge_round_interval", 1)))
        )

        self.environment_record = []

    def _maybe_autogen_scene_from_personas(self) -> None:

        cfg = self.config
        if not bool(cfg.get("autogen_scene_from_persona", False)):
            return

        persona_inline = cfg.get("persona_inline", None)
        persona_jsonl = str(cfg.get("persona_jsonl_path", "") or "").strip()
        persona_index = int(cfg.get("persona_index", 0))

        personas = []
        try:
            if persona_inline:
                if isinstance(persona_inline, dict):
                    personas = [self._normalize_persona(persona_inline)]
                elif isinstance(persona_inline, list) and len(persona_inline) > 0:
                    idx = max(0, min(persona_index, len(persona_inline)-1))
                    personas = [self._normalize_persona(persona_inline[idx])]
                else:
                    self.logger.warning("persona_inline is empty or has an unsupported type; skipping.")
            elif persona_jsonl:
                with open(persona_jsonl, "r", encoding="utf-8") as f:

                    for _i, line in enumerate(f):
                        if _i == persona_index:
                            import json as _json
                            personas = [self._normalize_persona(_json.loads(line))]
                            break
                if not personas:
                    raise ValueError(f"No persona found at index {persona_index} in {persona_jsonl}")
            else:

                return
        except Exception as e:
            self.logger.error(f"Failed to read persona: {e}")
            return

        if not personas:
            return

        event_hint = str(cfg.get("scene_event_hint", "") or "A small public interaction centered on a clear, specific goal")
        social_purpose = str(cfg.get("scene_social_purpose", "") or "Guide the protagonist to cover background, values, interests, personality, and experiences")
        language = str(cfg.get("scene_language", "en"))

        try:
            scene_json = EnvironmentAgent.generate_scene_from_personas(
                config=cfg,
                logger=self.logger,
                personas=personas,
                event_hint=event_hint,
                social_purpose=social_purpose,
                language=language,
            )
        except Exception as e:
            self.logger.error(f"EnvironmentAgent.generate_scene_from_personas failed: {e}")
            return

        out_dir = "output/scenes"
        os.makedirs(out_dir, exist_ok=True)

        main_name = None
        try:
            if personas and isinstance(personas, list):
                main_name = personas[0].get("name") or personas[0].get("user_name")
            if not main_name:
                main_name = scene_json.get("title")
        except Exception:
            pass

        if not main_name:
            main_name = "Protagonist"

        import re
        safe_name = re.sub(r"[^\w-]+", "_", str(main_name), flags=re.ASCII).strip("_-")
        if not safe_name:
            safe_name = "Protagonist"

        llm_name = str(cfg.get("character_llm", "llm"))
        safe_llm = re.sub(r"[^\w-]+", "_", llm_name, flags=re.ASCII).strip("_-")

        base_path = os.path.join(out_dir, f"autogen_scene_{safe_name}_{safe_llm}.json")
        tmp_scene_path = base_path
        suffix = 1
        while os.path.exists(tmp_scene_path):
            tmp_scene_path = os.path.join(out_dir, f"autogen_scene_{safe_name}_{safe_llm}_{suffix}.json")
            suffix += 1

        with open(tmp_scene_path, "w", encoding="utf-8") as f:
            json.dump(scene_json, f, ensure_ascii=False, indent=2)

        self.logger.info(f"[AutoGen] Scene generated: {tmp_scene_path}")
        cfg["scene_path"] = tmp_scene_path
        cfg["scene_id"] = 0 

    def _normalize_persona(self, persona: dict) -> dict:
        """Normalize a persona from jsonl into the fields needed by the scene generator."""
        p = persona or {}
        facts = p.get("facts", {}) or {}

        return {
            "id": int(p.get("user_id", 0)) if str(p.get("user_id", "")).isdigit() else 0,
            "name": p.get("user_name") or p.get("name") or "Protagonist",
            "user_name": p.get("user_name") or p.get("name") or "Protagonist",
            "narrative": p.get("narrative", ""),
            "facts": facts,
            "gender": facts.get("demographics", ""),
            "values": facts.get("values", []),
            "personality": facts.get("personality", []),
            "interests": facts.get("interests", []),
            "experiences": facts.get("experiences", []),
        }

    def load_simulator(self):
        """Load and initiate the simulator."""
        self.round_cnt = 0

        self._maybe_autogen_scene_from_personas()

        self.data = Data(self.config, self.logger)
        self.environment_record = []

        self.narrator, self.characters = self.agents_creation()

        self.character_record = {i: [] for i in self.characters.keys()}

        utils.ensure_dir(self.config["record_dir"])
        title = self.config['scene_path'].split("/")[-1].split(".")[0]
        utils.ensure_dir(os.path.join(self.config["record_dir"], title))
        self.plot_record = self.data.scenes[self.config['scene_id']]
        self.plot_record['generated_plot'] = []
        self.plot_record['synopsis'] = []

        self.social_purpose = self.data.scenes[self.config['scene_id']].get("social_purpose", "")
        protagonist_ids = self._infer_protagonist_ids()
        self.protagonist_ids = protagonist_ids
        if hasattr(self.narrator, "reset_checkpoint_tracker"):
            self.narrator.reset_checkpoint_tracker(protagonist_ids)

        setattr(self.narrator, "goal", self.social_purpose)
        for _id, _agent in self.characters.items():
            setattr(_agent, "goal", self.social_purpose)

        for i in self.characters.keys():
            self.acting[i] = True
        self.logger.info("Simulator loaded.")

    def relevance_score_fn(self, score: float) -> float:
        """Return a similarity score on a scale [0, 1]."""
        return 1.0 - score / math.sqrt(2)

    def create_new_memory_retriever(self):
        """Create a new vector store retriever unique to the agent."""
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {'device': 'cpu'}
        embeddings_model = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
        embedding_size = 768

        index = faiss.IndexFlatL2(embedding_size)
        vectorstore = FAISS(
            embeddings_model.embed_query,
            index,
            InMemoryDocstore({}),
            {},
            relevance_score_fn=self.relevance_score_fn,
        )
        return TimeWeightedVectorStoreRetriever(vectorstore=vectorstore, other_score_keys=["importance"], k=5)

    def pause(self):
        self.play_event.clear()

    def play(self):
        self.play_event.set()

    def get_round_info(self):
        history = ""
        for i in range(len(self.round_record)):
            history += self.round_record[i]['character_name'] + ":" + self.round_record[i]['detail']['text'] + '\n'
        return history

    def _current_round_protagonist_history(self) -> str:
        lines = []
        for item in self.round_record:
            if item.get("type") not in {"action", "dialogue"}:
                continue
            char_id = item.get("character_id")
            if char_id not in self.protagonist_ids:
                continue
            detail = item.get("detail", "")
            if isinstance(detail, dict):
                text = detail.get("text", "") or detail.get("response", "") or detail.get("action", "")
            else:
                text = str(detail)
            text = (text or "").strip()
            if not text:
                continue
            name = item.get("character_name", "")
            lines.append(f"{name}: {text}")
        return "\n".join(lines)

    def get_character_obs(self, agent_id):
        return ". ".join(self.round_obs[agent_id])  

    def _mentioned_character_ids(self, speaker_id: int, text: str) -> list[int]:
        if not text:
            return []
        candidates = []
        for cid, char in self.characters.items():
            if cid == speaker_id:
                continue
            name = (char.name or "").strip()
            if not name:
                continue
            idx = self._find_name_in_text(text, name)
            if idx is not None:
                candidates.append((idx, cid))
        candidates.sort(key=lambda x: x[0])
        return [cid for _, cid in candidates]

    @staticmethod
    def _find_name_in_text(text: str, name: str) -> int | None:
        if not text or not name:
            return None
        if all(ord(c) < 128 for c in name):
            pattern = r"\b" + re.escape(name) + r"\b"
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                first = name.split()[0]
                if first and first != name:
                    pattern = r"\b" + re.escape(first) + r"\b"
                    match = re.search(pattern, text, flags=re.IGNORECASE)
            return match.start() if match else None
        idx = text.find(name)
        return idx if idx != -1 else None

    def save_record(self):
        title = self.config['scene_path'].split("/")[-1].split(".")[0]

        if self.config['save_record']:
            env_by_round = {}
            for item in self.environment_record:
                env_by_round.setdefault(item.get("round", -1), []).append(item)

            # Sanitize model names for file paths (replace '/' with '_')
            narrator_llm_safe = self.config['narrator_llm'].replace('/', '_')
            character_llm_safe = self.config['character_llm'].replace('/', '_')

            utils.ensure_dir(os.path.join(self.config["record_dir"], title, "persona_detail"))
            with open(os.path.join(self.config["record_dir"], title, "persona_detail",
                                   narrator_llm_safe + "_" + character_llm_safe + "_" + self.config.get(
                                       "character_record_path", "character.jsonl")), "a", encoding='utf8') as f:
                data = {"title": title, "scene_id": self.config['scene_id'], "scene": self.data.init_scene,
                        "characters": self.data.init_characters, "record": self.character_record}
                json.dump(data, f, default=lambda o: o.__dict__, ensure_ascii=False)
                f.write("\n")
            print("Character record saved in:",
                  os.path.join(self.config["record_dir"], title, "persona_detail",
                               narrator_llm_safe + "_" + character_llm_safe + "_" + self.config.get(
                                   "character_record_path", "character.jsonl")))

            utils.ensure_dir(os.path.join(self.config["record_dir"], title, "round_detail"))
            with open(os.path.join(self.config["record_dir"], title, "round_detail",
                                   narrator_llm_safe + "_" + character_llm_safe + "_" + self.config.get(
                                       "round_record_path", "round.jsonl")), "a", encoding='utf8') as f:
                data = {"title": title, "scene_id": self.config['scene_id'], "scene": self.data.init_scene,
                        "characters": self.data.init_characters, "record": self.records}
                json.dump(data, f, default=lambda o: o.__dict__, ensure_ascii=False)
                f.write("\n")
            print("Round record saved in:",
                  os.path.join(self.config["record_dir"], title, "round_detail",
                               narrator_llm_safe + "_" + character_llm_safe + "_" + self.config.get(
                                   "round_record_path", "round.jsonl")))

            sim_record_path = self.config.get("sim_record_path", "simulation.txt")
            utils.ensure_dir(os.path.join(self.config["record_dir"], title, "simulation"))
            sim_path = os.path.join(
                self.config["record_dir"],
                title,
                "simulation",
                f"{narrator_llm_safe}_{character_llm_safe}_{sim_record_path}",
            )
            with open(sim_path, "a", encoding="utf8") as f:
                for round_idx, round_items in enumerate(self.records, start=1):
                    f.write(f"round {round_idx}:\n")
                    for item in round_items:
                        item_type = item.get("type", "")
                        if item_type not in {"action", "dialogue", "reaction"}:
                            continue
                        name = item.get("character_name", "")
                        char_id = item.get("character_id")
                        if char_id is None:
                            continue
                        if (char_id not in self.protagonist_ids) and (not self._is_npc(char_id)):
                            continue
                        detail = item.get("detail", "")
                        if isinstance(detail, dict):
                            content = detail.get("text", "")
                            if not content:
                                content = detail.get("response", "") or detail.get("action", "")
                        else:
                            content = str(detail)
                        if item_type in {"reaction", "dialogue"}:
                            react_to = item.get("react_to", "")
                            if react_to:
                                f.write(f"{name} ({item_type}) to {react_to}: {content}\n")
                                continue
                        f.write(f"{name} ({item_type}): {content}\n")
                    for env_item in env_by_round.get(round_idx, []):
                        eval_detail = env_item.get("detail", {}) or {}
                        decision = eval_detail.get("decision", "")
                        reason = eval_detail.get("reason", "")
                        summary = eval_detail.get("summary", "")
                        remaining = eval_detail.get("remaining_checkpoints", {})
                        f.write(
                            "checkpoint: decision="
                            f"{decision}; summary={summary}; reason={reason}; remaining={remaining}\n"
                        )
                        characters = eval_detail.get("characters", []) or []
                        for char_report in characters:
                            name = char_report.get("name", "")
                            checkpoints = char_report.get("checkpoints", {}) or {}
                            evidence = char_report.get("evidence", {}) or {}
                            for ckpt, met in checkpoints.items():
                                if not met:
                                    continue
                                ev_list = evidence.get(ckpt, []) or []
                                ev_text = " | ".join(ev_list)
                                f.write(
                                    f"evidence: {name} {ckpt} -> {ev_text}\n"
                                )
                    f.write("\n")
            print("Simulation record saved in:", sim_path)

    def one_step(self, agent_id, interacted_result):

        self.play_event.wait()
        agent = self.characters[agent_id]
        name = agent.name
        message = []

        round_interaction_result = ""

        temp_obs = self.get_character_obs(agent_id)
        full_obs = temp_obs + interacted_result
        action, detail = agent.take_action(full_obs, self.synopsis[name], self.now)
        action_info = {"round": self.round_cnt, "character_id": agent_id, "character_name": name,
                       "type": "action", "detail": detail}
        self.round_record.append(action_info)
        self.character_record[agent_id].append(action_info)
        action_message = Message(agent_id=0, action="POST", content=f"{name}: {action}.")
        message.append(action_message)
        self.round_message.append(action_message)
        self.logger.info(f"{name}: {action}")  

        response, detail = agent.generate_dialogue(action, self.synopsis[name], self.now)
        self.logger.info(f"{name}: \"{response}\" ")  
        response_message = Message(agent_id=agent_id, action="POST", content=f"{name}: \"{response}\"")
        message.append(response_message)
        self.round_message.append(response_message)
        self.round_obs[agent_id].clear()  
        action_info = {"round": self.round_cnt, "character_id": agent_id, "character_name": name,
                       "type": "dialogue", "detail": detail}
        self.character_record[agent_id].append(action_info)
        self.round_record.append(action_info)

        reacted_ids = set()

        def handle_influence(source_type, influence_id, inf_action):
            nonlocal round_interaction_result

            if influence_id is None or influence_id == agent.id:
                return

            inf_agent = self.characters[influence_id]
            reactions = []
            actor_payload = response if source_type == "dialogue" else action
            reactions.append({"Actor:": name, "Action:": actor_payload})
            self.logger.info(f"{name} influenced {inf_agent.name}: {inf_action}")  

            self.round_obs[influence_id].append(f"{name}:{action}")
            self.round_obs[influence_id].append(f"{name}: {response}")
            self.round_obs[influence_id].append(f"{inf_action}")

            if source_type == "dialogue":
                reaction, detail = inf_agent.generate_dialogue(
                    self.get_character_obs(influence_id),
                    self.synopsis.get(inf_agent.name, ""),
                    self.now,
                )
            else:
                reaction, detail = inf_agent.take_reaction(self.get_character_obs(influence_id), "", self.now)  

            inf_agent.memory.add_memory(f"{name}: {action}", now=self.now)
            inf_agent.memory.add_memory(f"{name}: {inf_action}", now=self.now)
            inf_agent.memory.add_memory(f"{name}: {response}", now=self.now)
            inf_agent.memory.add_memory(f"{inf_agent.name}: {reaction}", now=self.now)

            if influence_id not in self.acted:
                self.acted[influence_id] = True
                self.acting.pop(influence_id)

            reaction_type = "dialogue" if source_type == "dialogue" else "reaction"
            action_info = {
                "round": self.round_cnt,
                "character_id": influence_id,
                "character_name": inf_agent.name,
                "type": reaction_type,
                "detail": detail,
                "react_to": name,
            }
            self.character_record[influence_id].append(action_info)
            self.round_record.append(action_info)
            self.logger.info(f"{inf_agent.name} react to {name}: {reaction}") 

            reactions.append({"Actor:": inf_agent.name, "Action:": reaction})
            reactions_info = ";".join([f"{r['Actor:']}:{r['Action:']}" for r in reactions])
            result, detail = self.narrator.analyze_result(reactions_info, self.now, verbose=True)
            round_interaction_result = result  
            action_info = {"round": self.round_cnt, "character_id": -1, "character_name": "Narrator",
                           "type": "Analyze Result", "detail": detail}
            self.round_record.append(action_info)
            self.narrator_record.append(action_info)
            self.logger.info(f"Interaction Result:{result}") 

            agent.memory.add_memory(result, now=self.now)
            inf_agent.memory.add_memory(result, now=self.now)

            inf_position, inf_states, detail = self.narrator.update_character(
                self.characters[influence_id].name, reactions_info + ";" + result, self.now, verbose=True)

            self.round_obs[influence_id].clear() 

            action_info = {"round": self.round_cnt, "character_id": -1, "character_name": "Narrator",
                           "type": "Update Character", "detail": detail}
            self.narrator_record.append(action_info)
            self.logger.info(
                f"update_character: name:{self.characters[influence_id].name}\n position:{inf_position}\n states:{inf_states}\n")  

            position, states, detail = self.narrator.update_character(
                self.characters[agent_id].name, reactions_info + ";" + result, self.now, verbose=True)
            action_info = {"round": self.round_cnt, "character_id": -1, "character_name": "Narrator",
                           "type": "Update Character", "detail": detail}
            self.narrator_record.append(action_info)
            self.logger.info(
                f"update_character: name:{self.characters[agent_id].name}\n position:{position}\n states:{states}\n")

            if position is not None and states is not None:
                for char_obj in [self.characters[influence_id], self.narrator.characters[influence_id],
                                 self.data.characters[influence_id], self.data.characters_list[influence_id]]:
                    char_obj.position = inf_position
                    char_obj.states = inf_states
                for char_obj in [self.characters[agent_id], self.narrator.characters[agent_id],
                                 self.data.characters[agent_id], self.data.characters_list[agent_id]]:
                    char_obj.position = position
                    char_obj.states = states

            reacted_ids.add(influence_id)

        action_influence_id = None
        for source_type, act_text, dlg_text in (
            ("action", action, ""),
            ("dialogue", "", response),
        ):
            if not act_text and not dlg_text:
                continue
            if source_type == "dialogue":
                if action_influence_id is None:
                    continue
                influence_id = action_influence_id
                inf_action = f"{agent.name} follows up with {self.characters[influence_id].name} in dialogue."
                influence_type = "dialogue"
                detail = {"text": f"Dialogue routed to action responder: {self.characters[influence_id].name}"}
            else:
                influence_id, inf_action, influence_type, detail = self.narrator.analyze_action_influence(
                    agent.name, act_text, self.now, dialogue=dlg_text, source_type=source_type, verbose=True
                )
                action_influence_id = influence_id
            influence_label = "Action Influence" if source_type == "action" else "Dialogue Influence"
            action_info = {"round": self.round_cnt, "character_id": -1, "character_name": "Narrator",
                           "type": influence_label, "detail": detail}
            self.narrator_record.append(action_info)
            print(
                f"influence_id:{influence_id}, inf_action:{inf_action}, influence_type:{influence_type}, "
                f"source_type:{source_type}"
            )
            handle_influence(source_type, influence_id, inf_action)

        if not reacted_ids:
            position, states, detail = self.narrator.update_character(self.characters[agent_id].name, action, self.now,
                                                                      verbose=True)
            action_info = {"round": self.round_cnt, "character_id": -1, "character_name": "Narrator",
                           "type": "Update Character", "detail": detail}
            self.narrator_record.append(action_info)
            self.logger.info(
                f"update_character: name:{self.characters[agent_id].name}\n position:{position}\n states:{states}\n")
            for char_obj in [self.characters[agent_id], self.narrator.characters[agent_id],
                             self.data.characters[agent_id], self.data.characters_list[agent_id]]:
                char_obj.position = position
                char_obj.states = states

        return message, round_interaction_result

    def round(self):
        """
        Run one step for all agents.
        """
        messages = []
        self.logger.info("Round {} start.".format(self.round_cnt))

        for i in self.characters.keys():
            self_belief = self.characters[i].update_self_belief(self.last_round, self.now)  
            other_character = self.get_other_agents(self.data.characters_list, i) 
            env_belief = self.characters[i].update_env_belief(self.last_round, other_character, self.now) 

        sequence = [self.characters[i].name for i in self.acting.keys()]
        self.synopsis = self.narrator.generate_synopsis(self.last_round, sequence, self.plot_record['synopsis'],
                                                        self.now)

        for i in self.characters.keys():
            if self.characters[i].name not in self.synopsis:
                self.synopsis[self.characters[i].name] = ""
        self.plot_record["synopsis"].append({"round": self.round_cnt, "synopsis": self.synopsis})
        self.logger.info(f"Round Synopsis:{self.synopsis}")

        step_results = []  
        inter_result = ""
        while len(self.acted) < len(self.characters):
            actor = self.acting.popitem(last=False)[0]
            msgs, interaction_result = self.one_step(actor, inter_result)
            inter_result = interaction_result
            step_results.append(interaction_result) 
            messages.append(msgs)
            self.acted[actor] = True

            if self.terminated:
                break

        self.last_round = self.get_round_info()
        new_scene, detail = self.narrator.update_scene(self.last_round, verbose=True)
        action_info = {"round": self.round_cnt, "character_id": -1, "character_name": "Narrator",
                       "type": "Update Scene", "detail": detail}
        self.narrator_record.append(action_info)
        self.logger.info(f"Scene:{new_scene}")  

        round_plot = self.narrator.summary_plot(self.last_round)
        self.plot_record["generated_plot"].append({"round": self.round_cnt, "plot": round_plot})
        self.logger.info(f"Round Plot:{round_plot}")  

        for i in self.characters.keys():
            self.characters[i].scene = new_scene

        if not self.terminated:
            last_evidence = ""
            for r in reversed(step_results):
                if r:
                    last_evidence = r
                    break
            if not last_evidence:
                last_evidence = round_plot if isinstance(round_plot, str) else self.last_round

            should_evaluate = (self.round_cnt % self.checkpoint_round_interval == 0)
            if should_evaluate and hasattr(self.narrator, "evaluate_round"):
                current_round_history = self._current_round_protagonist_history()
                decision, eval_detail = self.narrator.evaluate_round(
                    self.data.scenes[self.config['scene_id']]["event"],
                    self.social_purpose,
                    current_round_history,
                    "",
                    self.round_cnt,
                )

                remaining = eval_detail.get("remaining_checkpoints", {})
                self.logger.info(
                    "[EnvironmentAgent] round=%s decision=%s remaining=%s",
                    self.round_cnt,
                    decision,
                    remaining,
                )

                summary_text = eval_detail.get("summary", "")
                guidance_text = eval_detail.get("npc_guidance_summary", "")
                env_text = (
                    f"[EnvironmentAgent] decision={decision}; {summary_text}"
                    if summary_text
                    else f"[EnvironmentAgent] decision={decision}"
                )
                if guidance_text:
                    env_text = f"{env_text}; npc_guidance={guidance_text}"
                env_message = Message(agent_id=-2, action="POST", content=env_text)
                self.round_message.append(env_message)
                messages.append([env_message])
                if guidance_text:
                    for cid in self.characters.keys():
                        if not self._is_npc(cid):
                            continue
                        self.round_obs.setdefault(cid, []).append(
                            "NPC Guidance: In the current context, naturally guide the protagonist to cover missing checkpoints. "
                            f"Suggested directions: {guidance_text}"
                        )

                env_info = {
                    "round": self.round_cnt,
                    "character_id": -2,
                    "character_name": "EnvironmentAgent",
                    "type": "Checkpoint Evaluation",
                    "detail": eval_detail,
                }
                self.environment_record.append(env_info)

                if decision == "complete":
                    self.terminated = True
                    self.termination = {"decision": decision, "reason": eval_detail.get("reason", "")}

        self.records.append(copy.copy(self.round_record))
        self.round_message.clear()
        self.round_record.clear()
        self.acting, self.acted = self.acted, self.acting 
        return messages

    def _is_npc(self, char_id: int) -> bool:
        try:
            scene = self.data.scenes[self.config['scene_id']]
            for ch in scene.get("characters", []):
                if int(ch.get("id", -1)) == int(char_id):
                    return bool(ch.get("is_npc", False))
        except Exception:
            pass
        try:
            return bool(getattr(self.data.characters[char_id], "is_npc", False))
        except Exception:
            return False

    def create_character(self, i, api_key, api_base="") -> Character:

        is_npc = self._is_npc(i)

        if is_npc:
            llm_name = self.config.get("npc_llm", self.config.get("character_llm"))
            use_api_key = self.config.get("npc_api_key", self.config.get("character_api_key", ""))
            use_api_base = self.config.get("npc_api_base", self.config.get("character_api_base", ""))
            role_name = "npc"
        else:
            llm_name = self.config.get("character_llm")
            use_api_key = self.config.get("character_api_key", "")
            use_api_base = self.config.get("character_api_base", "")
            role_name = "character"

        LLM = utils.get_llm(
            llm_name,
            config=self.config,
            logger=self.logger,
            api_key=use_api_key,
            api_base=use_api_base,
            role=role_name,
        )

        agent_memory = GenerativeAgentMemory(
            llm=LLM,
            memory_retriever=self.create_new_memory_retriever(),
            verbose=False,
            reflection_threshold=10,
            max_tokens_limit=2500,
            importance_weight=0.1
        )
        agent = Character(
            id=self.data.characters[i].id,
            name=self.data.characters[i].name,
            scene=SceneInfo(**self.data.scenes[self.config['scene_id']]),
            position=self.data.characters[i].position,
            states=self.data.characters[i].states,
            traits=self.data.characters[i].description,
            llm=LLM,
            memory=agent_memory,
            status="",
        )
        setattr(agent, "is_npc", bool(is_npc))
        goal = self.data.scenes[self.config['scene_id']].get("social_purpose", "")
        setattr(agent, "goal", goal)
        return agent

    def create_human_player(self, id, api_key, api_base=None):
        LLM = utils.get_llm(self.config['character_llm'], config=self.config, logger=self.logger,
                            api_key=api_key, api_base=api_base, role="human_character")

        MemoryClass = GenerativeAgentMemory
        agent_memory = MemoryClass(
            llm=LLM,
            memory_retriever=self.create_new_memory_retriever(),
            verbose=False,
            reflection_threshold=15,
        )

        agent = HumanPlayer(
            id=id,
            name=self.data.characters[id].name,
            scene=SceneInfo(**self.data.scenes[self.config['scene_id']]),
            position=self.data.characters[id].position,
            states=self.data.characters[id].states,
            traits=self.data.characters[id].description,
            llm=LLM,
            memory=agent_memory,
            status="",
        )

        return agent

    def create_narrator(self, config, logger, api_key, api_base=None):
        LLM = utils.get_llm(self.config['narrator_llm'], config=self.config, logger=self.logger,
                            api_key=api_key, api_base=api_base, role="narrator")

        MemoryClass = GenerativeAgentMemory
        agent_memory = MemoryClass(
            llm=LLM,
            memory_retriever=self.create_new_memory_retriever(),
            verbose=False,
            reflection_threshold=15,
        )

        narrator = EnvironmentAgent(
            id=0,
            name="EnvironmentAgent",
            traits="",
            status="",
            llm=LLM,
            memory=agent_memory,
            scene=SceneInfo(**self.data.scenes[self.config['scene_id']]),
            characters=self.data.characters_list,
        )
        goal = self.data.scenes[self.config['scene_id']].get("social_purpose", "")
        setattr(narrator, "goal", goal)
        return narrator

    def get_other_agents(self, characters, agent_id):
        return [characters[i] for i in range(len(characters)) if characters[i].id != agent_id]

    def agents_creation(self):
        """
        Create agents in parallel 
        """
        characters = {}
        character_api_key = self.config.get("character_api_key", "")
        character_api_base = self.config.get("character_api_base", "")
        self.last_round = ""

        for action in self.data.init_actions:
            self.last_round += action["character"] + ":" + action['action'] + ";" + action["character"] + ":" + action[
                'dialogue'] + '\n'

        npc_human_control = self.config.get("npc_human_control", False)
        for i in tqdm(self.data.characters.keys()):
            if npc_human_control and self._is_npc(i):
                agent = self.create_human_player(i, character_api_key, character_api_base)
                characters[agent.id] = agent
                self.round_obs[agent.id] = []
                self.round_obs[agent.id].append(self.last_round)
                continue
            if self.config["play_role"] is not None and i == self.config["play_role"]:
                if not npc_human_control:
                    self.logger.warning(
                        "play_role=%s set but npc_human_control is false; using LLM for this character.",
                        i,
                    )
                elif not self._is_npc(i):
                    self.logger.warning(
                        "play_role=%s is not an NPC; keeping model-controlled character for this id.",
                        i,
                    )
                else:
                    role_id = self.config["play_role"]
                    agent = self.create_human_player(role_id, character_api_key, character_api_base)
                    characters[role_id] = agent
                    self.round_obs[agent.id] = []
                    self.round_obs[agent.id].append(self.last_round)
                    self.data.role_id = role_id
                    continue

            agent = self.create_character(i, character_api_key, character_api_base)
            characters[agent.id] = agent
            self.round_obs[agent.id] = []
            self.round_obs[agent.id].append(self.last_round)

        narrator_api_key = self.config.get("narrator_api_key", "")
        narrator_api_base = self.config.get("narrator_api_base", "")
        narrator = self.create_narrator(self.config, self.logger, narrator_api_key, narrator_api_base)

        return narrator, characters

    def _infer_protagonist_ids(self) -> list[int]:
        config_ids = self.config.get("protagonist_ids", None)
        if config_ids is not None:
            if isinstance(config_ids, (int, float)):
                return [int(config_ids)]
            if isinstance(config_ids, (list, tuple)):
                return [int(x) for x in config_ids]
            try:
                return [int(x) for x in list(config_ids)]
            except TypeError:
                pass

        scene = self.data.scenes.get(self.config['scene_id'], {}) if hasattr(self, 'data') else {}
        candidates = []
        for char in scene.get("characters", []):
            if not char.get("is_npc", False):
                candidates.append(int(char.get("id", 0)))
        if candidates:
            return candidates

        first_char = next(iter(scene.get("characters", [])), None)
        if first_char is not None:
            return [int(first_char.get("id", 0))]
        return [0]

    def reset(self):
        self.pause()
        self.round_cnt = 0
        self.load_simulator()
        return "The system is reset, and the historic records are removed."


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--character_llm", type=str, default=None, help="character llm name/alias")
    parser.add_argument("--scene_path", type=str, default=None, help="path to scene json")
    parser.add_argument("-c", "--config_file", type=str, required=True, help="Path to config file")
    parser.add_argument("-i", "--scene_id", type=int, default=None, help="scene id in json")
    parser.add_argument("-l", "--log_file", type=str, default=None, help="Path to log file")
    parser.add_argument("-n", "--log_name", type=str, default=None, help="Name of logger")
    parser.add_argument("-p", "--play_role", type=int, default=None, help="Add a user controllable role")

    parser.add_argument(
        "--npc_human_control",
        action="store_true",
        help="Allow a specified NPC (play_role) to be controlled by a human",
    )

    parser.add_argument("--persona_jsonl_path", type=str, default="", help="Path to persona jsonl for autogen scene")
    parser.add_argument("--persona_index", type=int, default=None, help="Which persona line to use (0-based)")
    parser.add_argument(
        "--persona_indices",
        type=str,
        default="",
        help="Batch persona selector, e.g. '0-12' or '0,2,5'. When set, batch mode is enabled.",
    )
    parser.add_argument(
        "--experiment_parallelism",
        type=int,
        default=1,
        help="How many persona tasks to run concurrently in batch mode",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="In batch mode, skip tasks that already have .done marker or completed log",
    )
    parser.add_argument("--autogen_scene_from_persona", action="store_true", help="Autogen scene from persona first")

    parser.add_argument("--npc_llm", type=str, default=None, help="npc llm name/alias")
    parser.add_argument("--npc_api_key", type=str, default=None, help="api key for npc llm")
    parser.add_argument("--npc_api_base", type=str, default=None, help="api base for npc llm")

    parser.add_argument("--scene_language", type=str, choices=["zh", "en"], default=None)

    args = parser.parse_args()
    return args


def _parse_persona_indices(spec: str) -> list[int]:
    values: list[int] = []
    seen = set()
    text = (spec or "").strip()
    if not text:
        return values
    for part in text.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            seg = token.split("-", 1)
            if len(seg) != 2 or (not seg[0].isdigit()) or (not seg[1].isdigit()):
                raise ValueError(f"Invalid persona_indices token: {token}")
            start = int(seg[0])
            end = int(seg[1])
            if start > end:
                raise ValueError(f"Invalid persona_indices range: {token}")
            for i in range(start, end + 1):
                if i not in seen:
                    values.append(i)
                    seen.add(i)
        else:
            if not token.isdigit():
                raise ValueError(f"Invalid persona_indices token: {token}")
            i = int(token)
            if i not in seen:
                values.append(i)
                seen.add(i)
    return values


def _run_single(args, persona_index_override: int | None = None, log_file_override: str | None = None, log_name_override: str | None = None):
    # NOTE: this keeps the original single-persona execution flow.
    # Legacy main logic is retained here, then reused by batch mode.

    config = CfgNode(new_allowed=True)
    config.merge_from_file(args.config_file)
    if (args.log_file is not None) and (log_file_override is None):
        config = utils.add_variable_to_config(config, "log_file", args.log_file)
    if (args.log_name is not None) and (log_name_override is None):
        config = utils.add_variable_to_config(config, "log_name", args.log_name)
    if args.play_role is not None:
        config = utils.add_variable_to_config(config, "play_role", args.play_role)

    if args.character_llm is not None:
        config['character_llm'] = args.character_llm
    if args.scene_path is not None:
        config['scene_path'] = args.scene_path
    if args.scene_id is not None:
        config['scene_id'] = args.scene_id
    if args.scene_language is not None:
        config["scene_language"] = args.scene_language

    if args.persona_jsonl_path:
        config["persona_jsonl_path"] = args.persona_jsonl_path
    if args.autogen_scene_from_persona:
        config["autogen_scene_from_persona"] = True
    if persona_index_override is not None:
        config["persona_index"] = persona_index_override
    elif args.persona_index is not None:
        config["persona_index"] = args.persona_index

    if args.npc_llm is not None:
        config["npc_llm"] = args.npc_llm
    if args.npc_api_key is not None:
        config["npc_api_key"] = args.npc_api_key
    if args.npc_api_base is not None:
        config["npc_api_base"] = args.npc_api_base
    if args.npc_human_control:
        config["npc_human_control"] = True

    if "max_rounds" not in config:
        config["max_rounds"] = 20
    if "checkpoint_round_interval" not in config:
        config["checkpoint_round_interval"] = config.get("judge_round_interval", 1)

    title = config['scene_path'].split("/")[-1].split(".")[0] if config.get("scene_path") else "autogen"
    log_file = log_file_override or (config.get("log_file", "") or "")
    log_name = log_name_override or config.get("log_name") or str(os.getpid())
    if log_file == "":
        # Sanitize model names for file paths (replace '/' with '_')
        narrator_llm_safe = config['narrator_llm'].replace('/', '_')
        character_llm_safe = config['character_llm'].replace('/', '_')
        utils.ensure_dir(f"output/log/simulation/{title}")
        log_file = (
            f"output/log/simulation/{title}/"
            f"{narrator_llm_safe}_{character_llm_safe}_{config.get('scene_id', 0)}_character_simulation.log"
        )

    config = utils.add_variable_to_config(config, "log_file", log_file)
    config = utils.add_variable_to_config(config, "log_name", log_name)
    logger = utils.set_logger(log_file, log_name)
    logger.info(f"os.getpid()={os.getpid()}")
    logger.info(f"\n{config}")

    role_agent = Simulator(config, logger)
    role_agent.load_simulator()

    role_agent.play()

    messages = []
    st = time.time()

    max_rounds = int(config.get("max_rounds", config.get("round", 20)))
    while (not role_agent.terminated) and (role_agent.round_cnt < max_rounds):
        role_agent.round_cnt += 1
        role_agent.logger.info(f"Round {role_agent.round_cnt}")
        msg = role_agent.round()
        messages.append(msg)

    if not role_agent.terminated:
        role_agent.terminated = True
        role_agent.termination = {"decision": "continue", "reason": f"Reached max_rounds={max_rounds}"}

    role_agent.duration = time.time() - st
    logger.info(f"Duration:{role_agent.duration:.3f}s")
    print(f"Duration:{role_agent.duration:.3f}s")

    role_agent.save_record()
    return True


def _run_batch(args):
    persona_indices = _parse_persona_indices(args.persona_indices)
    if not persona_indices:
        raise ValueError("Batch mode requires non-empty --persona_indices")

    parallelism = max(1, int(args.experiment_parallelism))
    cfg_base = os.path.basename(args.config_file).replace(".yaml", "")
    done_dir = os.path.join("output", "batch_done")
    utils.ensure_dir(done_dir)
    utils.ensure_dir(os.path.join("output", "log"))

    print(
        f"[BATCH] config={args.config_file} personas={len(persona_indices)} "
        f"parallelism={parallelism} skip_existing={1 if args.skip_existing else 0}",
        flush=True,
    )

    def one_persona(idx: int):
        done_file = os.path.join(done_dir, f"{cfg_base}_p{idx}.done")
        log_rel = f"batch_{cfg_base}_p{idx}.log"
        log_full = os.path.join("output", "log", log_rel)

        if args.skip_existing:
            if os.path.exists(done_file):
                print(f"[SKIP] {cfg_base} persona_index={idx} (found done)", flush=True)
                return
            if os.path.exists(log_full):
                try:
                    with open(log_full, "r", encoding="utf-8") as f:
                        if "Duration:" in f.read():
                            print(f"[SKIP] {cfg_base} persona_index={idx} (completed log)", flush=True)
                            with open(done_file, "w", encoding="utf-8"):
                                pass
                            return
                except Exception:
                    pass

        print(f"[START] {cfg_base} persona_index={idx}", flush=True)
        _run_single(
            args,
            persona_index_override=idx,
            log_file_override=log_rel,
            log_name_override=f"{os.getpid()}_{cfg_base}_{idx}",
        )
        with open(done_file, "w", encoding="utf-8"):
            pass
        print(f"[DONE]  {cfg_base} persona_index={idx}", flush=True)

    failed = 0
    if parallelism == 1:
        for idx in persona_indices:
            try:
                one_persona(idx)
            except Exception as e:
                failed += 1
                print(f"[FAIL]  {cfg_base} persona_index={idx}: {e}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=parallelism) as executor:
            futures = {executor.submit(one_persona, idx): idx for idx in persona_indices}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Batch", unit="persona"):
                idx = futures[future]
                try:
                    future.result()
                except Exception as e:
                    failed += 1
                    print(f"[FAIL]  {cfg_base} persona_index={idx}: {e}", flush=True)

    if failed > 0:
        raise RuntimeError(f"Batch finished with failures: {failed}")


def main():
    args = parse_args()

    # Legacy behavior is preserved: without --persona_indices, run a single persona task.
    if (args.persona_indices or "").strip():
        _run_batch(args)
        return

    _run_single(args)


if __name__ == "__main__":
    main()
