from __future__ import annotations

import json
import re
import copy
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from utils import utils
from utils.character import CharacterInfo

from .narrator import Narrator


def _norm_ev(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[，,。.\!！\?？：:\-—；;【】\[\]\(\)（）\"'“”‘’]", "", s)
    return s


class EnvironmentAgent(Narrator):

    CHECKPOINTS: "OrderedDict[str, str]" = OrderedDict(
        [
            (
                "background",
                "Background identity: any identifiable information such as occupation/role, origin/culture, "
                "family/education, relationships with others, or current situation/occasion. Also counts when "
                "the protagonist mentions what they do or what they are doing (e.g., “I do…”, “when I’m doing…”), "
                "since this implies identity and should be treated as background. Evidence can come from "
                "self-statements or inferable scene/objects (e.g., badge, school emblem, uniform, on duty in a hospital).",
            ),
            ("values", "Values: any stated or inferable stance/principle/standard/boundary about people or things."),
            (
                "personality",
                "Personality traits: stable characteristics reflected through language, actions, decision style, "
                "or emotional reactions (e.g., cautious, considerate, impulsive, decisive, humorous).",
            ),
            (
                "interests",
                "Interests/preferences: any reaction showing like/dislike/preference/frequency/collection/attention/choice "
                "(must indicate liking or preference).",
            ),
            (
                "experiences",
                "Experiences: things done/participated in/learned/visited/encountered/obtained or lost, in the recent or past; "
                "time can be vague. Evidence may be self-stated or inferred from objects/photos/scars; success or failure both count.",
            ),
        ]
    )

    protagonist_ids: List[int] = []
    checkpoint_state: Dict[int, Dict[str, bool]] = {}
    checkpoint_evidence: Dict[int, Dict[str, List[str]]] = {}
    checkpoint_met_counts: Dict[int, Dict[str, int]] = {}
    evaluation_history: List[Dict[str, Any]] = []

    # ---------- Helpers ----------
    def reset_checkpoint_tracker(self, protagonist_ids: Optional[List[int]] = None) -> None:
        if protagonist_ids is not None:
            self.protagonist_ids = protagonist_ids
        if not getattr(self, "protagonist_ids", None):
            self.protagonist_ids = [0]

        self.checkpoint_state = {
            pid: {key: False for key in self.CHECKPOINTS.keys()} for pid in self.protagonist_ids
        }
        self.checkpoint_evidence = {
            pid: {key: [] for key in self.CHECKPOINTS.keys()} for pid in self.protagonist_ids
        }
        self.checkpoint_met_counts = {
            pid: {key: 0 for key in self.CHECKPOINTS.keys()} for pid in self.protagonist_ids
        }
        self.evaluation_history = []

    @staticmethod
    def _names_of(characters: List[CharacterInfo]) -> List[str]:
        names = []
        for c in characters or []:
            n = (getattr(c, "name", None) or "").strip()
            if n:
                names.append(n)
        return names

    @staticmethod
    def _filter_speaker_history(text: str, allowed_names: List[str]) -> str:

        if not text:
            return ""
        names_pat = "|".join(re.escape(n) for n in allowed_names if n)
        if not names_pat:
            return ""
        pat = re.compile(rf"^\s*(?:{names_pat})\s*:", re.UNICODE)
        kept_lines = []
        for line in text.splitlines():
            if pat.match(line):
                kept_lines.append(line.strip())
        return "\n".join(kept_lines)

    def _stop_rule_for_pid(self, pid: int) -> Tuple[bool, Optional[str]]:

        state = self.checkpoint_state.get(pid, {}) or {}
        counts = self.checkpoint_met_counts.get(pid, {}) or {}

        met_keys = [k for k, v in state.items() if bool(v)]
        met_num = len(met_keys)

        met_ge3 = [k for k in met_keys if counts.get(k, 0) >= 3]

        if len(met_ge3) >= 5:
            return True, "ALL5"

        met_ge5 = [k for k in met_keys if counts.get(k, 0) >= 4]
        if len(met_ge5) >= 4:
            return True, "4x4+"

        met_ge10 = [k for k in met_keys if counts.get(k, 0) >= 8]
        if len(met_ge10) >= 3:
            return True, "3x8+"

        return False, None

    # ---------- Checkpoint evaluation ----------
    def evaluate_round(
        self,
        event: str,
        social_purpose: str,
        round_history: str,
        latest_summary: str,
        round_index: int,
    ) -> tuple[str, Dict[str, Any]]:

        protagonists: List[CharacterInfo] = []
        all_chars: List[CharacterInfo] = getattr(self, "characters", []) or []
        pid_set = set(self.protagonist_ids or [0])
        for character in all_chars:
            is_npc = bool(getattr(character, "is_npc", False))
            if (character.id in pid_set) and (not is_npc):
                protagonists.append(character)

        if not protagonists:
            non_npc = [c for c in all_chars if not bool(getattr(c, "is_npc", False))]
            if non_npc:
                protagonists = [non_npc[0]]
            elif all_chars:
                protagonists = [all_chars[0]]

        allowed_names = self._names_of(protagonists)
        filtered_history = self._filter_speaker_history(round_history or "", allowed_names)

        lines = (latest_summary or "").splitlines()
        speaker_like = sum(1 for ln in lines if (":" in ln) or ("：" in ln))
        if speaker_like >= max(3, len(lines) // 2):
            filtered_summary = self._filter_speaker_history(latest_summary or "", allowed_names)
        else:
            filtered_summary = latest_summary or ""

        raw_status = self._extract_checkpoint_status(
            event, social_purpose, filtered_history, filtered_summary, protagonists
        )

        for status in raw_status:
            pid = status.get("character_id")
            if pid not in self.checkpoint_state:
                continue
            checkpoints = status.get("checkpoints", {})
            for key in self.CHECKPOINTS.keys():
                node = checkpoints.get(key, {})
                met = bool(node.get("met", False))
                evidence = (node.get("evidence") or "").strip()

                if met:
                    self.checkpoint_state[pid][key] = True

                    added_new_evidence = False
                    if evidence:
                        bucket = self.checkpoint_evidence[pid][key]
                        if evidence not in bucket:
                            norm = _norm_ev(evidence)
                            if all(_norm_ev(x) != norm for x in bucket):
                                bucket.append(evidence[:200])
                                added_new_evidence = True
                               
                                try:
                                    char_name = next(
                                        (c.name for c in getattr(self, "characters", []) if c.id == pid),
                                        str(pid)
                                    )
                                    self.logger.info(
                                        "[CheckpointEval] ✓ met | pid=%s name=%s | checkpoint=%s | evidence=%s",
                                        pid, char_name, key,
                                        evidence if len(evidence) <= 300 else evidence[:300] + "…"
                                    )
                                except Exception:
                                    pass

                    
                    if added_new_evidence:
                        try:
                            self.checkpoint_met_counts[pid][key] += 1
                        except KeyError:
                            self.checkpoint_met_counts.setdefault(pid, {}).setdefault(key, 0)
                            self.checkpoint_met_counts[pid][key] += 1

      
        stop_rules_map: Dict[int, Optional[str]] = {}
        should_stop_all = True
        for character in protagonists:
            pid = character.id
            should_stop, rule_name = self._stop_rule_for_pid(pid)
            stop_rules_map[pid] = rule_name
            if not should_stop:
                should_stop_all = False

        decision = "complete" if should_stop_all else "continue"
        if decision == "complete":
            reason = "All protagonists triggered an early-stop rule (ALL5 / 4x5+ / 3x10+)."
        else:
            reason = "No early-stop rule met; continue the interaction."

        
        character_reports: List[Dict[str, Any]] = []
        summary_lines: List[str] = []
        guidance_lines: List[str] = []
        remaining_overall: Dict[int, List[str]] = {}
        guidance_overall: Dict[int, str] = {}

        for character in protagonists:
            pid = character.id

            
            state_live = self.checkpoint_state.get(pid, {}) or {}
            state_snapshot = {k: bool(state_live.get(k, False)) for k in self.CHECKPOINTS.keys()}

            
            evidences_live = self.checkpoint_evidence.get(pid, {}) or {}
            evidences_snapshot = {k: list(evidences_live.get(k, [])) for k in self.CHECKPOINTS.keys()}

           
            counts_live = self.checkpoint_met_counts.get(pid, {}) or {}
            counts_snapshot = {k: int(counts_live.get(k, 0)) for k in self.CHECKPOINTS.keys()}

            
            missing_keys = [name for name, met in state_snapshot.items() if not met]
            missing_labels = [self.CHECKPOINTS[k] for k in missing_keys]

            character_reports.append(
                {
                    "id": pid,
                    "name": character.name,
                    "checkpoints": state_snapshot,
                    "evidence": evidences_snapshot,
                    "met_counts": counts_snapshot,        
                    "stop_rule": stop_rules_map.get(pid), 
                    "missing": missing_labels,
                }
            )

            if missing_labels:
                summary_lines.append(f"{character.name} missing: {', '.join(missing_labels)}")
                remaining_overall[pid] = missing_labels
                guidance = f"Guide the protagonist to cover: {', '.join(missing_labels)}"
                guidance_lines.append(f"{character.name}: {guidance}")
                guidance_overall[pid] = guidance
            else:
                summary_lines.append(f"{character.name} has covered all checkpoints")
                guidance_lines.append(f"{character.name}: No additional guidance needed")
                guidance_overall[pid] = "No additional guidance needed"

        detail = {
            "round": round_index,
            "event": event,
            "social_purpose": social_purpose,
            "decision": decision,
            "summary": " | ".join(summary_lines),
            "reason": reason,
            "characters": character_reports,
            "remaining_checkpoints": remaining_overall,
            "npc_guidance": guidance_overall,
            "npc_guidance_summary": " | ".join(guidance_lines),
            "stop_rules": stop_rules_map,  
        }

        try:
            self.logger.info("[CheckpointEval] round=%s decision=%s summary=%s",
                             round_index, decision, detail["summary"])
        except Exception:
            pass

        self.evaluation_history.append(copy.deepcopy(detail))
        return decision, detail

    @staticmethod
    def _augment_desc_with_persona_facts(orig_desc: str, persona: Dict[str, Any]) -> str:
        desc = (orig_desc or "").strip()
        facts = (persona or {}).get("facts", {}) or {}

        def join_items(lst):
            if isinstance(lst, (list, tuple)):
                return ", ".join([str(x) for x in lst if str(x).strip()])
            return str(lst).strip()

        labels = {
            "demographics": "",
            "occupation": "Occupation: ",
            "personality": "Personality: ",
            "values": "Values: ",
            "interests": "Interests: ",
            "experiences": "Experiences: ",
        }
        stop_mark = "; "

        pieces: List[str] = []

        demo = str(facts.get("demographics", "")).strip()
        if demo and demo not in desc:
            pieces.append(demo)

        for key in ["occupation", "personality", "values", "interests", "experiences"]:
            val = facts.get(key, [] if key in {"personality", "values", "interests", "experiences"} else "")
            text = join_items(val)
            if text:
                seg = f"{labels[key]}{text}"
                if seg not in desc:
                    pieces.append(seg)

        if pieces:
            return (desc + (stop_mark if desc else "") + stop_mark.join(pieces)).strip("; ").strip()
        return desc

    def _pick_persona_for_main(self, personas: List[Dict[str, Any]], main_name: str) -> Optional[Dict[str, Any]]:
        for p in personas or []:
            cand = (p.get("name") or p.get("user_name") or "").strip()
            if cand and cand == (main_name or "").strip():
                return p
        return personas[0] if personas else None

    def _extract_checkpoint_status(
        self,
        event: str,
        social_purpose: str,
        round_history: str,
        latest_summary: str,
        protagonists: List[CharacterInfo],
    ) -> List[Dict[str, Any]]:
        if not protagonists:
            return []

        protagonist_block = "\n".join(
            [f"- id={c.id}, name={c.name}" for c in protagonists]
        )
        checkpoint_list = "\n".join([f"  - {key}: {label}" for key, label in self.CHECKPOINTS.items()])

        prompt = (
            "You evaluate whether protagonists have revealed required personal information.\n"
            "Consider ONLY the protagonists listed below. Ignore any NPCs entirely.\n"
            "Information can accumulate across rounds.\n"
            "For each protagonist, determine if each checkpoint is clearly evidenced by Dialogue or ACTIONS so far.\n"
            "Event: {event}\nSocial Goal: {goal}\n"
            "Protagonists:\n{protagonists}\n"
            "Required checkpoints:\n{checkpoint_list}\n\n"
            "Judging rules:\n"
            "- 'interests' can be satisfied by explicit statements (e.g., \"I like...\", \"I prefer...\") "
            "  OR by consistent choices/behaviors showing preference (e.g., repeatedly picking X over Y, "
            "  accepting/asking for specific style/flavor/option).\n"
            "- 'background' should be marked true if the protagonist implies a regular role or identity through phrasing "
            "  like \"when I dance/teach/perform/work\", \"on shift\", or similar habitual activity language.\n"
            "- Evidence MUST be a short quote or action snippet taken from the Conversation history below.\n"
            "- Do NOT use character descriptions, states, or any external info as evidence.\n"
            "- If the evidence is not explicitly present in the Conversation history, set met=false.\n\n"
            "Conversation history (protagonist-only speaker:text):\n{history}\n\n"
            # "Latest summary/evidence (may be filtered):\n{summary}\n\n"
            "Return ONLY a JSON array. Each item must be:\n"
            "{{\n"
            "  \"character_id\": int,\n"
            "  \"character_name\": str,\n"
            "  \"checkpoints\": {{\n"
            "     \"background\": {{\"met\": bool, \"evidence\": str}},\n"
            "     \"values\": {{\"met\": bool, \"evidence\": str}},\n"
            "     \"interests\": {{\"met\": bool, \"evidence\": str}},\n"
            "     \"personality\": {{\"met\": bool, \"evidence\": str}},\n"
            "     \"experiences\": {{\"met\": bool, \"evidence\": str}}\n"
            "  }}\n"
            "}}\n"
            "If unsure or absent, set met=false and evidence=\"\".\n"
            "Output nothing else."
        ).format(
            event=event,
            goal=social_purpose,
            protagonists=protagonist_block,
            checkpoint_list=checkpoint_list,
            history=round_history or "(empty)",
            summary=latest_summary or "(none)",
        )

        try:
            response = self.llm.invoke(prompt)
            content = getattr(response, "content", None) or str(response)

            try:
                preview = content if len(content) < 1500 else content[:1500] + " ...<truncated>"
                self.logger.info("[CheckpointEval] raw LLM output preview: %s", preview)
            except Exception:
                pass

            data = self._parse_json_array(content)

            try:
                self.logger.info("[CheckpointEval] parsed JSON: %s", json.dumps(data, ensure_ascii=False))
            except Exception:
                pass

        except Exception as e:
            self.logger.error("[CheckpointEval] LLM invoke/parse failed: %s", e)
            return []

        cleaned: List[Dict[str, Any]] = []
        valid_ids = {c.id for c in protagonists}
        for item in data or []:
            try:
                cid = int(item.get("character_id"))
            except (TypeError, ValueError):
                continue
            if cid not in valid_ids:
                continue

            checkpoints = item.get("checkpoints", {}) or {}
            normalized = {}
            for key in self.CHECKPOINTS.keys():
                node = checkpoints.get(key) or {}
                normalized[key] = {
                    "met": bool(node.get("met", False)),
                    "evidence": (node.get("evidence") or "").strip(),
                }

            cleaned_item = {
                "character_id": cid,
                "character_name": item.get("character_name", ""),
                "checkpoints": normalized
            }
            cleaned.append(cleaned_item)

            try:
                self.logger.info(
                    "[CheckpointEval] character_id=%s name=%s checkpoints=%s",
                    cid,
                    cleaned_item["character_name"],
                    json.dumps(normalized, ensure_ascii=False),
                )
            except Exception:
                pass

        return cleaned

    @staticmethod
    def _parse_json_array(text: str) -> List[Dict[str, Any]]:
        if not text:
            return []
        s = text.strip()
        fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, flags=re.IGNORECASE | re.DOTALL)
        if fenced:
            s = fenced.group(1).strip()
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        match = re.search(r"\[(?:.|\n)*\]", s)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                return []
        return []

    @staticmethod
    def _clean_ws(s: Optional[str]) -> str:
        return re.sub(r"\s+", " ", (s or "").strip())

    @staticmethod
    def _index_characters_by_name(characters: List[Dict[str, Any]]) -> Dict[str, int]:
        name2id = {}
        for i, c in enumerate(characters):
            c["id"] = i
            nm = (c.get("name") or "").strip()
            if nm:
                name2id[nm] = i
        return name2id

    @classmethod
    def _postprocess_scene_output(
        cls,
        data: Dict[str, Any],
        personas: List[Dict[str, Any]],
        protagonist_ids: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        if not isinstance(data, dict) or "scenes" not in data or not data["scenes"]:
            raise ValueError("Scene JSON invalid: missing non-empty 'scenes'.")
        scene = data["scenes"][0]

        for k in ["id","event","time","location","description","characters","actions","plot","social_purpose","chunk"]:
            scene.setdefault(k, 0 if k in ["id"] else ([] if k in ["characters","actions"] else ""))

        for txt in ["event","time","location","description","plot","social_purpose"]:
            scene[txt] = cls._clean_ws(scene.get(txt))

        raw_chars = scene.get("characters") or []
        characters: List[Dict[str, Any]] = []
        for c in raw_chars:
            characters.append({
                "id": -1,
                "name": cls._clean_ws(c.get("name")),
                "gender": cls._clean_ws(c.get("gender")),
                "description": cls._clean_ws(c.get("description")),
                "position": cls._clean_ws(c.get("position")),
                "states": cls._clean_ws(c.get("states")),
            })
        scene["characters"] = characters
        name2id = cls._index_characters_by_name(scene["characters"])

        raw_actions = scene.get("actions") or []
        actions: List[Dict[str, Any]] = []
        for a in raw_actions:
            ch_name = cls._clean_ws(a.get("character"))
            if ch_name not in name2id:
                raise ValueError(f"Action references unknown character: {ch_name}")
            actions.append({
                "action_id": -1,
                "character": ch_name,
                "character_id": name2id[ch_name],
                "action": cls._clean_ws(a.get("action")),
                "dialogue": cls._clean_ws(a.get("dialogue")),
            })
        for i, a in enumerate(actions):
            a["action_id"] = i
        scene["actions"] = actions

        chunk = scene.get("chunk") or {}
        scene["chunk"] = {"id": 0, "text": cls._clean_ws(chunk.get("text") or chunk.get("content") or "")}
        scene["id"] = 0

        main_id = 0
        if protagonist_ids and len(protagonist_ids) > 0:
            main_id = int(protagonist_ids[0])
        if not (0 <= main_id < len(scene["characters"])):
            main_id = 0

        main_name = scene["characters"][main_id]["name"] or ""
        persona_for_main: Optional[Dict[str, Any]] = None
        for p in personas or []:
            cand = (p.get("name") or p.get("user_name") or "").strip()
            if cand and cand == main_name:
                persona_for_main = p
                break
        if persona_for_main is None and personas:
            persona_for_main = personas[0]

        if persona_for_main:
            scene["characters"][main_id]["description"] = cls._augment_desc_with_persona_facts(
                scene["characters"][main_id].get("description", ""),
                persona_for_main,
            )

        npcs: List[int] = []
        for i, c in enumerate(scene["characters"]):
            c["is_npc"] = (i != main_id)
            if c["is_npc"]:
                npcs.append(i)
        scene["npcs"] = npcs

        if not data.get("title"):
            main_name = scene["characters"][main_id]["name"] or None
            if not main_name:
                for p in personas:
                    main_name = p.get("name") or p.get("user_name")
                    if main_name:
                        break
            data["title"] = cls._clean_ws(main_name or "Protagonist")

        data["scenes"] = [scene]
        return data

    @classmethod
    def generate_scene_from_personas(
        cls,
        config: Dict[str, Any],
        logger,
        personas: List[Dict[str, Any]],
        event_hint: str = "",
        social_purpose: str = "",
        language: str = "",
    ) -> Dict[str, Any]:
        if not personas:
            raise ValueError("personas must not be empty")

        narrator_llm = config.get("environment_scene_llm", config.get("narrator_llm"))
        api_key = config.get("environment_scene_api_key", config.get("narrator_api_key", ""))
        api_base = config.get("environment_scene_api_base", config.get("narrator_api_base", ""))

        llm = utils.get_llm(
            narrator_llm, config=config, logger=logger, api_key=api_key, api_base=api_base, role="environment_scene",
        )

        prompt = cls._build_scene_prompt(personas, event_hint, social_purpose, language)
        response = llm.invoke(prompt)
        content = getattr(response, "content", None) or str(response)
        raw = cls._parse_scene_output(content)

        protagonist_ids = None
        if "protagonist_ids" in config:
            val = config.get("protagonist_ids")
            if isinstance(val, (list, tuple)):
                protagonist_ids = [int(x) for x in val]
            else:
                try:
                    protagonist_ids = [int(val)]
                except Exception:
                    pass

        scene = cls._postprocess_scene_output(raw, personas, protagonist_ids=protagonist_ids)
        return scene

    @staticmethod
    def _build_scene_prompt(
        personas: List[Dict[str, Any]],
        event_hint: str,
        social_purpose: str,
        language: str,
    ) -> str:
        persona_lines = []
        for persona in personas:
            pid = persona.get("id")
            name = persona.get("name") or persona.get("user_name") or f"Character {pid}"
            narrative = persona.get("narrative") or persona.get("background", "")
            facts = persona.get("facts", {})
            entry = {
                "id": pid,
                "name": name,
                "gender": persona.get("gender", facts.get("gender", "")),
                "narrative": narrative,
                "facts": facts,
                "values": persona.get("values") or facts.get("values", []),
                "personality": persona.get("personality") or facts.get("personality", []),
                "interests": persona.get("interests") or facts.get("interests", []),
                "experiences": persona.get("experiences") or facts.get("experiences", []),
            }
            persona_lines.append(json.dumps(entry, ensure_ascii=False))
        persona_block = "\n".join(persona_lines)

        instruction = (
            "Create a realistic social scenario grounded in the personas below. Ensure the main protagonist and 2-3 supporting characters engage around a shared goal, with actions consistent with their traits."
        )
        format_hint = (
            "Return JSON with structure:\n"
            "{\n  \"title\": ...,\n  \"scenes\": [\n    {\n      \"id\": 0,\n      \"event\": ...,\n      \"time\": ...,\n      \"location\": ...,\n      \"description\": ...,\n"
            "      \"characters\": [\n        {\"id\": int, \"name\": str, \"gender\": str, \"description\": str, \"position\": str, \"states\": str}\n      ],\n"
            "      \"actions\": [\n        {\"action_id\": int, \"character\": str, \"character_id\": int, \"action\": str, \"dialogue\": str}\n      ],\n"
            "      \"plot\": str,\n      \"social_purpose\": str,\n      \"chunk\": {\"id\": int, \"text\": str}\n    }\n  ]\n}\n"
            "Characters must include id, name, gender, description, position, states. Actions must include action_id, character, character_id, action, dialogue."
        )
        goal_line = social_purpose or "(unspecified)"
        event_line = event_hint or "(none)"

        prompt = (
            f"{instruction}\n\n"
            f"Personas (one JSON per line):\n{persona_block}\n\n"
            f"Scene event hint: {event_line}\n"
            f"Social goal hint: {goal_line}\n"
            f"{format_hint}\n"
            "Only output JSON. Do not add any explanation."
        )
        return prompt

    @classmethod
    def _parse_scene_output(cls, content: str) -> Dict[str, Any]:
        if not content:
            raise ValueError("Empty response when generating scene")
        try:
            data = json.loads(content)
            if isinstance(data, dict) and "scenes" in data:
                return data
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{(?:.|\n)*\}", content)
        if not match:
            raise ValueError("Failed to parse scene JSON output")
        data = json.loads(match.group(0))
        if "scenes" not in data:
            raise ValueError("Scene JSON missing 'scenes' key")
        return data
