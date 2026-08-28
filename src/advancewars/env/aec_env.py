"""PettingZoo-style AEC environment."""

from __future__ import annotations

from typing import Any

from advancewars.engine.actions import Action
from advancewars.engine.coordinates import Coord
from advancewars.engine.game import Game
from advancewars.env.action_codec import ActionCodec
from advancewars.env.observations import board_planes
from advancewars.env.rewards import rewards_for
from advancewars.env.structured_action_codec import (
    StructuredActionCodec,
    semantic_to_dict,
)
from advancewars.utils.defendpeace_map import parse_defendpeace_map

try:
    from gymnasium.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete
except Exception:  # pragma: no cover - exercised when optional dep is absent
    Box = Dict = MultiBinary = None
    from advancewars.env.spaces import Discrete, MultiDiscrete

try:
    from pettingzoo import AECEnv
except Exception:  # pragma: no cover - exercised when optional dep is absent
    class AECEnv:  # type: ignore[no-redef]
        @property
        def unwrapped(self):
            return self


class AdvanceWarsAECEnv(AECEnv):
    metadata = {"name": "advancewars_v0", "render_modes": [None, "ansi"]}

    def __init__(
        self,
        map_name: str = "duel",
        ruleset: str = "defendpeace_awbw",
        max_turns: int = 100,
        render_mode: str | None = None,
        observation_mode: str = "planes",
        reward_mode: str = "win_loss",
        fog: bool = False,
        weather: str = "clear",
        commanders=None,
        tag_mode: bool = False,
        luck: bool = False,
        seed: int | None = None,
        config=None,
        enabled_units=None,
        strict_units: bool | None = None,
        action_mode: str = "flat",
    ):
        if observation_mode != "planes":
            raise ValueError("Only observation_mode='planes' is implemented.")
        if action_mode not in {"flat", "structured"}:
            raise ValueError("action_mode must be 'flat' or 'structured'.")
        self.game = Game.from_map(
            map_name,
            ruleset=ruleset,
            max_turns=max_turns,
            fog=fog,
            weather=weather,
            commanders=commanders,
            tag_mode=tag_mode,
            luck=luck,
            seed=seed,
            config=config,
            enabled_units=enabled_units,
            strict_units=strict_units,
        )
        self.render_mode = render_mode
        self.observation_mode = observation_mode
        self.reward_mode = reward_mode
        self.action_mode = action_mode
        self.codec = ActionCodec()
        self.structured_codec: StructuredActionCodec | None = None
        parsed = parse_defendpeace_map(self.game.map_text)
        self.possible_agents: list[str] = [
            f"player_{pid}" for pid in sorted(parsed.player_ids)
        ]
        self.agents: list[str] = []
        self.agent_selection: str | None = None
        self.rewards: dict[str, float] = {}
        self._cumulative_rewards: dict[str, float] = {}
        self.terminations: dict[str, bool] = {}
        self.truncations: dict[str, bool] = {}
        self.infos: dict[str, dict[str, Any]] = {}
        self._action_maps: dict[str, dict[Any, Action]] = {}
        self._action_spaces: dict[str, Any] = {}
        self._observation_spaces: dict[str, Any] = {}

    @property
    def num_agents(self) -> int:
        return len(self.agents)

    @property
    def max_num_agents(self) -> int:
        return len(self.possible_agents)

    def reset(self, seed: int | None = None, options: dict | None = None):
        del options
        state = self.game.reset(seed=seed)
        self.possible_agents = [f"player_{pid}" for pid in sorted(state.players)]
        self.agents = list(self.possible_agents)
        self.agent_selection = f"player_{state.current_player}"
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = dict(self.rewards)
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._init_spaces()
        self._refresh_action_maps()

    def agent_iter(self, max_iter: int | None = None):
        count = 0
        while self.agents and (max_iter is None or count < max_iter):
            yield self.agent_selection
            count += 1

    def last(self, observe: bool = True):
        agent = self._require_agent()
        obs = self.observe(agent) if observe else None
        return (
            obs,
            self.rewards.get(agent, 0.0),
            self.terminations.get(agent, False),
            self.truncations.get(agent, False),
            self.infos.get(agent, {}),
        )

    def step(self, action: int | None):
        agent = self._require_agent()
        if self.terminations.get(agent) or self.truncations.get(agent):
            return
        if action is None:
            raise ValueError("Action cannot be None for an active agent.")
        action_map = self._action_maps.get(agent, {})
        action_key = self._normalize_env_action(action)
        if action_key not in action_map:
            raise ValueError(f"Illegal encoded action {action} for {agent}.")
        engine_action = action_map[action_key]
        if self.action_mode == "structured":
            _state, events = self.game.step_semantic(engine_action)
        else:
            _state, events = self.game.step(engine_action)
        state = self.game.state
        assert state is not None
        self.rewards = rewards_for(state, self.reward_mode, events)
        done = state.done
        truncated = state.truncated
        self.agents = [] if done or truncated else list(self.possible_agents)
        status_agents = self.possible_agents if done or truncated else self.agents
        self.terminations = {a: done for a in status_agents}
        self.truncations = {a: truncated for a in status_agents}
        self.rewards = {a: self.rewards.get(a, 0.0) for a in status_agents}
        self._cumulative_rewards = dict(self.rewards)
        self.infos = {a: self.infos.get(a, {}) for a in status_agents}
        self.agent_selection = (
            None if not self.agents else f"player_{state.current_player}"
        )
        self._refresh_action_maps()

    def observe(self, agent: str):
        state = self.game.state
        if state is None:
            raise RuntimeError("Environment has not been reset.")
        player_id = self._agent_id(agent)
        visible_coords = self.game.visible_coords(player_id)
        visible_unit_ids = {unit.id for unit in self.game.observed_units(player_id)}
        observation = {
            "observation": board_planes(
                state,
                player_id,
                visible_coords,
                visible_unit_ids,
            ),
        }
        if self.action_mode == "flat":
            mask, _ = self.codec.encode_legal(self.game.legal_actions(player_id))
            observation["action_mask"] = mask
        else:
            masks, _mapping, _semantic_actions = self._structured_legal(player_id)
            observation.update(masks)
        return observation

    def render(self):
        state = self.game.state
        if state is None:
            return ""
        rows = []
        for y, row in enumerate(state.map.tiles):
            cells = []
            for x, tile in enumerate(row):
                unit = state.unit_at(Coord(x, y))
                if unit is not None:
                    cells.append(f"{unit.owner}{unit.unit_type[0].upper()}")
                else:
                    owner = "." if tile.owner is None else str(tile.owner)
                    cells.append(f"{owner}{tile.definition.code}")
            rows.append(" ".join(cells))
        output = "\n".join(rows)
        if self.render_mode == "ansi":
            return output
        print(output)
        return None

    def close(self):
        return None

    def action_space(self, agent: str):
        return self._action_spaces[agent]

    def observation_space(self, agent: str):
        return self._observation_spaces[agent]

    def _init_spaces(self) -> None:
        if Box is None or Dict is None or MultiBinary is None:
            self._action_spaces = {
                agent: self._make_action_space() for agent in self.agents
            }
            self._observation_spaces = {agent: None for agent in self.agents}
            return
        state = self.game.state
        assert state is not None
        player_id = self._agent_id(self.agents[0])
        sample = board_planes(
            state,
            player_id,
            self.game.visible_coords(player_id),
            {unit.id for unit in self.game.observed_units(player_id)},
        )
        obs_items = {
            "observation": Box(
                low=-1.0,
                high=32.0,
                shape=sample.shape,
                dtype=sample.dtype,
            ),
        }
        if self.action_mode == "flat":
            obs_items["action_mask"] = MultiBinary(self.codec.max_actions)
        else:
            structured_codec = self._ensure_structured_codec()
            obs_items.update(
                {
                    "action_type_mask": MultiBinary(
                        structured_codec.empty_masks()["action_type_mask"].shape
                    ),
                    "source_mask": MultiBinary(
                        structured_codec.empty_masks()["source_mask"].shape
                    ),
                    "destination_mask": MultiBinary(
                        structured_codec.empty_masks()["destination_mask"].shape
                    ),
                    "target_mask": MultiBinary(
                        structured_codec.empty_masks()["target_mask"].shape
                    ),
                    "payload_mask": MultiBinary(
                        structured_codec.empty_masks()["payload_mask"].shape
                    ),
                }
            )
        obs_space = Dict(obs_items)
        action_space = self._make_action_space()
        self._action_spaces = {agent: action_space for agent in self.agents}
        self._observation_spaces = {agent: obs_space for agent in self.agents}

    def _refresh_action_maps(self) -> None:
        state = self.game.state
        if state is None:
            return
        self._action_maps = {}
        if not self.agents:
            return
        for player_id in sorted(state.players):
            agent = f"player_{player_id}"
            if agent not in self.agents:
                continue
            if self.action_mode == "flat":
                legal_actions = self.game.legal_actions(player_id)
                mask, mapping = self.codec.encode_legal(legal_actions)
                self._action_maps[agent] = mapping
                self.infos[agent] = {
                    "action_mask": mask,
                    "legal_action_indices": list(mapping),
                    "legal_actions": list(mapping.values()),
                    "current_player": state.current_player,
                    "turn": state.turn,
                }
            else:
                legal_actions = self.game.legal_semantic_actions(player_id)
                masks, mapping, semantic_actions = self._structured_legal_from_actions(
                    legal_actions,
                    state,
                )
                self._action_maps[agent] = mapping
                self.infos[agent] = {
                    **masks,
                    "action_masks": masks,
                    "legal_structured_actions": list(mapping),
                    "legal_semantic_actions": [
                        semantic_to_dict(action) for action in semantic_actions
                    ],
                    "legal_actions": list(mapping.values()),
                    "current_player": state.current_player,
                    "turn": state.turn,
                }

    def _make_action_space(self):
        if self.action_mode == "flat":
            return Discrete(self.codec.max_actions)
        return MultiDiscrete(self._ensure_structured_codec().nvec)

    def _normalize_env_action(self, action):
        if self.action_mode == "flat":
            return int(action)
        return self._ensure_structured_codec().normalize(action)

    def _structured_legal(
        self,
        player_id: int,
    ):
        state = self.game.state
        if state is None:
            raise RuntimeError("Environment has not been reset.")
        return self._structured_legal_from_actions(
            self.game.legal_semantic_actions(player_id),
            state,
        )

    def _structured_legal_from_actions(
        self,
        legal_actions: list[Action],
        state,
    ):
        del state
        return self._ensure_structured_codec().encode_legal_semantic(legal_actions)

    def _ensure_structured_codec(self) -> StructuredActionCodec:
        if self.structured_codec is None:
            state = self.game.state
            if state is None:
                raise RuntimeError("Environment has not been reset.")
            self.structured_codec = StructuredActionCodec(
                width=state.map.width,
                height=state.map.height,
            )
        return self.structured_codec

    def _require_agent(self) -> str:
        if self.agent_selection is None:
            raise RuntimeError("No active agent. Did you reset, or is the game done?")
        return self.agent_selection

    @staticmethod
    def _agent_id(agent: str) -> int:
        return int(agent.split("_", 1)[1])
