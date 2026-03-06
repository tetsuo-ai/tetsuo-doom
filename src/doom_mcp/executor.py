"""Autonomous executor for real-time Doom gameplay.

Runs a background thread at game speed (~35 Hz) that handles combat,
navigation, and item collection autonomously. The LLM acts as a
high-level director, setting objectives and strategy via MCP tools.
"""

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import vizdoom as vzd

from .objects import get_object_info
from .state import extract_objects


class ExecutorState(Enum):
    IDLE = "idle"
    EXPLORING = "exploring"
    FIGHTING = "fighting"
    COLLECTING = "collecting"
    RETREATING = "retreating"
    MOVING_TO = "moving_to"


class ObjectiveType(Enum):
    EXPLORE = "explore"
    KILL = "kill"
    MOVE_TO_POS = "move_to_pos"
    MOVE_TO_OBJ = "move_to_obj"
    COLLECT = "collect"
    USE_OBJECT = "use_object"
    RETREAT = "retreat"
    HOLD_POSITION = "hold_position"


@dataclass
class Objective:
    type: ObjectiveType
    params: dict = field(default_factory=dict)
    priority: int = 0
    timeout_tics: int = 0
    tics_active: int = 0


@dataclass
class Strategy:
    aggression: float = 0.5
    health_retreat_threshold: int = 20
    health_collect_threshold: int = 50
    ammo_switch_threshold: int = 5
    engage_range: float = 1500.0
    collect_range: float = 800.0
    prefer_cover: bool = False


@dataclass
class ExecutorEvent:
    tic: int
    event_type: str
    detail: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class _TickSnapshot:
    px: float
    py: float
    pz: float
    pa: float
    health: float
    armor: float
    ammo: float
    weapon: float
    attack_ready: bool
    dead: bool
    tic: int
    threats: list[dict]
    items: list[dict]
    depth: np.ndarray | None
    objects: list[dict]


# Threat classification constants
_THREAT_WEIGHTS = {"none": 0, "low": 1, "medium": 2, "high": 3}
_ATTACK_URGENCY = {"hitscan": 3, "projectile": 2, "melee": 1, "none": 0}

# Navigation constants
_WALL_CLOSE = 15.0
_WALL_NEAR = 40.0
_STUCK_WINDOW = 20
_STUCK_THRESHOLD = 15.0
_ARRIVE_DISTANCE = 64.0

# Stuck recovery phases
_STUCK_PHASE_TICS = 5
_STUCK_PHASES = 4

# Combat constants
_AIM_TOLERANCE = 5.0
_MAX_TURN_SPEED = 45.0
_STRAFE_INTERVAL = 20
_MELEE_BACKPEDAL_RANGE = 200.0

# Error limits
_MAX_CONSECUTIVE_ERRORS = 10


class AutonomousExecutor:
    """Background thread that plays Doom autonomously at game speed."""

    def __init__(
        self,
        game: vzd.DoomGame,
        buttons: list[vzd.Button],
        variable_names: list[str],
        nav_memory: Any,
        game_lock: threading.Lock,
    ) -> None:
        self._game = game
        self._buttons = buttons
        self._variable_names = variable_names
        self._nav_memory = nav_memory
        self._game_lock = game_lock

        # Director state (protected by _director_lock)
        self._director_lock = threading.Lock()
        self._objectives: list[Objective] = []
        self._strategy = Strategy()
        self._events: deque[ExecutorEvent] = deque(maxlen=100)
        self._event_cursor = 0

        # Executor internal state
        self._state = ExecutorState.IDLE
        self._prev_health: float = 100.0
        self._prev_ammo: float = 0.0
        self._prev_killcount: float = 0.0
        self._position_history: list[tuple[float, float]] = []
        self._stuck_phase = 0
        self._stuck_tics = 0
        self._strafe_tic_counter = 0
        self._strafe_sign = 1.0
        self._current_target_id: int | None = None
        self._turn_bias = 0.0

        # Thread control
        self._paused = threading.Event()
        self._paused.set()  # set = NOT paused
        self._pause_ack = threading.Event()
        self._pause_ack.set()
        self._stop_flag = threading.Event()
        self._thread: threading.Thread | None = None
        self._consecutive_errors = 0

    @property
    def state(self) -> ExecutorState:
        return self._state

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_flag.clear()
        self._paused.set()
        self._thread = threading.Thread(
            target=self._run_loop, name="doom-executor", daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_flag.set()
        self._paused.set()  # unblock if paused
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def pause(self) -> None:
        self._pause_ack.clear()
        self._paused.clear()
        self._pause_ack.wait(timeout=2.0)

    def resume(self) -> None:
        self._paused.set()

    def reset(self) -> None:
        with self._director_lock:
            self._objectives.clear()
        self._state = ExecutorState.IDLE
        self._position_history.clear()
        self._stuck_phase = 0
        self._stuck_tics = 0
        self._strafe_tic_counter = 0
        self._current_target_id = None
        self._turn_bias = 0.0
        self._consecutive_errors = 0
        self._prev_health = 100.0
        self._prev_ammo = 0.0
        self._prev_killcount = 0.0

    # ------------------------------------------------------------------
    # Objective management
    # ------------------------------------------------------------------

    def push_objective(self, objective: Objective) -> None:
        with self._director_lock:
            self._objectives.append(objective)
            self._objectives.sort(key=lambda o: o.priority, reverse=True)

    def clear_objectives(self) -> None:
        with self._director_lock:
            self._objectives.clear()

    def get_objectives(self) -> list[dict]:
        with self._director_lock:
            return [
                {
                    "type": o.type.value,
                    "params": o.params,
                    "priority": o.priority,
                    "timeout_tics": o.timeout_tics,
                    "tics_active": o.tics_active,
                }
                for o in self._objectives
            ]

    def _get_current_objective(self) -> Objective | None:
        with self._director_lock:
            return self._objectives[0] if self._objectives else None

    # ------------------------------------------------------------------
    # Strategy
    # ------------------------------------------------------------------

    def set_strategy(self, **kwargs: Any) -> None:
        with self._director_lock:
            for key, value in kwargs.items():
                if hasattr(self._strategy, key):
                    setattr(self._strategy, key, value)

    def get_strategy(self) -> dict:
        with self._director_lock:
            s = self._strategy
            return {
                "aggression": s.aggression,
                "health_retreat_threshold": s.health_retreat_threshold,
                "health_collect_threshold": s.health_collect_threshold,
                "ammo_switch_threshold": s.ammo_switch_threshold,
                "engage_range": s.engage_range,
                "collect_range": s.collect_range,
                "prefer_cover": s.prefer_cover,
            }

    # ------------------------------------------------------------------
    # Event log
    # ------------------------------------------------------------------

    def _log_event(self, tic: int, event_type: str, detail: str) -> None:
        with self._director_lock:
            self._events.append(ExecutorEvent(tic, event_type, detail))

    def get_recent_events(self) -> list[dict]:
        with self._director_lock:
            snapshot = list(self._events)
        result = [
            {
                "tic": e.tic,
                "event_type": e.event_type,
                "detail": e.detail,
                "timestamp": e.timestamp,
            }
            for e in snapshot
            if id(e) not in (0,)  # always true, just iterate
        ]
        # Cursor-based: return events since last call
        new_events = result[self._event_cursor:]
        self._event_cursor = len(result)
        return new_events

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        while not self._stop_flag.is_set():
            # Check pause
            if not self._paused.is_set():
                self._pause_ack.set()
                self._paused.wait()
                if self._stop_flag.is_set():
                    return
                continue

            try:
                self._tick()
                self._consecutive_errors = 0
            except Exception as e:
                self._consecutive_errors += 1
                self._log_event(0, "error", str(e))
                if self._consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                    self._log_event(0, "error", "Too many consecutive errors, executor stopping")
                    self._state = ExecutorState.IDLE
                    return
                time.sleep(0.1)

    # ------------------------------------------------------------------
    # Tick: one game tic
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        # Phase 1: Read state under lock
        snap = self._read_state()
        if snap is None:
            # Episode finished or dead
            if self._state != ExecutorState.IDLE:
                self._state = ExecutorState.IDLE
            time.sleep(0.01)
            return

        # Phase 2: Decide (no lock held)
        strategy = self._get_strategy_copy()
        self._update_state(snap, strategy)
        action = self._compute_action(snap, strategy)

        # Check if paused between Phase 2 and Phase 3
        if not self._paused.is_set():
            return

        # Phase 3: Apply action + update nav memory under lock
        with self._game_lock:
            if self._game.is_episode_finished():
                return
            self._game.make_action(action, 1)
            self._nav_memory.update(snap.px, snap.py, snap.pa)

        # Post: Track changes and log events
        self._post_tick(snap)

        # Idle sleep to avoid busy-spin
        if self._state == ExecutorState.IDLE:
            time.sleep(0.01)

    def _read_state(self) -> _TickSnapshot | None:
        with self._game_lock:
            if self._game.is_episode_finished():
                return None

            state = self._game.get_state()
            if state is None:
                return None

            px = self._game.get_game_variable(vzd.GameVariable.POSITION_X)
            py = self._game.get_game_variable(vzd.GameVariable.POSITION_Y)
            pz = self._game.get_game_variable(vzd.GameVariable.POSITION_Z)
            pa = self._game.get_game_variable(vzd.GameVariable.ANGLE)
            health = self._game.get_game_variable(vzd.GameVariable.HEALTH)
            armor = self._game.get_game_variable(vzd.GameVariable.ARMOR)
            ammo = self._game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO)
            weapon = self._game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON)
            attack_ready = bool(self._game.get_game_variable(vzd.GameVariable.ATTACK_READY))
            dead = bool(self._game.get_game_variable(vzd.GameVariable.DEAD))

            if dead:
                return None

            objects = extract_objects(state, player_x=px, player_y=py, player_angle=pa)
            depth = state.depth_buffer if state.depth_buffer is not None else None
            tic = state.tic

        threats = self._classify_threats(objects)
        items = self._classify_items(objects, px, py)

        return _TickSnapshot(
            px=px, py=py, pz=pz, pa=pa,
            health=health, armor=armor, ammo=ammo, weapon=weapon,
            attack_ready=attack_ready, dead=dead, tic=tic,
            threats=threats, items=items, depth=depth, objects=objects,
        )

    def _get_strategy_copy(self) -> Strategy:
        with self._director_lock:
            s = self._strategy
            return Strategy(
                aggression=s.aggression,
                health_retreat_threshold=s.health_retreat_threshold,
                health_collect_threshold=s.health_collect_threshold,
                ammo_switch_threshold=s.ammo_switch_threshold,
                engage_range=s.engage_range,
                collect_range=s.collect_range,
                prefer_cover=s.prefer_cover,
            )

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def _update_state(self, snap: _TickSnapshot, strategy: Strategy) -> None:
        old_state = self._state

        # Health critical -> RETREATING (always)
        if snap.health <= strategy.health_retreat_threshold:
            self._state = ExecutorState.RETREATING
        # Nearby visible threats + aggression check
        elif snap.threats and snap.threats[0]["distance"] <= strategy.engage_range:
            # Always fight if aggression >= threshold or threat is very close
            closest_dist = snap.threats[0]["distance"]
            if strategy.aggression >= 0.3 or closest_dist < 300:
                self._state = ExecutorState.FIGHTING
            else:
                self._state = ExecutorState.RETREATING
        else:
            # Check objective
            objective = self._get_current_objective()
            if objective is not None:
                # Tick the objective
                with self._director_lock:
                    if self._objectives:
                        self._objectives[0].tics_active += 1
                        # Timeout check
                        if objective.timeout_tics > 0 and self._objectives[0].tics_active >= objective.timeout_tics:
                            expired = self._objectives.pop(0)
                            self._log_event(snap.tic, "objective_failed", f"Timeout: {expired.type.value}")
                            objective = self._objectives[0] if self._objectives else None

                if objective is not None:
                    state_map = {
                        ObjectiveType.EXPLORE: ExecutorState.EXPLORING,
                        ObjectiveType.KILL: ExecutorState.FIGHTING,
                        ObjectiveType.MOVE_TO_POS: ExecutorState.MOVING_TO,
                        ObjectiveType.MOVE_TO_OBJ: ExecutorState.MOVING_TO,
                        ObjectiveType.COLLECT: ExecutorState.COLLECTING,
                        ObjectiveType.USE_OBJECT: ExecutorState.MOVING_TO,
                        ObjectiveType.RETREAT: ExecutorState.RETREATING,
                        ObjectiveType.HOLD_POSITION: ExecutorState.IDLE,
                    }
                    self._state = state_map.get(objective.type, ExecutorState.EXPLORING)
                else:
                    self._state = self._default_state(snap, strategy)
            else:
                self._state = self._default_state(snap, strategy)

        if self._state != old_state:
            self._log_event(snap.tic, "state_change", f"{old_state.value} -> {self._state.value}")

    def _default_state(self, snap: _TickSnapshot, strategy: Strategy) -> ExecutorState:
        # Health low + health items nearby -> COLLECTING
        if snap.health <= strategy.health_collect_threshold:
            health_items = [i for i in snap.items if i["category"] == "health"]
            if health_items:
                return ExecutorState.COLLECTING

        # Ammo low + ammo nearby -> COLLECTING
        if snap.ammo <= strategy.ammo_switch_threshold:
            ammo_items = [i for i in snap.items if i["category"] == "ammo"]
            if ammo_items:
                return ExecutorState.COLLECTING

        return ExecutorState.EXPLORING

    # ------------------------------------------------------------------
    # Action computation
    # ------------------------------------------------------------------

    def _compute_action(self, snap: _TickSnapshot, strategy: Strategy) -> list[float]:
        # Stuck detection first
        self._position_history.append((snap.px, snap.py))
        if len(self._position_history) > _STUCK_WINDOW:
            self._position_history.pop(0)

        if self._is_stuck() and self._state not in (ExecutorState.IDLE, ExecutorState.FIGHTING):
            return self._stuck_recovery_action(snap)

        if self._state == ExecutorState.FIGHTING:
            return self._fight_action(snap, strategy)
        elif self._state == ExecutorState.EXPLORING:
            return self._explore_action(snap)
        elif self._state == ExecutorState.COLLECTING:
            return self._collect_action(snap, strategy)
        elif self._state == ExecutorState.RETREATING:
            return self._retreat_action(snap)
        elif self._state == ExecutorState.MOVING_TO:
            return self._move_to_action(snap)
        elif self._state == ExecutorState.IDLE:
            return self._idle_action(snap)
        return self._build_action([])

    def _is_stuck(self) -> bool:
        if len(self._position_history) < _STUCK_WINDOW:
            return False
        xs = [p[0] for p in self._position_history]
        ys = [p[1] for p in self._position_history]
        spread = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        return spread < _STUCK_THRESHOLD

    # ------------------------------------------------------------------
    # Combat
    # ------------------------------------------------------------------

    def _fight_action(self, snap: _TickSnapshot, strategy: Strategy) -> list[float]:
        if not snap.threats:
            return self._explore_action(snap)

        self._strafe_tic_counter += 1

        # Pick target: current target if still valid, otherwise highest priority
        target = None
        if self._current_target_id is not None:
            for t in snap.threats:
                if t["id"] == self._current_target_id:
                    target = t
                    break
        if target is None:
            target = snap.threats[0]
            self._current_target_id = target["id"]

        angle = target["angle_to_aim"]
        dist = target["distance"]
        actions: dict[str, float] = {}

        # Aim
        clamped = max(-_MAX_TURN_SPEED, min(_MAX_TURN_SPEED, angle))
        actions["TURN_LEFT_RIGHT_DELTA"] = clamped

        # Fire when aligned and ready
        if abs(angle) <= _AIM_TOLERANCE and snap.attack_ready:
            actions["ATTACK"] = 1

        # Dodge strafe (alternate every ~20 tics)
        if self._strafe_tic_counter % _STRAFE_INTERVAL == 0:
            self._strafe_sign *= -1
        actions["MOVE_LEFT_RIGHT_DELTA"] = self._strafe_sign * 15.0

        # Backpedal from melee enemies
        if target.get("attack_type") == "melee" and dist < _MELEE_BACKPEDAL_RANGE:
            actions["MOVE_FORWARD_BACKWARD_DELTA"] = -15

        # Weapon switch when out of ammo
        if snap.ammo <= 0:
            actions["SELECT_NEXT_WEAPON"] = 1
            actions.pop("ATTACK", None)
            self._log_event(snap.tic, "weapon_switch", "Out of ammo, switching weapon")

        return self._build_action(actions)

    # ------------------------------------------------------------------
    # Exploration
    # ------------------------------------------------------------------

    def _explore_action(self, snap: _TickSnapshot) -> list[float]:
        actions: dict[str, float] = {}

        if snap.depth is not None:
            h, w = snap.depth.shape
            band_h = max(h // 6, 1)
            mid = h // 2
            band = snap.depth[mid - band_h:mid + band_h, :]
            third = w // 3
            left_score = float(band[:, :third].mean())
            center_score = float(band[:, third:2*third].mean())
            right_score = float(band[:, 2*third:].mean())

            if center_score < _WALL_CLOSE:
                if abs(left_score - right_score) > 3:
                    self._turn_bias = -25.0 if left_score > right_score else 25.0
                elif self._turn_bias == 0.0:
                    self._turn_bias = -25.0 if left_score >= right_score else 25.0
                actions["TURN_LEFT_RIGHT_DELTA"] = self._turn_bias
            elif center_score < _WALL_NEAR:
                if abs(left_score - right_score) > 3:
                    self._turn_bias = -10.0 if left_score > right_score else 10.0
                elif self._turn_bias == 0.0:
                    self._turn_bias = -10.0 if left_score >= right_score else 10.0
                actions["TURN_LEFT_RIGHT_DELTA"] = self._turn_bias
                actions["MOVE_FORWARD_BACKWARD_DELTA"] = 15
            else:
                self._turn_bias *= 0.5
                if abs(self._turn_bias) < 1.0:
                    self._turn_bias = 0.0
                actions["TURN_LEFT_RIGHT_DELTA"] = self._turn_bias
                actions["MOVE_FORWARD_BACKWARD_DELTA"] = 25
        else:
            actions["MOVE_FORWARD_BACKWARD_DELTA"] = 25

        return self._build_action(actions)

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------

    def _collect_action(self, snap: _TickSnapshot, strategy: Strategy) -> list[float]:
        if not snap.items:
            return self._explore_action(snap)

        # Prioritize health if low, otherwise nearest
        target_item = None
        if snap.health <= strategy.health_collect_threshold:
            health_items = [i for i in snap.items if i["category"] == "health"]
            if health_items:
                target_item = min(health_items, key=lambda i: i["distance"])
        if target_item is None:
            target_item = min(snap.items, key=lambda i: i["distance"])

        angle = target_item["angle_to_aim"]
        clamped = max(-_MAX_TURN_SPEED, min(_MAX_TURN_SPEED, angle))

        actions: dict[str, float] = {"TURN_LEFT_RIGHT_DELTA": clamped}
        if abs(angle) < 30:
            actions["MOVE_FORWARD_BACKWARD_DELTA"] = 25
            actions["SPEED"] = 1

        return self._build_action(actions)

    # ------------------------------------------------------------------
    # Retreat
    # ------------------------------------------------------------------

    def _retreat_action(self, snap: _TickSnapshot) -> list[float]:
        actions: dict[str, float] = {}

        if snap.threats:
            # Turn away from nearest threat
            nearest = snap.threats[0]
            away_angle = nearest["angle_to_aim"]
            # Turn 180 from threat direction
            if away_angle >= 0:
                turn = away_angle - 180
            else:
                turn = away_angle + 180
            clamped = max(-_MAX_TURN_SPEED, min(_MAX_TURN_SPEED, turn))
            actions["TURN_LEFT_RIGHT_DELTA"] = clamped
            actions["MOVE_FORWARD_BACKWARD_DELTA"] = 25
            actions["SPEED"] = 1
        else:
            # No threats, backpedal
            actions["MOVE_FORWARD_BACKWARD_DELTA"] = -20

        return self._build_action(actions)

    # ------------------------------------------------------------------
    # Move to target
    # ------------------------------------------------------------------

    def _move_to_action(self, snap: _TickSnapshot) -> list[float]:
        objective = self._get_current_objective()
        if objective is None:
            return self._explore_action(snap)

        target_x: float | None = None
        target_y: float | None = None

        if objective.type == ObjectiveType.MOVE_TO_POS:
            target_x = objective.params.get("x")
            target_y = objective.params.get("y")
        elif objective.type in (ObjectiveType.MOVE_TO_OBJ, ObjectiveType.USE_OBJECT):
            obj_id = objective.params.get("object_id")
            if obj_id is not None:
                for obj in snap.objects:
                    if obj["id"] == obj_id:
                        target_x = obj["position_x"]
                        target_y = obj["position_y"]
                        # Check arrival
                        if obj["distance"] < _ARRIVE_DISTANCE:
                            with self._director_lock:
                                if self._objectives:
                                    completed = self._objectives.pop(0)
                                    self._log_event(snap.tic, "objective_complete", f"{completed.type.value}")
                            return self._build_action([])
                        break

        if target_x is None or target_y is None:
            return self._explore_action(snap)

        # Compute angle to target
        dx = target_x - snap.px
        dy = target_y - snap.py
        target_angle = math.degrees(math.atan2(dy, dx))
        diff = (target_angle - snap.pa + 180) % 360 - 180
        angle = -diff  # match TURN_LEFT_RIGHT_DELTA convention

        dist = math.hypot(dx, dy)
        if dist < _ARRIVE_DISTANCE:
            with self._director_lock:
                if self._objectives:
                    completed = self._objectives.pop(0)
                    self._log_event(snap.tic, "objective_complete", f"{completed.type.value}")
            return self._build_action([])

        clamped = max(-_MAX_TURN_SPEED, min(_MAX_TURN_SPEED, angle))
        actions: dict[str, float] = {"TURN_LEFT_RIGHT_DELTA": clamped}
        if abs(angle) < 30:
            actions["MOVE_FORWARD_BACKWARD_DELTA"] = 25

        return self._build_action(actions)

    # ------------------------------------------------------------------
    # Idle
    # ------------------------------------------------------------------

    def _idle_action(self, snap: _TickSnapshot) -> list[float]:
        # Scan for threats, fight if any appear
        if snap.threats:
            self._state = ExecutorState.FIGHTING
            return self._fight_action(snap, self._get_strategy_copy())
        return self._build_action([])

    # ------------------------------------------------------------------
    # Stuck recovery
    # ------------------------------------------------------------------

    def _stuck_recovery_action(self, snap: _TickSnapshot) -> list[float]:
        self._stuck_tics += 1
        actions: dict[str, float] = {}

        # 4-phase cycle: turn right, turn left, 180, strafe+turn
        phase = self._stuck_phase % _STUCK_PHASES
        if phase == 0:
            actions["TURN_LEFT_RIGHT_DELTA"] = 30
        elif phase == 1:
            actions["TURN_LEFT_RIGHT_DELTA"] = -30
        elif phase == 2:
            actions["TURN_LEFT_RIGHT_DELTA"] = 45
            actions["MOVE_FORWARD_BACKWARD_DELTA"] = 10
        elif phase == 3:
            actions["MOVE_LEFT_RIGHT_DELTA"] = 20
            actions["TURN_LEFT_RIGHT_DELTA"] = 15

        if self._stuck_tics >= _STUCK_PHASE_TICS:
            self._stuck_phase += 1
            self._stuck_tics = 0
            self._position_history.clear()
            self._log_event(snap.tic, "stuck", f"Recovery phase {phase}")

        return self._build_action(actions)

    # ------------------------------------------------------------------
    # Classification helpers
    # ------------------------------------------------------------------

    def _classify_threats(self, objects: list[dict]) -> list[dict]:
        threats = []
        for obj in objects:
            info = get_object_info(obj["name"])
            if info["type"] != "monster":
                continue
            if not obj.get("is_visible", False):
                continue

            dist = max(obj["distance"], 1.0)
            threat_w = _THREAT_WEIGHTS.get(info["threat"], 0)
            attack_u = _ATTACK_URGENCY.get(info["attack"], 0)
            proximity = 1000.0 / dist
            score = threat_w * 10 + attack_u * 5 + proximity + 5.0

            if obj["name"] == "Archvile":
                score += 100

            threats.append({
                "id": obj["id"],
                "name": obj["name"],
                "distance": round(dist, 1),
                "angle_to_aim": round(obj["angle_to_aim"], 1),
                "attack_type": info["attack"],
                "priority_score": round(score, 1),
            })

        threats.sort(key=lambda t: t["priority_score"], reverse=True)
        return threats

    def _classify_items(self, objects: list[dict], px: float, py: float) -> list[dict]:
        items = []
        for obj in objects:
            info = get_object_info(obj["name"])
            if info["type"] not in ("item", "ammo", "weapon"):
                continue
            if not obj.get("is_visible", False):
                continue

            dist = obj["distance"]

            # Categorize
            name = obj["name"].lower()
            if any(h in name for h in ("health", "medikit", "stimpack", "soulsphere", "megasphere", "berserk")):
                category = "health"
            elif info["type"] == "ammo" or "ammo" in name:
                category = "ammo"
            elif info["type"] == "weapon":
                category = "weapon"
            else:
                category = "other"

            items.append({
                "id": obj["id"],
                "name": obj["name"],
                "distance": round(dist, 1),
                "angle_to_aim": round(obj["angle_to_aim"], 1),
                "category": category,
            })

        return items

    # ------------------------------------------------------------------
    # Post-tick tracking
    # ------------------------------------------------------------------

    def _post_tick(self, snap: _TickSnapshot) -> None:
        # Track health changes
        if snap.health < self._prev_health:
            damage = self._prev_health - snap.health
            self._log_event(snap.tic, "damage_taken", f"{damage:.0f} damage")
        self._prev_health = snap.health
        self._prev_ammo = snap.ammo

        # Reset stuck if we moved
        if not self._is_stuck():
            self._stuck_phase = 0
            self._stuck_tics = 0

    # ------------------------------------------------------------------
    # Action building
    # ------------------------------------------------------------------

    def _build_action(self, actions: dict[str, float] | list) -> list[float]:
        action_list = [0.0] * len(self._buttons)
        if isinstance(actions, list):
            return action_list
        button_index = {b.name: i for i, b in enumerate(self._buttons)}
        for name, value in actions.items():
            upper = name.upper()
            if upper in button_index:
                action_list[button_index[upper]] = float(value)
        return action_list
