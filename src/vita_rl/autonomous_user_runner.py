"""Non-interactive VitaBench control flow for the deterministic-user ablation.

The stock VitaBench orchestrator alternates every non-tool assistant message
through a user simulator.  That is correct for ordinary interactive users but
cannot represent an oracle-intent, user-speaks-once treatment without injecting
fake user content.  This module dispatches only the registered deterministic
user to an autonomous agent/environment loop; all other user implementations
continue through upstream VitaBench unchanged.
"""

from __future__ import annotations

import asyncio
import inspect
from copy import deepcopy
from typing import Any, Callable, Optional

from loguru import logger

from vita.agent.base import is_valid_agent_history_message
from vita.data_model.message import AssistantMessage, Message, MultiToolMessage, ToolMessage, UserMessage
from vita.data_model.simulation import SimulationRun, TerminationReason
from vita.orchestrator.orchestrator import Orchestrator, Role
from vita.user.base import is_valid_user_history_message
from vita.utils.utils import get_now

from vita_rl.deterministic_user import DETERMINISTIC_USER_NAME, DeterministicTaskUser


AUTONOMOUS_EXECUTION_DIRECTIVE = (
    "This is a non-interactive execution session. The initial user message contains "
    "the complete request and, when supplied, the user's public profile. No further "
    "user responses will arrive. Use available tools to complete the request. Treat "
    "the profile as user-provided context only when the request leaves a detail "
    "unspecified; do not invent missing facts. Do not rely on default fields: explicitly "
    "resolve and set every destination, product attribute, quantity, timing, and order "
    "status needed by the request. A personal request after work is for home unless it "
    "explicitly says work or another destination. When a request refers to preferences, "
    "frequent/recent orders, or availability, use tools to inspect the relevant public "
    "task-visible information before acting. Do not add constraints that the request or "
    "profile does not support. After attempted completion, you will receive two "
    "fact-free requests to inspect and review your work. Before the final response, "
    "re-check that every requirement in the request and applicable profile context has "
    "been carried out. When no further tool action is needed, end your final response "
    "with ###STOP###."
)


def _await(value: Any) -> Any:
    """Resolve the upstream async user contract from synchronous VitaBench code."""
    if not inspect.isawaitable(value):
        return value
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(value)
    raise RuntimeError("The deterministic VitaBench runner cannot run inside an active event loop")


class AutonomousDeterministicOrchestrator(Orchestrator):
    """One user disclosure followed only by agent/environment transitions."""

    def _append_autonomous_directive(self) -> None:
        system_messages = getattr(self.agent_state, "system_messages", None)
        if not system_messages:
            raise TypeError("The deterministic treatment requires an LLM-style agent state")
        # SGLang's OpenAI endpoint permits one leading system message, whereas
        # multiple system messages are rejected even when contiguous. Preserve
        # the upstream policy verbatim and extend that same leading message.
        system_messages[0].content = "\n\n".join(
            filter(None, [system_messages[0].content, AUTONOMOUS_EXECUTION_DIRECTIVE])
        )

    def initialize(self):
        message_history = (
            deepcopy(self.task.message_history)
            if self.task is not None and self.task.message_history is not None
            else []
        )
        for message in message_history:
            message.turn_idx = None
        message_history = self._add_timestamps(message_history)

        if self.seed is not None:
            self.agent.set_seed(self.seed)
            self.user.set_seed(self.seed)

        self.agent_state = self.agent.get_init_state(
            message_history=[
                message for message in message_history if is_valid_agent_history_message(message)
            ]
        )
        self.user_state = _await(
            self.user.get_init_state(
                message_history=[
                    message for message in message_history if is_valid_user_history_message(message)
                ]
            )
        )
        self._append_autonomous_directive()
        self.trajectory = list(message_history)

        if not message_history:
            initial_message, self.user_state = _await(self.user.initial_message(self.user_state))
            initial_message.timestamp = get_now()
            self.trajectory.append(initial_message)
            self.message = initial_message
            self.from_role = Role.USER
            self.to_role = Role.AGENT
            return

        last_message = message_history[-1]
        self.message = last_message
        if isinstance(last_message, UserMessage):
            self.from_role, self.to_role = Role.USER, Role.AGENT
        elif isinstance(last_message, ToolMessage) and last_message.requestor == "assistant":
            self.from_role, self.to_role = Role.ENV, Role.AGENT
        elif isinstance(last_message, AssistantMessage) and last_message.is_tool_call():
            self.from_role, self.to_role = Role.AGENT, Role.ENV
        elif isinstance(last_message, AssistantMessage):
            self.done = True
            self.termination_reason = TerminationReason.AGENT_STOP
        else:
            raise ValueError(
                "A non-interactive deterministic run cannot resume after a user-requested tool"
            )

    def _generate_agent_message(self) -> AssistantMessage:
        original_state = deepcopy(self.agent_state)
        for attempt in range(3):
            current_state = deepcopy(original_state)
            agent_message, updated_state = self.agent.generate_next_message(
                self.message, current_state
            )
            if agent_message.has_text_content() or agent_message.is_tool_call():
                self.agent_state = updated_state
                return agent_message
            logger.warning("Agent generated invalid message (attempt {}/3): {}", attempt + 1, agent_message)
        self.done = True
        self.termination_reason = TerminationReason.INVALID_AGENT_MESSAGE
        raise RuntimeError("Agent generated three invalid messages")

    def step(self):
        if self.done:
            raise ValueError("Simulation is done")
        if self.from_role in (Role.USER, Role.ENV) and self.to_role == Role.AGENT:
            agent_message = self._generate_agent_message()
            agent_message.validate()
            self.trajectory.append(agent_message)
            self.message = agent_message
            self.from_role = Role.AGENT
            if agent_message.is_tool_call():
                self.to_role = Role.ENV
            elif self.user_state.review_turns_remaining:
                review_message, self.user_state = _await(self.user.review_message(self.user_state))
                review_message.timestamp = get_now()
                self.trajectory.append(review_message)
                self.message = review_message
                self.from_role, self.to_role = Role.USER, Role.AGENT
            else:
                # A non-tool response is the agent's only possible terminal
                # completion after the bounded fact-free review.
                self.done = True
                self.termination_reason = TerminationReason.AGENT_STOP
        elif self.from_role == Role.AGENT and self.to_role == Role.ENV:
            tool_messages = [self.environment.get_response(call) for call in self.message.tool_calls]
            self.trajectory.extend(tool_messages)
            self.message = (
                MultiToolMessage(role="tool", tool_messages=tool_messages)
                if len(tool_messages) > 1
                else tool_messages[0]
            )
            self.from_role, self.to_role = Role.ENV, Role.AGENT
        else:
            raise ValueError(
                f"Invalid autonomous role combination: {self.from_role} -> {self.to_role}"
            )
        self.step_count += 1


def run_autonomous_deterministic_task(
    domain: str,
    task: Any,
    agent: str,
    user: str,
    llm_agent: Optional[str] = None,
    llm_args_agent: Optional[dict] = None,
    llm_user: Optional[str] = None,
    llm_args_user: Optional[dict] = None,
    max_steps: int = 100,
    max_errors: int = 10,
    evaluation_type: Any = "trajectory",
    seed: Optional[int] = None,
    enable_think: bool = False,
    llm_evaluator: Optional[str] = None,
    llm_args_evaluator: Optional[dict] = None,
    language: str = None,
) -> SimulationRun:
    """Root-owned equivalent of VitaBench's task runner for this user only."""
    import vita.run as vita_run

    vita_run._clear_global_state()
    registry = vita_run.registry
    logger.info(
        "STARTING AUTONOMOUS DETERMINISTIC SIMULATION: Domain: {}, Task: {}, Agent: {}",
        domain, task.id, agent,
    )
    environment = (
        vita_run.get_cross_environment(domain, task.environment, language)
        if "," in domain
        else registry.get_env_constructor(domain)(task.environment, language)
    )
    agent_constructor = registry.get_agent_constructor(agent)
    current_time = environment.tools.db.time
    vita_run.global_time = current_time
    if issubclass(agent_constructor, vita_run.LLMAgent):
        agent_instance = agent_constructor(
            tools=environment.get_tools(), domain_policy=environment.get_policy(),
            llm=llm_agent, llm_args=llm_args_agent, time=current_time,
            enable_think=enable_think, language=language,
        )
    else:
        raise ValueError("The deterministic autonomous treatment supports LLM agents only")

    user_constructor = registry.get_user_constructor(user)
    user_instance = user_constructor(
        persona=str(task.user_scenario.user_profile), instructions=str(task.instructions),
        llm=llm_user, llm_args=llm_args_user, language=language,
    )
    if not isinstance(user_instance, DeterministicTaskUser):
        raise TypeError("Autonomous deterministic runner received a non-deterministic user")
    orchestrator = AutonomousDeterministicOrchestrator(
        domain=domain, agent=agent_instance, user=user_instance, environment=environment,
        task=task, max_steps=max_steps, max_errors=max_errors, seed=seed, language=language,
    )
    simulation = orchestrator.run()
    simulation.reward_info = vita_run.evaluate_simulation(
        domain=domain, task=task, simulation=simulation, evaluation_type=evaluation_type,
        llm_evaluator=llm_evaluator, llm_args_evaluator=llm_args_evaluator, language=language,
    )
    logger.info(
        "FINISHED AUTONOMOUS DETERMINISTIC SIMULATION: Domain: {}, Task: {}, Reward: {}",
        domain, task.id, simulation.reward_info.reward,
    )
    return simulation


def install_autonomous_deterministic_runner() -> None:
    """Patch VitaBench's private task hook only for the registered user name."""
    import vita.run as vita_run

    if getattr(vita_run._run_task_internal, "_vita_rl_autonomous_dispatch", False):
        return
    original: Callable[..., SimulationRun] = vita_run._run_task_internal
    signature = inspect.signature(original)

    def dispatch(*args: Any, **kwargs: Any) -> SimulationRun:
        bound = signature.bind_partial(*args, **kwargs)
        if bound.arguments.get("user") != DETERMINISTIC_USER_NAME:
            return original(*args, **kwargs)
        return run_autonomous_deterministic_task(**bound.arguments)

    dispatch._vita_rl_autonomous_dispatch = True  # type: ignore[attr-defined]
    vita_run._run_task_internal = dispatch
