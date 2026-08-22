"""Native play environment: a text topic world for autonomous exploration.

Phase 3 play organ.  The world implements the ``TaijiEnvironment`` protocol:
every transition genuinely depends on the organism's action -- at a choice
moment the afforded actions are the distinct opening bytes of the topic pool
(minus any topic locked out by the anti-lock-in scaffold), and the organism's
own motor policy decides which direction to explore.  Once a topic is chosen
the world streams its bytes as sensations while the organism acts to complete
each position; completion rewards reward-modulated motor learning, but the
sensory stream never waits for a "correct" answer -- the environment is an
outcome provider, not a teacher.

Anti-lock-in scaffold (B1-bis semantics): when the organism picks the same
topic ``force_switch_streak`` episodes in a row, that topic is withdrawn from
the affordances for the next episode, so exploration cannot ossify on one
direction.  The scaffold shapes what is afforded, never what is chosen.

The ``play`` driver runs the canonical active sequence from the architecture
document: ``observe(cue) -> act(affordances) -> env.step(action) ->
settle_action(reward) -> observe(outcome.sensation)``, closing every pending
experience with a final boundary observation so the episodic field receives
real ``experienced`` engrams ready for endogenous replay.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from taiji import EnvironmentOutcome

from .model import Seed

COMPLETION_REWARD = 1.0
MISS_REWARD = 0.0


class TopicWorld:
    """A topic pool whose transitions branch on the organism's byte actions."""

    def __init__(
        self,
        topics: Sequence[bytes],
        *,
        boundary_symbol: int = 256,
        force_switch_streak: int = 5,
        recency_window: int = 0,
    ) -> None:
        if not topics:
            raise ValueError("topic pool cannot be empty")
        for topic in topics:
            if not topic:
                raise ValueError("topics cannot be empty")
        self.topics: List[bytes] = [bytes(topic) for topic in topics]
        self.boundary_symbol = int(boundary_symbol)
        if force_switch_streak <= 0:
            raise ValueError("force_switch_streak must be positive")
        if recency_window < 0:
            raise ValueError("recency_window cannot be negative")
        self.force_switch_streak = int(force_switch_streak)
        self.recency_window = int(recency_window)
        self.streak = 0
        self.last_topic_index: Optional[int] = None
        self.topics_visited: List[int] = []
        self.forced_switches = 0
        self._excluded: Optional[int] = None
        self._choosing = True
        self._index: Optional[int] = None
        self._position = 0
        self._previous_index: Optional[int] = None

    @property
    def choosing(self) -> bool:
        return self._choosing

    def _blocked_indices(self) -> set:
        blocked = set()
        if self._excluded is not None:
            blocked.add(self._excluded)
        if self.recency_window > 0 and self.topics_visited:
            blocked.update(self.topics_visited[-self.recency_window :])
        return blocked

    def available_actions(self) -> List[int]:
        """Afforded choice bytes: one per un-blocked topic's opening byte.

        Blocking comes from two anti-lock-in scaffolds: the streak exclusion
        (one episode penalty) and the recency window (recently visited topics
        step aside so exploration keeps moving).  Scaffolds shape what is
        afforded; if everything were blocked the pool is afforded whole, since
        the organism must never face an empty choice.
        """

        blocked = self._blocked_indices()
        actions: List[int] = []
        for index, topic in enumerate(self.topics):
            if index in blocked:
                continue
            symbol = int(topic[0])
            if symbol not in actions:
                actions.append(symbol)
        if not actions:
            for topic in self.topics:
                symbol = int(topic[0])
                if symbol not in actions:
                    actions.append(symbol)
        return actions

    def reset(self) -> Tuple[int, Sequence[int]]:
        return self.reset_choice()

    def reset_choice(self) -> Tuple[int, Sequence[int]]:
        """Enter a choice moment: boundary sensation plus afforded directions."""

        self._choosing = True
        self._index = None
        self._position = 0
        self.last_topic_index = None
        return self.boundary_symbol, self.available_actions()

    def step(self, action_symbol: int) -> EnvironmentOutcome:
        action_symbol = int(action_symbol)
        if self._choosing:
            # An exclusion is served exactly once -- by this choice.
            self._excluded = None
            blocked = self._blocked_indices()
            candidates = [
                index
                for index, topic in enumerate(self.topics)
                if index not in blocked and topic[0] == action_symbol
            ]
            if not candidates:
                raise ValueError("chosen byte affords no topic")
            index = candidates[0]
            self._choosing = False
            self._index = index
            self._position = 1
            self.last_topic_index = index
            self.topics_visited.append(index)
            if self._previous_index == index:
                self.streak += 1
            else:
                self.streak = 1
            self._previous_index = index
            if self.streak >= self.force_switch_streak:
                self._excluded = index
                self.forced_switches += 1
            terminal = self._position >= len(self.topics[index])
            if terminal:
                self._choosing = True
                self._index = None
            return EnvironmentOutcome(
                sensation=int(self.topics[index][0]),
                reward=MISS_REWARD,
                terminal=terminal,
            )
        index = self._index
        assert index is not None
        topic = self.topics[index]
        expected = int(topic[self._position])
        reward = COMPLETION_REWARD if action_symbol == expected else MISS_REWARD
        self._position += 1
        terminal = self._position >= len(topic)
        if terminal:
            self._choosing = True
            self._index = None
        return EnvironmentOutcome(
            sensation=expected, reward=reward, terminal=terminal
        )


def play(
    seed: Seed,
    world: TopicWorld,
    *,
    episodes: int,
    sample: bool = True,
    learn: bool = True,
) -> Dict[str, object]:
    """Run the canonical observe/act/step/settle/observe loop over episodes.

    Each episode ends with a boundary observation that closes the last pending
    experience, so every lived step becomes an endogenous episodic write with
    provenance ``experienced`` -- the raw material of later sleep.
    """

    if episodes <= 0:
        raise ValueError("episodes must be positive")
    boundary = seed.substrate.config.boundary_symbol
    actions = 0
    reward_sum = 0.0
    crashes = 0
    topic_sequence: List[int] = []
    for episode in range(int(episodes)):
        sensation, affordances = world.reset()
        seed.reset_dynamics(episode_id=f"play-{episode}")
        seed.observe(sensation, learn=bool(learn), learn_motor=False)
        try:
            decision = seed.act(affordances, sample=bool(sample))
            outcome = world.step(decision.action_symbol)
            seed.settle_action(
                outcome.reward, learn=bool(learn), provenance="experienced"
            )
            seed.observe(
                outcome.sensation, learn=bool(learn), learn_motor=False
            )
            actions += 1
            reward_sum += float(outcome.reward)
            while not outcome.terminal:
                decision = seed.act(
                    range(seed.substrate.config.alphabet_size),
                    sample=bool(sample),
                )
                outcome = world.step(decision.action_symbol)
                seed.settle_action(
                    outcome.reward, learn=bool(learn), provenance="experienced"
                )
                seed.observe(
                    outcome.sensation, learn=bool(learn), learn_motor=False
                )
                actions += 1
                reward_sum += float(outcome.reward)
            seed.observe(boundary, learn=bool(learn), learn_motor=False)
            topic_sequence.append(int(world.last_topic_index))
        except Exception:
            crashes += 1
            seed.reset_dynamics(episode_id=f"play-crash-{episode}")
            seed.observe(boundary, learn=False)
    return {
        "episodes": int(episodes),
        "actions": actions,
        "crashes": crashes,
        "mean_reward": reward_sum / max(1, actions),
        "distinct_topics": len(set(topic_sequence)),
        "topic_sequence": topic_sequence,
        "forced_switches": world.forced_switches,
    }

