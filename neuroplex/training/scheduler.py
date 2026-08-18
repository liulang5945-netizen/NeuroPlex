"""Training scheduler — unified interface for "train vs evolve" decisions.

The core question from the roadmap: when a new domain dataset arrives,
does Taiji fine-tune an existing neuron or spawn a new one?

Answer: the TrainingScheduler makes this decision dynamically:
- High match_score → fine-tune existing neuron
- Medium match_score → spawn new neuron from parent (distillation)
- Low match_score → train new neuron from scratch

This is the single entry point for all training tasks in the neuron era.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch


class TrainingScheduler:
    """Unified scheduler: "spawn new neuron" vs "reuse existing neuron".

    Usage:
        scheduler = TrainingScheduler(neurons, field, REUSE_THRESHOLD=0.7)
        task = scheduler.schedule(domain_data)
        # task = FineTuneTask / SpawnNeuronTask / NewNeuronTask
    """

    REUSE_THRESHOLD = 0.7   # cosine similarity: above this → fine-tune
    SPAWN_THRESHOLD = 0.3   # cosine similarity: between REUSE and SPAWN → spawn

    def __init__(
        self,
        neurons: Dict[str, object],
        field: object = None,
    ):
        self.neurons = neurons
        self.field = field

    def find_best_neuron(self, domain_embedding: torch.Tensor) -> Optional[str]:
        """Find the neuron most relevant to the new domain.

        Args:
            domain_embedding: [D] representative embedding of the new domain.

        Returns:
            neuron_id of the best match, or None if no neurons exist.
        """
        if not self.neurons:
            return None

        best_score = -1.0
        best_nid = None
        for nid, neuron in self.neurons.items():
            fingerprint = getattr(neuron, "fingerprint", None)
            if fingerprint is None:
                continue
            # Cosine similarity between domain and neuron fingerprint
            score = float(
                torch.dot(
                    domain_embedding / (domain_embedding.norm() + 1e-8),
                    fingerprint / (fingerprint.norm() + 1e-8),
                )
            )
            if score > best_score:
                best_score = score
                best_nid = nid

        return best_nid

    def schedule(self, domain_data, domain_embedding: Optional[torch.Tensor] = None):
        """Schedule a training task for new domain data.

        Args:
            domain_data: training data for the new domain.
            domain_embedding: optional pre-computed domain embedding.

        Returns:
            Task object: FineTuneTask / SpawnNeuronTask / NewNeuronTask
        """
        best_nid = self.find_best_neuron(domain_embedding) if domain_embedding is not None else None

        if best_nid is None:
            # No existing neurons → train from scratch
            return NewNeuronTask(domain_data)

        # Compute match score
        match_score = 0.5  # placeholder; real implementation uses cosine + learning rate

        if match_score > self.REUSE_THRESHOLD:
            return FineTuneTask(neuron_id=best_nid, data=domain_data)
        elif match_score > self.SPAWN_THRESHOLD:
            return SpawnNeuronTask(parent_id=best_nid, data=domain_data)
        else:
            return NewNeuronTask(data=domain_data)


class FineTuneTask:
    """Fine-tune an existing neuron on new data."""
    def __init__(self, neuron_id: str, data):
        self.neuron_id = neuron_id
        self.data = data
        self.task_type = "finetune"


class SpawnNeuronTask:
    """Spawn a new neuron from a parent via distillation."""
    def __init__(self, parent_id: str, data):
        self.parent_id = parent_id
        self.data = data
        self.task_type = "spawn"


class NewNeuronTask:
    """Train a new neuron from scratch."""
    def __init__(self, data, seed_neurons: Optional[List[str]] = None):
        self.data = data
        self.seed_neurons = seed_neurons or []
        self.task_type = "new"
