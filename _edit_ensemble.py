import pathlib

p = pathlib.Path('taiji/resonance/ensemble.py')
text = p.read_text(encoding='utf-8', errors='replace')

# The block to replace: the else branch (resonance-score softmax) + the weighting loop.
# We replace from 'else:' (line 282) through the final_weights dict (line 300).
old_block = """            else:
                # Default: resonance-score softmax weighting
                score_tensor = torch.tensor(
                    [final_scores.get(nid, 0.0) for nid in all_logits.keys()],
                    device=shared_embeddings.device,
                )
                weights = F.softmax(score_tensor * 2.0, dim=0)

            weighted_logits = None
            for i, (nid, logits) in enumerate(all_logits.items()):
                w = weights[i]
                if weighted_logits is None:
                    weighted_logits = w * logits
                else:
                    weighted_logits = weighted_logits + w * logits
            result["weighted_logits"] = weighted_logits
            result["final_weights"] = {
                nid: float(weights[i]) for i, nid in enumerate(all_logits.keys())
            }"""

assert old_block in text, 'old weighting block not found'

new_block = """            else:
                # Per-position routing (v2): logit-entropy weighting + complementarity.
                # Each position independently picks the neuron that is most confident.
                # Complementarity scores boost neurons bringing new information.
                # Memory-efficient: process one neuron at a time for entropy.
                neuron_ids = list(all_logits.keys())
                entropies = []
                for nid in neuron_ids:
                    log_probs = F.log_softmax(all_logits[nid], dim=-1)
                    probs = torch.exp(log_probs)
                    ent = -(probs * log_probs).sum(dim=-1)  # [B, L]
                    entropies.append(ent)
                ent_stack = torch.stack(entropies)  # [N, B, L]
                # Lower entropy = more confident = higher weight
                confidence = 1.0 / (ent_stack + 1e-8)  # [N, B, L]
                position_weights = F.softmax(confidence * 2.0, dim=0)  # [N, B, L]

                # Boost complementary neurons (v2)
                if hasattr(self.field, 'complementarity_score'):
                    comp_scores = []
                    for nid in neuron_ids:
                        v = vectors.get(nid, torch.zeros(1, device=shared_embeddings.device))
                        comp_scores.append(self.field.complementarity_score(v))
                    comp_boost = torch.tensor(comp_scores, device=shared_embeddings.device)
                    comp_boost = (1.0 + comp_boost).unsqueeze(-1).unsqueeze(-1)  # [N, 1, 1]
                    position_weights = position_weights * comp_boost
                    position_weights = position_weights / position_weights.sum(dim=0, keepdim=True)

                # Apply per-position weights (memory-efficient: one at a time)
                weighted_logits = None
                for i, (nid, logits) in enumerate(all_logits.items()):
                    w = position_weights[i]  # [B, L]
                    if weighted_logits is None:
                        weighted_logits = w.unsqueeze(-1) * logits
                    else:
                        weighted_logits = weighted_logits + w.unsqueeze(-1) * logits
                result["weighted_logits"] = weighted_logits
                result["final_weights"] = {
                    nid: float(position_weights[i].mean().item())
                    for i, nid in enumerate(neuron_ids)
                }"""

text = text.replace(old_block, new_block, 1)
p.write_text(text, encoding='utf-8')
print('ensemble.py updated successfully')
