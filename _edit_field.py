import pathlib

p = pathlib.Path('taiji/resonance/field.py')
text = p.read_text(encoding='utf-8', errors='replace')

# Add complementarity_score and combined_score after score() method
anchor = '        return float(torch.dot(v_norm, f_norm).item())'
assert anchor in text, 'score return anchor not found'

new_methods = anchor + '''

    def complementarity_score(self, vector: torch.Tensor) -> float:
        """Compute complementarity: how much *new* information a neuron brings.

        Measures the orthogonal component of the neuron's field vector relative
        to the current field state.
        - High score (close to 1): neuron contributes information not already in the field
        - Low score (close to 0): neuron echoes what the field already has
        Score in [0, 1].
        """
        if vector.dim() == 2:
            vector = vector.mean(dim=0)
        v_norm = vector / (vector.norm() + 1e-8)
        if self.state.norm() < 1e-8:
            return 1.0  # empty field: everything is new
        f_norm = self.state / (self.state.norm() + 1e-8)
        alignment = float(torch.dot(v_norm, f_norm).item())
        # Orthogonal component: what v adds beyond its projection onto f
        orthogonal = v_norm - alignment * f_norm
        return float(orthogonal.norm().item())

    def combined_score(self, vector: torch.Tensor, alpha: float = 0.5) -> float:
        """Blend alignment and complementarity.

        alpha=0: pure alignment (original behavior).
        alpha=1: pure complementarity.
        alpha=0.5: balanced.
        """
        align = self.score(vector)
        comp = self.complementarity_score(vector)
        align_01 = (align + 1.0) / 2.0
        return (1.0 - alpha) * align_01 + alpha * comp'''

text = text.replace(anchor, new_methods, 1)
p.write_text(text, encoding='utf-8')
print('field.py updated successfully')
