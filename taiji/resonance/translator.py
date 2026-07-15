"""Token translator and tokenizer hub for domain-specific tokenizers.

Each neuron can have its own domain-optimized tokenizer (32K-48K tokens)
while the general 256K tokenizer serves as the common I/O protocol.

TokenizerHub manages hot-swappable domain tokenizers — adding a new domain
tokenizer does not affect any existing neurons or the resonance field.

Based on the three-layer architecture:
- Layer 1: General tokenizer (256K) — I/O format, can be replaced
- Layer 2: Domain tokenizer (32K-48K) — per-neuron internal representation
- Layer 3: Resonance field (4096-dim) — completely independent of tokenizers
"""

from __future__ import annotations

from typing import Dict, Optional


class TokenTranslator:
    """Bidirectional translation between domain tokenizer and general tokenizer.

    Each neuron has its own translator instance.
    """

    def __init__(self, domain_vocab_size: int, general_vocab_size: int = 256000):
        self.domain_vocab_size = domain_vocab_size
        self.general_vocab_size = general_vocab_size

        # Alignment table: domain_token_id → [general_token_ids]
        self.alignment: Dict[int, list[int]] = {}

    def build_alignment(self, domain_tokenizer, general_tokenizer) -> None:
        """Build alignment table by encoding each domain token into general tokens.

        Args:
            domain_tokenizer: domain-specific SentencePiece processor.
            general_tokenizer: general 256K SentencePiece processor.
        """
        from sentencepiece import SentencePieceProcessor

        self.alignment = {}
        for domain_id in range(self.domain_vocab_size):
            word = domain_tokenizer.id_to_piece(domain_id)
            # Re-encode using general tokenizer
            general_ids = general_tokenizer.encode(word)
            self.alignment[domain_id] = general_ids

    def domain_to_general(self, domain_ids: list[int]) -> list[int]:
        """Convert domain token IDs to general token IDs.

        Args:
            domain_ids: list of domain token IDs.

        Returns:
            list of corresponding general token IDs.
        """
        result = []
        for did in domain_ids:
            if did in self.alignment:
                result.extend(self.alignment[did])
            else:
                # OOV fallback: return empty (will be handled by caller)
                pass
        return result

    def general_to_domain(self, general_ids: list[int], domain_tokenizer) -> list[int]:
        """Convert general token IDs to domain token IDs (for output).

        Args:
            general_ids: list of general token IDs.
            domain_tokenizer: domain tokenizer for re-encoding.

        Returns:
            list of domain token IDs.
        """
        # Decode general tokens to text, then re-encode with domain tokenizer
        # This is handled at a higher level since we need the general tokenizer for decoding
        raise NotImplementedError("Use domain tokenizer encode(decode_text) pattern")


class TokenizerHub:
    """Central registry for all domain tokenizers.

    Supports hot-swap: adding a new domain tokenizer does not affect
    any existing neurons or the resonance field.

    Usage:
        hub = TokenizerHub()
        hub.register_domain("rust", rust_tokenizer)
        tokens = hub.encode("fn main() {}", domain="rust")
    """

    def __init__(self, general_tokenizer=None):
        """Args:
            general_tokenizer: the general 256K tokenizer (I/O protocol). Optional.
        """
        self.tokenizers: Dict[str, object] = {}
        self.translators: Dict[str, TokenTranslator] = {}
        self.general_tokenizer = general_tokenizer

        # Register general tokenizer if provided
        if general_tokenizer is not None:
            self.tokenizers["general"] = general_tokenizer

    def register_domain(self, domain: str, domain_tokenizer) -> None:
        """Register a new domain tokenizer (hot-swap).

        Does not affect any existing neurons or tokenizers.

        Args:
            domain: domain name (e.g., "zh", "code", "rust").
            domain_tokenizer: SentencePiece processor for this domain.
        """
        self.tokenizers[domain] = domain_tokenizer
        print(f"[TokenizerHub] registered {domain} tokenizer")

    def get_tokenizer(self, domain: str):
        """Get tokenizer for a domain. Falls back to general if domain not found.

        Args:
            domain: domain name.

        Returns:
            Tokenizer instance or None.
        """
        if domain in self.tokenizers:
            return self.tokenizers[domain]
        return self.tokenizers.get("general")

    def encode(self, text: str, domain: str = "general") -> list[int]:
        """Encode text using the appropriate domain tokenizer.

        Args:
            text: input text.
            domain: domain name (falls back to "general").

        Returns:
            list of token IDs.
        """
        tok = self.get_tokenizer(domain)
        if tok is None:
            raise ValueError(f"No tokenizer for domain '{domain}' and no general fallback")
        return tok.encode(text)

    def list_domains(self) -> list[str]:
        """List all registered domains."""
        return list(self.tokenizers.keys())

    def build_translator(self, domain: str) -> Optional[TokenTranslator]:
        """Build a translator between a domain tokenizer and the general tokenizer.

        Args:
            domain: domain name to build translator for.

        Returns:
            TokenTranslator instance, or None if general tokenizer is not available.
        """
        if "general" not in self.tokenizers or domain not in self.tokenizers:
            return None

        domain_tok = self.tokenizers[domain]
        general_tok = self.tokenizers["general"]
        vocab_size = getattr(domain_tok, "vocab_size", getattr(domain_tok, "GetPieceSize", lambda: 0)())

        translator = TokenTranslator(vocab_size, 256000)
        translator.build_alignment(domain_tok, general_tok)
        self.translators[domain] = translator
        return translator
