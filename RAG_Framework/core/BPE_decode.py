"""
BPE byte-level decoding utilities for GPT-2-style tokenizers.

Single source of truth — import BPEDecoder from here everywhere.
Previously duplicated across agents/decoder.py, components/generators/standard.py,
and components/generators/LRM.py.
"""

_SKIP_SPECIAL = {'<s>', '</s>', '<unk>', '<pad>'}


class BPEDecoder:
    """BPE byte-level decoding utilities for GPT-2-style tokenizers.

    Handles tokenizers where the vocabulary uses Ġ (U+0120) as the space
    marker but the decoder expects ▁ (U+2581), which causes tokenizer.decode()
    to strip all spaces from output.

    In byte-level BPE, each byte (0-255) is mapped to a Unicode character.
    Printable ASCII maps to itself; non-printable bytes (space, newline, etc.)
    map to characters starting at U+0100 (e.g. space→Ġ, newline→Ċ).
    The methods here build and use the reverse mapping to recover proper UTF-8.
    """

    _byte_decoder = None  # Lazy-initialized inverse of GPT-2 bytes_to_unicode
    _inv_vocab = None     # Lazy-initialized inverse vocabulary {id: string}
    _special_ids = None   # Lazy-initialized set of special token IDs

    @staticmethod
    def build_byte_decoder():
        """Build the inverse of the GPT-2 byte-level BPE bytes_to_unicode mapping."""
        bs = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
        )
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        return {chr(c): b for b, c in zip(bs, cs)}

    @staticmethod
    def ensure_vocab(tokenizer):
        """Lazily build and cache inverse vocabulary and special token set."""
        if BPEDecoder._byte_decoder is None:
            BPEDecoder._byte_decoder = BPEDecoder.build_byte_decoder()
        if BPEDecoder._inv_vocab is None:
            vocab = tokenizer.get_vocab()
            BPEDecoder._inv_vocab = {v: k for k, v in vocab.items()}
            special = set(getattr(tokenizer, 'all_special_ids', []))
            if hasattr(tokenizer, 'added_tokens_encoder'):
                special.update(tokenizer.added_tokens_encoder.values())
            BPEDecoder._special_ids = special

    @staticmethod
    def decode_tokens(tokenizer, tokens):
        """Decode token IDs to text, bypassing tokenizer.decode() entirely.

        Handles tokenizers where the BPE vocabulary uses Ġ (U+0120) as the
        space marker but the decoder expects ▁ (U+2581), which causes
        tokenizer.decode() to strip all spaces from output.
        """
        if not tokens:
            return ""
        BPEDecoder.ensure_vocab(tokenizer)

        inv_vocab = BPEDecoder._inv_vocab
        special_ids = BPEDecoder._special_ids
        byte_decoder = BPEDecoder._byte_decoder

        parts = []
        byte_buf = bytearray()

        for tok_id in tokens:
            tok_str = inv_vocab.get(tok_id, '')

            if tok_id in special_ids:
                if byte_buf:
                    parts.append(byte_buf.decode('utf-8', errors='ignore'))
                    byte_buf = bytearray()
                if tok_str not in _SKIP_SPECIAL:
                    parts.append(tok_str)
            else:
                for c in tok_str:
                    if c == '\u2581':
                        # Metaspace ▁ → space byte (handles mixed encoding)
                        byte_buf.append(0x20)
                    elif c in byte_decoder:
                        byte_buf.append(byte_decoder[c])
                    else:
                        byte_buf.extend(c.encode('utf-8'))

        if byte_buf:
            parts.append(byte_buf.decode('utf-8', errors='ignore'))

        return ''.join(parts)

    @staticmethod
    def fix_bpe_artifacts(text):
        """Post-process a string to fix BPE decoding artifacts.

        When tokenizer.decode() has a decoder mismatch (e.g. Ministral),
        generate() returns raw BPE chars like Ġ (space), Ċ (newline),
        ðŁĺĬ (emoji bytes). Applies the GPT-2 byte decoder char-by-char
        to recover proper UTF-8 text.
        Only activates when Ġ (U+0120) or ▁ (U+2581) artifacts are present.
        """
        if '\u0120' not in text and '\u2581' not in text:
            return text
        if BPEDecoder._byte_decoder is None:
            BPEDecoder._byte_decoder = BPEDecoder.build_byte_decoder()
        byte_decoder = BPEDecoder._byte_decoder
        byte_buf = bytearray()
        parts = []
        for c in text:
            if c == '\u2581':
                byte_buf.append(0x20)
            elif c in byte_decoder:
                byte_buf.append(byte_decoder[c])
            else:
                if byte_buf:
                    parts.append(byte_buf.decode('utf-8', errors='ignore'))
                    byte_buf = bytearray()
                parts.append(c)
        if byte_buf:
            parts.append(byte_buf.decode('utf-8', errors='ignore'))
        return ''.join(parts)
