"""Reference-based text separation; this does not identify voices from a waveform."""

import hashlib
import re
import unicodedata

SEPARATION_VERSION = "device-reference-v1"


def segments(text: str) -> list[dict]:
    result = []
    for match in re.finditer(r"[^。！？!?\n]+[。！？!?]*", text):
        raw = match.group()
        if not raw.strip():
            continue
        start = match.start() + len(raw) - len(raw.lstrip())
        end = match.start() + len(raw.rstrip())
        result.append({"id": len(result) + 1, "start": start, "end": end, "text": text[start:end]})
    return result


def _canonical(text: str) -> tuple[str, list[tuple[int, int]]]:
    """Matching-only normalization with a map back to untouched source characters."""
    chars, offsets = [], []
    for cluster in re.finditer(r".[\u3099\u309a\uff9e\uff9f]*", text, re.S):
        for char in unicodedata.normalize("NFKC", cluster.group()):
            if char.isspace() or unicodedata.category(char)[0] in "PZC":
                continue
            chars.append(char)
            offsets.append((cluster.start(), cluster.end()))
    value = "".join(chars)
    # Spelling variants only: do not turn a shock announcement into completion.
    for old, new in (("コネクター", "コネクタ"), ("直ちに", "ただちに")):
        matches = list(re.finditer(re.escape(old), value))
        for match in reversed(matches):
            a, b = match.span()
            span = (offsets[a][0], offsets[b - 1][1])
            value = value[:a] + new + value[b:]
            offsets[a:b] = [span] * len(new)
    return value, offsets


_LABEL = r"参加者[0-9０-９]*|人間|救助者[0-9０-９]*|AED|LED|機器|指導者|不明"
_LABELED_LINE = re.compile(
    rf"(?m)^[ \t]*(?:\[(?P<bracket>{_LABEL})\]|【(?P<wide>{_LABEL})】|(?P<plain>{_LABEL})[:：])[^\n]*",
    re.I,
)


def separate_speech(text: str, reference_text: str) -> dict:
    phrases = list(dict.fromkeys(_canonical(p["text"])[0] for p in segments(reference_text)))
    phrases = [phrase for phrase in phrases if phrase]
    if not phrases:
        raise ValueError("機器音声の参照TXTにアナウンスがありません")
    canonical, offsets = _canonical(text)
    matches = []
    for phrase in phrases:
        pos = 0
        while (pos := canonical.find(phrase, pos)) >= 0:
            matches.append((offsets[pos][0], offsets[pos + len(phrase) - 1][1]))
            pos += len(phrase)
    # Prefer the whole announcement to a suffix also present in the reference.
    matched = []
    for start, end in sorted(set(matches), key=lambda span: (span[0], -span[1])):
        if not matched or start >= matched[-1][1]:
            matched.append((start, end))

    labels = []
    for match in _LABELED_LINE.finditer(text):
        label = next(v for v in match.groupdict().values() if v).upper()
        role = ("device" if label in ("AED", "LED", "機器") else
                "instructor" if label == "指導者" else "unknown" if label == "不明" else "participant")
        labels.append((match.start(), match.end(), role))

    parts = []
    for part in segments(text):
        role = next((r for a, b, r in labels if a <= part["start"] < b), None)
        boundaries = {part["start"], part["end"]}
        if role is None:
            for a, b in matched:
                if a < part["end"] and b > part["start"]:
                    boundaries.update((max(a, part["start"]), min(b, part["end"])))
        edges = sorted(boundaries)
        for a, b in zip(edges, edges[1:]):
            if not _canonical(text[a:b])[0]:
                # Keep punctuation attached to the preceding span when possible.
                if parts and parts[-1]["end"] == a:
                    parts[-1].update(end=b, text=text[parts[-1]["start"]:b])
                continue
            is_match = any(start <= a and b <= end for start, end in matched)
            parts.append({"id": len(parts) + 1, "start": a, "end": b, "text": text[a:b],
                          "source_role": role or ("device" if is_match else "candidate"),
                          "source": "speaker_label" if role else "reference_match" if is_match else "unmatched"})

    return {"version": SEPARATION_VERSION,
            "reference_sha256": hashlib.sha256(reference_text.encode()).hexdigest(),
            "reference_text": reference_text,
            "segments": parts,
            "human_candidate_text": "\n".join(p["text"] for p in parts if p["source_role"] in ("candidate", "participant")),
            "device_text": "\n".join(p["text"] for p in parts if p["source_role"] == "device"),
            "excluded_text": "\n".join(p["text"] for p in parts if p["source_role"] in ("instructor", "unknown"))}
