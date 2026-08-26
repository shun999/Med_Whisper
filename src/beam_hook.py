# # src/beam_hook.py
# # ============================================================
# # openai/whisper BeamSearchDecoder.update hook
# # - domain term next-token bias (safe)
# # - beam prefix log (best-effort)
# # - inspect top-k decoded candidates (pre/post bias)
# #
# # Key points:
# # - "pre_bias" and "post_bias" are both logged so you can prove bias effect.
# # - ban_token_ids (e.g., [15553]) is applied as hard -inf to logprobs.
# # - state["step"] always increments every update (even if enable_log=False).
# # ============================================================

# from __future__ import annotations

# from dataclasses import dataclass
# from typing import List, Dict, Any, Optional, Tuple, Union
# import time
# import json

# import torch
# from whisper.decoding import BeamSearchDecoder


# # ----------------------------
# # Config
# # ----------------------------
# @dataclass
# class BiasConfig:
#     enable_bias: bool = True
#     bias_strength: float = 3.0
#     max_suffix_tokens: int = 32
#     disable_bias_if_no_prefix: bool = True  # IMPORTANT
#     clamp_bias: float = 3.0                 # per-token clamp for safety


# @dataclass
# class LogConfig:
#     enable_log: bool = True
#     topk: int = 10


# @dataclass
# class InspectConfig:
#     enable_inspect: bool = True
#     topk: int = 200
#     targets: Optional[List[str]] = None
#     store_topk_list: bool = True
#     decoded_max_len: int = 40
#     ban_token_ids: Optional[List[int]] = None  # hard-ban these token_ids (set -inf)


# # ----------------------------
# # Trie (real prefix tree)
# # ----------------------------
# class TrieNode:
#     __slots__ = ("children", "terminal")

#     def __init__(self):
#         self.children: Dict[int, "TrieNode"] = {}
#         self.terminal: bool = False


# def _safe_encode(tokenizer, text: str) -> List[int]:
#     """
#     Whisper tokenizer.encode expects allowed_special as a set/frozenset in some versions.
#     Use empty frozenset() to disallow specials.
#     """
#     try:
#         return tokenizer.encode(text, allowed_special=frozenset())
#     except TypeError:
#         return tokenizer.encode(text)
#     except Exception:
#         try:
#             return tokenizer.encode(text)
#         except Exception:
#             return []


# def build_trie(tokenizer, terms: List[str], max_len: int) -> TrieNode:
#     root = TrieNode()
#     for t in (terms or []):
#         if not t:
#             continue
#         ids = _safe_encode(tokenizer, t)
#         if 0 < len(ids) <= max_len:
#             node = root
#             for tok in ids:
#                 tok = int(tok)
#                 node = node.children.setdefault(tok, TrieNode())
#             node.terminal = True
#     return root


# def suffix_next_tokens(root: TrieNode, suffix: List[int]) -> List[int]:
#     node = root
#     for tok in suffix:
#         node = node.children.get(int(tok))
#         if node is None:
#             return []
#     return list(node.children.keys())


# def compute_next_token_bias(
#     root: TrieNode,
#     prefix_ids: List[int],
#     vocab_size: int,
#     cfg: BiasConfig,
#     device: torch.device,
# ) -> torch.Tensor:
#     """
#     Bias next tokens that continue any domain term using suffix matching.
#     Complexity O(L^2) where L <= max_suffix_tokens (independent of #terms).
#     """
#     bias = torch.zeros((vocab_size,), dtype=torch.float32, device=device)

#     if not prefix_ids or vocab_size <= 0:
#         return bias

#     base = prefix_ids[-cfg.max_suffix_tokens:]
#     L = len(base)

#     for i in range(L):
#         suf = base[i:L]
#         nxts = suffix_next_tokens(root, suf)
#         if not nxts:
#             continue
#         idx = torch.tensor(nxts, dtype=torch.long, device=device)
#         bias.index_add_(
#             0,
#             idx,
#             torch.full((len(nxts),), float(cfg.bias_strength), device=device),
#         )

#     if cfg.clamp_bias and cfg.clamp_bias > 0:
#         bias = torch.clamp(bias, 0.0, float(cfg.clamp_bias))
#     return bias


# # ----------------------------
# # Beam prefix fetch (best-effort)
# # ----------------------------
# def _get_beam_prefixes_from_decoder_attrs(decoder: Any, beam_size: int) -> Optional[List[List[int]]]:
#     candidates = ["seqs", "sequences", "sequence", "tokens", "current_sequences"]
#     seqs = None
#     for name in candidates:
#         if hasattr(decoder, name):
#             seqs = getattr(decoder, name)
#             if seqs is not None:
#                 break
#     if seqs is None:
#         return None

#     try:
#         if isinstance(seqs, torch.Tensor):
#             if seqs.ndim >= 2:
#                 return [seqs[b].tolist() for b in range(min(beam_size, seqs.shape[0]))]
#             return None

#         if isinstance(seqs, list):
#             out: List[List[int]] = []
#             for b in range(min(beam_size, len(seqs))):
#                 s = seqs[b]
#                 if isinstance(s, torch.Tensor):
#                     out.append(s.tolist())
#                 elif hasattr(s, "tolist"):
#                     out.append(s.tolist())
#                 else:
#                     out.append(list(s))
#             while len(out) < beam_size:
#                 out.append([])
#             return out
#     except Exception:
#         return None

#     return None


# def _tokens_to_text(tokenizer, tokens: List[int]) -> str:
#     try:
#         return tokenizer.decode(tokens).strip()
#     except Exception:
#         return ""


# def _token_kind(tokenizer, tid: int) -> str:
#     # openai-whisper tokenizer has timestamp_begin / eot
#     ts_begin = getattr(tokenizer, "timestamp_begin", None)
#     eot = getattr(tokenizer, "eot", None)
#     if ts_begin is not None and eot is not None:
#         if int(ts_begin) <= int(tid) < int(eot):
#             return "timestamp"

#     # common specials
#     specials = ["sot", "eot", "no_speech", "translate", "transcribe"]
#     for name in specials:
#         v = getattr(tokenizer, name, None)
#         if v is not None and int(tid) == int(v):
#             return f"special:{name}"
#     return "normal"


# def _is_float_logprobs_tensor(x: Any) -> bool:
#     return (
#         torch.is_tensor(x)
#         and x.dtype.is_floating_point
#         and x.ndim in (1, 2)
#     )


# def _is_int_tokens_tensor(x: Any) -> bool:
#     return (
#         torch.is_tensor(x)
#         and x.dtype in (torch.int32, torch.int64)
#         and x.ndim == 2
#     )


# def _decode_next_with_context(tokenizer, prefix_ids: List[int], tid: int, tail_n: int = 24, cap: int = 80) -> str:
#     """
#     Return incremental decoded surface of appending tid given the last tail_n tokens context.
#     (best-effort; used for inspect readability)
#     """
#     tail = prefix_ids[-tail_n:] if prefix_ids else []
#     try:
#         before = tokenizer.decode(tail)
#         after = tokenizer.decode(tail + [int(tid)])
#         if after.startswith(before):
#             inc = after[len(before):]
#         else:
#             inc = after
#         inc = inc.replace("\n", "\\n")
#         if cap and len(inc) > cap:
#             inc = inc[:cap] + "…"
#         return inc
#     except Exception:
#         return ""


# # ----------------------------
# # Hook install/uninstall
# # ----------------------------
# def install_beam_hook(
#     tokenizer,
#     domain_terms: List[str],
#     bias_cfg: BiasConfig,
#     log_cfg: LogConfig,
#     inspect_cfg: Optional[InspectConfig] = None,
# ) -> Tuple[Any, Dict[str, Any]]:
#     """
#     Monkeypatch BeamSearchDecoder.update.
#     Returns (original_update, state).
#     """
#     original_update = BeamSearchDecoder.update
#     trie_root = build_trie(tokenizer, domain_terms, max_len=bias_cfg.max_suffix_tokens)

#     state: Dict[str, Any] = {
#         "step": 0,
#         "beam_log": [],
#         "inspect_log": [],
#         "trie_terms": len(domain_terms or []),
#     }

#     def _emit_inspect_log(phase: str, logprobs_2d: torch.Tensor, prefixes: Optional[List[List[int]]]) -> None:
#         """
#         phase: "pre_bias" or "post_bias"
#         logprobs_2d: shape (beam, vocab)
#         """
#         if inspect_cfg is None or not inspect_cfg.enable_inspect:
#             return
#         try:
#             beam_size, vocab_size = logprobs_2d.shape
#             k = max(1, min(int(inspect_cfg.topk), vocab_size))
#             targets = inspect_cfg.targets or []
#             ban_ids = set(int(x) for x in (inspect_cfg.ban_token_ids or []))

#             for b in range(beam_size):
#                 row = logprobs_2d[b]
#                 vals, idxs = torch.topk(row, k=k, dim=-1)
#                 idxs_l = idxs.detach().cpu().tolist()
#                 vals_l = vals.detach().cpu().tolist()

#                 top1_lp = float(vals_l[0]) if vals_l else float("nan")
#                 pref = prefixes[b] if prefixes is not None and b < len(prefixes) else []

#                 decoded_list = []
#                 best_by_target: Dict[str, Dict[str, Any]] = {}
#                 hit_any = False

#                 for rank, (tid, lp) in enumerate(zip(idxs_l, vals_l), start=1):
#                     tid_i = int(tid)
#                     dec = _decode_next_with_context(tokenizer, pref, tid_i, tail_n=24, cap=inspect_cfg.decoded_max_len)

#                     if inspect_cfg.store_topk_list:
#                         decoded_list.append({
#                             "rank": int(rank),
#                             "token_id": tid_i,
#                             "logprob": float(lp),
#                             "gap_to_top1": float(top1_lp - float(lp)),
#                             "decoded": dec,
#                             "kind": _token_kind(tokenizer, tid_i),
#                             "banned": bool(tid_i in ban_ids),
#                         })

#                     # banned token is excluded from hit check (but still appears in topk_list with banned=true)
#                     if tid_i in ban_ids:
#                         continue

#                     if targets:
#                         for t in targets:
#                             if t and (t in dec):
#                                 hit_any = True
#                                 if t not in best_by_target:
#                                     best_by_target[t] = {
#                                         "target": t,
#                                         "rank": int(rank),
#                                         "token_id": tid_i,
#                                         "logprob": float(lp),
#                                         "gap_to_top1": float(top1_lp - float(lp)),
#                                         "decoded": dec,
#                                         "kind": _token_kind(tokenizer, tid_i),
#                                     }

#                 top1_tid = int(idxs_l[0]) if idxs_l else None
#                 top1_rec = None
#                 if top1_tid is not None:
#                     top1_rec = {
#                         "token_id": top1_tid,
#                         "logprob": float(vals_l[0]),
#                         "decoded": _decode_next_with_context(tokenizer, pref, top1_tid, tail_n=24, cap=inspect_cfg.decoded_max_len),
#                         "kind": _token_kind(tokenizer, top1_tid),
#                         "banned": bool(top1_tid in ban_ids),
#                     }

#                 state["inspect_log"].append({
#                     "t_ms": int(time.time() * 1000),
#                     "step": int(state["step"]),
#                     "beam": int(b),
#                     "phase": phase,
#                     "topk": int(k),
#                     "targets": targets,
#                     "hit": bool(hit_any),
#                     "hits_best": list(best_by_target.values())[:50],
#                     "top1": top1_rec,
#                     "topk_list": decoded_list if inspect_cfg.store_topk_list else None,
#                 })
#         except Exception:
#             pass

#     def hooked_update(self, *args, **kwargs):
#         # ------------------------------------
#         # 1) find logprobs tensor in args/kwargs
#         # ------------------------------------
#         logprobs = None
#         logprobs_where: Optional[Tuple[str, Union[int, str]]] = None  # ("args", idx) or ("kwargs", key)

#         for i, a in enumerate(args):
#             if _is_float_logprobs_tensor(a):
#                 logprobs = a
#                 logprobs_where = ("args", i)
#                 break

#         if logprobs is None:
#             for k, v in kwargs.items():
#                 if _is_float_logprobs_tensor(v):
#                     logprobs = v
#                     logprobs_where = ("kwargs", k)
#                     break

#         if logprobs is None or logprobs_where is None:
#             return original_update(self, *args, **kwargs)

#         original_was_1d = (logprobs.ndim == 1)
#         if original_was_1d:
#             logprobs = logprobs.unsqueeze(0)

#         beam_size, vocab_size = logprobs.shape

#         # ------------------------------------
#         # 2) Hard BAN token ids (e.g., replacement char "�")
#         # ------------------------------------
#         if inspect_cfg is not None and inspect_cfg.ban_token_ids:
#             try:
#                 for bid in inspect_cfg.ban_token_ids:
#                     bid = int(bid)
#                     if 0 <= bid < vocab_size:
#                         logprobs[:, bid] = -1e9  # effectively -inf
#             except Exception:
#                 pass

#         # ------------------------------------
#         # 3) find token prefixes (beam, t)
#         # ------------------------------------
#         prefixes: Optional[List[List[int]]] = None
#         prefix_source: Optional[str] = None

#         for i, a in enumerate(args):
#             if _is_int_tokens_tensor(a) and a.shape[0] == beam_size:
#                 prefixes = [a[b].tolist() for b in range(min(beam_size, a.shape[0]))]
#                 prefix_source = f"args[{i}]"
#                 break

#         if prefixes is None:
#             for k, v in kwargs.items():
#                 if _is_int_tokens_tensor(v) and v.shape[0] == beam_size:
#                     prefixes = [v[b].tolist() for b in range(min(beam_size, v.shape[0]))]
#                     prefix_source = f"kwargs['{k}']"
#                     break

#         if prefixes is None:
#             prefixes = _get_beam_prefixes_from_decoder_attrs(self, beam_size)
#             prefix_source = "decoder_attr" if prefixes is not None else None

#         # ------------------------------------
#         # 4) inspect BEFORE bias
#         # ------------------------------------
#         _emit_inspect_log("pre_bias", logprobs, prefixes)

#         # ------------------------------------
#         # 5) apply bias (safe)
#         # ------------------------------------
#         if bias_cfg.enable_bias and vocab_size > 0:
#             if prefixes is None and bias_cfg.disable_bias_if_no_prefix:
#                 pass
#             else:
#                 try:
#                     for b in range(beam_size):
#                         pref = prefixes[b] if prefixes is not None and b < len(prefixes) else []
#                         bvec = compute_next_token_bias(
#                             trie_root, pref, vocab_size, bias_cfg, device=logprobs.device
#                         )
#                         logprobs[b] = logprobs[b] + bvec
#                 except Exception:
#                     pass

#         # ------------------------------------
#         # 6) inspect AFTER bias (to prove bias effect)
#         # ------------------------------------
#         _emit_inspect_log("post_bias", logprobs, prefixes)

#         # ------------------------------------
#         # 7) put back modified logprobs into args/kwargs
#         # ------------------------------------
#         if logprobs_where[0] == "args":
#             idx = int(logprobs_where[1])  # type: ignore[arg-type]
#             new_args = list(args)
#             new_args[idx] = logprobs.squeeze(0) if original_was_1d else logprobs
#             args = tuple(new_args)
#         else:
#             key = str(logprobs_where[1])  # type: ignore[arg-type]
#             kwargs[key] = logprobs.squeeze(0) if original_was_1d else logprobs

#         out = original_update(self, *args, **kwargs)

#         # ------------------------------------
#         # 8) log beam prefix (best-effort)
#         # ------------------------------------
#         if log_cfg.enable_log:
#             try:
#                 pref_list = prefixes or [[] for _ in range(beam_size)]
#                 k = min(log_cfg.topk, beam_size)
#                 for b in range(k):
#                     pref_ids = pref_list[b] if b < len(pref_list) else []
#                     state["beam_log"].append({
#                         "t_ms": int(time.time() * 1000),
#                         "step": int(state["step"]),
#                         "beam": int(b),
#                         "logprobs_where": f"{logprobs_where[0]}:{logprobs_where[1]}",
#                         "prefix_source": prefix_source,
#                         "prefix_len": int(len(pref_ids)),
#                         "prefix_tail_ids": pref_ids[-16:],
#                         "prefix": _tokens_to_text(tokenizer, pref_ids),
#                     })
#             except Exception:
#                 pass

#         # ★ step is always advanced (even if enable_log=False)
#         state["step"] += 1
#         return out

#     BeamSearchDecoder.update = hooked_update
#     return original_update, state


# def uninstall_beam_hook(original_update):
#     BeamSearchDecoder.update = original_update


# def save_beam_log_jsonl(state: Dict[str, Any], path: str) -> None:
#     with open(path, "w", encoding="utf-8") as f:
#         for rec in state.get("beam_log", []):
#             f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# def save_inspect_log_jsonl(state: Dict[str, Any], path: str) -> None:
#     with open(path, "w", encoding="utf-8") as f:
#         for rec in state.get("inspect_log", []):
#             f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# src/beam_hook.py
# ============================================================
# openai/whisper BeamSearchDecoder.update hook
# - (A) optional domain-term next-token bias via suffix-trie
# - (B) optional "force target if present in top-k" (rank flip)
# - (C) optional inspect logging (pre/post bias) to JSONL
#
# 目的:
#   「top-k に目的語(例: 傷/傷病者/傷病者発見)が見えているなら、
#    それを強制的に top1 にする（= logprob を top1 超えにする）」
#
# 注意:
# - Whisperのtokenはサブワード。日本語は「傷」単体tokenがある場合もあるが、
#   prefix文脈込みのdecode増分で判定する方が安定する。
# - このrank-flipは強力なので、発動条件(topk/max_gap/step範囲など)を絞ると安全。
# ============================================================



# src/beam_hook.py
# ============================================================
# openai/whisper BeamSearchDecoder.update hook
# - BAN: 特定token_idを -inf にして除外（例: "�" のような壊れトークン）
# - INSPECT: 各step/beamの top-k を JSONL に吐く（pre/post の両方）
# - BIAS: 既存の trie suffix next-token bias（任意）
# - FORCE: 「top-k 内に target がいれば、それを次トークンとして最大にする」
#          さらに target が multi-token なら、残りtokenを数step lock して最後まで出す
#
# 重要:
# - “target文字列がtop-kにあるなら最大にする” を安定してやるには、
#   「文字列一致」ではなく「targetの token 列（tokenizer.encode）」で誘導するのが最短です。
# - 文字列一致は prefix 依存で揺れる可能性があるので、FORCE は基本 token 列で lock します。
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Union
import time
import json

import torch
from whisper.decoding import BeamSearchDecoder


# ----------------------------
# Config
# ----------------------------
@dataclass
class BiasConfig:
    enable_bias: bool = True
    bias_strength: float = 3.0
    max_suffix_tokens: int = 32
    disable_bias_if_no_prefix: bool = True
    clamp_bias: float = 3.0  # per-token clamp


@dataclass
class LogConfig:
    enable_log: bool = True
    topk: int = 10


@dataclass
class InspectConfig:
    enable_inspect: bool = True
    topk: int = 200
    targets: Optional[List[str]] = None
    store_topk_list: bool = True
    decoded_max_len: int = 40
    ban_token_ids: Optional[List[int]] = None

    # ---- NEW: FORCE controls ----
    enable_force: bool = True
    # hard: 次トークンを完全に固定（それ以外は -inf）
    # soft: 選んだ token の logprob を +force_bonus して勝たせる（完全固定はしない）
    force_mode: str = "hard"  # "hard" or "soft"
    force_bonus: float = 50.0  # soft用 / hardでも上書き前の保険
    # target が multi-token の場合、残り token を lock して最後まで出す
    lock_after_hit: bool = True


# ----------------------------
# Trie (suffix next-token bias)
# ----------------------------
class TrieNode:
    __slots__ = ("children", "terminal")
    def __init__(self):
        self.children: Dict[int, "TrieNode"] = {}
        self.terminal: bool = False


def _safe_encode(tokenizer, text: str) -> List[int]:
    try:
        return tokenizer.encode(text, allowed_special=frozenset())
    except TypeError:
        return tokenizer.encode(text)
    except Exception:
        try:
            return tokenizer.encode(text)
        except Exception:
            return []


def build_trie(tokenizer, terms: List[str], max_len: int) -> TrieNode:
    root = TrieNode()
    for t in (terms or []):
        if not t:
            continue
        ids = _safe_encode(tokenizer, t)
        if 0 < len(ids) <= max_len:
            node = root
            for tok in ids:
                tok = int(tok)
                node = node.children.setdefault(tok, TrieNode())
            node.terminal = True
    return root


def suffix_next_tokens(root: TrieNode, suffix: List[int]) -> List[int]:
    node = root
    for tok in suffix:
        node = node.children.get(int(tok))
        if node is None:
            return []
    return list(node.children.keys())


def compute_next_token_bias(
    root: TrieNode,
    prefix_ids: List[int],
    vocab_size: int,
    cfg: BiasConfig,
    device: torch.device,
) -> torch.Tensor:
    bias = torch.zeros((vocab_size,), dtype=torch.float32, device=device)
    if not prefix_ids or vocab_size <= 0:
        return bias

    base = prefix_ids[-cfg.max_suffix_tokens:]
    L = len(base)

    for i in range(L):
        suf = base[i:L]
        nxts = suffix_next_tokens(root, suf)
        if not nxts:
            continue
        idx = torch.tensor(nxts, dtype=torch.long, device=device)
        bias.index_add_(
            0,
            idx,
            torch.full((len(nxts),), float(cfg.bias_strength), device=device),
        )

    if cfg.clamp_bias and cfg.clamp_bias > 0:
        bias = torch.clamp(bias, 0.0, float(cfg.clamp_bias))
    return bias


# ----------------------------
# Beam prefix fetch (best-effort)
# ----------------------------
def _get_beam_prefixes_from_decoder_attrs(decoder: Any, beam_size: int) -> Optional[List[List[int]]]:
    candidates = ["seqs", "sequences", "sequence", "tokens", "current_sequences"]
    seqs = None
    for name in candidates:
        if hasattr(decoder, name):
            seqs = getattr(decoder, name)
            if seqs is not None:
                break
    if seqs is None:
        return None

    try:
        if isinstance(seqs, torch.Tensor):
            if seqs.ndim >= 2:
                return [seqs[b].tolist() for b in range(min(beam_size, seqs.shape[0]))]
            return None

        if isinstance(seqs, list):
            out: List[List[int]] = []
            for b in range(min(beam_size, len(seqs))):
                s = seqs[b]
                if isinstance(s, torch.Tensor):
                    out.append(s.tolist())
                elif hasattr(s, "tolist"):
                    out.append(s.tolist())
                else:
                    out.append(list(s))
            while len(out) < beam_size:
                out.append([])
            return out
    except Exception:
        return None

    return None


def _tokens_to_text(tokenizer, tokens: List[int]) -> str:
    try:
        return tokenizer.decode(tokens).strip()
    except Exception:
        return ""


def _token_kind(tokenizer, tid: int) -> str:
    ts_begin = getattr(tokenizer, "timestamp_begin", None)
    eot = getattr(tokenizer, "eot", None)
    if ts_begin is not None and eot is not None:
        if ts_begin <= tid < eot:
            return "timestamp"

    specials = ["sot", "eot", "no_speech", "translate", "transcribe"]
    for name in specials:
        v = getattr(tokenizer, name, None)
        if v is not None and tid == int(v):
            return f"special:{name}"
    return "normal"


def _decode_next_with_context(tokenizer, prefix_ids: List[int], tid: int, tail_n: int = 24) -> str:
    """
    prefix末尾 tail_n と次tokenを足したときの「増分テキスト」を返す。
    これ自体は inspect 用。FORCE の判定は基本 token 列で行う。
    """
    tail = prefix_ids[-tail_n:] if prefix_ids else []
    try:
        before = tokenizer.decode(tail)
        after  = tokenizer.decode(tail + [int(tid)])
        if after.startswith(before):
            inc = after[len(before):]
        else:
            inc = after[-80:]
        inc = inc.replace("\n", "\\n")
        return inc
    except Exception:
        return ""


def _is_float_logprobs_tensor(x: Any) -> bool:
    return torch.is_tensor(x) and x.dtype.is_floating_point and x.ndim in (1, 2)


def _is_int_tokens_tensor(x: Any) -> bool:
    return torch.is_tensor(x) and x.dtype in (torch.int32, torch.int64) and x.ndim == 2


# ----------------------------
# Hook install/uninstall
# ----------------------------
def install_beam_hook(
    tokenizer,
    domain_terms: List[str],
    bias_cfg: BiasConfig,
    log_cfg: LogConfig,
    inspect_cfg: Optional[InspectConfig] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Monkeypatch BeamSearchDecoder.update.
    Returns (original_update, state).
    """
    original_update = BeamSearchDecoder.update
    trie_root = build_trie(tokenizer, domain_terms, max_len=bias_cfg.max_suffix_tokens)

    # targets を token列にして、先頭token->候補列の辞書を作る（FORCE/LOCK用）
    target_seqs: Dict[int, List[List[int]]] = {}
    if inspect_cfg is not None and inspect_cfg.targets:
        for t in inspect_cfg.targets:
            if not t:
                continue
            ids = _safe_encode(tokenizer, t)
            if not ids:
                continue
            first = int(ids[0])
            target_seqs.setdefault(first, []).append([int(x) for x in ids])

        # 先頭token同一なら「長い（=より具体的）」を優先するために降順ソート
        for k in list(target_seqs.keys()):
            target_seqs[k].sort(key=lambda seq: len(seq), reverse=True)

    state: Dict[str, Any] = {
        "step": 0,
        "beam_log": [],
        "inspect_log": [],
        "trie_terms": len(domain_terms or []),
        # beamごとに「次に強制したい token 列（残り）」を保持
        "lock_remaining": [],  # set in runtime once beam_size is known
    }

    def _emit_inspect_log(phase: str, logprobs_2d: torch.Tensor, prefixes: Optional[List[List[int]]]):
        if inspect_cfg is None or not inspect_cfg.enable_inspect:
            return
        try:
            beam_size, vocab_size = logprobs_2d.shape
            k = max(1, min(int(inspect_cfg.topk), vocab_size))
            targets = inspect_cfg.targets or []
            ban_ids = set(int(x) for x in (inspect_cfg.ban_token_ids or []))

            for b in range(beam_size):
                row = logprobs_2d[b]
                vals, idxs = torch.topk(row, k=k, dim=-1)
                idxs_l = idxs.detach().cpu().tolist()
                vals_l = vals.detach().cpu().tolist()
                top1_lp = float(vals_l[0]) if vals_l else float("nan")
                pref = prefixes[b] if prefixes is not None and b < len(prefixes) else []

                decoded_list = []
                best_by_target = {}
                hit_any = False

                for rank, (tid, lp) in enumerate(zip(idxs_l, vals_l), start=1):
                    tid_i = int(tid)
                    inc = _decode_next_with_context(tokenizer, pref, tid_i, tail_n=24)

                    if inspect_cfg.store_topk_list:
                        decoded_list.append({
                            "rank": int(rank),
                            "token_id": tid_i,
                            "logprob": float(lp),
                            "gap_to_top1": float(top1_lp - float(lp)),
                            "decoded": inc,
                            "kind": _token_kind(tokenizer, tid_i),
                            "banned": bool(tid_i in ban_ids),
                        })

                    if tid_i in ban_ids:
                        continue

                    if targets:
                        for t in targets:
                            if t and (t in inc):
                                hit_any = True
                                if t not in best_by_target:
                                    best_by_target[t] = {
                                        "target": t,
                                        "rank": int(rank),
                                        "token_id": tid_i,
                                        "logprob": float(lp),
                                        "gap_to_top1": float(top1_lp - float(lp)),
                                        "decoded": inc,
                                        "kind": _token_kind(tokenizer, tid_i),
                                    }

                state["inspect_log"].append({
                    "t_ms": int(time.time() * 1000),
                    "step": int(state["step"]),
                    "beam": int(b),
                    "phase": phase,
                    "topk": int(k),
                    "targets": targets,
                    "hit": bool(hit_any),
                    "hits_best": list(best_by_target.values())[:50],
                    "top1": None if not vals_l else {
                        "token_id": int(idxs_l[0]),
                        "logprob": float(vals_l[0]),
                        "decoded": _decode_next_with_context(tokenizer, pref, int(idxs_l[0]), tail_n=24),
                        "kind": _token_kind(tokenizer, int(idxs_l[0])),
                        "banned": bool(int(idxs_l[0]) in ban_ids),
                    },
                    "topk_list": decoded_list if inspect_cfg.store_topk_list else None,
                })
        except Exception:
            pass

    def _apply_ban(logprobs_2d: torch.Tensor):
        if inspect_cfg is None:
            return
        ban_ids = (inspect_cfg.ban_token_ids or []) if hasattr(inspect_cfg, "ban_token_ids") else []
        if not ban_ids:
            return
        try:
            beam_size, vocab_size = logprobs_2d.shape
            for bid in ban_ids:
                bid = int(bid)
                if 0 <= bid < vocab_size:
                    logprobs_2d[:, bid] = -1e9
        except Exception:
            pass

    def _force_token_for_beam(
        logprobs_2d: torch.Tensor,
        b: int,
        token_id: int,
        mode: str,
        bonus: float,
    ):
        """
        mode="hard": token_id以外を -inf にして確定
        mode="soft": token_idのlogprobを +bonus して勝ちやすく
        """
        try:
            row = logprobs_2d[b]
            if mode == "hard":
                # hard force
                row[:] = -1e9
                row[int(token_id)] = 0.0  # 0にしても他が-1e9なので必ず勝つ
            else:
                row[int(token_id)] = row[int(token_id)] + float(bonus)
        except Exception:
            pass

    def _choose_target_from_topk(
        logprobs_2d: torch.Tensor,
        b: int,
        vocab_size: int,
    ) -> Optional[Tuple[int, List[int]]]:
        """
        top-k の中に target の「先頭token」があれば、それを選ぶ。
        戻り値: (chosen_tid, lock_remaining_ids)
        """
        if inspect_cfg is None or not inspect_cfg.enable_force:
            return None
        if not target_seqs:
            return None

        try:
            # ここは軽くしたいので inspect.topk を流用（小さめ推奨）
            k = max(1, min(int(inspect_cfg.topk), vocab_size))
            row = logprobs_2d[b]
            vals, idxs = torch.topk(row, k=k, dim=-1)
            idxs_l = idxs.detach().cpu().tolist()

            ban_ids = set(int(x) for x in (inspect_cfg.ban_token_ids or []))

            # topk を rank順に見て、target先頭tokenに一致するtidを採用
            for tid in idxs_l:
                tid_i = int(tid)
                if tid_i in ban_ids:
                    continue
                if tid_i in target_seqs:
                    # 先頭tokenが一致する target token列が複数ある場合は最長を採用
                    seq = target_seqs[tid_i][0]  # already sorted by len desc
                    lock_remaining = seq[1:] if (inspect_cfg.lock_after_hit and len(seq) > 1) else []
                    return tid_i, lock_remaining
        except Exception:
            return None

        return None

    def hooked_update(self, *args, **kwargs):
        # ------------------------------------
        # 1) find logprobs tensor in args/kwargs
        # ------------------------------------
        logprobs = None
        logprobs_where: Optional[Tuple[str, Union[int, str]]] = None

        for i, a in enumerate(args):
            if _is_float_logprobs_tensor(a):
                logprobs = a
                logprobs_where = ("args", i)
                break

        if logprobs is None:
            for k, v in kwargs.items():
                if _is_float_logprobs_tensor(v):
                    logprobs = v
                    logprobs_where = ("kwargs", k)
                    break

        if logprobs is None or logprobs_where is None:
            return original_update(self, *args, **kwargs)

        original_was_1d = (logprobs.ndim == 1)
        if original_was_1d:
            logprobs = logprobs.unsqueeze(0)

        beam_size, vocab_size = logprobs.shape

        # lock_remaining 初期化
        if not state["lock_remaining"] or len(state["lock_remaining"]) != beam_size:
            state["lock_remaining"] = [[] for _ in range(beam_size)]

        # ------------------------------------
        # 2) find token prefixes (beam, t)
        # ------------------------------------
        prefixes: Optional[List[List[int]]] = None
        prefix_source: Optional[str] = None

        for i, a in enumerate(args):
            if _is_int_tokens_tensor(a) and a.shape[0] == beam_size:
                prefixes = [a[b].tolist() for b in range(min(beam_size, a.shape[0]))]
                prefix_source = f"args[{i}]"
                break

        if prefixes is None:
            for k, v in kwargs.items():
                if _is_int_tokens_tensor(v) and v.shape[0] == beam_size:
                    prefixes = [v[b].tolist() for b in range(min(beam_size, v.shape[0]))]
                    prefix_source = f"kwargs['{k}']"
                    break

        if prefixes is None:
            prefixes = _get_beam_prefixes_from_decoder_attrs(self, beam_size)
            prefix_source = "decoder_attr" if prefixes is not None else None

        # ------------------------------------
        # 3) BAN first (so inspect sees it too)
        # ------------------------------------
        _apply_ban(logprobs)

        # ------------------------------------
        # 4) inspect pre (optional)
        # ------------------------------------
        _emit_inspect_log("pre_bias", logprobs, prefixes)

        # ------------------------------------
        # 5) FORCE/LOCK (最優先)
        # ------------------------------------
        if inspect_cfg is not None and inspect_cfg.enable_force:
            try:
                for b in range(beam_size):
                    # (a) lock が残っているなら、それを最優先で固定
                    if state["lock_remaining"][b]:
                        next_tid = int(state["lock_remaining"][b].pop(0))
                        _force_token_for_beam(
                            logprobs, b, next_tid,
                            mode=inspect_cfg.force_mode,
                            bonus=inspect_cfg.force_bonus,
                        )
                        continue

                    # (b) lockが無いなら、「top-kにtarget先頭tokenがあれば」それを最大に
                    chosen = _choose_target_from_topk(logprobs, b, vocab_size)
                    if chosen is not None:
                        tid_i, lock_rem = chosen
                        _force_token_for_beam(
                            logprobs, b, tid_i,
                            mode=inspect_cfg.force_mode,
                            bonus=inspect_cfg.force_bonus,
                        )
                        if lock_rem:
                            state["lock_remaining"][b] = list(lock_rem)
            except Exception:
                pass

        # ------------------------------------
        # 6) BIAS (任意): trie suffix next-token bias
        #    ※ FORCE が hard の場合は実質ここは影響しにくい
        # ------------------------------------
        if bias_cfg.enable_bias and vocab_size > 0:
            if prefixes is None and bias_cfg.disable_bias_if_no_prefix:
                pass
            else:
                try:
                    for b in range(beam_size):
                        pref = prefixes[b] if prefixes is not None and b < len(prefixes) else []
                        bvec = compute_next_token_bias(
                            trie_root, pref, vocab_size, bias_cfg, device=logprobs.device
                        )
                        logprobs[b] = logprobs[b] + bvec
                except Exception:
                    pass

        # ------------------------------------
        # 7) inspect post (optional)
        # ------------------------------------
        _emit_inspect_log("post_bias", logprobs, prefixes)

        # ------------------------------------
        # 8) put back modified logprobs into args/kwargs
        # ------------------------------------
        if logprobs_where[0] == "args":
            idx = int(logprobs_where[1])  # type: ignore[arg-type]
            new_args = list(args)
            new_args[idx] = logprobs.squeeze(0) if original_was_1d else logprobs
            args = tuple(new_args)
        else:
            key = str(logprobs_where[1])  # type: ignore[arg-type]
            kwargs[key] = logprobs.squeeze(0) if original_was_1d else logprobs

        out = original_update(self, *args, **kwargs)

        # ------------------------------------
        # 9) beam prefix log (best-effort)
        # ------------------------------------
        if log_cfg.enable_log:
            try:
                pref_list = prefixes or [[] for _ in range(beam_size)]
                k = min(log_cfg.topk, beam_size)
                for b in range(k):
                    pref_ids = pref_list[b] if b < len(pref_list) else []
                    #追加０８２３
                    # デコーダ引数(args/kwargs)またはselfから累積スコアを取得
                    current_score = 0.0
                    try:
                        if len(args) >= 3 and args[2] is not None and b < len(args[2]):
                            current_score = float(args[2][b].item())
                        elif "sum_logprobs" in kwargs and kwargs["sum_logprobs"] is not None and b < len(kwargs["sum_logprobs"]):
                            current_score = float(kwargs["sum_logprobs"][b].item())
                        elif hasattr(self, "sum_logprobs") and self.sum_logprobs is not None and b < len(self.sum_logprobs):
                            current_score = float(self.sum_logprobs[b].item())
                    except Exception:
                        current_score = 0.0                    
                    #追加終わり
                    state["beam_log"].append({
                        "t_ms": int(time.time() * 1000),
                        "step": int(state["step"]),
                        "beam": int(b),
                        #追加０８２３
                        "score": current_score,                  # ★ スコアを追加
                        "sum_logprob": current_score,            # ★ 互換用
                        #追加終わり
                        "logprobs_where": f"{logprobs_where[0]}:{logprobs_where[1]}",
                        "prefix_source": prefix_source,
                        "prefix_len": int(len(pref_ids)),
                        "prefix_tail_ids": pref_ids[-16:],
                        "prefix": _tokens_to_text(tokenizer, pref_ids),
                        "lock_remaining_len": int(len(state["lock_remaining"][b])) if b < len(state["lock_remaining"]) else 0,
                    })
            except Exception:
                pass

        # ★ step は毎回進める（inspect/force を安定させる）
        state["step"] += 1
        return out

    BeamSearchDecoder.update = hooked_update
    return original_update, state


def uninstall_beam_hook(original_update):
    BeamSearchDecoder.update = original_update


def save_beam_log_jsonl(state: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in state.get("beam_log", []):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def save_inspect_log_jsonl(state: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in state.get("inspect_log", []):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
