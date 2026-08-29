"""Configuration-driven resolution of foundry SPICE model decks for ngspice.

NDA / LICENSING NOTE -- READ BEFORE EDITING
-------------------------------------------
Foundry decks (TSMC CRN65GPLUS in particular) are proprietary, NDA-encumbered
data. This module references such a deck BY PATH ONLY. It contains no model
parameter values, and it must never be extended to contain any. Device names
(nch_mac, pch_mac) and corner section names (TT, SS, FF, SF, FS) are public
interface identifiers, not model data, and are therefore fine to name here.

Everything is resolved at runtime from configuration:
  1. environment variable override, per PDK, e.g. ASIC_AI_TSMC65_DECK
  2. configs/eda_tools.yaml  -> pdk.<id>.ngspice
  3. built-in fallback defaults in _BUILTIN_PDKS below

When the deck is absent, every entry point here returns None or False rather
than raising, so a machine without the PDK stays fully functional and the test
suite stays green by skipping.

ngspice deck path constraints (measured, not assumed)
-----------------------------------------------------
  - ngspice cannot open a UNC path in a .lib line: it eats one leading
    separator and the \\\\host\\share prefix is destroyed.
  - ngspice truncates any .lib path at the first SPACE. Quoting does not help.
  - Windows MAX_PATH (260) silently breaks nested includes.
Consequently the deck is copied once into a short, space-free local cache
directory outside the repository (%LOCALAPPDATA%\\asic_ai\\pdk on Windows,
~/.cache/asic_ai/pdk otherwise). THAT CACHE IS A VERBATIM COPY OF NDA FOUNDRY
DATA. It is deliberately placed outside the repo and outside anything git
tracks. Override the location with ASIC_AI_PDK_CACHE; purge it by deleting the
directory.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

# Longest deck path we are willing to hand to ngspice. Windows MAX_PATH is 260
# and the deck resolves nested sections relative to its own directory.
MAX_DECK_PATH_LEN = 200

# An X instance of a foundry MOS wrapper subcircuit, e.g. "nch_mac",
# "pch_25_mac". Matching by shape rather than by an enumerated list keeps zero
# foundry data in the repo while still recognising the wrappers.
_MAC_DEVICE_RE = re.compile(r"^[np]ch(?:_[a-z0-9]+)*_mac$", re.IGNORECASE)

# Built-in defaults. Overridden by configs/eda_tools.yaml when that file has a
# pdk.<id>.ngspice block. Paths only.
_BUILTIN_PDKS: dict[str, dict[str, Any]] = {
    "tsmc65": {
        "enabled": True,
        "description": "TSMC 65nm CRN65GPLUS HSPICE BSIM4 deck (NDA)",
        "deck_env_var": "ASIC_AI_TSMC65_DECK",
        "deck_candidates": [
            "\\\\wsl.localhost\\Alma_EDA\\opt\\eda\\PDK\\CRN65GPNEW\\CRN65GPNEW"
            "\\models\\hspice\\crn65gplus_2d5_lk_v1d0.l",
            "/opt/eda/PDK/CRN65GPNEW/CRN65GPNEW/models/hspice/"
            "crn65gplus_2d5_lk_v1d0.l",
        ],
        "cache_name": "tsmc65",
        # 'hsa' selects ngspice's HSPICE compatibility mode, which the deck's
        # single-quoted expression syntax needs. 'ps' must NOT be used: it
        # downgrades ".lib file section" to a plain include and the deck's
        # self-referential section calls then fail.
        "ngbehavior": "hsa",
        "corner_sections": {
            "tt": "TT", "ss": "SS", "ff": "FF", "sf": "SF", "fs": "FS",
        },
        # Statistical sections. 'stat' defines the global process parameters
        # that the 'MC' section consumes, so it must be included first.
        "mc_sections": ["stat", "MC"],
        # The MC section does not define the local mismatch draws, so the
        # caller supplies them. Standard HSPICE syntax, no foundry values.
        "mc_params": {
            "parn1": "agauss(0,1,1)", "parn2": "agauss(0,1,1)",
            "parp1": "agauss(0,1,1)", "parp2": "agauss(0,1,1)",
        },
        # Every _mac subcircuit is declared l=length w=width, and the deck
        # defines neither symbol anywhere. Without these two dummy globals the
        # subcircuit defaults cannot be evaluated. Instance l=/w= override them,
        # so the values themselves are irrelevant.
        "global_params": {"length": "1u", "width": "1u"},
        # The corner sections draw local mismatch from AGAUSS at every netlist
        # parse, so a nominal run is non-deterministic (about 8 pct spread on
        # drive current) unless sigma is pinned. This is applied per instance.
        "nominal_instance_params": {"sigma": "0"},
        "supply_voltage": 1.0,
        # Nominal PVT attached to each corner name, used when a caller names a
        # corner without giving a voltage and temperature. Matches the corner
        # table already carried in data/pdk_knowledge.py.
        "corner_pvt": {
            "tt": {"voltage": 1.0, "temperature": 27.0},
            "ss": {"voltage": 0.9, "temperature": -40.0},
            "ff": {"voltage": 1.1, "temperature": 125.0},
            "sf": {"voltage": 1.0, "temperature": 27.0},
            "fs": {"voltage": 1.0, "temperature": 27.0},
        },
        "devices": {
            "nmos": "nch_mac", "pmos": "pch_mac",
            "nmos_hvt": "nch_hvt_mac", "pmos_hvt": "pch_hvt_mac",
            "nmos_lvt": "nch_lvt_mac", "pmos_lvt": "pch_lvt_mac",
            "nmos_18": "nch_18_mac", "pmos_18": "pch_18_mac",
            "nmos_25": "nch_25_mac", "pmos_25": "pch_25_mac",
            "nmos_33": "nch_33_mac", "pmos_33": "pch_33_mac",
        },
    },
    "sky130": {
        # Present for completeness: sky130 is Apache-2.0 and therefore the
        # distributable fallback. Resolves only if the release is unpacked
        # locally and pointed at by the env var or the yaml.
        "enabled": True,
        "description": "SkyWater sky130 open PDK (Apache-2.0)",
        "deck_env_var": "ASIC_AI_SKY130_DECK",
        "deck_candidates": [],
        "cache_name": "sky130",
        "ngbehavior": "",
        "corner_sections": {
            "tt": "tt", "ss": "ss", "ff": "ff", "sf": "sf", "fs": "fs",
        },
        "mc_sections": [],
        "mc_params": {},
        "global_params": {},
        "nominal_instance_params": {},
        "supply_voltage": 1.8,
        "corner_pvt": {
            "tt": {"voltage": 1.8, "temperature": 27.0},
            "ss": {"voltage": 1.62, "temperature": -40.0},
            "ff": {"voltage": 1.98, "temperature": 125.0},
            "sf": {"voltage": 1.8, "temperature": 27.0},
            "fs": {"voltage": 1.8, "temperature": 27.0},
        },
        "devices": {
            "nmos": "sky130_fd_pr__nfet_01v8",
            "pmos": "sky130_fd_pr__pfet_01v8",
        },
    },
}

_ALIASES = {
    "tsmc65gp": "tsmc65",
    "crn65gp": "tsmc65",
    "crn65gplus": "tsmc65",
}

_config_cache: Optional[dict[str, dict[str, Any]]] = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _config_path() -> Path:
    env = os.environ.get("ASIC_AI_EDA_CONFIG")
    if env:
        return Path(env)
    return _repo_root() / "configs" / "eda_tools.yaml"


def _load_config(force: bool = False) -> dict[str, dict[str, Any]]:
    """Merge configs/eda_tools.yaml pdk.<id>.ngspice over the built-in defaults."""
    global _config_cache
    if _config_cache is not None and not force:
        return _config_cache

    merged: dict[str, dict[str, Any]] = {
        k: dict(v) for k, v in _BUILTIN_PDKS.items()
    }
    path = _config_path()
    try:
        if path.exists():
            import yaml  # local import: config is optional
            with open(path, "r", encoding="utf-8") as fh:
                doc = yaml.safe_load(fh) or {}
            for pdk_id, block in (doc.get("pdk") or {}).items():
                if not isinstance(block, dict):
                    continue
                ng = block.get("ngspice")
                if not isinstance(ng, dict):
                    continue
                base = dict(merged.get(pdk_id, {}))
                base.update(ng)
                merged[pdk_id] = base
    except Exception as exc:  # pragma: no cover - config must never be fatal
        log.warning("could not read PDK config %s: %s", path, exc)

    _config_cache = merged
    return merged


def reload_config() -> None:
    """Drop the cached configuration. Used by tests that repoint the config."""
    global _config_cache
    _config_cache = None


def list_pdks() -> list[str]:
    """All configured PDK ids."""
    return sorted(_load_config().keys())


def get_pdk_config(pdk_id: str) -> Optional[dict[str, Any]]:
    """Configuration block for a PDK id, or None when it is unknown."""
    if not pdk_id:
        return None
    key = _ALIASES.get(pdk_id.lower(), pdk_id.lower())
    return _load_config().get(key)


# ----------------------------------------------------------------------------
# Deck location and local caching
# ----------------------------------------------------------------------------

def _cache_root() -> Path:
    env = os.environ.get("ASIC_AI_PDK_CACHE")
    if env:
        return Path(env)
    local = os.environ.get("LOCALAPPDATA")
    if local:
        return Path(local) / "asic_ai" / "pdk"
    return Path.home() / ".cache" / "asic_ai" / "pdk"


def _is_ngspice_safe(path: Path) -> bool:
    """True when ngspice can actually open this path from a .lib line."""
    text = str(path)
    if " " in text:
        return False
    if text.startswith("\\\\") or text.startswith("//"):
        return False
    if len(text) > MAX_DECK_PATH_LEN:
        return False
    try:
        text.encode("ascii")
    except UnicodeEncodeError:
        return False
    return True


def find_deck(pdk_id: str) -> Optional[Path]:
    """Locate the source deck for a PDK. Returns None when it is not installed.

    Resolution order: per-PDK env var, then the configured candidate paths.
    """
    cfg = get_pdk_config(pdk_id)
    if not cfg or not cfg.get("enabled", True):
        return None

    env_var = cfg.get("deck_env_var")
    if env_var:
        override = os.environ.get(env_var)
        if override:
            p = Path(override)
            return p if p.is_file() else None

    for cand in cfg.get("deck_candidates") or []:
        try:
            p = Path(cand)
            if p.is_file():
                return p
        except OSError:
            continue
    return None


def ensure_local_deck(pdk_id: str) -> Optional[Path]:
    """Return a deck path ngspice can actually open, copying it if required.

    The source deck usually lives on a UNC share or under a path containing
    spaces, neither of which ngspice can open. In that case it is copied once
    into the local cache (see the module docstring for the NDA implications).
    Returns None when the deck is not installed.
    """
    src = find_deck(pdk_id)
    if src is None:
        return None
    if _is_ngspice_safe(src):
        return src

    cfg = get_pdk_config(pdk_id) or {}
    root = _cache_root()
    if " " in str(root):
        raise RuntimeError(
            "PDK cache directory contains a space, which ngspice cannot open: "
            f"{root}. Set ASIC_AI_PDK_CACHE to a short, space-free directory."
        )
    dest_dir = root / str(cfg.get("cache_name", pdk_id))
    dest = dest_dir / src.name
    if len(str(dest)) > MAX_DECK_PATH_LEN:
        raise RuntimeError(
            f"PDK cache path is too long for ngspice ({len(str(dest))} chars): "
            f"{dest}. Set ASIC_AI_PDK_CACHE to a shorter directory."
        )

    try:
        s = src.stat()
        if dest.is_file():
            d = dest.stat()
            if d.st_size == s.st_size:
                return dest
        dest_dir.mkdir(parents=True, exist_ok=True)
        log.info("caching PDK deck %s -> %s (%.1f MB)", pdk_id, dest,
                 s.st_size / 1e6)
        shutil.copyfile(str(src), str(dest))
        return dest
    except OSError as exc:
        log.warning("could not cache PDK deck for %s: %s", pdk_id, exc)
        return None


def pdk_available(pdk_id: str) -> bool:
    """True when the deck for this PDK is installed and reachable."""
    try:
        return find_deck(pdk_id) is not None
    except Exception:
        return False


# ----------------------------------------------------------------------------
# Netlist prelude construction
# ----------------------------------------------------------------------------

def corner_section(pdk_id: str, corner: str) -> Optional[str]:
    """Library section name for a process corner, e.g. 'ss' -> 'SS'."""
    cfg = get_pdk_config(pdk_id)
    if not cfg:
        return None
    sections = cfg.get("corner_sections") or {}
    return sections.get((corner or "tt").lower())


def corner_pvt(pdk_id: str, corner: str) -> dict[str, float]:
    """Nominal voltage/temperature attached to a corner name, or {}.

    Used when a caller names a corner ('ss') without supplying a full PVT
    triple. It is configuration, not a guess: the values live in the PDK block.
    """
    cfg = get_pdk_config(pdk_id)
    if not cfg:
        return {}
    table = cfg.get("corner_pvt") or {}
    entry = table.get((corner or "").lower())
    return dict(entry) if isinstance(entry, dict) else {}


def ngbehavior(pdk_id: str) -> str:
    """ngspice compatibility mode this deck needs ('' means leave it unset)."""
    cfg = get_pdk_config(pdk_id)
    return str((cfg or {}).get("ngbehavior") or "")


def supply_voltage(pdk_id: str) -> Optional[float]:
    """Nominal core supply for this PDK, or None when unknown."""
    cfg = get_pdk_config(pdk_id)
    if not cfg:
        return None
    v = cfg.get("supply_voltage")
    return float(v) if v is not None else None


def device_names(pdk_id: str) -> dict[str, str]:
    """Public device subcircuit names, e.g. {'nmos': 'nch_mac', ...}."""
    cfg = get_pdk_config(pdk_id)
    return dict((cfg or {}).get("devices") or {})


def lib_lines(pdk_id: str, corner: str = "tt", mc: bool = False) -> list[str]:
    """The exact .lib / .param lines to prepend to a netlist for this PDK.

    Returns [] when the deck is not installed, so a caller can degrade to a
    PDK-free netlist instead of failing.

    With mc=False the process corner section is included and nominal instance
    parameters (sigma=0) should also be applied to instances, giving a
    bit-reproducible nominal run.

    With mc=True the statistical sections are included instead of the corner
    section, together with the caller-supplied unit-normal mismatch draws. The
    per-run seed is applied separately by the adapter via '.option seed='.
    """
    cfg = get_pdk_config(pdk_id)
    if not cfg:
        return []
    deck = ensure_local_deck(pdk_id)
    if deck is None:
        return []
    deck_str = str(deck).replace("\\", "/")

    lines: list[str] = [f"* PDK {pdk_id}: model deck referenced by path only"]
    globals_ = cfg.get("global_params") or {}
    if globals_:
        lines.append(".param " + " ".join(f"{k}={v}" for k, v in globals_.items()))

    if mc:
        mc_params = cfg.get("mc_params") or {}
        if mc_params:
            lines.append(
                ".param " + " ".join(f"{k}={v}" for k, v in mc_params.items())
            )
        sections = cfg.get("mc_sections") or []
        if not sections:
            section = corner_section(pdk_id, corner)
            sections = [section] if section else []
        for sec in sections:
            lines.append(f".lib '{deck_str}' {sec}")
    else:
        section = corner_section(pdk_id, corner)
        if section is None:
            raise ValueError(
                f"PDK '{pdk_id}' has no section for corner '{corner}'. "
                f"Known: {sorted((cfg.get('corner_sections') or {}).keys())}"
            )
        lines.append(f".lib '{deck_str}' {section}")
    return lines


def apply_instance_params(netlist: str, pdk_id: str) -> str:
    """Pin the nominal instance parameters on foundry MOS wrapper instances.

    The TSMC corner sections draw local mismatch from AGAUSS at every netlist
    parse, so an unpinned nominal run varies by several percent between
    identical loads. That would inject invented noise straight into the GRPO
    reward. This appends sigma=0 to every X instance of a *_mac wrapper that
    does not already set sigma= or mismatchflag=. Instances of other devices
    and every non-X line are returned untouched.
    """
    cfg = get_pdk_config(pdk_id)
    if not cfg:
        return netlist
    extra = cfg.get("nominal_instance_params") or {}
    if not extra:
        return netlist

    out: list[str] = []
    for raw in netlist.splitlines():
        line = raw
        stripped = raw.strip()
        if stripped[:1].lower() == "x":
            tokens = stripped.split()
            if any(_MAC_DEVICE_RE.match(t) for t in tokens):
                low = stripped.lower()
                add = [
                    f"{k}={v}" for k, v in extra.items()
                    if f"{k.lower()}=" not in low and "mismatchflag=" not in low
                ]
                if add:
                    line = raw.rstrip() + " " + " ".join(add)
        out.append(line)
    return "\n".join(out)


def describe(pdk_id: str) -> dict[str, Any]:
    """Human/agent readable summary. Never includes model parameter values."""
    cfg = get_pdk_config(pdk_id) or {}
    src = find_deck(pdk_id)
    return {
        "pdk": pdk_id,
        "description": cfg.get("description", ""),
        "available": src is not None,
        "deck_found_at": str(src) if src else None,
        "ngbehavior": cfg.get("ngbehavior", ""),
        "corners": sorted((cfg.get("corner_sections") or {}).keys()),
        "devices": dict(cfg.get("devices") or {}),
        "supply_voltage": cfg.get("supply_voltage"),
        "mc_supported": bool(cfg.get("mc_sections")),
    }
