"""ngspice shared library adapter using KiCad's DLL via ctypes.

Runs ngspice in-process through the shared library API and extracts REAL
numeric vectors with ngGet_Vec_Info. Nothing here fabricates data: if a run
fails, or if the requested data is not present, an NgspiceError is raised.

Why this file is written the way it is
--------------------------------------
ngspice's shared library has three failure modes that all produce confident,
well formed, WRONG numbers. Each one is handled explicitly below:

  1. STALE PLOT. A netlist that fails to parse does not unload the previously
     loaded circuit, and "run" happily re-runs the old one into a fresh plot.
     Return codes stay 0. Mitigation: full teardown ("destroy all" plus
     "remcirc" until empty) at the START of every simulation, then require a
     NEW plot whose name is not 'const'.
  2. SWALLOWED FIRST LINE. SPICE consumes line 1 as the deck title. A model
     generated netlist that starts with a device line loses that device with no
     diagnostic at all, and the answer comes back as clean zeros. Mitigation:
     a title comment is prepended unconditionally, so the caller's first line is
     never eaten.
  3. NOISE PLOT SPLIT. ".noise" creates two plots and ngSpice_CurPlot() returns
     the totals plot, not the spectrum. Mitigation: noise plots are identified
     by their vector names, not by CurPlot().

Process singleton
-----------------
There is exactly one ngspice instance per process. Calling ngSpice_Init again
rebinds the console callbacks globally, which silently blinds any earlier
adapter, and letting the object that owns the CFUNCTYPE trampolines be garbage
collected while ngspice still holds them crashes the process with an access
violation. So the library handle and its callbacks live in a module level
singleton that is never released.

Known limitations, stated plainly
---------------------------------
  - config.timeout is NOT enforced on this path. ngSpice_Command("run") blocks
    synchronously inside the DLL and cannot be interrupted from Python. Cap
    .tran tstop/tstep at the caller.
  - Process corner variation requires a configured PDK deck (see pdk_deck.py).
    Without one, corners() varies temperature and supply only, and says so.
"""
from __future__ import annotations

import ctypes
import logging
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from asic_ai.adapters import measure, pdk_deck
from asic_ai.adapters.base import SimulatorAdapter, AdapterConfig
from asic_ai.tool_interface.schema import (
    DCResult, ACResult, TranResult, NoiseResult, StabilityResult,
    CornerResult, MonteCarloResult, SimParams, PVTCorner, SignalData,
)

log = logging.getLogger(__name__)

# sharedspice.h vector flags. ALWAYS test with a bitwise AND: the observed
# values are 129 and 130 because VF_PERMANENT (128) is set on everything.
VF_REAL = 1
VF_COMPLEX = 2
VF_PERMANENT = 128

# sharedspice.h v_type values, kept for reference / debugging.
VTYPE_TIME = 1
VTYPE_FREQUENCY = 2
VTYPE_VOLTAGE = 3
VTYPE_CURRENT = 4
VTYPE_NOISE_DENSITY = 5

DEFAULT_DLL_PATHS = [
    r"C:\Program Files\KiCad\10.0\bin\ngspice.dll",
    r"C:\Program Files\KiCad\9.0\bin\ngspice.dll",
    r"C:\Program Files\KiCad\8.0\bin\ngspice.dll",
]

NETLIST_TITLE = "* asic-ai netlist"

# Console substrings that mean the run produced nothing trustworthy.
_FATAL_MARKERS = (
    "circuit not parsed",
    "could not find a valid modelname",
    "unknown parameter",
    "undefined parameter",
    "formula() error",
    "error on line",
    "no job (tran, ac, op etc.) defined",
    "run simulation not started",
    ".end statement is missing",
    "aren't any circuits loaded",
    "simulation interrupted due to error",
    "run simulation(s) aborted",
    "doanalyses:",
    "dc solution failed",
    "timestep too small",
    "could not find library file",
    "could not find include file",
    "cannot recover and awaits",
)

# Console substrings that are suspicious but do NOT invalidate the run. ngspice
# falls through gmin stepping and source stepping to a transient operating
# point rescue and often converges anyway; rejecting these throws away good
# runs.
_WARN_MARKERS = (
    "singular matrix",
    "gmin stepping failed",
    "source stepping failed",
    "transient op",
)

_ERROR_LINE_RE = re.compile(r"error on line\s+(\d+)", re.IGNORECASE)

# Temperature attached to a corner NAME when the caller supplies nothing else
# and no PDK is configured. Same convention as data/pdk_knowledge.py. Voltage
# stays unspecified (0.0) because a bare corner name genuinely says nothing
# about the supply, and forcing one would corrupt an IO-voltage netlist.
#
# SIGN-OFF CONVENTION: a corner is the WORST CASE of one thing, so every axis
# must push the same way.
#   SS = slow silicon + LOW supply  + HOT  (125 C): slowest, least drive.
#   FF = fast silicon + HIGH supply + COLD (-40 C): fastest, most drive, worst
#        leakage-free overdrive and worst hold time.
# These temperatures used to be the other way round, which cancelled part of
# the corner spread against itself: SS ran slow-process/low-VDD at the cold
# (fast) temperature and FF ran fast-process/high-VDD at the hot (slow) one, so
# the reported corner-to-corner spread understated the real one.
_GENERIC_CORNER_TEMPS = {
    "tt": 27.0, "ss": 125.0, "ff": -40.0, "sf": 27.0, "fs": 27.0,
}


class NgspiceError(RuntimeError):
    """A simulation failed, or produced no usable data."""


def find_ngspice_dll() -> str | None:
    """Auto-detect ngspice DLL from KiCad installation."""
    env = os.environ.get("ASIC_AI_NGSPICE_DLL")
    if env and Path(env).exists():
        return env
    for p in DEFAULT_DLL_PATHS:
        if Path(p).exists():
            return p
    return None


# ---------------------------------------------------------------------------
# ctypes plumbing (exactly per sharedspice.h)
# ---------------------------------------------------------------------------

class ngcomplex_t(ctypes.Structure):
    _fields_ = [
        ("cx_real", ctypes.c_double),
        ("cx_imag", ctypes.c_double),
    ]


class vector_info(ctypes.Structure):
    _fields_ = [
        ("v_name", ctypes.c_char_p),
        ("v_type", ctypes.c_int),
        ("v_flags", ctypes.c_short),
        ("v_realdata", ctypes.POINTER(ctypes.c_double)),
        ("v_compdata", ctypes.POINTER(ngcomplex_t)),
        ("v_length", ctypes.c_int),
    ]


pvector_info = ctypes.POINTER(vector_info)

_SEND_CHAR = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p
)
_SEND_STAT = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p
)
_CTRL_EXIT = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_int, ctypes.c_bool, ctypes.c_bool,
    ctypes.c_int, ctypes.c_void_p
)


def _charpp_to_list(ptr) -> list[str]:
    """Decode a NULL terminated char** returned by ngspice."""
    out: list[str] = []
    if not ptr:
        return out
    i = 0
    while True:
        item = ptr[i]
        if not item:
            break
        out.append(item.decode("utf-8", errors="replace"))
        i += 1
        if i > 100000:  # paranoia against a missing terminator
            break
    return out


def _clean_console_line(line: str) -> str:
    """Strip the optional 'stdout '/'stderr ' tag and CR that ngspice emits."""
    s = line.replace("\r", "").strip()
    for prefix in ("stdout ", "stderr "):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    return s.strip()


class _NgspiceLibrary:
    """Process wide singleton wrapper around ngspice.dll.

    Never instantiate directly, use _NgspiceLibrary.instance(). The callbacks
    are stored on the instance and the instance is stored in a module global,
    so neither can be collected while ngspice still holds pointers to them.
    """

    _instance: Optional["_NgspiceLibrary"] = None

    def __init__(self, dll_path: str):
        self.dll_path = dll_path
        self.console: list[str] = []
        self.exit_events: list[tuple[int, bool, bool]] = []

        dll_dir = Path(dll_path).parent
        lib_dir = dll_dir.parent / "lib" / "ngspice"
        os.environ["SPICE_LIB_DIR"] = str(lib_dir)
        if str(dll_dir) not in os.environ.get("PATH", ""):
            os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory") and dll_dir.exists():
            try:
                self._dll_dir_cookie = os.add_dll_directory(str(dll_dir))
            except (OSError, AttributeError):  # pragma: no cover
                self._dll_dir_cookie = None

        self.dll = ctypes.CDLL(dll_path)
        self._declare_signatures()
        self._install_callbacks()

    # -- setup -------------------------------------------------------------

    def _declare_signatures(self) -> None:
        d = self.dll
        d.ngSpice_Init.restype = ctypes.c_int
        d.ngSpice_Init.argtypes = [
            _SEND_CHAR, _SEND_STAT, _CTRL_EXIT,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ]
        d.ngSpice_Command.restype = ctypes.c_int
        d.ngSpice_Command.argtypes = [ctypes.c_char_p]
        d.ngSpice_Circ.restype = ctypes.c_int
        d.ngSpice_Circ.argtypes = [ctypes.POINTER(ctypes.c_char_p)]
        d.ngSpice_CurPlot.restype = ctypes.c_char_p
        d.ngSpice_CurPlot.argtypes = []
        d.ngSpice_AllPlots.restype = ctypes.POINTER(ctypes.c_char_p)
        d.ngSpice_AllPlots.argtypes = []
        d.ngSpice_AllVecs.restype = ctypes.POINTER(ctypes.c_char_p)
        d.ngSpice_AllVecs.argtypes = [ctypes.c_char_p]
        # Without an explicit restype ctypes truncates this 64 bit pointer to
        # a 32 bit int and every vector read lands on a garbage address.
        d.ngGet_Vec_Info.restype = pvector_info
        d.ngGet_Vec_Info.argtypes = [ctypes.c_char_p]
        d.ngSpice_running.restype = ctypes.c_bool
        d.ngSpice_running.argtypes = []

    def _install_callbacks(self) -> None:
        @_SEND_CHAR
        def on_char(msg, ident, userdata):
            try:
                self.console.append(msg.decode("utf-8", errors="replace"))
            except Exception:
                pass
            return 0

        @_SEND_STAT
        def on_stat(msg, ident, userdata):
            return 0

        @_CTRL_EXIT
        def on_exit(status, immediate, quit_, ident, userdata):
            self.exit_events.append((int(status), bool(immediate), bool(quit_)))
            return 0

        # Held forever: ngspice keeps raw pointers to these trampolines.
        self._callbacks = (on_char, on_stat, on_exit)
        rc = self.dll.ngSpice_Init(on_char, on_stat, on_exit, None, None, None, None)
        if rc != 0:
            raise RuntimeError(f"ngSpice_Init failed: {rc}")

    @classmethod
    def instance(cls, dll_path: str) -> "_NgspiceLibrary":
        if cls._instance is None:
            cls._instance = _NgspiceLibrary(dll_path)
        elif os.path.normcase(cls._instance.dll_path) != os.path.normcase(dll_path):
            log.warning(
                "ngspice already initialised from %s; ignoring request for %s "
                "(one shared library instance per process)",
                cls._instance.dll_path, dll_path,
            )
        return cls._instance

    @classmethod
    def reload(cls) -> "_NgspiceLibrary":
        """Unload the wedged DLL and load a fresh one.

        A deck ngspice cannot parse (the 824g eval's first bad deck put an
        AC-only output name in a context that made it print 'Undefined
        parameter [vdb]') can leave the engine in 'cannot recover and awaits
        to be reset or detached'. In that state EVERY later load reports
        \"there aren't any circuits loaded\": one bad deck from the model
        poisoned all 76 remaining eval tasks. ngspice's own documentation
        offers exactly one way back -- detach the library -- so this frees
        the module handle and reinitialises. Windows-only by construction
        (this whole adapter drives ngspice.dll through ctypes).
        """
        if cls._instance is None:
            raise RuntimeError("reload() called before instance()")
        path = cls._instance.dll_path
        handle = cls._instance.dll._handle
        cls._instance = None
        try:
            ctypes.windll.kernel32.FreeLibrary(
                ctypes.c_void_p(handle))
        except OSError as exc:  # pragma: no cover
            log.warning("FreeLibrary on wedged ngspice failed: %s", exc)
        log.warning("ngspice engine was unrecoverable; DLL reloaded")
        cls._instance = _NgspiceLibrary(path)
        return cls._instance

    # -- primitives --------------------------------------------------------

    def clear_console(self) -> None:
        self.console.clear()

    def console_lines(self) -> list[str]:
        return [_clean_console_line(x) for x in self.console]

    def command(self, cmd: str) -> int:
        return int(self.dll.ngSpice_Command(cmd.encode("utf-8")))

    def load_circuit(self, lines: Sequence[str]) -> int:
        """Load a netlist from memory. No temp file, no path length limits."""
        buf = (ctypes.c_char_p * (len(lines) + 1))()
        for i, line in enumerate(lines):
            buf[i] = line.encode("utf-8")
        buf[len(lines)] = None
        return int(self.dll.ngSpice_Circ(buf))

    def cur_plot(self) -> Optional[str]:
        raw = self.dll.ngSpice_CurPlot()
        if not raw:
            return None
        return raw.decode("utf-8", errors="replace")

    def all_plots(self) -> list[str]:
        return _charpp_to_list(self.dll.ngSpice_AllPlots())

    def all_vecs(self, plot: str) -> list[str]:
        return _charpp_to_list(self.dll.ngSpice_AllVecs(plot.encode("utf-8")))

    def vector(self, fq_name: str) -> Optional[list]:
        """Read one vector by its fully qualified '{plot}.{vector}' name.

        Returns a list of float for a real vector, a list of complex for a
        complex one, or None when the vector does not exist.
        """
        ptr = self.dll.ngGet_Vec_Info(fq_name.encode("utf-8"))
        if not ptr:
            return None
        info = ptr.contents
        n = int(info.v_length)
        if n <= 0:
            return []
        flags = int(info.v_flags)
        if flags & VF_COMPLEX:
            if not info.v_compdata:
                return None
            data = info.v_compdata
            return [complex(data[i].cx_real, data[i].cx_imag) for i in range(n)]
        if not info.v_realdata:
            return None
        data = info.v_realdata
        return [float(data[i]) for i in range(n)]

    def teardown(self) -> None:
        """Return ngspice to a clean slate.

        'destroy all' frees every plot and resets the plot name counter to 1.
        'remcirc' removes ONE circuit, so it has to be repeated: 'source' and
        ngSpice_Circ both stack, and leaving 300 circuits loaded costs 5x
        throughput and leaks about 59 kB per simulation.
        """
        self.command("destroy all")
        for _ in range(256):
            self.clear_console()
            self.command("remcirc")
            text = " ".join(self.console_lines()).lower()
            if "no circuit loaded" in text or "aren't any circuits loaded" in text:
                break
        self.clear_console()


# ---------------------------------------------------------------------------
# Run bookkeeping
# ---------------------------------------------------------------------------

@dataclass
class _SimRun:
    """Everything one ngspice load-and-run produced."""

    plots: dict[str, dict[str, list]] = field(default_factory=dict)
    plot_order: list[str] = field(default_factory=list)
    console: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    netlist: str = ""

    def find_plot(self, *prefixes: str) -> Optional[str]:
        """First new plot whose name starts with one of the given prefixes."""
        for name in self.plot_order:
            low = name.lower()
            for pre in prefixes:
                if low.startswith(pre):
                    return name
        return None

    def vectors(self, plot: Optional[str]) -> dict[str, list]:
        if not plot:
            return {}
        return self.plots.get(plot, {})


# ngspice names the vector it swept "<kind>-sweep" and puts it LAST in the
# ngSpice_AllVecs list. Verified against the KiCad 10 DLL:
#
#   .dc v1 0 1.8 0.45   -> ['v1#branch','b','a','vdd','v-sweep']   v_type 3
#   .dc i1 0 1m 0.25m   -> ['out','i-sweep']                       v_type 4
#   .dc temp -40 120 40 -> ['v1#branch','out','vdd','temp-sweep']  v_type 14
#   .dc r2 5k 20k 5k    -> ['v1#branch','out','vdd','res-sweep']   v_type 15
#
# Only "v-sweep" used to be recognised, and the fallback picked the LONGEST
# vector -- but in a DC sweep every vector has the same length, so max() just
# returned whichever key ngspice happened to list first. That silently put a
# branch current in AMPERES on the x axis of a temperature sweep, and
# transposed a current sweep completely. There is no length heuristic that can
# work here; the designated name is the only signal there is.
_DC_SWEEP_NAMES = ("v-sweep", "i-sweep", "temp-sweep", "res-sweep")


def _dc_sweep_axis(vecs: dict[str, list]) -> Optional[str]:
    """Name of the vector ngspice designated as the DC sweep axis, or None."""
    for name in _DC_SWEEP_NAMES:
        if name in vecs:
            return name
    for name in vecs:
        if name.lower().endswith("-sweep"):
            return name
    return None


def _axis_reversals(x: Sequence[float]) -> int:
    """How many times a sweep axis changes direction.

    Zero for any single sweep, in either direction. A NESTED '.dc' -- ngspice
    flattens the two loops into one vector -- reverses once per outer step.
    """
    signs = [1 if b > a else (-1 if b < a else 0)
             for a, b in zip(x, list(x)[1:])]
    signs = [s for s in signs if s]
    return sum(1 for a, b in zip(signs, signs[1:]) if a != b)


def _is_monotone_axis(x: Sequence[float]) -> bool:
    """True when a sweep axis never doubles back on itself."""
    return _axis_reversals(x) == 0


def _real(values: Sequence) -> list[float]:
    """Real projection of a vector that may be real or complex."""
    out: list[float] = []
    for v in values:
        if isinstance(v, complex):
            out.append(float(v.real))
        else:
            out.append(float(v))
    return out


def _is_complex(values: Sequence) -> bool:
    return bool(values) and isinstance(values[0], complex)


def _finite_pairs(x: Sequence[float], y: Sequence[float],
                  label: str = "") -> tuple[list[float], list[float], int]:
    """(x, y, dropped) keeping only the samples where BOTH are finite.

    A pydantic result is a SERIALIZATION boundary. The metric layer is right to
    use -inf for a magnitude that is genuinely zero and NaN for a phase that
    genuinely does not exist -- those are ordered, arithmetic-safe values and
    every scan in measure.py handles them. But json.dumps writes them as
    -Infinity and NaN, which is not JSON: json.loads with a strict
    parse_constant, jq, JavaScript and the HuggingFace datasets loader all
    reject it, and rl_env hands exactly that text to the model and on into
    trajectories and SFT files. On an ordinary deck -- one AC input and one
    DC-only supply rail -- three of seven AC signals are non-finite at all 61
    samples, so this is the common case and not an edge case.

    Dropping the sample rather than flooring it keeps the pair honest: the
    signal is reported at the frequencies where it is defined, with its own
    x_values, and nothing downstream has to guess what -6000 dB meant.
    """
    n = min(len(x), len(y))
    xs: list[float] = []
    ys: list[float] = []
    for i in range(n):
        xv, yv = float(x[i]), float(y[i])
        if math.isfinite(xv) and math.isfinite(yv):
            xs.append(xv)
            ys.append(yv)
    dropped = n - len(xs)
    if dropped and label:
        log.warning(
            "%s: %d of %d samples are not finite and are dropped from the "
            "result; a non-finite float cannot be serialised as JSON and would "
            "reach a trajectory and an SFT file as -Infinity or NaN",
            label, dropped, n,
        )
    return xs, ys, dropped


# ---------------------------------------------------------------------------
# Netlist helpers
# ---------------------------------------------------------------------------

_SUPPLY_NODES = ("vdd", "vcc", "vdda", "vddd", "avdd", "dvdd", "vpwr", "vsup")
_GROUND_NODES = ("0", "gnd", "gnd!", "vss", "vgnd")


def netlist_text(source: Any) -> str:
    """Accept either netlist TEXT or a PATH and return the netlist text.

    The frozen SimulatorInterface declares `netlist: str` (text) and
    training/rl_env.py passes text, while the scripts and tests in this repo
    pass a file path. Both are supported. Detection is deliberately
    conservative: anything multi-line, anything longer than a legal Windows
    path, and anything that is not an existing file is treated as text.
    """
    if isinstance(source, Path):
        return source.read_text(encoding="utf-8")
    if not isinstance(source, str):
        raise TypeError(f"netlist must be str or Path, got {type(source).__name__}")
    if "\n" in source or "\r" in source or "\x00" in source:
        return source
    if len(source) > 260:
        return source
    try:
        candidate = Path(source)
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")
    except (OSError, ValueError):
        pass
    return source


def _param(params: Any, name: str, default: Any = None) -> Any:
    """Read a field from a SimParams, a plain dict, or None."""
    if params is None:
        return default
    if isinstance(params, dict):
        value = params.get(name, default)
        return default if value is None else value
    value = getattr(params, name, default)
    return default if value is None else value


def _options(params: Any) -> dict:
    opts = _param(params, "options", None)
    if isinstance(opts, dict):
        return opts
    if isinstance(params, dict):
        # RL passes a flat args dict; treat it as the options bag too.
        return params
    return {}


def _code_lines(netlist: str) -> list[str]:
    """Netlist lines with comments and blanks removed, lowercased, stripped."""
    out = []
    for raw in netlist.splitlines():
        s = raw.strip()
        if not s or s.startswith("*") or s.startswith(";"):
            continue
        out.append(s.lower())
    return out


def _has_card(netlist: str, *cards: str) -> bool:
    for line in _code_lines(netlist):
        for card in cards:
            if line.startswith(card):
                return True
    return False


def _has_control_run(netlist: str) -> bool:
    """True when the deck carries a .control block that runs the sim itself."""
    inside = False
    for line in _code_lines(netlist):
        if line.startswith(".control"):
            inside = True
            continue
        if line.startswith(".endc"):
            inside = False
            continue
        if inside and (line == "run" or line.startswith("run ")):
            return True
    return False


def _strip_lines(netlist: str, *prefixes: str) -> str:
    keep = []
    for raw in netlist.splitlines():
        low = raw.strip().lower()
        if any(low.startswith(p) for p in prefixes):
            continue
        keep.append(raw)
    return "\n".join(keep)


def _body_without_end(netlist: str) -> list[str]:
    """Netlist lines with any bare '.end' removed ('.ends'/'.endc' kept)."""
    keep = []
    for raw in netlist.splitlines():
        if raw.strip().lower() == ".end":
            continue
        keep.append(raw.rstrip())
    return keep


_VSOURCE_RE = re.compile(
    r"^(?P<name>[vV][\w.:$#\-]*)\s+(?P<pos>\S+)\s+(?P<neg>\S+)\s*(?P<rest>.*)$"
)
_DC_VALUE_RE = re.compile(r"(?i)\b(dc)\b\s*(=?)\s*(?P<val>[-+]?[\d.]+(?:e[-+]?\d+)?\w*)")
_LEADING_NUM_RE = re.compile(r"^(?P<val>[-+]?[\d.]+(?:e[-+]?\d+)?\w*)")


def set_supply_voltage(netlist: str, voltage: float,
                       source: Optional[str] = None) -> tuple[str, int]:
    """Rewrite the DC value of the supply voltage source(s).

    When `source` is given only that source is rewritten. Otherwise every
    independent voltage source from a supply-looking node to ground is
    rewritten. Returns (netlist, number_of_sources_changed) so the caller can
    fail loudly instead of silently simulating the wrong supply.
    """
    changed = 0
    out: list[str] = []
    for raw in netlist.splitlines():
        stripped = raw.strip()
        m = _VSOURCE_RE.match(stripped)
        if not m or stripped.startswith("*"):
            out.append(raw)
            continue
        name = m.group("name")
        pos = m.group("pos").lower()
        neg = m.group("neg").lower()
        rest = m.group("rest")
        if source is not None:
            if name.lower() != source.lower():
                out.append(raw)
                continue
        else:
            if pos not in _SUPPLY_NODES or neg not in _GROUND_NODES:
                out.append(raw)
                continue
        value = f"{voltage:g}"
        dc_match = _DC_VALUE_RE.search(rest)
        if dc_match:
            new_rest = rest[:dc_match.start("val")] + value + rest[dc_match.end("val"):]
        else:
            num_match = _LEADING_NUM_RE.match(rest.strip())
            if num_match and not rest.strip().lower().startswith(("ac", "pulse", "sin", "pwl", "exp")):
                tail = rest.strip()[num_match.end("val"):]
                new_rest = f"DC {value}{tail}"
            else:
                new_rest = f"DC {value} {rest}".strip()
        out.append(f"{name} {m.group('pos')} {m.group('neg')} {new_rest}".rstrip())
        changed += 1
    return "\n".join(out), changed


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class NgspiceSharedAdapter(SimulatorAdapter):
    """ngspice adapter using the shared library (DLL) via ctypes.

    Every analysis method accepts either netlist TEXT or a PATH as its first
    argument, and `params` may be a SimParams, a plain dict of arguments (what
    training/rl_env.py passes), or None.

    AC signal naming scheme (kept consistent everywhere in this class):
      ACResult.signals["vdb(<vector>)"] -- magnitude in dB, x = frequency [Hz]
      ACResult.signals["vp(<vector>)"]  -- phase in degrees, x = frequency [Hz]
    The names follow ngspice's own vdb()/vp() post-processing functions, so
    "vdb(out)" is the dB magnitude of node 'out' and "vp(v1#branch)" is the
    phase of the current through V1.
    """

    def __init__(self, config: AdapterConfig, pdk: Optional[str] = None,
                 corner: str = "tt"):
        super().__init__(config)
        self._output: list[str] = []
        self._last_netlist: str = ""

        dll_path = config.binary_path
        if not dll_path or not Path(dll_path).exists():
            dll_path = find_ngspice_dll()
        if not dll_path:
            raise FileNotFoundError(
                "ngspice.dll not found. Install KiCad, set binary_path, or set "
                "ASIC_AI_NGSPICE_DLL."
            )

        self._ng = _NgspiceLibrary.instance(dll_path)
        self._lib = self._ng.dll
        self.dll_path = dll_path
        self.pdk = pdk or os.environ.get("ASIC_AI_PDK") or None
        self.corner = corner
        self.last_warnings: list[str] = []
        if self.pdk and not pdk_deck.pdk_available(self.pdk):
            log.warning("PDK '%s' requested but its model deck was not found; "
                        "netlists will be simulated without it", self.pdk)
        log.info("ngspice shared library loaded: %s (pdk=%s)", dll_path, self.pdk)

    # -- PDK -------------------------------------------------------------

    def pdk_ready(self, pdk: Optional[str] = None) -> bool:
        """True when a PDK model deck is configured AND present on this machine."""
        target = pdk or self.pdk
        return bool(target) and pdk_deck.pdk_available(target)

    def _pdk_selection(self, params: Any) -> tuple[Optional[str], str]:
        opts = _options(params)
        pdk = opts.get("pdk", self.pdk)
        corner = opts.get("corner", self.corner) or "tt"
        if pdk and not pdk_deck.pdk_available(pdk):
            return None, corner
        return pdk, corner

    # -- netlist assembly -------------------------------------------------

    def _analysis_card(self, kind: str, params: Any, netlist: str) -> Optional[str]:
        """Synthesise the analysis directive when the deck does not carry one.

        Every alternative below that is listed FIRST is a name from the frozen
        tool contract (sim.ac: start_freq/stop_freq/points_per_decade,
        sim.tran: stop_time/step_time, sim.noise: output_node/input_source).
        The generic start/stop/step/points spellings are kept as fallbacks.
        Until 2026-09-01 only the generic spellings were read, so a model
        calling sim.ac EXACTLY as the contract documents it was still refused
        with "params do not supply start/stop frequency" -- the fifth
        contract-vs-implementation split in this repo, and the only one where
        the contract side was the one nobody had ever exercised.
        """
        opts = _options(params)

        def p(*names: str) -> Any:
            for n in names:
                v = _param(params, n)
                if v is not None:
                    return v
            return None

        step = p("step")
        points = p("points")
        sweep = p("sweep_var")

        if kind == "dc":
            start, stop = p("start"), p("stop")
            if sweep and start is not None and stop is not None and step:
                return f".dc {sweep} {float(start):g} {float(stop):g} {float(step):g}"
            return ".op"
        if kind == "ac":
            start = p("start_freq", "start")
            stop = p("stop_freq", "stop")
            n = p("points_per_decade", "points")
            if start is None or stop is None:
                raise NgspiceError(
                    "ac(): the netlist has no .ac card and params do not supply "
                    "start/stop frequency. Refusing to invent a sweep."
                )
            variant = str(opts.get("ac_variant", "dec"))
            return f".ac {variant} {int(n or 10)} {float(start):g} {float(stop):g}"
        if kind == "tran":
            stop = p("stop_time", "stop")
            tstep = p("step_time", "step")
            if stop is None:
                raise NgspiceError(
                    "tran(): the netlist has no .tran card and params do not "
                    "supply a stop time. Refusing to invent a sweep."
                )
            tstep = tstep or (float(stop) / 1000.0)
            return f".tran {float(tstep):g} {float(stop):g}"
        if kind == "noise":
            out_node = opts.get("output_node") or opts.get("output") or opts.get("out")
            src = opts.get("input_source") or opts.get("source")
            start = p("start_freq", "start")
            stop = p("stop_freq", "stop")
            if not out_node or not src or start is None or stop is None:
                raise NgspiceError(
                    "noise(): the netlist has no .noise card and params do not "
                    "supply output/source/start/stop. Refusing to invent one."
                )
            n = p("points_per_decade", "points")
            return (f".noise v({out_node}) {src} dec {int(n or 100)} "
                    f"{float(start):g} {float(stop):g}")
        return None

    def _build_netlist(self, source: Any, kind: str, params: Any = None,
                       pdk: Optional[str] = None, corner: str = "tt",
                       temperature: Optional[float] = None,
                       supply: Optional[float] = None,
                       mc: bool = False, seed: Optional[int] = None) -> str:
        body = netlist_text(source)

        if supply is not None:
            src_name = _options(params).get("supply_source")
            body, n_changed = set_supply_voltage(body, supply, src_name)
            if n_changed == 0:
                raise NgspiceError(
                    f"cannot apply supply voltage {supply} V: no independent "
                    "voltage source from a supply node to ground was found. "
                    "Name it explicitly via options['supply_source']."
                )

        prelude: list[str] = []
        if pdk:
            if not mc:
                body = pdk_deck.apply_instance_params(body, pdk)
            prelude = pdk_deck.lib_lines(pdk, corner=corner, mc=mc)

        directives: list[str] = []
        if temperature is not None:
            body = _strip_lines(body, ".temp", ".option temp")
            directives.append(f".temp {temperature:g}")
        if seed is not None:
            body = _strip_lines(body, ".option seed")
            directives.append(f".option seed={int(seed)}")

        cards = {
            "dc": (".dc", ".op"),
            "ac": (".ac",),
            "tran": (".tran",),
            "noise": (".noise",),
        }.get(kind, ())
        if cards and not _has_card(body, *cards):
            card = self._analysis_card(kind, params, body)
            if card:
                directives.append(card)

        # A title comment is prepended unconditionally so that SPICE cannot
        # swallow the caller's first device line. See module docstring, item 2.
        lines = [NETLIST_TITLE]
        lines.extend(prelude)
        lines.extend(_body_without_end(body))
        lines.extend(directives)
        lines.append(".end")
        return "\n".join(lines)

    # -- execution --------------------------------------------------------

    def _execute(self, netlist: str, pdk: Optional[str]) -> _SimRun:
        ng = self._ng
        ng.teardown()

        behavior = pdk_deck.ngbehavior(pdk) if pdk else ""
        if behavior:
            ng.command(f"set ngbehavior={behavior}")
        else:
            ng.command("unset ngbehavior")

        before = set(ng.all_plots())
        ng.clear_console()

        lines = netlist.splitlines()
        ng.load_circuit(lines)
        if not _has_control_run(netlist):
            ng.command("run")

        console = ng.console_lines()
        self._output = list(console)

        lowered = [c.lower() for c in console]
        fatal = [c for c, low in zip(console, lowered)
                 if any(m in low for m in _FATAL_MARKERS)]
        warns = [c for c, low in zip(console, lowered)
                 if any(m in low for m in _WARN_MARKERS)]

        cur = ng.cur_plot()
        after = ng.all_plots()
        new_plots = [p for p in after if p not in before and p != "const"]

        if fatal or not new_plots or cur in (None, "", "const"):
            # A wedged engine poisons every FUTURE run in this process (each
            # would fail with "there aren't any circuits loaded" no matter how
            # good its deck is), so recover it before reporting this failure.
            # The failure itself is still raised: recovery must never turn a
            # bad deck into a silent retry.
            if any("cannot recover and awaits" in low for low in lowered):
                self._ng = _NgspiceLibrary.reload()
                self._lib = self._ng.dll
            raise NgspiceError(self._failure_message(fatal, cur, new_plots, pdk))

        # Remembered so measure_idd can tell a 0 V sense source from a rail.
        # Only ever applied to a result whose element names still match it.
        self._last_netlist = netlist

        run = _SimRun(plot_order=list(new_plots), console=console,
                      warnings=warns, netlist=netlist)
        for plot in new_plots:
            vecs: dict[str, list] = {}
            for name in ng.all_vecs(plot):
                data = ng.vector(f"{plot}.{name}")
                if data is None:
                    continue
                vecs[name] = data
            run.plots[plot] = vecs

        if warns:
            log.warning("ngspice reported %d convergence warning(s); first: %s",
                        len(warns), warns[0])
        self.last_warnings = warns
        return run

    def _failure_message(self, fatal: list[str], cur: Optional[str],
                         new_plots: list[str], pdk: Optional[str]) -> str:
        """Build a failure message, redacting deck text when a PDK is loaded.

        ngspice quotes offending deck expressions verbatim in its diagnostics.
        With an NDA foundry deck loaded those lines are model data, so only the
        marker category is reported, never the raw text.
        """
        if pdk and pdk_deck.pdk_available(pdk):
            categories = sorted({
                m for line in fatal for m in _FATAL_MARKERS if m in line.lower()
            })
            lines = sorted({m.group(1) for line in fatal
                            for m in [_ERROR_LINE_RE.search(line)] if m})
            detail = ", ".join(categories) if categories else "no new plot produced"
            where = f" at netlist line(s) {', '.join(lines)}" if lines else ""
            return (
                f"ngspice simulation failed with PDK '{pdk}' [{detail}]{where}. "
                "Diagnostic text is withheld because it can quote proprietary "
                "model deck content; re-run without the PDK to see it. Common "
                "causes: a device geometry outside the model bins (an IO device "
                "at a core gate length), or a device name that is not a "
                "subcircuit wrapper."
            )
        if fatal:
            return "ngspice simulation failed: " + " | ".join(fatal[:5])
        return (
            f"ngspice produced no new plot (cur_plot={cur!r}, new={new_plots}). "
            "The netlist most likely failed to parse or declared no analysis."
        )

    def _run(self, source: Any, kind: str, params: Any = None,
             temperature: Optional[float] = None, supply: Optional[float] = None,
             mc: bool = False, seed: Optional[int] = None,
             corner_override: Optional[str] = None) -> _SimRun:
        pdk, corner = self._pdk_selection(params)
        if corner_override:
            corner = corner_override
        netlist = self._build_netlist(
            source, kind, params, pdk=pdk, corner=corner,
            temperature=temperature, supply=supply, mc=mc, seed=seed,
        )
        return self._execute(netlist, pdk)

    # -- result builders ---------------------------------------------------

    @staticmethod
    def _build_dc(run: _SimRun) -> Optional[DCResult]:
        op_plot = run.find_plot("op")
        dc_plot = run.find_plot("dc")
        if not op_plot and not dc_plot:
            return None

        op_points: dict[str, float] = {}
        for name, data in run.vectors(op_plot).items():
            if len(data) == 1:
                v = _real(data)[0]
                if not math.isfinite(v):
                    log.warning(
                        "operating point %r is %r, which is not a number and "
                        "cannot be serialised as JSON; it is left out of the "
                        "result. The run did not converge here.", name, v,
                    )
                    continue
                op_points[name] = v

        sweeps: dict[str, SignalData] = {}
        vecs = run.vectors(dc_plot)
        if vecs:
            x_name = _dc_sweep_axis(vecs)
            if x_name is None:
                log.warning(
                    "DC plot has no '*-sweep' vector, so ngspice designated no "
                    "sweep axis; sweeps are dropped rather than plotted against "
                    "a guessed axis (vectors: %s)", sorted(vecs),
                )
            else:
                x_values = _real(vecs[x_name])
                # A NESTED sweep -- '.dc V1 0 1 0.5 V2 0 1 0.5' -- is written by
                # ngspice as ONE flat vector per signal, with the inner sweep
                # repeated once per outer value, so the axis reads
                # [0, 0.5, 1, 0, 0.5, 1, 0, 0.5, 1]. The DATA is complete and a
                # device I-V family is a legitimate analysis, so it is kept --
                # but it is a 2-D grid flattened, not a curve, and nothing said
                # so. Everything that interpolates along it, or takes a span
                # across it as a signal excursion, is silently answering a
                # different question. Flagged here and refused downstream (see
                # spec_extract, which will not report an output_swing across a
                # grid), because a plain max-minus-min over a nested sweep
                # includes the outer variable's own excursion.
                reversals = _axis_reversals(x_values)
                if reversals:
                    log.warning(
                        "DC sweep axis %r doubles back %d time(s) over %d "
                        "points: this is a NESTED '.dc' -- two swept sources -- "
                        "flattened into one vector, so every signal here is a "
                        "2-D grid and NOT a curve. The samples are kept, but "
                        "any metric that interpolates along this axis, or takes "
                        "max-minus-min across it as a signal excursion, is "
                        "measuring across the outer sweep as well. Run one "
                        "'.dc' per outer value to get curves.",
                        x_name, reversals, len(x_values),
                    )
                for name, data in vecs.items():
                    if name == x_name or len(data) != len(x_values):
                        continue
                    xs, ys, _ = _finite_pairs(x_values, _real(data),
                                              f"DC sweep {name!r}")
                    if not xs:
                        continue
                    sweeps[name] = SignalData(name=name, x_values=xs,
                                              y_values=ys)
        if not op_points and not sweeps:
            return None
        return DCResult(op_points=op_points, sweeps=sweeps)

    @staticmethod
    def _ac_frequencies(vecs: dict[str, list]) -> list[float]:
        raw = vecs.get("frequency")
        if raw is None:
            raise NgspiceError("AC plot has no 'frequency' vector")
        # The AC frequency axis is COMPLEX with a zero imaginary part, while
        # the noise frequency axis is REAL. Both must work here.
        return _real(raw)

    @classmethod
    def _build_ac(cls, run: _SimRun, params: Any = None) -> Optional[ACResult]:
        plot = run.find_plot("ac")
        vecs = run.vectors(plot)
        if not vecs:
            return None
        freqs = cls._ac_frequencies(vecs)

        divisor = None
        ref = _options(params).get("input_signal")
        if ref and ref in vecs:
            divisor = vecs[ref]

        signals: dict[str, SignalData] = {}
        for name, data in vecs.items():
            if name == "frequency" or len(data) != len(freqs):
                continue
            values = [complex(v) if not isinstance(v, complex) else v for v in data]
            gain_db, phase_deg = measure.transfer_function(values, divisor)
            # A sample is kept only where BOTH the magnitude and the phase are
            # finite, so vdb() and vp() always share an x axis and neither can
            # carry a -inf or a NaN into the result. A vector with no AC
            # content at all -- a DC-only supply rail -- is -inf everywhere and
            # is dropped entirely rather than reported as a signal of nothing.
            keep = [i for i in range(len(freqs))
                    if math.isfinite(gain_db[i]) and math.isfinite(phase_deg[i])
                    and math.isfinite(freqs[i])]
            keep_f = [float(freqs[i]) for i in keep]
            keep_g = [float(gain_db[i]) for i in keep]
            keep_p = [float(phase_deg[i]) for i in keep]
            dropped_g = len(freqs) - len(keep)
            if not keep_f:
                log.warning(
                    "AC vector %r has no finite transfer function at any of "
                    "%d frequencies (its response is identically zero, or the "
                    "stimulus is); it is left out of the result rather than "
                    "reported as -inf dB", name, len(freqs),
                )
                continue
            if dropped_g:
                log.warning(
                    "AC vector %r: %d of %d samples have no transfer function "
                    "and are dropped; the signal carries its own x_values",
                    name, dropped_g, len(freqs),
                )
            key_m = f"vdb({name})"
            key_p = f"vp({name})"
            signals[key_m] = SignalData(name=key_m, x_values=list(keep_f),
                                        y_values=keep_g)
            signals[key_p] = SignalData(name=key_p, x_values=list(keep_f),
                                        y_values=keep_p)
        return ACResult(frequencies=freqs, signals=signals)

    @staticmethod
    def _build_tran(run: _SimRun) -> Optional[TranResult]:
        plot = run.find_plot("tran")
        vecs = run.vectors(plot)
        if not vecs:
            return None
        if "time" not in vecs:
            raise NgspiceError("transient plot has no 'time' vector")
        t = _real(vecs["time"])
        signals: dict[str, SignalData] = {}
        for name, data in vecs.items():
            if name == "time" or len(data) != len(t):
                continue
            xs, ys, _ = _finite_pairs(t, _real(data), f"transient {name!r}")
            if not xs:
                continue
            signals[name] = SignalData(name=name, x_values=xs, y_values=ys)
        return TranResult(time=[v for v in t if math.isfinite(v)],
                          signals=signals)

    @staticmethod
    def _noise_plots(run: _SimRun) -> tuple[Optional[str], Optional[str]]:
        """Identify the spectrum plot and the totals plot by their vectors.

        ngSpice_CurPlot() returns the TOTALS plot for a .noise run, so it must
        not be used to find the spectrum.
        """
        spectrum = totals = None
        for name in run.plot_order:
            names = set(run.plots.get(name, {}))
            if any(v.endswith("_spectrum") for v in names):
                spectrum = spectrum or name
            elif any(v.endswith("_total") for v in names):
                totals = totals or name
        return spectrum, totals

    # -- SimulatorInterface methods ---------------------------------------

    def dc(self, netlist: str, params: Any = None) -> DCResult:
        """DC operating point and/or DC sweep.

        `.op` fills op_points from the length-1 vectors. `.dc` fills sweeps,
        using the real swept variable ('v-sweep') as x_values. When the deck
        carries both, both are populated from a single load.
        """
        run = self._run(netlist, "dc", params)
        result = self._build_dc(run)
        if result is None:
            raise NgspiceError(
                f"dc(): run produced no operating point or sweep data "
                f"(plots: {run.plot_order})"
            )
        return result

    def ac(self, netlist: str, params: Any = None) -> ACResult:
        """Small-signal AC analysis. See the class docstring for signal naming."""
        run = self._run(netlist, "ac", params)
        result = self._build_ac(run, params)
        if result is None:
            raise NgspiceError(f"ac(): run produced no AC plot (plots: {run.plot_order})")
        return result

    def tran(self, netlist: str, params: Any = None) -> TranResult:
        """Transient analysis. x_values of every signal is the real time vector."""
        run = self._run(netlist, "tran", params)
        result = self._build_tran(run)
        if result is None:
            raise NgspiceError(
                f"tran(): run produced no transient plot (plots: {run.plot_order})"
            )
        return result

    def noise(self, netlist: str, params: Any = None) -> NoiseResult:
        """Noise analysis.

        Returns the SPECTRAL DENSITY curves (V/sqrt(Hz)), which live in the
        first noise plot. ngspice's own integrated inoise_total/onoise_total
        are grid dependent and read about 12 pct high at 'dec 10', so they are
        deliberately not returned; use measure.integrate_noise() on the
        returned spectra instead, with >= 100 points/decade.
        """
        run = self._run(netlist, "noise", params)
        spectrum_plot, _totals = self._noise_plots(run)
        vecs = run.vectors(spectrum_plot)
        if not vecs or "frequency" not in vecs:
            raise NgspiceError(
                f"noise(): no noise spectrum plot found (plots: {run.plot_order})"
            )
        freqs = _real(vecs["frequency"])
        inoise = vecs.get("inoise_spectrum")
        onoise = vecs.get("onoise_spectrum")
        if inoise is None or onoise is None:
            raise NgspiceError("noise(): spectrum plot lacks inoise/onoise vectors")
        fi, yi, _ = _finite_pairs(freqs, _real(inoise), "inoise_spectrum")
        fo, yo, _ = _finite_pairs(freqs, _real(onoise), "onoise_spectrum")
        return NoiseResult(
            frequencies=[v for v in freqs if math.isfinite(v)],
            input_noise=SignalData(name="inoise_spectrum", x_values=fi,
                                   y_values=yi),
            output_noise=SignalData(name="onoise_spectrum", x_values=fo,
                                    y_values=yo),
        )

    def stb(self, netlist: str, params: Any = None) -> StabilityResult:
        """Loop stability from a broken-loop AC response.

        ngspice has no native .stb analysis, so this is the classical
        broken-loop method: the deck must already break the loop and drive it
        with an AC source. The loop response is out/in, taken from
        options['loop_out'] (default 'out') divided by options['loop_in']
        (default 'in' when that vector exists).

        The phase is unwrapped, and a whole 180 deg inversion is removed only
        when the phase at the BOTTOM of the sweep departs by 180 deg from what
        the magnitude slope there implies (measure.phase_inversion_shift). It
        is NOT normalised to 0 deg at any single sample: doing that at the
        peak-gain sample fabricated an inversion on every resonant loop, which
        turned a phase margin of -68.2 deg into +111.8 deg and, by destroying
        the -180 deg crossing, an unstable loop into an infinite gain margin.

        phase_margin is taken at the LOOP CLOSURE -- the last frequency at
        which the gain is above 0 dB -- not at the first 0 dB crossing.

        Raises when the loop gain never crosses 0 dB, because the phase margin
        is then undefined and returning 0.0 would be a fabrication. float('inf')
        is reported for the gain margin ONLY when ac_metrics says the phase
        never reaches -180 deg at all; a phase that sits at or below -180 deg
        gets a real, finite, negative margin.
        """
        opts = _options(params)
        out_name = str(opts.get("loop_out", "out"))
        in_name = opts.get("loop_in", "in")

        run = self._run(netlist, "ac", params)
        plot = run.find_plot("ac")
        vecs = run.vectors(plot)
        if not vecs:
            raise NgspiceError(f"stb(): no AC plot produced (plots: {run.plot_order})")
        freqs = self._ac_frequencies(vecs)
        if out_name not in vecs:
            raise NgspiceError(
                f"stb(): loop output vector '{out_name}' not in AC plot "
                f"{sorted(vecs)}. Set options['loop_out']."
            )
        num = [complex(v) for v in vecs[out_name]]
        den = None
        if in_name and in_name in vecs:
            den = [complex(v) for v in vecs[in_name]]
        gain_db, phase_deg = measure.transfer_function(num, den)
        m = measure.ac_metrics(freqs, gain_db, phase_deg)

        if m["phase_margin"] is None:
            # Say what is ACTUALLY wrong. The old message claimed "never
            # crosses 0 dB" even for a loop whose peak gain was 40 dB, because
            # the ugb guard was keyed on the gain at the bottom of the sweep
            # instead of on the peak.
            notes = m["notes"]
            reason = (notes.get("phase_margin") or notes.get("ugb")
                      or notes.get("*")
                      or "the phase is not defined at the unity-gain frequency")
            peak, f_peak = m["peak_gain_db"], m["f_peak"]
            detail = (f" (peak loop gain {peak:.2f} dB at {f_peak:g} Hz)"
                      if peak is not None and f_peak is not None else "")
            raise NgspiceError(f"stb(): no phase margin. {reason}{detail}.")
        gm = m["gain_margin"]
        lf, lg, _ = _finite_pairs(freqs, gain_db, "stb loop gain")
        return StabilityResult(
            phase_margin=float(m["phase_margin"]),
            gain_margin=float(gm) if gm is not None else float("inf"),
            loop_gain=SignalData(name="loop_gain_db", x_values=lf,
                                 y_values=lg),
        )

    def corners(self, netlist: str, pvt_list: Sequence[Any]) -> list[CornerResult]:
        """Run a real simulation per PVT corner.

        Each corner varies:
          - TEMPERATURE via a '.temp' directive (always),
          - SUPPLY via a rewrite of the supply source's DC value (when the
            corner declares a non-zero voltage),
          - PROCESS via the PDK model deck's corner section (ONLY when a PDK
            deck is configured and present; without one ngspice has no process
            corner to switch to and this is logged as a warning rather than
            silently pretended).

        Accepts PVTCorner objects or plain corner-name strings ('tt', 'ss',
        ...), because training/rl_env.py passes strings. A bare NAME carries no
        supply information, so only its process and temperature are applied;
        pass a PVTCorner to also vary the supply. Whichever analyses the deck
        declares (.op/.dc, .ac, .tran) are harvested from one load per corner.
        """
        pdk, _ = self._pdk_selection(None)
        if not pdk:
            log.warning("corners(): no PDK model deck configured; process corner "
                        "is NOT varied, only temperature and supply")

        results: list[CornerResult] = []
        for entry in pvt_list:
            corner = self._as_pvt(entry, pdk)
            supply = corner.voltage if corner.voltage and corner.voltage > 0 else None
            run = self._run(
                netlist, "dc", None,
                temperature=corner.temperature, supply=supply,
                corner_override=corner.process.lower() if pdk else None,
            )
            results.append(CornerResult(
                corner=corner,
                dc=self._build_dc(run),
                ac=self._build_ac(run),
                tran=self._build_tran(run),
            ))
        return results

    def _as_pvt(self, entry: Any, pdk: Optional[str]) -> PVTCorner:
        if isinstance(entry, PVTCorner):
            return entry
        if isinstance(entry, dict):
            return PVTCorner(**entry)
        name = str(entry).lower()
        # A bare corner name carries no voltage. 0.0 means "not specified" and
        # suppresses the supply rewrite rather than forcing a 0 V supply.
        temperature = 27.0
        if pdk:
            temperature = float(
                pdk_deck.corner_pvt(pdk, name).get("temperature", temperature)
            )
        else:
            temperature = _GENERIC_CORNER_TEMPS.get(name, temperature)
        return PVTCorner(process=name, voltage=0.0, temperature=temperature)

    def mc(self, netlist: str, n: Any = 10, seed: int = 0,
           params: Any = None) -> MonteCarloResult:
        """Monte Carlo by running n real, individually seeded simulations.

        HOW THE RANDOMNESS IS SOURCED, and its limits:
          - With a PDK whose config declares statistical sections (TSMC65
            declares 'stat' then 'MC'), those sections are included instead of
            the process corner, the caller-side unit-normal mismatch draws are
            declared, and each run gets '.option seed=<seed+i>'. That yields
            global process plus local mismatch variation and is bit-for-bit
            reproducible for a given seed.
          - Without a PDK, the netlist itself must contain a distribution
            function (agauss/gauss/unif/aunif). ngspice re-evaluates those at
            every netlist parse, so seeding per run gives genuine variation.
          - With neither, there is NOTHING to vary. This method then RAISES
            rather than returning n identical runs dressed up as Monte Carlo.
            'set rndseed' does not reach the parse-time RNG and is not used.

        The per-run metrics are generic (operating point node values, and
        min/max/last of each swept or transient signal). Domain metrics such as
        gain or bandwidth should be computed by the caller from `results`.

        `n` also accepts the raw args dict that training/rl_env.py passes, in
        which case n/seed are read out of it.
        """
        if isinstance(n, dict):
            args = n
            params = params or args
            seed = int(args.get("seed", seed) or 0)
            n = int(args.get("n", args.get("runs", args.get("iterations", 10))))
        n = int(n)
        seed = int(seed or 0)

        pdk, corner = self._pdk_selection(params)
        body = netlist_text(netlist)
        pdk_mc = bool(pdk) and bool(
            (pdk_deck.get_pdk_config(pdk) or {}).get("mc_sections")
        )
        netlist_random = bool(
            re.search(r"\b(a?gauss|a?unif)\s*\(", body, re.IGNORECASE)
        )
        if not pdk_mc and not netlist_random:
            raise NgspiceError(
                "mc(): no source of statistical variation. Configure a PDK with "
                "statistical model sections, or write distribution functions "
                "(agauss/gauss/unif) into the netlist. Refusing to return "
                f"{n} identical runs as Monte Carlo."
            )

        results: list[dict[str, Any]] = []
        for i in range(n):
            run_seed = seed + i
            run = self._run(body, "dc", params, mc=pdk_mc, seed=run_seed,
                            corner_override=corner)
            metrics = self._summarize_run(run)
            metrics["run"] = i
            metrics["seed"] = run_seed
            results.append(metrics)
        return MonteCarloResult(seed=seed, runs=n, results=results)

    @staticmethod
    def _summarize_run(run: _SimRun) -> dict[str, Any]:
        """Generic per-run scalar summary used by mc().

        The sweep axis is the vector ngspice DESIGNATED as the axis
        (_dc_sweep_axis), not "v-sweep if present else time". That guess is the
        one _build_dc had to stop making: a '.dc temp -40 125' run designates
        'temp-sweep', a '.dc i1 ...' run designates 'i1-sweep', and neither is
        'v-sweep', so the axis was summarised as though it were a measured
        circuit quantity and every Monte Carlo result carried a
        'temp-sweep.min = -40' that reads as a node voltage.
        """
        metrics: dict[str, Any] = {}
        op_plot = run.find_plot("op")
        for name, data in run.vectors(op_plot).items():
            if len(data) == 1:
                metrics[f"op.{name}"] = _real(data)[0]
        for prefix in ("dc", "tran"):
            plot = run.find_plot(prefix)
            vecs = run.vectors(plot)
            axis = _dc_sweep_axis(vecs) if prefix == "dc" else "time"
            for name, data in vecs.items():
                if name == axis or not data:
                    continue
                y = _real(data)
                metrics[f"{name}.min"] = min(y)
                metrics[f"{name}.max"] = max(y)
                metrics[f"{name}.last"] = y[-1]
        return metrics

    # -- measurement convenience ------------------------------------------

    @staticmethod
    def measure_ac(result: ACResult, signal: str = "out") -> dict[str, Any]:
        """dc_gain_db / bandwidth_3db / ugb / phase_margin / gain_margin.

        `signal` is a raw vector name such as 'out'; the vdb()/vp() keys are
        looked up for you.

        See measure.ac_metrics for the full key list. Anything the response
        does not define comes back as None with the reason under "notes" --
        including dc_gain_db, which is None whenever the sweep does not
        actually reach DC. Read notes before treating a None as a failure.

        THE SIGNAL'S OWN x_values ARE THE FREQUENCY AXIS, not result.frequencies.
        _build_ac drops the samples where a signal has no transfer function and
        gives that signal its own axis, so the two lengths legitimately differ.
        Pairing a shortened y with the full global axis attributes every sample
        to the wrong frequency: on `.ac LIN 1001 0 1e6` through an AC-coupled
        stage the f = 0 sample of out is exactly zero and is dropped, leaving
        1000 y values against 1001 frequencies, and every sample was read one
        grid step low --

            bandwidth_3db   101510.30 Hz   against 100316.74 Hz (analytic 100 kHz)
            passband_gain    39.8909 dB    against  39.9862 dB
            rolloff          None          against -17.033 dB/dec

        -- a shift of one grid step, which on a coarse LIN sweep is arbitrarily
        large. spec_extract was fixed to honour the signal's own axis; this
        helper and measure_tran were not.
        """
        mag = result.signals.get(f"vdb({signal})")
        if mag is None:
            raise KeyError(
                f"no magnitude signal 'vdb({signal})' in ACResult "
                f"(have {sorted(result.signals)})"
            )
        freqs = list(result.frequencies)
        if mag.x_values and len(mag.x_values) == len(mag.y_values):
            freqs = list(mag.x_values)
        ph = result.signals.get(f"vp({signal})")
        phase = ph.y_values if ph else None
        if phase is not None and len(phase) != len(mag.y_values):
            # vdb() and vp() share an axis by construction in _build_ac; if they
            # ever do not, the phase belongs to different frequencies than the
            # magnitude and no margin taken from the pair would mean anything.
            log.warning(
                "measure_ac: 'vp(%s)' has %d samples against %d for 'vdb(%s)'; "
                "the phase is dropped rather than paired with the wrong "
                "frequencies", signal, len(phase), len(mag.y_values), signal,
            )
            phase = None
        return measure.ac_metrics(freqs, mag.y_values, phase)

    @staticmethod
    def measure_tran(result: TranResult, signal: str = "out",
                     input_signal: Optional[str] = None) -> dict[str, Any]:
        """rise_time / fall_time / overshoot / settling_time / slew_rate.

        The signal's own x_values are the time axis whenever they are present
        and consistent, for the same reason measure_ac uses them: _build_tran
        drops the samples where a signal is non-finite and gives that signal
        its own axis, and pairing a shortened y with the full global time
        vector shifts every sample against its time.

        The STIMULUS is passed through whenever the result carries one on the
        same axis. It is the only vector that can say a flat tail is the drive
        having been removed rather than the circuit having settled; see
        measure.drive_truncation_note().
        """
        sig = result.signals.get(signal)
        if sig is None:
            raise KeyError(
                f"no signal '{signal}' in TranResult (have {sorted(result.signals)})"
            )
        t = list(result.time)
        if sig.x_values and len(sig.x_values) == len(sig.y_values):
            t = list(sig.x_values)
        y_in = None
        for cand in (input_signal, "in", "vin", "input", "in_p"):
            drive = result.signals.get(cand) if cand else None
            if (drive is not None and cand != signal
                    and len(drive.y_values) == len(t)
                    and (not drive.x_values or list(drive.x_values) == t)):
                y_in = list(drive.y_values)
                break
        return measure.tran_metrics(t, sig.y_values, y_in)

    def measure_idd(self, result: DCResult,
                    sources: Optional[Iterable[str]] = None,
                    netlist: Optional[str] = None) -> Optional[float]:
        """Total supply current magnitude in amperes, or None when unavailable.

        ngspice reports a source branch current with the passive sign
        convention, so a 1.8 V supply feeding 20 kOhm reports -90 uA. This
        returns the magnitude, 90e-6.

        Which sources are SUPPLIES is not knowable from an operating point: a
        0 V ammeter is spelled exactly like a rail, and a current-source-biased
        block has no supply branch vector at all. So the deck is consulted --
        `netlist` when given, otherwise the deck of the last run, and only when
        its element names still match this result, so a stale netlist from a
        later run can never be applied to an earlier one. Name the supplies via
        `sources` whenever you know them; that always wins.

        Returns None, with the reason logged, when the supply cannot be
        identified. That is deliberate: this call used to report a 1 uA sense
        source as the supply current of a 1 mA stage, silently.
        """
        return measure.supply_current(
            result.op_points,
            list(sources) if sources else None,
            netlist if netlist is not None else self._netlist_for(result),
        )

    def _netlist_for(self, result: DCResult) -> Optional[str]:
        """The last run's deck, but only if it really produced `result`.

        Matched on element names: every '<name>#branch' in the operating point
        must be an element card in that deck. A result carried over from an
        earlier, different circuit fails this and gets no netlist rather than
        the wrong one.

        A branch INSIDE a subcircuit instance is spelled hierarchically --
        'v.x1.vsense#branch' -- and no deck has an element card called that.
        The match is therefore on the LOCAL name (the last dotted component)
        against every element card at any level of hierarchy, including
        subcircuit bodies. Comparing the hierarchical name against top-level
        cards only made the deck fail to match its OWN result the moment a
        subcircuit held a V or an L, and a rejected deck means no netlist,
        which means the 0 V ammeter inside that subcircuit is summed as if it
        were a second supply: exactly 2x.
        """
        netlist = getattr(self, "_last_netlist", "") or ""
        if not netlist:
            return None
        wanted = {k.split("#")[0].lower().split(".")[-1]
                  for k in result.op_points if "#branch" in k.lower()}
        if not wanted:
            return None
        present = measure.parse_deck_sources(netlist).elements
        return netlist if wanted <= present else None
