"""What the engine would do with a model file on a Mac of some size, worked out without loading anything.

    ./strata-mlx --check                              the files that have been run, on this Mac
    ./strata-mlx --check --machine-memory 48          on a Mac with 48 GB
    ./strata-mlx --check --native <shard 1>           a file that is on the disk, by its own header

The arithmetic is the engine's (paged.py): a Mac's memory, less what is left to macOS and to this process beyond
its weights, less the dense weights and the draft head, is the experts'. If all of them fit, the model is held
whole. If not, the rest is read from the SSD as tokens ask for it. If under half a GB is left for them, the
Mac is too small.

Where experts are read from the SSD, the dense tensors of two types stay the file's blocks (dense.BlockLinear)
and take that much less than the rule's arrays; that is counted for such a Mac, as the engine finds it when it
starts.

The weights of the Q6_K layers as floats, 3.3 GB with the IQ3_S file, are kept from the first prompt on only
where they fit beside the whole model (dense.py, load.py); a Mac that holds the file whole and lacks just that
makes them anew for each chunk of a prompt, and the report says so.

Memory is in the GB of About This Mac (2^30 bytes), sizes on disk in the GB of Finder (10^9). None of this is a
measurement: what has been measured, and on which Mac, is in the README. It is for the engine's default floats,
float16: with `--dtype float32` the floats kept for prompts are twice as large and the rest larger too, and the
engine, which counts the first, may not keep them where this says it does.
"""

import argparse
import os
import platform
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from .dense import kept_floats, spared
from .gguf import shards
from .paged import DENSE_AS_ARRAYS, DRAFT_HEAD, LEAST, WORKING, all_fit, left_to_system, machine_for, machine_memory, sizes

GB = 2**30
HEAD_ON_DISK = 4.9e9  # the draft head's 31 tensors, as tools/mtp_fetch.py leaves them
CACHES = 30e3  # bytes a token of a conversation takes: the attention caches, in float16
RUN_PAGED = ("Q2_0",)  # the files that have been run with experts read from the SSD
_NAME = re.compile(r"-([A-Za-z0-9_]+)-\d{5}-of-\d{5}\.gguf$")


@dataclass
class Model:
    name: str
    experts: int  # bytes, as the file has them
    dense: int  # the dense tensors, as the file has them
    floats: int  # its Q6_K layers but the output layer as float16: held once a prompt has been read
    disk: int
    spared: int = 0  # what the dense tensors take less as blocks, where experts are read from the SSD


# The files that have been run, for a Mac that does not have them yet: what `of` reads from their headers.
KNOWN = (Model("Q2_0", 33_973_862_400, 3_638_853_120, 205_783_040, 66_423_878_624, 1_739_248_640),
         Model("IQ3_S", 50_292_326_400, 4_514_173_440, 3_306_946_560, 83_617_662_656, 567_336_960))


def of(path: str, files: list | None = None) -> Model:
    """The sizes of the model whose first shard is at `path`, from the headers of its files."""
    files = files or shards(path)
    experts, dense = sizes(files)
    m = _NAME.search(path)
    return Model(m.group(1) if m else Path(path).stem, experts, dense, kept_floats(files[0].tensors),
                 sum(os.path.getsize(f.path) for f in files), spared(files[0].tensors))


@dataclass
class Plan:
    whole: bool  # every expert in memory: what the engine decides
    floats: bool  # and the Q6_K layers' floats kept for prompts beside them
    needs: int  # the memory of a Mac that holds the file whole and keeps the floats
    kept: int  # the experts' share of this Mac's memory, where they do not all fit
    least: int  # the memory of the smallest Mac that leaves the experts anything


def plan(m: Model, machine: int) -> Plan:
    fixed = int(WORKING + DENSE_AS_ARRAYS * m.dense + DRAFT_HEAD)  # besides what macOS is left
    paged = fixed - m.spared
    return Plan(all_fit(machine, m.experts, m.dense), all_fit(machine, m.experts, m.dense, m.floats),
                machine_for(fixed + m.experts + m.floats), machine - left_to_system(machine) - paged,
                machine_for(paged + LEAST))


DEFAULT_PARKED = 4096 << 20  # of parked conversations, where the model with its floats is held whole without it (#12)


def parking_budget(m: Model, machine: int) -> int:
    """What the engine parks conversations in by default (conversations.py): 4 GiB where the model and the floats
    kept for prompts are still held whole with that much less, nothing where it would come out of the experts'.
    So on a 64 GB Mac Q2_0 is parked in and IQ3_S is not. Held whole with 4,096 MiB of parking, Q2_0 did not
    swap on a 64 GB M5 Pro (#8); a 96 GB M2 Max lost a 40,000-token conversation to every request of another
    client while parking was off (#12)."""
    return DEFAULT_PARKED if plan(m, machine - DEFAULT_PARKED).floats else 0


def lines(m: Model, machine: int, parked: int | None = None) -> list:
    """The report's lines for a model on a Mac of `machine` bytes; `parked` is what parked conversations take
    first, the default's (`parking_budget`) when None, and then the report says what the default is."""
    default = parked is None
    if default:
        parked = parking_budget(m, machine)
    machine -= parked  # parked conversations are memory the model cannot have
    p, out = plan(m, machine), [f"{m.name}, {m.disk / 1e9:.1f} GB on disk"]
    if p.floats:
        out += [f"  held whole in memory: needs {p.needs / GB:.1f} GB, {(machine - p.needs) / GB:.1f} GB to spare",
                "  nothing is read from the disk while it runs"]
    elif p.whole:
        out += [f"  held whole in memory, without the {m.floats / GB:.1f} GB its Q6_K layers take as floats for prompts:",
                "  they are made anew for each chunk of a prompt, which reads prompts about a tenth slower;",
                f"  nothing is read from the disk while it runs, and {p.needs / GB:.1f} GB would keep the floats too"]
    elif p.kept >= LEAST:
        out += [f"  read from the SSD in part: about {p.kept / GB:.1f} GB of its {m.experts / GB:.1f} GB of experts stay in "
                f"memory ({p.kept / m.experts:.0%}),",
                f"  the others are read as tokens ask for them; holding it whole takes a Mac with {p.needs / GB:.1f} GB"]
        if m.name not in RUN_PAGED:
            out.append(f"  not run this way yet: only {' and '.join(RUN_PAGED)} has been")
    else:
        out += [f"  too small: what always stays in memory leaves the experts under {LEAST / GB:.1f} GB;",
                f"  this file takes a Mac with {p.least / GB:.1f} GB at least"]
    if default:
        out.append(f"  conversations parked in {parked / GB:.1f} GB by default (--conversation-cache-mib 0 for none)" if parked
                   else "  no conversations parked by default: the memory would be the experts' (--conversation-cache-mib 4096 to park anyway)")
    return out


def report(models: list, machine: int, this_mac: bool = False, parked: int | None = None) -> str:
    out = [f"{'This Mac has' if this_mac else 'A Mac with'} {machine / GB:g} GB of memory", ""]
    for m in models:
        out += [*lines(m, machine, parked), ""]
    out += [f"Counted: {left_to_system(machine) / GB:.1f} GB left to macOS and other programs (a quarter, 8.5 GB at least), {WORKING / GB:.1f} GB of working memory, "
            f"the draft head ({DRAFT_HEAD / GB:.1f} GB)"
            + (f", {parked / GB:.1f} GB of parked conversations (--conversation-cache-mib)." if parked else "."),
            f"Not counted: the conversation, {CACHES / 1e3:.0f} KB a token (1 GB for {GB / CACHES / 1e3:.0f},000 tokens).",
            f"On disk the draft head is {HEAD_ON_DISK / 1e9:.1f} GB more."]
    if this_mac:
        out.append(f"Free on the disk of your home folder: {shutil.disk_usage(Path.home()).free / 1e9:.0f} GB.")
        if platform.machine() != "arm64":
            out.append("This is not an Apple Silicon Mac: the engine needs one.")
        version = platform.mac_ver()[0]
        if version and int(version.split(".")[0]) < 27:
            out.append(f"This Mac runs macOS {version}: the engine has only been run on macOS 27.")
    out.append("Worked out from sizes for float16, not measured: see the README for the speeds measured, and on which Mac.")
    return "\n".join(out)


def main(argv: list) -> int:
    p = argparse.ArgumentParser(prog="strata-mlx --check", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--native", help="shard 1 of a GGUF on the disk; without it, the files that have been run")
    p.add_argument("--machine-memory", type=float, help="GB of memory of the Mac to work it out for; without it, this Mac")
    p.add_argument("--conversation-cache-mib", default="auto",
                   help="the engine's budget for parked conversations; auto (the default) says what the engine would take")
    args, _ = p.parse_known_args(argv)  # the engine's other arguments may stand beside these
    machine = int(args.machine_memory * GB) if args.machine_memory else machine_memory()
    parked = None if args.conversation_cache_mib == "auto" else int(args.conversation_cache_mib) << 20
    print(report([of(args.native)] if args.native else list(KNOWN), machine, this_mac=not args.machine_memory,
                 parked=parked))
    return 0
