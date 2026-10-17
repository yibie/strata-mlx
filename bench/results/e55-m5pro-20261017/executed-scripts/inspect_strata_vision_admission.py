"""Header-only budget arithmetic; never constructs or loads the text model."""
import json
import sys
from pathlib import Path
import mlx.core as mx
from strata_mlx.dense import kept_floats
from strata_mlx.gguf import shards
from strata_mlx.paged import DENSE_AS_ARRAYS, DRAFT_HEAD, WORKING, all_fit, machine_memory, sizes
from strata_mlx.residency import SYSTEM_HEADROOM
files=shards(sys.argv[1])
a,b=sizes(files)
floats=kept_floats(files[0].tensors,2)
physical=machine_memory()
limit=min(physical-SYSTEM_HEADROOM,mx.device_info()['max_recommended_working_set_size'])
rows=[]
for mib in (512,1024):
    parked=mib<<20;vision=512<<20;machine=physical-parked-vision
    weights=int(a+DENSE_AS_ARRAYS*b+DRAFT_HEAD)
    if all_fit(machine,a,b,floats):weights+=floats
    required=weights+parked+WORKING+vision
    rows.append(dict(cache_mib=mib,weight_bytes=weights,working_bytes=WORKING,
                     vision_reserve_bytes=vision,required_bytes=required,limit_bytes=limit,
                     within_static_allowance=required<=limit))
Path(sys.argv[2]).write_text(json.dumps(rows,indent=2)+'\n')
