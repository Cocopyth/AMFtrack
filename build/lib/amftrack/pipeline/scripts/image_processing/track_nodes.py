import sys
from amftrack.util.sys import temp_path
import pandas as pd
from amftrack.pipeline.functions.image_processing.node_id import create_corresp
from amftrack.pipeline.functions.image_processing.experiment_class_surf import (
    Experiment,
)

directory = str(sys.argv[1])
i = int(sys.argv[-1])
op_id = int(sys.argv[-2])
run_info = pd.read_json(
    f"{temp_path}/{op_id}.json", convert_dates=True, dtype={"unique_id": str}
)
run_info["datetime"] = pd.to_datetime(run_info["date"], format="%d.%m.%Y, %H:%M:")
folders = run_info.sort_values("datetime")
select = folders.iloc[i : i + 2]
exp = Experiment(directory)
exp.load(select, suffix="_width")
create_corresp(exp)
