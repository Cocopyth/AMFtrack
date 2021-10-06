from path import path_code_dir
import sys  
sys.path.insert(0, path_code_dir)
from amftrack.util import get_dates_datetime
import os
from amftrack.pipeline.functions.experiment_class_surf import Experiment
# from experiment_class_surftest import Experiment, clean_exp_with_hyphaes
from amftrack.pipeline.functions.hyphae_id_surf import (
    clean_and_relabel,
    get_mother,
    save_hyphaes,
    resolve_ambiguity_two_ends,
    clean_obvious_fake_tips,
    width_based_cleaning
)
# from hyphae_id_surftest import (
#     clean_and_relabel,
#     get_mother,
#     save_hyphaes,
#     resolve_ambiguity_two_ends,
#     solve_degree4,
#     clean_obvious_fake_tips,
# )

plate = int(sys.argv[1])
begin = int(sys.argv[2])
end = int(sys.argv[3])
directory = str(sys.argv[4])

dates_datetime = get_dates_datetime(directory,plate)
dates_datetime.sort()
dates_datetime_chosen = dates_datetime[begin : end + 1]
dates = dates_datetime_chosen
exp = Experiment(plate, directory)
exp.load(dates)
#when no width is included
# width_based_cleaning(exp)
exp_clean = clean_and_relabel(exp)
to_remove = []
for hyph in exp_clean.hyphaes:
    hyph.update_ts()
    if len(hyph.ts) == 0:
        to_remove.append(hyph)
for hyph in to_remove:
    exp_clean.hyphaes.remove(hyph)
get_mother(exp_clean.hyphaes)
# solve_two_ends = resolve_ambiguity_two_ends(exp_clean.hyphaes)
# solved = solve_degree4(exp_clean)
clean_obvious_fake_tips(exp_clean)
dirName = f"{directory}Analysis_Plate{plate}_{dates[0]}_{dates[-1]}"
try:
    os.mkdir(f"{directory}Analysis_Plate{plate}_{dates[0]}_{dates[-1]}")
    print("Directory ", dirName, " Created ")
except FileExistsError:
    print("Directory ", dirName, " already exists")
hyphs, gr_inf = save_hyphaes(
    exp_clean, f"{directory}Analysis_Plate{plate}_{dates[0]}_{dates[-1]}/"
)
exp_clean.save(f"{directory}Analysis_Plate{plate}_{dates[0]}_{dates[-1]}/")
exp_clean.pickle_save(f"{directory}Analysis_Plate{plate}_{dates[0]}_{dates[-1]}/")
