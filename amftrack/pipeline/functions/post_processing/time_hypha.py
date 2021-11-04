from amftrack.pipeline.functions.image_processing.hyphae_id_surf import get_pixel_growth_and_new_children
from amftrack.pipeline.functions.post_processing.util import get_length_um


def get_time_since_start(hypha, t, tp1, args):
    exp = hypha.experiment
    seconds = (exp.dates[tp1]-exp.dates[hypha.ts[0]]).total_seconds()
    return("time_since_emergence",seconds/3600)


def get_time(hypha,t,tp1,args):
    exp = hypha.experiment
    seconds = (exp.dates[tp1]-exp.dates[t]).total_seconds()
    return("time",seconds/3600)


def get_speed(hypha,t,tp1,args):
    try:
        pixels,nodes = get_pixel_growth_and_new_children(hypha,t,tp1)
        speed = np.sum([get_length_um(seg) for seg in pixels])/get_time(hypha,t,tp1,None)[1]
        return('speed',speed)
    except:
        print('not_connected',hypha.end.label,hypha.get_root(tp1).label)
        return('speed',None)