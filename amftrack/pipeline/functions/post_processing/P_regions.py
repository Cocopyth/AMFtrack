from amftrack.pipeline.functions.post_processing.area_hulls_util import *
from amftrack.notebooks.P_experiment.helper import get_polygons, get_regions
from amftrack.util.geometry import create_polygon
from shapely import affinity


def get_length_density_in_region(exp, t, args):
    i = args["i"]
    polygons = get_regions(exp, 0)
    polygons =[affinity.scale(polygon, xfact=1000, yfact=1000, origin=(0, 0)) for polygon in polygons]
    if i  <= len(polygons):
        shape = polygons[i]
        length = get_length_shape_fast(exp, t,shape)
        return (f"length_density_region_{i}", length)
    else:
        return (f"LA_region_{i}", None)

def get_surface_area_in_region(exp, t, args):
    i = args["i"]
    polygons = get_regions(exp, 0)
    polygons =[affinity.scale(polygon, xfact=1000, yfact=1000, origin=(0, 0)) for polygon in polygons]
    if i  <= len(polygons):
        shape = polygons[i]
        length = get_surface_area_shape_fast(exp, t,shape)
        return (f"SA_region_{i}", length)
    else:
        return (f"SA_region_{i}", None)