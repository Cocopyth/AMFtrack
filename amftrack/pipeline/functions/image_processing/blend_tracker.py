import bpy
from bpy import context
import numpy as np
import csv
import glob
import json

bpy.context.scene.frame_current = 1

csv_path = 'C:\\Users\\night\\OneDrive\\Documenten\\Internships\\AMOLF\\AMFtrack\\amftrack\\notebooks\\Simon\\blobs.csv'

directory= "E:\\AMOLF_Data\\TEMP\\"
tiffies = glob.glob(directory + '\\*.tiff')
tiffies_names = [x.split('\\')[-1] for x in tiffies]

bpy.context.area.ui_type = 'CLIP_EDITOR'

bpy.ops.clip.open(directory=directory, files=[{"name":tiffies_names[0]}, {"name":tiffies_names[0]}], relative_path=True)


with open(csv_path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    
    for i, row in enumerate(csv_reader):
#        print(row)
#        print(i)
        if i == 0:
            X_RES = float(row[0])
            Y_RES = float(row[1])
            continue
        
        rad = float(row[2])
#        
#        bpy.data.movieclips["0001.tiff"].(null) = rad

        bpy.ops.clip.add_marker(location=(float(row[1])/X_RES, 1-float(row[0])/Y_RES))
        tracker = bpy.context.selected_movieclip_tracks[0]
        
        
#        tracker.markers[0].pattern_bound_box[0][0] *= rad
        
#        tracker.pattern_bound_box = ((float(row[1])/X_RES + rad/X_RES, 1-float(row[0])/Y_RES + rad/Y_RES), (float(row[1])/X_RES - rad/X_RES, 1-float(row[0])/Y_RES - float(rad/Y_RES)))
#bpy.ops.clip.select_all(action='SELECT')
#selection_names = bpy.context.selected_movieclip_tracks
#for i, name in enumerate(selection_names):
#    rad = 
#    name.pattern_corners = []

