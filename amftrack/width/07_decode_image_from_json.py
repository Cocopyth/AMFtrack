"""Importing an image stored into a json file from the label me app"""

import json
import matplotlib.pyplot as plt
import base64

source = "/media/felix/AMFtopology02/prov/width1/labels/20220324_Plate907_001.json"

with open(source) as f:
    d = json.load(f)

a = d["imageData"]

decodeit = open("/media/felix/AMFtopology02/prov/width1/labels/try.jpeg", "wb")
decodeit.write(base64.b64decode((a)))
decodeit.close()
