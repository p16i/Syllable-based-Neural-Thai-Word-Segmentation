import glob
import json

import re

import os

# tag ref: https://drive.google.com/file/d/1bLOIOEMDNLPyWBOvwEEWCogUTWt_EPTN/view

bad_files = [
    "จดหมายเหตุรายวันของสมเด็จเจ้าฟ้ามหาวชิรุณหิศ"
]

files = glob.glob("./data/tnhc-raw/*.json")

os.makedirs("./data/tnhc-processed", exist_ok=True)

for f in files:
    filename = os.path.basename(f).replace(".json", "")

    if filename in bad_files:
        print(f"skipping {filename}")
        continue

    txt = f"./data/tnhc-processed/{filename}.txt"
    label = f"./data/tnhc-processed/{filename}.label"

    with open(f, "r", encoding="utf-8") as fh, \
         open(label, "w", encoding="utf-8") as fl:
        
        for line in json.load(fh):

            line = list(filter(lambda x: x and len(x) > 0, line))

            if len(line) <= 0:
                continue

            if ":" in line: # assume this is a meta tag line
                continue

            line = "|".join(line).strip()

            # for token in line:
            line = line.strip().lower()
            line = line.replace("<s>", " ")
            line = line.replace("\t", " ")
            line = line.replace("<p>", "")
            line = line.replace("ss", "")
            line = line.replace("sz", "")
            line = line.replace("/", "")
            line = line.replace("\\", "")
            line = line.replace("lgp", "")
            line = line.replace("cmn", "")
            line = line.replace("\ufeff", "") # byte-order mark
            

            line = re.sub("cc(.+?)cc", "", line) # this specifies type of poem
            line = re.sub("cm(.+?)cm", "", line) # this also something similar

            # some file has its title embedded in the content, usually in the same name.
            if line.replace("|", "") == filename:
                print(f"skip filename: {filename}")
                continue

            txt = line.replace("|", "")
            if len(txt) <= 1 or re.match("b ?\d+", txt) or re.match("\.{5,}", txt):
                print(f"skip short line or title date line, e.g. b2012: {line}")
                continue
        
            if len(line) > 0:
                line = re.sub("^[\| ]+", "", line.strip())
                line = re.sub("[ \|]+$", "", line)
                line = re.sub("\|+", "|", line)
                fl.write(f"{line}\n")


print("combine file")
os.makedirs("./data/tnhc-final", exist_ok=True)

with open("./data/tnhc-final/tnhc.label", "w") as fl, open("./data/tnhc-final/input.txt", "w") as fi:
    for f in glob.glob("./data/tnhc-processed/*.label"):
        with open(f, "r") as fh:
            for line in fh:
                line = line.strip()
                fl.write(f"{line}\n")

                txt = line.replace("|", "")
                fi.write(f"{txt}\n")