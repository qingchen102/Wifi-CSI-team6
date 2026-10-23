import os, glob
root = r"D:\Wifi CSI\数据集\发出给高校学生"
tot = {}
for scene in ["穿墙场景","公寓场景","小会议室场景"]:
    for p in glob.glob(os.path.join(root, scene, "*.txt")):
        n = 0
        with open(p, "rb") as f:
            while True:
                b = f.read(1 << 22)
                if not b: break
                n += b.count(b"\n")
        name = os.path.basename(p)
        lab = "exist" if "exist" in name else "none"
        d = tot.setdefault(scene, {"exist": [0,0], "none": [0,0]})
        d[lab][0] += 1; d[lab][1] += n
for s, d in tot.items():
    for lab, (c, n) in d.items():
        print("%s %s: files=%d frames=%d dur=%.1fmin" % (s, lab, c, n, n/50.0/60))
te = sum(v["exist"][1] for v in tot.values()); tn = sum(v["none"][1] for v in tot.values())
print("TOTAL exist frames=%d (%.2fh)  none frames=%d (%.2fh)" % (te, te/50/3600, tn, tn/50/3600))
