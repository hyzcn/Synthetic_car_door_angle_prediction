import random
def sample_data():
    fl = [x for x in range(-40, 1, 20)]
    fr = [x for x in range(0, 60, 20)]
    bl = [x for x in range(-40, 1, 20)]
    br = [x for x in range(0, 60, 20)]
    trunk = [x for x in range(0, 60, 20)] 
    az = [x for x in range(0, 361, 40)]
    el = [x for x in range(20, 90, 20)]
    dist = [400, 450]
    fl_spl = random.sample(fl, 1)
    fr_spl = random.sample(fr, 1)
    bl_spl = random.sample(bl, 1)
    br_spl = random.sample(br, 1)
    trunk_spl = random.sample(trunk, 1)
    az_spl = random.sample(az, 1)
    el_spl = random.sample(el, 1)
    dist_spl = random.sample(dist, 1)
    return str(fl_spl[0]), str(fr_spl[0]), str(bl_spl[0]), str(br_spl[0]), str(trunk_spl[0]), str(az_spl[0]), str(el_spl[0]), str(dist_spl[0])