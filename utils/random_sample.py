import random
import itertools
def sample_data():
    # texture settings
    fl = [x for x in range(0, 41, 20)]
    fr = [x for x in range(0, 60, 20)]
    bl = [x for x in range(0, 41, 20)]
    br = [x for x in range(0, 60, 20)]
    trunk = [x for x in range(0, 60, 20)] 
    az = [x for x in range(0, 361, 40)]
    el = [x for x in range(20, 90, 20)]
    dist = [400, 450]
    # spatial settings
    # fl = [x for x in range(-40, 1, 20)]
    # fr = [x for x in range(0, 60, 20)]
    # bl = [x for x in range(-40, 1, 20)]
    # br = [x for x in range(0, 60, 20)]
    # trunk = [x for x in range(0, 60, 20)] 
    # az = [x for x in range(0, 361, 40)]
    # el = [x for x in range(20, 90, 20)]
    # dist = [400, 450]
    fl_spl = random.sample(fl, 1)
    fr_spl = random.sample(fr, 1)
    bl_spl = random.sample(bl, 1)
    br_spl = random.sample(br, 1)
    trunk_spl = random.sample(trunk, 1)
    az_spl = random.sample(az, 1)
    el_spl = random.sample(el, 1)
    dist_spl = random.sample(dist, 1)
    return str(fl_spl[0]), str(fr_spl[0]), str(bl_spl[0]), str(br_spl[0]), str(trunk_spl[0]), str(az_spl[0]), str(el_spl[0]), str(dist_spl[0])

def get_samples(train_params, i):
    fl_spl, fr_spl, bl_spl, br_spl, trunk_spl, az_spl, el_spl, dist_spl = sample_data()
    name_data = []
    if i == 0:# sample for fl
        random_params = itertools.product(
            train_params["mesh_id"], train_params["fl"]
        )
        random_params = list(random_params)
        for ins, fl in random_params:
            name_data.append("{ins}_{fl}_{fr_spl}_{bl_spl}_{br_spl}_{trunk_spl}_{az_spl}_{el_spl}_{dist_spl}".format(**locals()))
    elif i == 1:# sample for fr
        random_params = itertools.product(
            train_params["mesh_id"], train_params["fr"]
        )
        random_params = list(random_params)
        for ins, fr in random_params:
            name_data.append("{ins}_{fl_spl}_{fr}_{bl_spl}_{br_spl}_{trunk_spl}_{az_spl}_{el_spl}_{dist_spl}".format(**locals()))
    elif i == 2:# sample for bl
        random_params = itertools.product(
            train_params["mesh_id"], train_params["bl"]
        )
        random_params = list(random_params)
        for ins, bl in random_params:
            name_data.append("{ins}_{fl_spl}_{fr_spl}_{bl}_{br_spl}_{trunk_spl}_{az_spl}_{el_spl}_{dist_spl}".format(**locals()))
    elif i == 3:# sample for br
        random_params = itertools.product(
            train_params["mesh_id"], train_params["br"]
        )
        random_params = list(random_params)
        for ins, br in random_params:
            name_data.append("{ins}_{fl_spl}_{fr_spl}_{bl_spl}_{br}_{trunk_spl}_{az_spl}_{el_spl}_{dist_spl}".format(**locals()))
    elif i == 4:# sample for trunk
        random_params = itertools.product(
            train_params["mesh_id"], train_params["trunk"]
        )
        random_params = list(random_params)
        for ins, trunk in random_params:
            name_data.append("{ins}_{fl_spl}_{fr_spl}_{bl_spl}_{br_spl}_{trunk}_{az_spl}_{el_spl}_{dist_spl}".format(**locals()))
    return name_data