import numpy as np
import time

try:
    import healpy as hp
    has_healpy = True
except ImportError:
    has_healpy = False

import hpgeom


ntrial = 10
np.random.seed(12345)

with open('benchmark_table.md', 'w') as bt:
    bt.write("|Function|Scheme|Nside|Size|Time (healpy)|Time (hpgeom)|hpgeom/healpy|\n")
    bt.write("|--------|------|-----|----|-------------|-------------|-------------|\n")

    function = 'angle_to_pixel'

    for scheme in ['nest', 'ring']:
        if scheme == 'nest':
            nest = True
        else:
            nest = False

        for size in [10, 10_000, 1_000_000]:
            lon = np.random.uniform(low=0.0, high=360.0, size=size)
            lat = np.random.uniform(low=-90.0, high=90.0, size=size)

            for nside in [128, 4096, 2**17]:
                start_time = time.time()
                for i in range(ntrial):
                    _ = hpgeom.angle_to_pixel(nside, lon, lat, nest=nest, lonlat=True)
                time_hpgeom = (time.time() - start_time)/ntrial

                start_time = time.time()
                for i in range(ntrial):
                    _ = hp.ang2pix(nside, lon, lat, nest=nest, lonlat=True)
                time_healpy = (time.time() - start_time)/ntrial

                time_ratio = time_hpgeom/time_healpy

                bt.write(f'|{function}|{scheme}|{nside}|{size}|{time_healpy}|{time_hpgeom}|{time_ratio}|\n')

    function = 'pixel_to_angle'

    for scheme in ['nest', 'ring']:
        if scheme == 'nest':
            nest = True
        else:
            nest = False

        for size in [10, 10_000, 1_000_000]:
            for nside in [128, 4096, 2**17]:
                pix = np.random.randint(low=0, high=12*nside*nside-1, size=size)

                start_time = time.time()
                for i in range(ntrial):
                    _ = hpgeom.pixel_to_angle(nside, pix, nest=nest, lonlat=True)
                time_hpgeom = (time.time() - start_time)/ntrial

                start_time = time.time()
                for i in range(ntrial):
                    _ = hp.pix2ang(nside, pix, nest=nest, lonlat=True)
                time_healpy = (time.time() - start_time)/ntrial

                time_ratio = time_hpgeom/time_healpy

                bt.write(f'|{function}|{scheme}|{nside}|{size}|{time_healpy}|{time_hpgeom}|{time_ratio}|\n')

    function = 'nest_to_ring'

    for scheme in ['nest']:
        for size in [10, 10_000, 1_000_000]:
            for nside in [128, 4096, 2**17]:
                pix = np.random.randint(low=0, high=12*nside*nside-1, size=size)

                start_time = time.time()
                for i in range(ntrial):
                    _ = hpgeom.nest_to_ring(nside, pix)
                time_hpgeom = (time.time() - start_time)/ntrial

                start_time = time.time()
                for i in range(ntrial):
                    _ = hp.nest2ring(nside, pix)
                time_healpy = (time.time() - start_time)/ntrial

                time_ratio = time_hpgeom/time_healpy

                bt.write(f'|{function}|{scheme}|{nside}|{size}|{time_healpy}|{time_hpgeom}|{time_ratio}|\n')

    function = 'ring_to_nest'

    for scheme in ['ring']:
        for size in [10, 10_000, 1_000_000]:
            for nside in [128, 4096, 2**17]:
                pix = np.random.randint(low=0, high=12*nside*nside-1, size=size)

                start_time = time.time()
                for i in range(ntrial):
                    _ = hpgeom.ring_to_nest(nside, pix)
                time_hpgeom = (time.time() - start_time)/ntrial

                start_time = time.time()
                for i in range(ntrial):
                    _ = hp.ring2nest(nside, pix)
                time_healpy = (time.time() - start_time)/ntrial

                time_ratio = time_hpgeom/time_healpy

                bt.write(f'|{function}|{scheme}|{nside}|{size}|{time_healpy}|{time_hpgeom}|{time_ratio}|\n')

    bt.write("\n")
    bt.write("|Function|Scheme|Nside|Radius|Time (healpy)|Time (hpgeom)|hpgeom/healpy|\n")
    bt.write("|--------|------|-----|----|-------------|-------------|-------------|\n")

    function = 'query_circle'

    for scheme in ['nest', 'ring', 'ring2nest']:
        if scheme == 'nest':
            nest = True
        else:
            nest = False

        for radius in np.deg2rad([0.1, 0.5, 1.0]):
            rad_deg = np.rad2deg(radius)
            for nside in [128, 1024, 4096]:
                start_time = time.time()
                for i in range(ntrial):
                    pixels = hpgeom.query_circle(
                        nside,
                        0.0,
                        0.0,
                        radius,
                        nest=nest,
                        lonlat=True,
                        degrees=False
                    )
                    if scheme == 'ring2nest':
                        _ = hpgeom.ring_to_nest(nside, pixels)
                time_hpgeom = (time.time() - start_time)/ntrial

                vec = hp.ang2vec(0.0, 0.0, lonlat=True)
                start_time = time.time()
                for i in range(ntrial):
                    pixels = hp.query_disc(nside, vec, radius, nest=nest)
                    if scheme == 'ring2nest':
                        _ = hp.ring2nest(nside, pixels)
                time_healpy = (time.time() - start_time)/ntrial

                time_ratio = time_hpgeom/time_healpy

                bt.write(f'|{function}|{scheme}|{nside}|{rad_deg}|'
                         f'{time_healpy}|{time_hpgeom}|{time_ratio}|\n')
