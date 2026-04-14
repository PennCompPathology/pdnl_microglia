
import os
import sys

import argparse
import numpy as np
import pdnl_sana.image
import pdnl_sana.process
import pdnl_sana.segment

from matplotlib import pyplot as plt

def main():

    debug = True

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', help="path to pdnl_process outputs", required=True)
    parser.add_argument('--checkpoint', action='store_true', help="saves intermediate data")
    args = parser.parse_args()

    stain = pdnl_sana.image.Frame(os.path.join(args.input_path, 'stain.npy'))
    pos = pdnl_sana.image.Frame(os.path.join(args.input_path, 'pos.png'))

    pos.img = pos.img // np.max(pos.img); pos.to_short()

    if args.checkpoint and os.path.exists(os.path.join(args.input_path, 'ctrs.npy')):
        soma_ctrs = np.load(os.path.join(args.input_path, 'ctrs.npy'))
    else:
        soma_ctrs = pdnl_sana.segment.detect_somas(
            pos, 
            minimum_soma_radius=5,
        )
        if args.checkpoint:
            np.save(os.path.join(args.input_path, 'ctrs.npy'), soma_ctrs)

    if args.checkpoint and os.path.exists(os.path.join(args.input_path, 'somas.png')):
        soma_mask = pdnl_sana.image.Frame(os.path.join(args.input_path, 'somas.png'))
    else:
        soma_mask = pdnl_sana.segment.segment_somas(pos, soma_ctrs, n_directions=8, stride=4, sigma=3, fm_threshold=12, npools=1, debug=debug)
        if args.checkpoint:
            soma_mask.save(os.path.join(args.input_path, 'somas.png'))

    soma_mask.img = (soma_mask.img / np.max(soma_mask.img)).astype(np.uint8)

    soma_polys = soma_mask.to_polygons()[0]

    # TODO: microglia_mask instead of pos, build this by thresholding pos.to_polygons() area

    microglia_instances = pdnl_sana.segment.segment_microglia(pos, soma_polys, debug=debug)
    if args.checkpoint:
        d = os.path.join(os.path.join(args.input_path, 'microglia'))
        if not os.path.exists(d):
            os.makedirs(d)
        for i, x in enumerate(microglia_instances):
            x.save(os.path.join(d, f'{i}.npz'))


    if debug:
        fig, ax = plt.subplots(1,1)
        ax.imshow(stain.img, cmap='gray')
        [ax.plot(*x.T, color='red') for x in soma_polys]
        plt.show()


if __name__ == "__main__":
    main()
