
import os
import sys

import argparse
import numpy as np
from tqdm import tqdm
import pdnl_sana.image
import pdnl_sana.process
import pdnl_sana.segment

from matplotlib import pyplot as plt

def main():

    debug = False

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', help="path to pdnl_process outputs", required=True)
    parser.add_argument('--checkpoint', action='store_true', help="saves intermediate data")
    parser.add_argument('--model_path', help="path to ML model", default=None)
    args = parser.parse_args()

    print('Processing chunks...', flush=True)
    for d in tqdm(os.listdir(args.input_path)):
        i, j = list(map(int, d.split('_')))

        in_d = os.path.join(args.input_path, f'{i}_{j}')
        frame = pdnl_sana.image.Frame(os.path.join(in_d.replace('dab', 'chunks'), 'frame.png'))
        stain = pdnl_sana.image.Frame(os.path.join(in_d, 'stain.npy'))
        pos = pdnl_sana.image.Frame(os.path.join(in_d, 'pos.png'))

        pos.img = pos.img // np.max(pos.img); pos.to_short()

        if args.checkpoint and os.path.exists(os.path.join(in_d, 'ctrs.npy')):
            soma_ctrs = np.load(os.path.join(in_d, 'ctrs.npy'))
        else:
            soma_ctrs = pdnl_sana.segment.detect_somas(
                pos, 
                minimum_soma_radius=5,
                debug=False,
            )
            if args.checkpoint:
                np.save(os.path.join(in_d, 'ctrs.npy'), soma_ctrs)

        if args.checkpoint and os.path.exists(os.path.join(in_d, 'somas.png')):
            soma_mask = pdnl_sana.image.Frame(os.path.join(in_d, 'somas.png'))
        else:
            soma_mask = pdnl_sana.segment.segment_somas(pos, soma_ctrs, n_directions=8, stride=4, sigma=3, fm_threshold=12, npools=1, debug=False)
            if args.checkpoint:
                soma_mask.save(os.path.join(in_d, 'somas.png'))

        soma_mask.img = (soma_mask.img / np.max(soma_mask.img)).astype(np.uint8)

        soma_polys = soma_mask.to_polygons()[0]

        # TODO: microglia_mask instead of pos, build this by thresholding pos.to_polygons() area
        if args.checkpoint and os.path.exists(os.path.join(in_d, 'microglia')):
            microglia_instances = []
            for f in os.listdir(os.path.join(in_d, 'microglia')):
                x = np.load(os.path.join(in_d, 'microglia', f), allow_pickle=True)

                # TODO: handle level!
                loc = pdnl_sana.geo.Point(*x['loc'], is_micron=False, level=0)
                size = pdnl_sana.geo.Point(*x['size'], is_micron=False, level=0)
                instance = pdnl_sana.segment.MicrogliaInstance(
                    soma=x['soma'], 
                    skeleton=x['skeleton'], 
                    mask=x['mask'], 
                    loc=loc, size=size
                )
                microglia_instances.append(instance)
        else:
            microglia_instances = pdnl_sana.segment.segment_microglia(pos, soma_polys, debug=False)
            if args.checkpoint:
                d = os.path.join(os.path.join(in_d, 'microglia'))
                if not os.path.exists(d):
                    os.makedirs(d)
                    for i, x in enumerate(microglia_instances):
                        x.save(os.path.join(d, f'{i}.npz'))

        features = np.array([x.to_features() for x in microglia_instances])

        if not args.model_path is None:
            import pickle
            ss = pickle.load(open(os.path.join(args.model_path, 'ss_v2.pkl'), 'rb'))
            clf = pickle.load(open(os.path.join(args.model_path, 'model_v2.pkl'), 'rb'))

            features = ss.transform(features)
            probs = clf.predict_proba(features)
            probs = np.array([prob[:,1] for prob in probs]).T

        if debug:
            fig, axs = plt.subplots(2,2, sharex=True, sharey=True)
            axs = axs.ravel()

            # somas
            axs[0].imshow(frame.img)
            [axs[0].plot(*x.T, color='red') for x in soma_polys]

            # skeletons
            skeleton_frame = frame.copy()
            for x in microglia_instances:
                x.overlay_skeleton(skeleton_frame, color=[255,0,0])
            axs[1].imshow(skeleton_frame.img)
        
            # microglia
            microglia_frame = frame.copy()
            for x in microglia_instances:
                x.overlay_microglia(microglia_frame, color=[255,0,0])
            axs[2].imshow(microglia_frame.img)

            # predictions
            if not args.model_path is None:
                pred = frame.copy()
                colors = [[255,0,0],[0,255,0],[0,0,255],[255,255,0],[0,255,255]]
                for i, x in enumerate(microglia_instances):
                    x.overlay_prediction(pred, probs[i], colors)
                axs[3].imshow(pred.img)

            plt.show()

    # TODO: some kind of aggregation?


if __name__ == "__main__":
    main()
