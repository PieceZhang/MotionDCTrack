from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json

import os
import argparse

from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
from toolkit.evaluation import OPEBenchmark


parser = argparse.ArgumentParser(description='tracking evaluation')
parser.add_argument('--tracker_path', '-p', default='./results', type=str,
                    help='tracker result path')
parser.add_argument('--dataset', '-d', default='my_endos', type=str,
                    help='dataset name')
parser.add_argument('--num', '-n', default=1, type=int,
                    help='number of thread to eval')
parser.add_argument('--tracker_prefix', '-t', default='checkpoint_',
                    type=str, help='tracker name')
parser.add_argument('--show_video_level', '-s', dest='show_video_level',
                    action='store_true')
parser.set_defaults(show_video_level=False)
parser.add_argument('--dir', type=str, help='dataset root')
parser.add_argument('--config', default='./config.yaml', type=str, help='config file')

args = parser.parse_args()


def save_result(success_ret, precision_ret, norm_precision_ret, save_path):
    for tracker in success_ret:
        result = {}
        for video in success_ret[tracker]:
            result[video] = {}
            result[video]['suc'] = list(success_ret[tracker][video])
            result[video]['prec'] = list(precision_ret[tracker][video])
            result[video]['nprec'] = list(norm_precision_ret[tracker][video])
        json.dump(result, open(os.path.join(save_path, tracker, 'result.json'), 'w'), indent=4, sort_keys=True)
        print("Result saved at", os.path.join(save_path, tracker, 'result.json'))


def main():
    tracker_dir = os.path.join(args.tracker_path, args.dataset)
    trackers = glob(os.path.join(args.tracker_path,
                                 args.dataset,
                                 args.tracker_prefix+'*'))  # load pth by prefix
    trackers = [x.split('/')[-1] for x in trackers]

    assert len(trackers) > 0
    args.num = min(args.num, len(trackers))

    vallist = 'training_dataset/my_endos/test.json'

    with open('../../' + vallist, 'r') as f:
        vallist = json.load(f)

    opebenchmark = OPEBenchmark(vallist, tracker_dir)

    success_ret = {}
    with Pool(processes=args.num) as pool:
        for ret in tqdm(pool.imap_unordered(opebenchmark.eval_success, trackers), desc='eval success', total=len(trackers), ncols=100):
            success_ret.update(ret)

    precision_ret = {}
    with Pool(processes=args.num) as pool:
        for ret in tqdm(pool.imap_unordered(opebenchmark.eval_precision, trackers), desc='eval precision', total=len(trackers), ncols=100):
            precision_ret.update(ret)

    norm_precision_ret = {}
    with Pool(processes=args.num) as pool:
        for ret in tqdm(pool.imap_unordered(opebenchmark.eval_norm_precision, trackers), desc='eval norm precision', total=len(trackers), ncols=100):
            norm_precision_ret.update(ret)

    opebenchmark.show_result(success_ret, precision_ret, norm_precision_ret, show_video_level=args.show_video_level)

    save_result(success_ret, precision_ret, norm_precision_ret, os.path.join(args.tracker_path, args.dataset))


if __name__ == '__main__':
    main()
