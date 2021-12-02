
import re
import glob
import os
import tqdm
import pickle as pkl

import pandas as pd
from nltk import ngrams
from nltk.lm.models import KneserNeyInterpolated
from nltk.lm.preprocessing import pad_both_ends

import random
random.seed(1001)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--delpher', default='/home/manjavacasema/data/Dutch/delphertxt')
    parser.add_argument('--models-dir', default='./models/')
    args = parser.parse_args()

    lms = {}
    for f in os.listdir(args.models_dir):
        start, stop = re.search('start=([^-]+)-stop=([^-]+).pkl', t).groups()
        with open(f, 'rb') as inp:
            lms[int(start), int(stop)] = pkl.load(inp)

    centers = {stop-start: (start, stop) for start, stop in lms.keys()}

    results = []
    for f in tqdm.tqdm(glob.glob(os.path.join(args.delpher, '/*'))):
        year = int(os.path.basename(f).split('_')[0])
        lm = lms[next(iter(sorted(centers, key=lambda center: abs(center - year))))]
        with open(f) as inp:
            lines = list(enumerate(inp))
            random.shuffle(lines)
            for idx, line in lines[:10]:
                score = lm.entropy(list(ngrams(list(pad_both_ends(line, 5)), 5)))
                results.append({'file': os.path.basename(f), 'line': int(idx), 'score': score})
    
    pd.DataFrame.from_dict(results).to_csv(args.output_file, index=False)