
import collections
import re
import glob
import os
import tqdm
import pickle as pkl

import pandas as pd
from nltk import ngrams
from nltk.lm.models import KneserNeyInterpolated
from nltk.lm.preprocessing import pad_both_ends, everygrams

import random
random.seed(1001)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-dir', required=True)
    parser.add_argument('--models-dir', required=True)
    parser.add_argument('--freqs-dir', required=True)
    parser.add_argument('--max-lines', default=10, type=int)
    parser.add_argument('--max-lines-per-file', type=int, default=5000)
    parser.add_argument('--output-path', required=True)
    args = parser.parse_args()

    print("- Loading LMs...")
    lms = {}
    for f in os.listdir(args.models_dir):
        start, stop = re.search('start=([^-]+)-stop=([^-]+).pkl', f).groups()
        with open(os.path.join(args.models_dir, f), 'rb') as inp:
            lms[int(start), int(stop)] = pkl.load(inp)
    print("Done!\n")

    centers = {start + ((stop-start)/2): (start, stop) for start, stop in lms.keys()}

    print("- Loading frequency tables...")
    decades = collections.defaultdict(collections.Counter)

    for start, stop in sorted(lms.keys()): # use ranges from language models
        for f in glob.glob(os.path.join('./freqs/', 'DBNL.[0-9]*.freq')):
            _, decade, _ = os.path.basename(f).split('.')
            decade = int(decade)
            if decade >= start and decade < 1800:
                with open(f) as inp:
                    counts = collections.Counter()
                    for line in inp:
                        c, w = line.strip().split()
                        counts[w] = int(c)
                    decades[start, stop].update(counts)

    # normalize vocab
    top_n = 400_000 # this is based on min vocab lengths
    for key, d in decades.items():
        decades[key] = {w: c for w, c in d.most_common(top_n)}

    bands = [top_n // div for div in [1e3, 1e2, 1e1]]

    decades_bands = collections.defaultdict(dict)
    for key, d in decades.items():
        for rank, w in enumerate(sorted(d, key=lambda w: d[w], reverse=True)):
            rank += 1
            for band_id, band in enumerate(bands):
                if rank < band:
                    decades_bands[key][w] = band_id
                    break
            else:
                decades_bands[key][w] = len(bands)

    print("Done!\n")

    results = []
    for f in tqdm.tqdm(glob.glob(os.path.join(args.target_dir, '*.txt'))):
        year = re.search('([0-9]+)', f)
        if not year:
            continue
        year = int(year.group())
        center = next(iter(sorted(centers, key=lambda center: abs(center - year))))
        lm = lms[centers[center]]
        lookup_d = decades_bands[centers[center]]

        with open(f) as inp:
            lines = []
            for idx, line in enumerate(inp):
                if idx > args.max_lines_per_file:
                    break
                lines.append((idx, line))
            random.shuffle(lines)
            for idx, line in lines[:args.max_lines]:
                # get entropy
                entropy_score = lm.entropy(list(ngrams(list(pad_both_ends(line, 5)), 5)))
                # get lookup scores
                found = set(
                    ''.join(gram) for w in line.split() for gram in everygrams(w, min_len=2)
                ).intersection(lookup_d)
                lookup = collections.Counter()
                for ngram in found:
                    lookup['band-{}'.format(lookup_d[ngram])] += 1
                size = len(line)
                snippet = line[max((size//2)-100, 0):(size//2)+100]
                results.append(
                    dict(file=os.path.basename(f), line=int(idx), snippet=snippet,
                         year=year, nChars=size, nWords=len(line.split()),
                         entropy=entropy_score, **lookup))
    
    pd.DataFrame.from_dict(results).to_csv(args.output_path, index=False)
