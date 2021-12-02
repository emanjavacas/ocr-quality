
import tqdm
import pickle as pkl
import random
import os
import collections
import glob
from nltk import collections
from nltk.tokenize import sent_tokenize
from nltk.lm.preprocessing import padded_everygram_pipeline, pad_both_ends
from nltk.lm.models import MLE, KneserNeyInterpolated
from nltk import ngrams


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dbnl', default='/home/manjavacasema/data/Dutch/DBNL/DBNLtxt/')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-sents', type=int, default=500000)
    parser.add_argument('--order', type=int, default=5)
    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    # get start and end dates for each model
    ranges = [(year, year+100) for year in range(1600, 1800, 50)]

    decades = collections.defaultdict(list)
    for f in glob.glob(os.path.join(args.dbnl, '[0-9]*.txt')):
        decade = int(os.path.basename(f).split('.')[0])
        if decade < 1600 or decade > 1800: continue
        decades[decade].append(f)

    for start, end in ranges:
        sents = []
        for decade, files in decades.items():
            if decade >= start and decade < end:
                for f in files:
                    with open(f) as inp:
                        for sent in sent_tokenize(inp.read()):
                            sents.append(sent)
            else:
                continue

        random.shuffle(sents)
        print(start, end, len(sents))

        train, padded = padded_everygram_pipeline(args.order, sents[:args.max_sents])
        lm = KneserNeyInterpolated(args.order)
        lm.fit(train, padded)

        output_file = os.path.join(
            args.output_dir,
            'DBNL-kneser-{}-start={}-stop={}.pkl'.format(args.order, start, end))
        with open(output_file, 'wb+') as f:
            pkl.dump(lm, f)
