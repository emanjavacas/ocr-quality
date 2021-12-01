
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

# get start and end dates for each model
ranges = [(year, year+100) for year in range(1600, 1800, 50)]

decades = collections.defaultdict(list)
for f in glob.glob('../../Leiden/Datasets/DBNL/DBNLtxttok/[0-9]*.txt'):
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

    train, padded = padded_everygram_pipeline(5, sents[:100000])
    lm = KneserNeyInterpolated(5)
    lm.fit(train, padded)

    with open('./DBNL-kneser-{}-{}.pkl'.format(start, end), 'wb+') as f:
        pkl.dump(lm, f)


with open('./test.pkl', 'rb') as f:
    lm2 = pkl.load(f)

results = {}
for f in tqdm.tqdm(glob.glob('./delpher/*')):
    with open(f) as inp:
        for idx, line in enumerate(inp):
            score = lm.entropy(list(ngrams(list(pad_both_ends(line, 5)), 5)))
            results[os.path.basename(f), idx] = (line, score)


keys = sorted(results.keys(), key=lambda key: results[key][1])
results[keys[500]]