
import time
import csv
import re
import glob
import os
import tqdm
import pickle as pkl
from joblib import Parallel, delayed
import multiprocessing

import random
random.seed(1001)


def read_lines(f, args):
    try:
        year = int(re.search('([0-9]{4})', f).group())
    except:
        print("Couldn't parse year: ", f)
        return

    with open(f) as inp:
        for line_idx, line in enumerate(inp):
            if not line.strip():
                continue
            yield (f, year, line_idx, line[:args.max_chars_per_line])


def process(lines, centers, lms):
    output = []
    last_year = None

    for fname, year, line_idx, line in lines:
        if last_year is None:
            center = next(iter(sorted(centers, key=lambda center: abs(center - year))))
            last_year = year
        elif last_year != year:
            center = next(iter(sorted(centers, key=lambda center: abs(center - year))))

        lm = lms[centers[center]]
        ent = lm.entropy(list(ngrams(list(pad_both_ends(line, 5)), 5)))
        output.append(
            {'fname': fname,
             'year': year,
             'lineNum': line_idx,
             'entropy': ent,
             'words': len(line.split())})

    return output


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-dir', required=True)
    parser.add_argument('--models-dir', required=True)
    parser.add_argument('--output-path', required=True)
    parser.add_argument('--max-chars-per-line', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=5)
    parser.add_argument('--n-processes', default=0, type=int)
    args = parser.parse_args()
    print(args)

    print("- Loading LMs...")
    lms = {}
    for f in os.listdir(args.models_dir):
        start, stop = re.search('start=([^-]+)-stop=([^-]+).pkl', f).groups()
        with open(os.path.join(args.models_dir, f), 'rb') as inp:
            lms[int(start), int(stop)] = pkl.load(inp)
    print("Done!\n")

    centers = {start + ((stop-start)/2): (start, stop) for start, stop in lms.keys()}

    fieldnames = ['fname', 'year', 'lineNum', 'entropy', 'words']

    # check what we've done already
    done, last = set(), None
    if os.path.isfile(args.output_path):
        with open(args.output_path) as f:
            for row in csv.DictReader(f, delimiter=','):
                if row['fname'] in done:
                    continue
                else:
                    done.add(row['fname'])
                    last = row['fname']
        done.remove(last)
    else:
        with open(args.output_path, 'w+') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    targets = glob.glob(os.path.join(args.target_dir, '*.txt'))
    targets = [f for f in targets if f not in done]
    to_do = len(targets)
    print("{} files to do".format(to_do))

    for i in tqdm.tqdm(range(0, to_do, args.batch_size), total=to_do // args.batch_size):

        batch = targets[i: i+args.batch_size]
        print("Reading...")
        start = time.time()
        lines = [line for f in batch for line in read_lines(f, args)]
        print(lines[:3])
        print("Took {} secs to read {} lines".format(time.time() - start, len(lines)))

        n_processes = args.n_processes or multiprocessing.cpu_count() - 1
        print("Running {} processes".format(n_processes))
        load = len(lines) // n_processes
        output = Parallel(n_jobs=n_processes)(
            delayed(process)(lines[start: start + load], centers, lms) 
            for start in range(0, len(lines), load))

        with open(args.output_path, 'a+') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            for result in output:
                for row in result:
                    writer.writerow(row)
