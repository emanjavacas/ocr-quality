
    parser.add_argument('--delpher', default='/home/manjavacasema/data/Dutch/delphertxt')
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
