import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import seaborn as sns
from scipy.spatial.distance import pdist
from itertools import combinations
from scipy.stats import wilcoxon
from functools import partial
import os
from tkinter import filedialog
from tkinter import *

def read_data_multicolor(f, n):
    d = pd.read_csv(f, skiprows=3)
    d = d.sort_values('ID').iloc[:, 0:3].reset_index(drop=True)
    d['Image'] = f.split('/')[-1].split(' - ')[0]
    if d.shape[0]%n!=0:
        raise ValueError("Number of measurements incompatible with number of\
                         probes! Please check input data for missing or extra\
                         spots. File %s." % f)
    d['colour'] = list(np.arange(n))*(d.shape[0]//n)
    d['group'] = np.repeat(np.arange(d.shape[0]//n), n)
    return d

def distance(group, probe_names):
    return pd.Series(pdist(group.iloc[:, :3]), index=combinations(probe_names, 2))

def make_distances_multicolour(coords, probe_names):
    dists = coords.groupby(['Image', 'group']).apply(distance, probe_names=probe_names)
    return pd.DataFrame(dists).reset_index()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--probes", default=None, type=str, required=False)
    parser.add_argument("--folder", default=None, type=str, required=False)
    args = parser.parse_args()
    if args.probes is None: #Get probes interactively
#         def get_number():
#             global w
#             n = w.get()
#             global N_probes
#             N_probes.set(int(n))
#             root.destroy()

#         root = Tk()
#         root.title('Number of probes')

#         w = Spinbox(root, from_=0, to=10)
#         w.pack()

#         b = Button(root, text="OK", command=get_number)
#         b.pack()

#         N_probes = IntVar()

#         root.mainloop()
#         N = N_probes.get()

        root = Tk()
        root.title('Probe names')
        probes = [StringVar() for i in range(4)]
        es = [Entry(root, textvariable=p) for p in probes]
        for e in es:
            e.pack()

        def callback():
            for e in es:
                e.get()
            root.destroy()

        b = Button(root, text="Done", width=10, command=callback)
        b.pack()

        mainloop()
        probes = [probe.get() for probe in probes]
        probes = list(filter(lambda x: bool(x) is True, probes))
        N = len(probes)
    else:
        probes = args.probes.split(',')
        N = len(probes)

    print('Got %s probes: %s' % (N, probes))

    if args.folder is None:
        def browse_button():
            # Allow user to select a directory and store it in global var
            # called folder_path
            global folder_path
            filename = filedialog.askdirectory()
            folder_path.set(filename)
            print(filename)

        root = Tk()
        folder_path = StringVar()
        browse_button()
        root.destroy()
        folder = folder_path.get()
    else:
        folder = args.folder

    print("Analysing folder %s" % folder)

    files = filter(lambda x: os.path.isfile and x.endswith('.csv'), glob.glob(folder+'/*/*') + glob.glob(folder+'/*'))

    f_dist = partial(make_distances_multicolour, probe_names=probes)
    f_read = partial(read_data_multicolor, n=N)
    data = pd.concat(map(f_dist, map(f_read, files))).reset_index(drop=True)
    f_sort = partial(sorted, key=lambda x: probes.index(x))
    data.columns = list(data.columns[:2])+[tuple(i) for i in map(sorted, data.columns[2:])]
    data.to_csv(os.path.join(folder, 'Distances.tsv'), index=False, sep='\t')

    order = data.columns[2:]
    sns.set_context('notebook')
    f, ax = plt.subplots(figsize=(8, 6))
    sns.swarmplot(data=data, order=order, ax=ax)
    sns.boxplot(data=data, order=order, palette=['grey'], showfliers=False,
                ax=ax)
    ax.set_ylabel('Distance, $\mu m$')
    ax.set_xlabel('Probe pair')
    ax.set_title('n=%s' % data.shape[0])
    plt.savefig(os.path.join(folder, 'Plot.png'), dpi=150)

    pvals = pd.DataFrame(index=order, columns=order)
    for i, j  in combinations(data.columns[2:], 2):
        _, p = wilcoxon(data[i], data[j])
        pvals.loc[i, j] = p
        pvals.loc[j, i] = p
    pvals.to_csv(os.path.join(folder, 'Pvals_Wilcoxon.tsv'), sep='\t')
