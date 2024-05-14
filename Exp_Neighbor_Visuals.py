import json
import os

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

import _colors
import _folders

_folders.set_experiment_folders('Neighbor_Hidden_Size')

def _get_grouped_data(save:bool=False):
    experiment_files = [f for f in os.listdir(_folders.RESULTS_PATH) if f.endswith('.json') and 'Training' in f]
    grouped_data = {}

    for exp in experiment_files:
        neighborhood = exp.split('_')[1]
        run = int(exp.split('_')[-1].split('.')[0])
        hs = int(exp.split('_')[-3])

        if neighborhood not in grouped_data:
            grouped_data[neighborhood] = {}

        if hs not in grouped_data[neighborhood]:
            grouped_data[neighborhood][hs] = []

        exp_path = os.path.join(_folders.RESULTS_PATH, exp)
        with open(exp_path, 'r') as file:
            data = json.load(file)

        steps = data['training_results']
        losses = [step['loss'] for step in steps]
        times = [step['time'] for step in steps]
        grouped_data[neighborhood][hs].append(losses)

    if save:
        with open(f'{_folders.RESULTS_PATH}/grouped_data.json', 'w') as file:
            json.dump(grouped_data, file)

    return grouped_data

def plot_training_results():
    grouped_data = _get_grouped_data()
    sizes = list(set(np.concatenate([list(map(int, grouped_data[neighborhood].keys())) for neighborhood in grouped_data])))            
    palette = _colors.create_palette(len(sizes)+1)

    fig, ax = plt.subplots(figsize=_colors.FIG_SIZE)
    for neighborhood in grouped_data:
        print(neighborhood)
        HS = np.array(list(grouped_data[neighborhood].keys())).astype(int)
        HS = HS[np.argsort(HS)]
        for hs in HS:
            losses = np.array(grouped_data[neighborhood][hs])
            size_lost = np.mean(losses, axis=0)
            size_std = np.std(losses, axis=0)
            
            n = 50

            smoothed_loss = size_lost[::n]
            std_dev = size_std[::n]
            # Include the last element
            smoothed_loss = np.append(smoothed_loss, size_lost[-1])
            std_dev = np.append(std_dev, size_std[-1])

            #split_normalized_loss = np.reshape(size_lost, (len(size_lost)//n, n))
            #smoothed_loss = np.mean(split_normalized_loss, axis=1)
            #std_dev = np.std(split_normalized_loss, axis=1)

            x_ticks = np.arange(0, len(smoothed_loss) * n, n)  # Create x ticks at every n steps
            
            #color
            linestyle = '-' if neighborhood == 'Manhattan' else '--'
            size_index = sizes.index(hs)+1
            color = palette[size_index]
            plt.fill_between(x_ticks, smoothed_loss - std_dev, smoothed_loss + std_dev, alpha=0.1, color=color)
            plt.plot(x_ticks, smoothed_loss, color=color, linestyle=linestyle)



    plt.plot([], [], '--', label=f'Moore', color='black')
    plt.plot([], [], '-', label=f'Von Neumann', color='black')
    
    #for i,hs in enumerate(sizes):
    #    plt.plot([], [], 'o', label=f'$H_S$:{hs}', color=palette[i+1])


    cmap = ListedColormap(palette)
    bounds = range(len(palette))
    norm = plt.Normalize(min(bounds), max(bounds))
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    ticks = [bound + 0.5 for bound in bounds[:-1]]

    # Specify the Axes (ax) for the Colorbar
    cbar = plt.colorbar(sm, ax=ax, ticks=ticks, boundaries=bounds, format='%1i', extend='neither', extendfrac='auto')
    cbar.set_label('Hidden Channels Size')
    cbar.ax.set_position(cbar.ax.get_position().translated(-0.025, 0))

    #plt.legend(ncol=4)
    plt.legend()
    plt.xlabel('Training Step')
    plt.ylabel('Loss [log scale]')
    plt.yscale('log')
    
    # 10 logaritmic values between 0.03 and 0.4, using np
    ticks = [0.03, 0.05, 0.1, 0.2,0.3,0.4]
    plt.yticks(ticks, ticks)

    file_path = f'{_folders.VISUALIZATIONS_PATH}/training_performance.png'
    plt.savefig(file_path, bbox_inches='tight', dpi=600)
    plt.show()
    plt.clf()

def plot_final_loss():
    grouped_data = _get_grouped_data()
    sizes = list(set(np.concatenate([list(map(int, grouped_data[neighborhood].keys())) for neighborhood in grouped_data])))            
    palette = _colors.create_palette(len(sizes)+1)

    mcm, mcs, ccm, ccs = [], [], [], []

    for neighborhood in grouped_data:
        print(neighborhood)
        HS = np.array(list(grouped_data[neighborhood].keys())).astype(int)
        HS = HS[np.argsort(HS)]
        for hs in HS:
            last_n = 100
            losses = np.array(grouped_data[neighborhood][hs])[:,-last_n:]
            losses = np.reshape(losses, (losses.shape[0]*losses.shape[1],))
            size_lost = np.mean(losses)
            size_std = np.std(losses)
            
            if neighborhood == 'Manhattan':
                mcm.append(size_lost)
                mcs.append(size_std)
            else:
                ccm.append(size_lost)
                ccs.append(size_std)

            
    mcm, mcs, ccm, ccs = np.array(mcm), np.array(mcs), np.array(ccm), np.array(ccs)
    

    plt.figure(figsize=_colors.FIG_SIZE)
    
    palette2 = _colors.create_palette(2)
    plt.plot(sizes, ccm, color='black', linestyle='--', label='Moore')
    plt.fill_between(sizes, ccm - ccs, ccm + ccs, alpha=0.1, color=palette[0])
    plt.plot(sizes, mcm, color='black', linestyle='-', label='Von Neumann')
    plt.fill_between(sizes, mcm - mcs, mcm + mcs, alpha=0.1, color=palette[-1])


    from itertools import zip_longest

    for i, (mm, mc) in enumerate(zip_longest(mcm, ccm)):
        size_index = sizes.index(i)+1
        color = palette[size_index]
        plt.plot(i, mm, color=color, linestyle='-', marker='o', markeredgecolor='black')
        plt.plot(i, mc, color=color, linestyle='-', marker='o', markeredgecolor='black')
    

    plt.legend()
    plt.xticks(sizes)

    plt.xlabel('Hidden Channels Size')
    plt.ylabel('Final Loss')

    file_path = f'{_folders.VISUALIZATIONS_PATH}/final_loss.png'
    plt.savefig(file_path, bbox_inches='tight', dpi=600)
    
    #plt.show()
    plt.clf()


plot_training_results()
plot_final_loss()