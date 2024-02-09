import gzip
import json
import os

import matplotlib.pyplot as plt
import numpy as np

import _colors
import _folders

experiment_name=f'Robustness'
_folders.set_experiment_folders(experiment_name)

def resultant_error():

    folder_name = 'Robustness\__Results'
    experiment_files = [f for f in os.listdir(folder_name) if f.endswith('.json') or f.endswith('.json.gz')]
    
    print(experiment_files)

    pallette = _colors.create_palette(2)
    plt.figure(figsize=_colors.FIG_SIZE)
    for exp_file in experiment_files:
        print(exp_file)
        if 'Robust' in exp_file:
            marker = '^'
            label = 'Robust,'
        else:
            marker = 'o'
            label = 'Standard,'
        
        if 'tetrominos' in exp_file:
            color = pallette[0]
            label = label + ' Tetrominoes'
        elif 'unknown' in exp_file:
            label = label + ' Unknown'
            color = pallette[1]

        if exp_file.endswith('.gz'):
            with gzip.open(os.path.join(folder_name, exp_file), 'r') as file:
                data = json.load(file)

        else:
            exp_path = os.path.join(folder_name, exp_file)
            with open(exp_path, 'r') as file:
                    data = json.load(file)

    
        by_percent = {}

        for batch in data:
            alive_percent = batch['alive_percent']
            dead_percent = 1 - alive_percent
            if dead_percent not in by_percent:
                by_percent[dead_percent] = []
            by_percent[dead_percent].append(batch)


        means = []
        stds = []

        for al in by_percent.keys():
            batches = by_percent[al]
            error_means = []
            for batch in batches:
                movements = batch['movements']
                for mov in movements:
                    last_mean = mov['means'][-1]
                    error_means.append(last_mean)
            
            error_means = np.array(error_means)
            means.append(np.mean(error_means))
            stds.append(np.std(error_means))

        means = np.array(means)
        stds = np.array(stds)
        alive_percents = np.array(list(by_percent.keys()))

        #plt.errorbar(alive_percents, means, yerr=stds, fmt='o', color=pallette[0], label='Tetrominoes')
        
        dead_percents = [1 - a for a in alive_percents]
        tick_labels = [i*100 for i in dead_percents]
        tick_labels = [f'{i:.0f}' for i in tick_labels]
       
        plt.plot(alive_percents, means, color=color, label=label, marker=marker)
        plt.fill_between(alive_percents, means - stds, means + stds, color=color, alpha=0.2)
        plt.xticks(dead_percents, tick_labels)

    #plot horizontal line at 0.5
    plt.axhline(0.5, color='black', linewidth=1, linestyle='dashed', label=f'Half tile: 0.5')

    plt.legend()
    plt.xlabel('Faulty tiles [%]')
    plt.ylabel('Estimation Error $(\mu_e)$')


    file_path = f'{_folders.VISUALIZATIONS_PATH}/robustness_error.png'
    plt.savefig(file_path, bbox_inches='tight', dpi=600)
    #plt.show()
    plt.clf()
  
resultant_error()
