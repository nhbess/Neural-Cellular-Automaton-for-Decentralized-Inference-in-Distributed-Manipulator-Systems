import json
import os

import matplotlib.pyplot as plt
import numpy as np

import _colors
import Environment.Shapes as Shapes
import _folders
import gzip


experiment_name=f'Performance'
_folders.set_experiment_folders(experiment_name)

def _get_index_stability(means:list, threshold = 0.01):
    for i in range(len(means), 0, -1):
        current_value = means[i - 1]
        final_value = means[-1]
        diff = abs(current_value - final_value)
        if diff > threshold*final_value:
            return i
    return 0

def old():
    folder_name = 'Performance\__Results'
    experiment_files = [f for f in os.listdir(folder_name) if f.endswith('.json')]
    #experiment_files = sorted(experiment_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    for exp in experiment_files:
        exp_path = os.path.join(folder_name, exp)
        with open(exp_path, 'r') as file:
            data = json.load(file)
        
        if exp == 'tetrominos.json':
            shapes = Shapes.tetrominos
            names_shapes = list(Shapes.tetros_dict.keys())
            file_name = 'tetrominos'
        elif exp == 'unknown_shapes.json':
            shapes = Shapes.unknown_shapes
            names_shapes = list(Shapes.unknown_shapes_dict.keys())
            file_name = 'unknown_shapes'
        else:
            raise Exception('Unknown shapes')
        
        palette = _colors.create_palette(len(shapes))
        #colors.show_palette(palette)

        for i, batch in enumerate(data):
            name_shape = batch['name_shape']
            scaler = batch['scaler']
            movements = batch['movements']

            color = palette[names_shapes.index(name_shape)]

            if False:
                means = np.array([mov['means'] for mov in movements])
                stds = np.array([mov['stds'] for mov in movements])
                
                #resize to a single list
                means = np.reshape(means, (means.shape[0] * means.shape[1],))
                stds = np.reshape(stds, (stds.shape[0] * stds.shape[1],))

                update_steps = len(means)
                x_ticks = np.arange(0, update_steps)

                plt.plot(x_ticks, means, color=color)
                #plt.fill_between(x_ticks, means - stds, means + stds, alpha=0.2, color=color)

            else:
                num_movements = len(movements)
                for mov in movements:
                    angle = mov['angle']
                    true_center = mov['true_center']
                    target_center = mov['target_center']
                    means = np.array(mov['means'])
                    stds = np.array(mov['stds'])

                    update_steps = len(means)
                    x_ticks = np.arange(0, update_steps)

                    alpha =1/num_movements
                    alpha =0.2

                    plt.plot(x_ticks, means, color = color, alpha =alpha)
                    #plt.fill_between(x_ticks, means - stds, means + stds, alpha=0.2)
        

        for i,ns in enumerate(names_shapes):
            plt.plot([], [], 'o', label=f'{ns}', color=palette[i])

        plt.legend()
        plt.xlabel('Update Steps')
        plt.ylabel('Error')
        #plt.yscale('log')
        
        #save
        file_path = f'{_folders.VISUALIZATIONS_PATH}/{file_name}.png'
        plt.savefig(file_path)
        plt.show()
        plt.clf()

def convergence_update_steps():
    experiment_files = [f for f in os.listdir(_folders.RESULTS_PATH) if f.endswith('.json') or f.endswith('.gz')]
    stabilization_steps = {'tetrominos': [], 
                           'unknown_shapes': []}
    
    for exp in experiment_files:
        exp_path = os.path.join(_folders.RESULTS_PATH, exp)
        
        if exp.endswith('.gz'):
            with gzip.open(exp_path, 'rb') as file:
                data = json.load(file)
                exp = exp.replace('.gz', '')
        else:
            with open(exp_path, 'r') as file:
                data = json.load(file)
        
        if exp == 'tetrominos.json':
            shapes = Shapes.tetrominos
            names_shapes = list(Shapes.tetros_dict.keys())
            file_name = 'tetrominos'
        elif exp == 'unknown_shapes.json':
            shapes = Shapes.unknown_shapes
            names_shapes = list(Shapes.unknown_shapes_dict.keys())
            file_name = 'unknown_shapes'
        else:
            raise Exception('Unknown shapes')
        
        for i, batch in enumerate(data):
            name_shape = batch['name_shape']
            scaler = batch['scaler']
            movements = batch['movements']
            for mov in movements:
                means = np.array(mov['means'])
                stds = np.array(mov['stds'])       
                stabilization_steps[file_name].append(_get_index_stability(means))                

    tetros_times = np.array(stabilization_steps['tetrominos'])
    unknown_times = np.array(stabilization_steps['unknown_shapes'])

    bins = 41
    pallette = _colors.create_palette(2)

    plt.hist(tetros_times, bins=bins, alpha=0.6, label='Tetrominos', color=pallette[0],weights=(np.ones_like(tetros_times) / len(tetros_times))*100)
    plt.hist(unknown_times, bins=bins, alpha=0.6, label='Unknown Shapes', color=pallette[1],weights=(np.ones_like(unknown_times) / len(unknown_times))*100)

    tetro_mean = np.mean(tetros_times)  
    unknown_mean = np.mean(unknown_times)
    tetro_std = np.std(tetros_times)
    unknown_std = np.std(unknown_times)
    
    #print(f'Tetrominos mean: {tetro_mean:.2f} +- {tetro_std:.2f}')
    #print(f'Unknown mean: {unknown_mean:.2f} +- {unknown_std:.2f}')
    
    plt.axvline(tetro_mean, color=pallette[0], linestyle='dashed', linewidth=1, label=f'Tetrominoes mean: {tetro_mean:.2f}')
    plt.axvline(unknown_mean, color=pallette[1], linestyle='dashed', linewidth=1, label=f'Unknown mean: {unknown_mean:.2f}')

    plt.legend()
    plt.xlabel('Update Steps')
    plt.ylabel('Frequency %')

    file_path = f'{_folders.VISUALIZATIONS_PATH}/convergence_histogram.png'
    plt.savefig(file_path)
    #plt.show()
    plt.clf()

def resultant_error():
    experiment_files = [f for f in os.listdir(_folders.RESULTS_PATH) if f.endswith('.json') or f.endswith('.gz')]

    final_values = {'tetrominos': [], 
                    'unknown_shapes': []}
    
    for exp in experiment_files:
        exp_path = os.path.join(_folders.RESULTS_PATH, exp)

        if exp.endswith('.gz'):
            with gzip.open(exp_path, 'rb') as file:
                data = json.load(file)
                exp = exp.replace('.gz', '')
        else:
            with open(exp_path, 'r') as file:
                data = json.load(file)
        
        if exp == 'tetrominos.json':
            shapes = Shapes.tetrominos
            names_shapes = list(Shapes.tetros_dict.keys())
            file_name = 'tetrominos'
        elif exp == 'unknown_shapes.json':
            shapes = Shapes.unknown_shapes
            names_shapes = list(Shapes.unknown_shapes_dict.keys())
            file_name = 'unknown_shapes'
        else:
            raise Exception('Unknown shapes')
        
        for i, batch in enumerate(data):
            name_shape = batch['name_shape']
            scaler = batch['scaler']
            movements = batch['movements']
            
            for mov in movements:
                last_mean = mov['means'][-1]
                final_values[file_name].append(last_mean)

    tetros_times = np.array(final_values['tetrominos'])
    unknown_times = np.array(final_values['unknown_shapes'])
    
    bins = 50
    pallette = _colors.create_palette(2)

    plt.subplots(figsize=_colors.FIG_SIZE)
    plt.hist(tetros_times, bins=bins, alpha=0.6, label='Tetrominoes', color=pallette[0],weights=100*np.ones_like(tetros_times) / len(tetros_times))
    plt.hist(unknown_times, bins=bins, alpha=0.6, label='Unknown Shapes', color=pallette[1],weights=100*np.ones_like(unknown_times) / len(unknown_times))

    tetro_mean = np.mean(tetros_times)  
    tetro_median = np.median(tetros_times)  
    unknown_mean = np.mean(unknown_times)
    unknown_median = np.median(unknown_times)
    
    #mode
    #print(f'Tetrominos mean: {tetro_mean:.2f} +- {tetro_std:.2f}')

    tetro_std = np.std(tetros_times)
    unknown_std = np.std(unknown_times)
    #print(f'Tetrominos mean: {tetro_mean:.2f} +- {tetro_std:.2f}')
    #print(f'Unknown mean: {unknown_mean:.2f} +- {unknown_std:.2f}')

    plt.axvline(tetro_mean, color=pallette[0],  linewidth=1, label=f'Tetrominoes $\mu_e$ mean: {tetro_mean:.2f}', alpha=0.6)
    plt.axvline(unknown_mean, color=pallette[1], linewidth=1, label=f'Unknown $\mu_e$ mean: {unknown_mean:.2f}', alpha=0.8)
    
    
    
    newpalette = _colors.create_palette(5)
    #colors.show_palette(newpalette)
    plt.axvline(0.5, color='black', linewidth=1, linestyle='dashed', label=f'Half tile: 0.5')
    #plt.axvline(tetro_median, color=pallette[0], linestyle='dotted', linewidth=1, label=f'Tetrominoes median: {tetro_median:.2f}')
    #plt.axvline(unknown_median, color=pallette[1], linestyle='dotted', linewidth=1, label=f'Unknown median: {unknown_median:.2f}')

    
    plt.legend()
    plt.xlabel('Estimation Error $(\mu_e)$')
    plt.ylabel('Frequency [%]')
    
    file_path = f'{_folders.VISUALIZATIONS_PATH}/resultant_error.png'
    plt.savefig(file_path, bbox_inches='tight', dpi=600)
    #plt.show()
    plt.clf()
    
def resultant_error_std():
    experiment_files = [f for f in os.listdir(_folders.RESULTS_PATH) if f.endswith('.json') or f.endswith('.gz')]

    final_values = {'tetrominos': [], 
                    'unknown_shapes': []}
    
    for exp in experiment_files:
        exp_path = os.path.join(_folders.RESULTS_PATH, exp)

        if exp.endswith('.gz'):
            with gzip.open(exp_path, 'rb') as file:
                data = json.load(file)
                exp = exp.replace('.gz', '')
        else:
            with open(exp_path, 'r') as file:
                data = json.load(file)
        
        if exp == 'tetrominos.json':
            file_name = 'tetrominos'
        elif exp == 'unknown_shapes.json':
            file_name = 'unknown_shapes'
        else:
            raise Exception('Unknown shapes')
        
        for i, batch in enumerate(data):
            movements = batch['movements']       
            for mov in movements:
                #last_mean = mov['means'][-1]
                last_std = mov['stds'][-1]
                final_values[file_name].append(last_std)

    tetros_times = np.array(final_values['tetrominos'])
    unknown_times = np.array(final_values['unknown_shapes'])
    
    bins = 50
    pallette = _colors.create_palette(2)

    plt.hist(tetros_times, bins=bins, alpha=0.6, label='Tetrominos', color=pallette[0],weights=100*np.ones_like(tetros_times) / len(tetros_times))
    plt.hist(unknown_times, bins=bins, alpha=0.6, label='Unknown Shapes', color=pallette[1],weights=100*np.ones_like(unknown_times) / len(unknown_times))

    tetro_mean = np.mean(tetros_times)  
    unknown_mean = np.mean(unknown_times)
    tetro_std = np.std(tetros_times)
    unknown_std = np.std(unknown_times)
    #print(f'Tetrominos mean: {tetro_mean:.2f} +- {tetro_std:.2f}')
    #print(f'Unknown mean: {unknown_mean:.2f} +- {unknown_std:.2f}')

    plt.axvline(tetro_mean, color=pallette[0], linestyle='dashed', linewidth=1, label=f'Tetrominoes mean: {tetro_mean:.2f}')
    plt.axvline(unknown_mean, color=pallette[1], linestyle='dashed', linewidth=1, label=f'Unknown mean: {unknown_mean:.2f}')

    plt.legend()
    plt.xlabel('Error Standard Deviation Distribution')
    plt.ylabel('Frequency %')

    file_path = f'{_folders.VISUALIZATIONS_PATH}/resultant_error_std.png'
    plt.savefig(file_path)
    #plt.show()
    plt.clf()
 
def by_shape():
    experiment_files = [f for f in os.listdir(_folders.RESULTS_PATH) if f.endswith('.json') or f.endswith('.gz')]
    shapes_data = {}
    for exp in experiment_files:
        exp_path = os.path.join(_folders.RESULTS_PATH, exp)
        
        if exp.endswith('.gz'):
            with gzip.open(exp_path, 'rb') as file:
                data = json.load(file)
                exp = exp.replace('.gz', '')
        else:
            with open(exp_path, 'r') as file:
                data = json.load(file)

        for i, batch in enumerate(data):
            name_shape = batch['name_shape']
            movements = batch['movements']
            for mov in movements:
                means = np.array(mov['means'])
                last_mean = means[-1]
                if name_shape not in shapes_data:
                    shapes_data[name_shape] = []
                
                shapes_data[name_shape].append(last_mean)
    
    #print(shapes_data.keys())

    tetros_shapes = list(Shapes.tetros_dict.keys())
    unknown_shapes = list(Shapes.unknown_shapes_dict.keys())
    
    shapes_to_plot = [tetros_shapes, unknown_shapes]
    shapes_to_plot = [tetros_shapes + unknown_shapes]

    for shapes, names in zip(shapes_to_plot, ['tetrominoes', 'unknown_shapes']):    
        palette = _colors.create_palette(len(shapes)//2)
        data_to_plot = [shapes_data[s] for s in shapes]
        fig, ax = plt.subplots(figsize=_colors.FIG_SIZE)
        vp = ax.violinplot(data_to_plot, 
                        showmedians=False,
                        showmeans=False, 
                        showextrema=True,
                        widths=0.5,
                        )

        for i, violin in enumerate(vp['bodies']):
            violin.set_facecolor(palette[i%len(palette)])  # Assign color from the palette
            violin.set_alpha(1)  # Set the desired alpha value (0.0 to 1.0)'
            violin.set_edgecolor('black')
            violin.set_linewidth(1)

        for partname in ('cbars','cmins','cmaxes',):
            vp[partname].set_edgecolor('black')
            vp[partname].set_linewidth(1)
        
    for i, violin in enumerate(vp['bodies']):
        x = i+1
        y = np.median(data_to_plot[i])
        z = np.mean(data_to_plot[i])
        #ax.plot(x, y, 'o', color='darkgrey', markersize=5, markeredgecolor='black')
        ax.plot(x, z, 'o', color='darkgrey', markersize=5, markeredgecolor='black')
        if i == 0:
            #ax.plot(x, y, 'o', color='darkgrey', markersize=5, markeredgecolor='black', label='Median')
            ax.plot(x, z, 'o', color='darkgrey', markersize=5, markeredgecolor='black', label='Shape $\mu_e$ mean')
                    

    #plot horizontal line
    ax.axhline(0.5, color='black', linestyle='dashed', linewidth=1, label=f'Half tile: 0.5')
    ax.set_xticks([i+1 for i in range(len(shapes))])
    ax.set_xticklabels(shapes)
    ax.set_ylabel('Estimation Error $(\mu_e)$')
    ax.legend()
    file_path = f'{_folders.VISUALIZATIONS_PATH}/{names}_violin.png'
    plt.savefig(file_path, bbox_inches='tight', dpi=600)
    #plt.show()
    plt.clf()

#convergence_update_steps()
resultant_error()
#resultant_error_std()
by_shape()