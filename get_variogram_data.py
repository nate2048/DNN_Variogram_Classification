import prepare_data as prepare
import numpy as np
import pandas as pd
import skgstat as skg

print(f'\nStep 1: Process input data\n')

# SGS Data
sgs = prepare.ImgData('Data/SGS')

# DGGAN Data
gan = prepare.ImgData('Data/DCGAN')

# Diffusion Data
diffusion = prepare.DiffusionData('Data/Diffusion.csv')

# Coordinates
coord = pd.read_csv("Data/coords_256.csv").values

print(f'\tFinished Processing Data\n')

print(f'Step 2: Compute Variograms\n')

maxlag = 50000
n_lags = 30
downsample = 0.1

cols = [f'Lag_{i}' for i in range(1,31)]
cols.append('Y')
vario_df = pd.DataFrame(columns = cols, index = range(300))

print(f'\tSGS Variograms\n')

for i in range(len(sgs)):

    vario = skg.Variogram(coord, sgs[i], bin_func = 'even', n_lags = n_lags,
                         maxlag = maxlag, normalize = False, samples = downsample)
    entry = vario.experimental
    entry = np.append(entry, 0)
    vario_df.loc[i] = entry

    print(f'\t\tSGS Variograms {i+1} Completed')

print(f'\tGAN Variograms\n')    

for i in range(len(gan)):

    vario = skg.Variogram(coord, gan[i], bin_func = 'even', n_lags = n_lags,
                         maxlag = maxlag, normalize = False, samples = downsample)
    entry = vario.experimental
    entry = np.append(entry, 1)
    vario_df.loc[(100 + i)] = entry

    print(f'\t\tGAN Variograms {i+1} Completed')


print(f'\Diffusion Variograms\n')    

for i in range(len(diffusion)):

    vario = skg.Variogram(coord, diffusion[i], bin_func = 'even', n_lags = n_lags,
                         maxlag = maxlag, normalize = False, samples = downsample)
    entry = vario.experimental
    entry = np.append(entry, 2)
    vario_df.loc[(200 + i)] = entry

    print(f'\t\tDiffusion Variograms {i+1} Completed')

vario_df.to_csv("Output/Variogram.csv", index=False)