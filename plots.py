import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Plot epoch history
def plot_epoch_history(history):
  font = {
    'family': 'serif',
    'color':'darkblue',
    'weight':'normal',
    'size': 14
  }

  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure(figsize=(6, 4))

  plt.xlabel('Epoch', fontdict=font)
  plt.ylabel('Mean Squared Error', fontdict=font)

  plt.plot(hist['epoch'], hist['loss'], label='Train Error')
  plt.plot(hist['epoch'], hist['val_loss'], label='Val Error')

  plt.legend()
  plt.grid()

  plt.show()


  # Plot partitions side by side
def plot_partitions(time, data, start, end, window_size, split_mode, estilo, cor):
  time_start = 0

  for i in range(start, end):
    time_end = time_start + window_size

    if split_mode == 1:
      time_end += 1

    plt.plot(time[time_start:time_end], data[i, :], estilo, c=cor)
    plt.axvline(time[time_start], color='g', linestyle='--')

    if split_mode == 1:
      time_start = time_end - 1
    else:
      time_start = time_end

def plot_results(data_expected, data_predicted, timestep_size, n_partitions,
                 window_size, well_index, well_type, well_output, split_mode, sample_index=None):
  # Define plot styles
  font = {
      'family': 'serif',
      'color':'darkblue',
      'weight':'normal',
      'size': 16
  }

  # Get number of plots
  if data_expected.ndim == 1:
    n_test_samples = 1
  else:
    n_test_samples = data_expected.shape[0]

  n_plots = n_test_samples // n_partitions

  # Set horizontal axis data
  time_horizon = timestep_size * window_size * n_partitions
  time = np.arange(0, time_horizon + timestep_size, timestep_size)

  # Plot figure
  col_start = 0

  for i in range(n_plots):
    col_end = col_start + n_partitions

    plt.figure(figsize = (6, 4))

    plot_partitions(time, data_expected, col_start, col_end, window_size, split_mode, 'o-', 'b')
    plot_partitions(time, data_predicted, col_start, col_end, window_size, split_mode, '*-', 'r')

    if (n_partitions == 1):
      plt.legend(handles=[plt.Line2D([], [], marker='s', markersize=6, markerfacecolor='blue', markeredgecolor='blue', label='Expected'),
                          plt.Line2D([], [], marker='s', markersize=6, markerfacecolor='red', markeredgecolor='red', label='Predicted')],
                 loc='upper right', handlelength=0,
                 shadow=True, fontsize='x-large')
    else:
      plt.legend(handles=[plt.Line2D([], [], marker='s', markersize=6, markerfacecolor='blue', markeredgecolor='blue', label='Expected'),
                          plt.Line2D([], [], marker='s', markersize=6, markerfacecolor='red', markeredgecolor='red', label='Predicted'),
                          plt.Line2D([], [], marker='s', markersize=6, markerfacecolor='green', markeredgecolor='green', label='Partition Border')],
                 loc='upper right', handlelength=0,
                 shadow=True, fontsize='x-large')

    if sample_index == None:
      test_index = i
    else:
      test_index = sample_index

    if well_type == 0:
      if well_output == 'ir':
        plt.title(f'Water Rate I-{well_index}, Test-{test_index+1}', fontdict=font)
        plt.ylabel('Psi', fontdict=font)
        plt.ylim([0, 1100])
    elif well_type == 1:
      if well_output == 'lr':
        plt.title(f'Liquid Rate P-{well_index}, Test-{test_index+1}', fontdict=font)
        plt.ylabel('bbl/day', fontdict=font)
        plt.ylim([0, 4500])
      elif well_output == 'ir':
        plt.title(f'BHP I-{well_index}, Test-{test_index+1}', fontdict=font)
        plt.ylabel('Psi', fontdict=font)
        plt.ylim([0, 1100])
    elif well_type == 2:
      if well_output == 'or':
        plt.title(f'Oil Rate P-{well_index}, Test-{test_index+1}', fontdict=font)
        plt.ylabel('bbl/day', fontdict=font)
        plt.ylim([0, 1000])
      elif well_output == 'wr':
        plt.title(f'Water Rate P-{well_index}, Test-{test_index+1}', fontdict=font)
        plt.ylabel('bbl/day', fontdict=font)
        plt.ylim([0, 2500])
      elif well_output == 'ir':
        plt.title(f'BHP I-{well_index}, Test-{test_index+1}', fontdict=font)
        plt.ylabel('Psi', fontdict=font)
        plt.ylim([39000, 47000])
    elif well_type == 3:
      if well_output == 'ir':
        plt.title(f'BHP I-{well_index}, Test-{test_index+1}', fontdict=font)
        plt.ylabel('Psi', fontdict=font)
        plt.ylim([0, 1100])

    plt.xlabel('Time (days)', fontdict=font)
    plt.xlim([0, time_horizon+20])

    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20

    plt.grid()
    plt.show()

    col_start = col_end