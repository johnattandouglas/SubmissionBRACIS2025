import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# Split dataset to partitions according to window size
def split_dataset_by_window(dataset, window_size, start, end, step, split_mode):
    sub_arrays = []

    n_samples = dataset.shape[0]
    n_timesteps = dataset.shape[1]
    n_steps = int(n_timesteps / window_size)

    for i in range(n_samples):
      # Sets the initial boundary of columns for the current range
      if split_mode == 2:
        col_start = 1
      else:
        col_start = 0

      for j in range(n_steps):
          # Sets the final boundary of columns for the current range
          col_end = col_start + window_size

          if split_mode == 1:
            col_end += 1

          # Slices the dataset using the limits defined above
          sub_array = dataset[i, col_start:col_end, start:end:step]

          # Updates col_start
          if split_mode == 1:
            col_start = col_end - 1
          else:
            col_start = col_end

          sub_arrays.append(sub_array)
          np.concatenate(sub_arrays, axis=0)

    return np.array(sub_arrays)

# Split dataset to TRAIN, VALIDATION and TEST sets
# The first dimension will be a multiple of
# the number of partitions for plotting reasons
#   TRAIN: about 70% of dataset
#   VALIDATION: about 20% of dataset
#   TEST: about 10% of dataset
def split_dataset(dataset, n_partitions, p_train=0.7, p_val=0.20, p_test=0.10):
    dim_1 = dataset.shape[0]
    dim_2 = dataset.shape[1]
    dim_3 = dataset.shape[2]

    size_1 = int(dim_1 * p_train) // n_partitions * n_partitions
    size_2 = int(dim_1 * p_val) // n_partitions * n_partitions
    size_3 = int(dim_1 * p_test) // n_partitions * n_partitions

    subarrays = [np.empty((size_1, dim_2, dim_3)) for i in range(3)]

    subarrays[0] = dataset[:size_1, :, :]
    subarrays[1] = dataset[size_1:size_1+size_2, :, :]
    subarrays[2] = dataset[size_1+size_2:size_1+size_2+size_3, :, :]

    return subarrays



# Define normalization type
# 0: MinMaxScaler
# 1: StandardScaler
# 2: RobustScaler
# 3: Custom Scaler (divides by dataset average)
# Check official doc for more info
def scaler(data, norm_type=0):
  if norm_type == 0:
    scaler = MinMaxScaler()
    return scaler.fit_transform(data), scaler
  elif norm_type == 1:
    scaler = StandardScaler()
    return scaler.fit_transform(data), scaler
  elif norm_type == 2:
    scaler = RobustScaler()
    return scaler.fit_transform(data), scaler
  elif norm_type == 3:
    max = np.amax(data)
    min = np.amin(data)

    scaler = (max + min) / 2
    data = data / scaler
    return data, scaler
  
def printFeatOutput():
  print()
  print("Feature output: {}".format(i+1))

  if window_size == 6:
    yhatE = Ay_hatE[:, :, i]
    ytest = Ay_test[:, :, i]
  else:
    yhatE = y_hat[:, :, i]
    ytest = y_test[:, :, i]

  #print(yhatE.shape,ytest.shape)
  vec=np.zeros((y_hat.shape[0], 1))

  for j in range(ytest.shape[0]):
    score = np.sqrt(metrics.mean_squared_error(yhatE[j, :], ytest[j, :]))
    #print("Amostra " +str(j+1)," --> Final score (RMSE) {}: ".format(score))
    vec[j, 0] = score

  def erro_mais_proximo(num, vetor):
      diferenca = abs(num - vetor[0])
      index = 0
      for k in range(1, len(vetor)):
          if abs(num - vetor[k]) < diferenca:
              diferenca = abs(num - vetor[k])
              index = k
      return index, vetor[index]

  erro_index = erro_mais_proximo(score_mean, vec)
  print("Amostra " + str(erro_index[0] + 1)," --> Final score (RMSE): {}".format(erro_index[1]))

  return erro_index

def find_by_average_error(value, data):
    result = data[0]
    index = 0

    for i, element in enumerate(data):
        if abs(element - value) < abs(result - value):
            result = element
            index = i

    return result, index