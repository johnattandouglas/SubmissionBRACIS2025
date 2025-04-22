# # Gerar gráfico com os resultados do grid search

import pandas as pd
import matplotlib.pyplot as plt

# Carregar os dados dos arquivos
df_early = pd.read_csv('files/resultadosEarlyGRU.csv')

# Filtrar os dados para incluir apenas as condições desejadas,
# excluindo os resultados com 50 neurônios
df_early_linear = df_early[(df_early['batch_size'] == 4) & 
                           (df_early['learning_rate'] == 0.001) & 
                           (df_early['activation'] == 'linear') &
                           (df_early['neurons'] != 50)]
df_early_tanh = df_early[(df_early['batch_size'] == 4) & 
                         (df_early['learning_rate'] == 0.001) & 
                         (df_early['activation'] == 'tanh') &
                         (df_early['neurons'] != 50)]
df_early_relu = df_early[(df_early['batch_size'] == 4) & 
                         (df_early['learning_rate'] == 0.001) & 
                         (df_early['activation'] == 'relu') &
                         (df_early['neurons'] != 50)]

# Encontrar o menor score para cada combinação de neurônios para linear
bsL = df_early_linear.groupby('neurons')['score'].min().reset_index()

# Encontrar o menor score para cada combinação de neurônios para tanh
bsT = df_early_tanh.groupby('neurons')['score'].min().reset_index()

# Encontrar o menor score para cada combinação de neurônios para relu
bsR = df_early_relu.groupby('neurons')['score'].min().reset_index()

# Ordenar os dados por número de neurônios
bsL = bsL.sort_values(by='neurons')
bsT = bsT.sort_values(by='neurons')
bsR = bsR.sort_values(by='neurons')

# Plotar o gráfico comparativo dos scores
plt.figure(figsize=(6, 2.5))

if not bsL.empty:
    plt.plot(bsL['neurons'],
             bsL['score'],
             marker='o',
             label='Linear')
if not bsT.empty:
    plt.plot(bsT['neurons'],
             bsT['score'],
             marker='o',
             label='Tanh')
if not bsR.empty:
    plt.plot(bsR['neurons'],
             bsR['score'],
             marker='o',
             label='ReLU')

plt.xlabel('Quantidade de Neurônios')
plt.ylabel('Score')
plt.legend(loc='lower left')
plt.grid(True)
plt.xticks(bsL['neurons'])
plt.ylim(0.003, 0.0120) 
plt.tight_layout()
plt.savefig('GridSearchGRU2.png')
plt.show()