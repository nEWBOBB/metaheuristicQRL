import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

script_dir = os.path.dirname(os.path.realpath(__file__))
script_dir = script_dir + "/resultsPLOTS/5x5/ACO"


# Liste aller CSV-Dateien im Verzeichnis
filepaths = glob.glob(os.path.join(script_dir, "*.csv"))
filename = filepaths[0] + ".png"

# Liste zum Speichern der DataFrames
dfs = []

# Lese jede Datei und füge sie zur Liste hinzu
for filepath in filepaths:
    df = pd.read_csv(filepath)
    dfs.append(df)

# Kombiniere alle DataFrames
combined_df = pd.concat(dfs)

##################################################

# Plotte die Daten
sns.lineplot(data=combined_df, x="generations", y="best", label='ACO', color='#1f77b4')

# Ändere den Namen der x-Achse
plt.xlabel("Iterations")

plt.ylabel("Best Score")



### Settings for 5x5 MiniGrid Environment
plt.axhline(y=0.955, xmin=0.0, xmax=1.0, color='r', linestyle='--', linewidth=0.7, alpha=0.9, label = 'Max Score')

# Setze die Grenzen der y-Achse auf 0.0 bis 1.0
plt.ylim(0.0, 1.0)

# Setze die yticks alle 0.1 Einheiten
plt.yticks(np.arange(0, 1.0 + 0.1, 0.1))
"""

### Settings for Cartpole Environment
plt.axhline(y=500, xmin=0.0, xmax=1.0, color='r', linestyle='--', linewidth=0.7, alpha=0.9, label = 'Max Score')
# Setze die Grenzen der y-Achse auf 0.0 bis 525.0
plt.ylim(0.0, 525.0)
"""

plt.legend()
#plt.savefig(filename)
plt.show()