# Sorry for the mess.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import matplotlib.ticker as ticker

# Verzeichnis des Skripts und Pfad zum Ordner "results"
script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, 'resultsPLOTS/5x5/')

# Iteriere über alle Ordner im "results"-Ordner
for folder in os.listdir(results_dir):
	# Finde alle CSV-Dateien im Ordner "results"
	resultfolder = os.path.join(results_dir, folder)
	csv_files = glob.glob(os.path.join(resultfolder, '*.csv'))

	# Leere Liste zum Speichern der Dataframes
	dataframes = []

	# Einlesen der CSV-Dateien
	for file in csv_files:
		df = pd.read_csv(file)
		dataframes.append(df)

	# Kombinieren der Dataframes zu einem einzigen Dataframe
	combined_df = pd.concat(dataframes, ignore_index=True)

	# Bestimmen des maximalen Wertes der Runtimes
	max_runtime = combined_df['runtime'].max()

	# Generieren von Bins in 50er-Schritten bis zum maximalen Wert der Runtimes
	step_size = 80
	bins = np.arange(0, max_runtime + step_size, step_size)

	# Erstellen einer neuen Spalte 'runtime_bin' basierend auf den generierten Bins
	combined_df['runtime_bin'] = pd.cut(combined_df['runtime'], bins)

	# Gruppieren der Daten nach 'runtime_bin' und Berechnen des Mittelwerts und des Konfidenzintervalls
	grouped_data = combined_df.groupby('runtime_bin')['best'].agg(['mean', 'std', 'count'])
	grouped_data['ci'] = 1.96 * grouped_data['std'] / np.sqrt(grouped_data['count'])

	# Berechnen der oberen und unteren Grenzen für die Fehlerbalken
	lower_error = grouped_data['mean'] - grouped_data['ci']
	upper_error = grouped_data['mean'] + grouped_data['ci']

	# Plotten der Daten mit Seaborn
	#plt.figure(figsize=(10, 6))
	print(folder)
	sns.lineplot(x=bins[:-1], y=grouped_data['mean'], label=folder)
	plt.fill_between(bins[:-1], lower_error, upper_error, alpha=0.2)

# Definiere eine Funktion, die Sekunden in hh:mm:ss umwandelt
def format_func(value, tick_number):
    hours = int(value // 3600)
    minutes = int((value % 3600) // 60)
    seconds = int(value % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


###ALLGEMEIN
maxRuntimePlot = 15000

plt.xlabel('Runtime (hh:mm:ss)')
plt.ylabel('Best Score')
plt.title('Best Score over Runtime')

# Setze die x-Achse Ticks und Labels
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MultipleLocator(1800))  # Setze die Ticks alle 2400 Sekunden
ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_func))  # Verwende die format_func, um die Labels zu formatieren
plt.xticks(rotation=45)

# xticks alle x sekunden (runtime)
#plt.xticks(np.arange(0, maxRuntimePlot + 20, 2400), rotation=45)

plt.axhline(y=0.955, xmin=0.0, xmax=1.0, color='r', linestyle='--', linewidth=0.7, alpha=0.9, label = 'Max Score')

# Setze die yticks alle 100 Einheiten
plt.yticks(np.arange(0, 1.1, 0.1))

# Limitiere die X-Achse auf maximal 500
plt.xlim(0, maxRuntimePlot)

# Setze die Grenzen der y-Achse auf 0.0 bis 1.0
plt.ylim(0.0, 1.0)

# Zeichne die Legende
plt.legend()

plt.tight_layout()
plt.show()