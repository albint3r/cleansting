import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def quick_analysis(data):
    columns = ['price', 'm2_const', 'm2_terreno', 'habitaciones', 'banos', 'autos']
    corr_tb = data[columns].corr()

    return corr_tb


def heatmap_corr(data):
    df = data
    corr_tb = quick_analysis(df)
    sns.set()

    fig, ax = plt.subplots(1, 1, figsize=(20, 5))
    sns.heatmap(corr_tb, annot=True, ax=ax)
    ax.set(xlabel='Amenidades', ylabel='Amenidades')
    ax.set_title('Grafica de Correlacion de Amenidades', y=1.05, fontsize=18)
    plt.show()
