import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from disentanglement_utils import *

import os
print(os.getcwd())
COLORS = {
    'blue':    '#377eb8', 
    'orange':  '#ff7f00',
    'green':   '#4daf4a',
    'pink':    '#f781bf',
    'brown':   '#a65628',
    'purple':  '#984ea3',
    'gray':    '#999999',
    'red':     '#e41a1c',
    'yellow':  '#dede00'
} 
 
FACTORS = {

'shape':'blue',
'scale':'orange',
'orientation':'green',
'posX':'pink',
'posY':'brown',
}


# aggregate results for the same hyperparameter and metric
def aggregate_seeds(data, name="Beta", value=str(0.25)):
	# return dictionary key is hyperparameter, values list of metric value (one for each seed)
	
	aggregated = {}

	for index, row in data.iterrows():
		if index == len(data):
			break

		hyperparameter = row[name]
		
		# create list

		if hyperparameter not in aggregated:
			#print(row[value])
			aggregated[hyperparameter] = [row[value]]
		else:
		# append
			aggregated[hyperparameter].append(row[value])
	return aggregated



def plot_metric_factors(info, w=0.2, title=r"Metric $\alpha=0.25$", set_lim = True):

	for factor, color in FACTORS.items():

		data = aggregate_seeds(info, value=factor)
		#print(data)
		clr = plt.cm.Blues(0.9)

		#ax.set(adjustable='box')
		plt.title(title, fontsize = 14, fontweight = 'bold')
	
	
		x = list(data.keys()) # hyperparameter list
		#print(x)
		#print(list(data.values())[0])
		#print([elem[w] for hyper, values in data.items() for elem in values])
		y_m = [np.mean([elem[w] for elem in values]) for hyper, values in data.items() ]	
		std = [np.std([elem[w] for elem in values]) for hyper, values in data.items() ] # each single list metric values fixed hyperparameter
	
		y_l = [ mean + std for mean, std in zip(y_m, std)]
		y_u = [ mean - std for mean, std in zip(y_m, std)]

		plt.plot(x, y_m, label = factor, color = COLORS[color])
		#plt.fill_between(x, y_l, y_u, alpha=0.3, edgecolor=clr, facecolor=clr, label="std")
		plt.ylabel('Value', fontsize = 'medium')
		plt.xlabel('Hyperparameter', fontsize = 'medium')
		#plt.tick_params(axis='both', labelsize='small')
		#plt.spines['right'].set_visible(False)
		#plt.spines['top'].set_visible(False)
	
		if set_lim:
			plt.ylim([-0.005, 1.005])


def plot_metric(data, title, set_lim = True):
	clr = plt.cm.Blues(0.9)

	#ax.set(adjustable='box')
	plt.title(title, fontsize = 14, fontweight = 'bold')
	
	
	x = list(data.keys()) # hyperparameter list
	
	y_m = [np.mean(values) for hyper, values in data.items()]	
	std = [np.std(values) for hyper, values in data.items()] # each single list metric values fixed hyperparameter
	
	y_l = [ mean + std for mean, std in zip(y_m, std)]
	y_u = [ mean - std for mean, std in zip(y_m, std)]

	plt.plot(x, y_m, label = 'mean', color = clr)
	plt.fill_between(x, y_l, y_u, alpha=0.3, edgecolor=clr, facecolor=clr, label="std")
	plt.ylabel('Value', fontsize = 'medium')
	plt.xlabel('Hyperparameter', fontsize = 'medium')
	#plt.tick_params(axis='both', labelsize='small')
	#plt.spines['right'].set_visible(False)
	#plt.spines['top'].set_visible(False)
	
	if set_lim:
		plt.ylim([-0.005, 1.005])


def main():

    info = pd.read_csv("couples_representations_perfect/info.csv", header=0)
    #info = pd.DataFrame(info)
    info["Scores"]=None
    w_list=[1.0, 0.75, 0.50, 0.25, 0.10, 0.05, 0.0]
    
    score_w_list =[]
    score_w_factors_list =[]
    dead_list = []

    for index, row in info.iterrows():
        elem = int(row["Num"])
        
        X = np.load("couples_representations_perfect/X_train_representation_{}.npz".format(elem))
        Y = np.load("couples_representations_perfect/Y_train_representation_{}.npz".format(elem))
        score_w={str(w):0.0 for w in w_list}
        score_w_factors = {factor:{}for factor in FACTORS.keys()}

        for w in w_list:
                score, is_dead, dict_association, factors_score = get_score(X, Y, w)
                score_w[str(w)]=score
                if w ==0.25:
                  dead_list.append(len(is_dead))
                for f, s in factors_score.items():
                    score_w_factors[f][w]=s
        score_w_list.append(score_w)
        score_w_factors_list.append(score_w_factors)

        X.close()
        Y.close()

    info["Dead"]=dead_list
    #info["Scores"]=score_w_list
    #info["Factors_Scores"]=score_w_factors_list


    info = info.assign(**pd.DataFrame(score_w_list))
    info = info.assign(**pd.DataFrame(score_w_factors_list))
    #info=info.reset_index(drop=True)
    
    info.to_csv("info_updated.csv",index=False)

    # plot 
    for w in w_list:

        aggregated_data = aggregate_seeds(info, value=str(w))
        plot_metric(aggregated_data, "Metric: {}".format(w))
        plt.legend()
        plt.savefig("resuls_mean_representation_{}.png".format(w))
        plt.clf()
        
        
        plot_metric_factors(info, w=w, title="Metric: {}".format(w))
        plt.legend()
        plt.savefig("score_factors_{}.png".format(w))
        plt.clf()

    data = aggregate_seeds(info, value="Dead")
    plot_metric(data, "Dead", False)
    plt.legend()
    plt.savefig("resuls_dead.png")

    


if __name__ == '__main__':
    main()

    
