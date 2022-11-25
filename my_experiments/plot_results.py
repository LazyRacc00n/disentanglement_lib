import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from absl import app
from absl import flags

metrics_list = {"BetaVae Score": "beta_score", "Factor VAE Score":"factorvae_score", "MIG":"mig", "DCI Disentanglement":"dis", "Modularity":"mod", "SAP":"sap"}
represetation_list= ["Mean", "Sampled"]

FLAGS = flags.FLAGS
flags.DEFINE_string("input_path", "results.json",
                    "String with path to input json file.")

flags.DEFINE_integer("model_nums", 12,
                    "Number of models to aggregate.")

# aggregate results for the same hyperparameter and metric
def aggregate_seeds(data, metric_name):
	# return dictionary key is hyperparameter, values list of metric value (one for each seed)
	
	aggregated = {}
	for model_num, info in data.items():
		hyperparameter = info["hyperparameter"]
		
		# create list
		if hyperparameter not in aggregated:
			aggregated[hyperparameter] = [info[metric_name]]
		else:
		# append
			aggregated[hyperparameter].append(info[metric_name])
	

	return aggregated

def plot_metric(ax, data, title):
	clr = plt.cm.Blues(0.9)

	ax.set(adjustable='box')
	ax.set_title(title, fontsize = 14, fontweight = 'bold')
	
	
	x = list(data.keys()) # hyperparameter list
	
	y_m = [np.mean(values) for hyper, values in data.items()]	
	std = [np.std(values) for hyper, values in data.items()] # each single list metric values fixed hyperparameter
	
	y_l = [ mean + std for mean, std in zip(y_m, std)]
	y_u = [ mean - std for mean, std in zip(y_m, std)]

	ax.plot(x, y_m, label = 'mean', color = clr)
	ax.fill_between(x, y_l, y_u, alpha=0.3, edgecolor=clr, facecolor=clr, label="std")
	ax.set_ylabel('Value', fontsize = 'medium')
	ax.set_xlabel('Hyperparameter', fontsize = 'medium')
	ax.tick_params(axis='both', labelsize='small')
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	
	ax.set_ylim([0.005, 1.005])


def main(unused_argv):
	
	# number of model to consider
	model_nums = FLAGS.model_nums

	# num random seed used to train models
	#n_seed = 2

	results = pd.read_json(FLAGS.input_path)


	score_dict = {}
	for i in range(model_nums):
		
		
		# select model with model_num == i
		score_dict[i] = {}
		model_data = results[results["train_config.model.model_num"]==i]
		
		score_dict[i]["seed"] = np.float64(model_data["train_config.model.random_seed"])
		score_dict[i]["hyperparameter"] = np.float64(list(model_data["train_config.vae.beta"])[0])


		# chissà se capisco come estrarre le metriche

		# Beta-vae score on test set
		beta_score = model_data[["evaluation_results.eval_accuracy", "path"]]
		score_dict[i]["beta_score_mean"] = np.float64(beta_score.loc[beta_score["path"].str.contains("mean") & beta_score["path"].str.contains("beta_vae") ] ["evaluation_results.eval_accuracy"])
		score_dict[i]["beta_score_sampled"] = np.float64(beta_score.loc[beta_score["path"].str.contains("sampled") & beta_score["path"].str.contains("beta_vae") ] ["evaluation_results.eval_accuracy"])

		# Factor VAE test set 
		factorvae_score = model_data[["evaluation_results.eval_accuracy", "path"]]
		score_dict[i]["factorvae_score_mean"] = np.float64(factorvae_score.loc[factorvae_score["path"].str.contains("mean") & factorvae_score["path"].str.contains("factor_vae_metric") ] ["evaluation_results.eval_accuracy"] )
		score_dict[i]["factorvae_score_sampled"] = np.float64(factorvae_score.loc[factorvae_score["path"].str.contains("sampled") & factorvae_score["path"].str.contains("factor_vae_metric") ] ["evaluation_results.eval_accuracy"])

		# mig score
		mig_score = model_data[["evaluation_results.discrete_mig", "path"]]
		score_dict[i]["mig_mean"] = np.float64(mig_score.loc[mig_score["path"].str.contains("mean") & mig_score["path"].str.contains("mig") ] ["evaluation_results.discrete_mig"] )
		score_dict[i]["mig_sampled"] = np.float64(mig_score.loc[mig_score["path"].str.contains("sampled") & mig_score["path"].str.contains("mig") ] ["evaluation_results.discrete_mig"])

		# DCI
		# disentanglement on test set
		dis_score = model_data[["evaluation_results.disentanglement" , "path"]]
		score_dict[i]["dis_mean"] = np.float64(dis_score.loc[dis_score["path"].str.contains("mean") & dis_score["path"].str.contains("dci") ] ["evaluation_results.disentanglement"] )
		score_dict[i]["dis_sampled"] = np.float64(dis_score.loc[dis_score["path"].str.contains("sampled") & dis_score["path"].str.contains("dci") ] ["evaluation_results.disentanglement"])

		
		# modularity
		mod_score = model_data[["evaluation_results.modularity_score" , "path"]]
		score_dict[i]["mod_mean"] = np.float64(mod_score.loc[mod_score["path"].str.contains("mean") & mod_score["path"].str.contains("modularity") ] ["evaluation_results.modularity_score"] )
		score_dict[i]["mod_sampled"] = np.float64(mod_score.loc[mod_score["path"].str.contains("sampled") & mod_score["path"].str.contains("modularity") ] ["evaluation_results.modularity_score"])

		# SAP
		sap_score = model_data[['evaluation_results.SAP_score' , "path"]]
		score_dict[i]["sap_mean"] = np.float64(sap_score.loc[sap_score["path"].str.contains("mean") & sap_score["path"].str.contains("sap_score") ] ["evaluation_results.SAP_score"] )
		score_dict[i]["sap_sampled"] = np.float64(sap_score.loc[sap_score["path"].str.contains("sampled") & sap_score["path"].str.contains("sap_score") ] ["evaluation_results.SAP_score"])


	# PLOT SCORES MEAN REPRESENTATION
	fig, axs = plt.subplots(1, len(metrics_list), sharex=True, sharey=False, figsize = (25, 5)) # 5, 1

	fig.tight_layout(pad = 5.0)
	title = f'Metrics wrt hyperparameter'
	fig.suptitle(title, fontsize = 'xx-large',  fontweight = 'bold')

	for i, ax in enumerate(axs.flatten()):
		data = aggregate_seeds(score_dict, list(metrics_list.values())[i] + "_mean")
		plot_metric(ax, data, list(metrics_list.keys())[i])
	plt.legend()
	plt.savefig("resuls_mean_representation.png")


	fig, axs = plt.subplots(1, len(metrics_list), sharex=True, sharey=False, figsize = (25, 5))

	fig.tight_layout(pad = 5.0)
	title = f'Metrics wrt hyperparameter'
	fig.suptitle(title, fontsize = 'xx-large',  fontweight = 'bold')

	for i, ax in enumerate(axs.flatten()):
		data = aggregate_seeds(score_dict, list(metrics_list.values())[i] + "_sampled")
		plot_metric(ax, data, list(metrics_list.keys())[i])
	plt.legend()
	plt.savefig("resuls_sample_representation.png")

if __name__ == "__main__":
	app.run(main)

