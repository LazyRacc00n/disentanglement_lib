from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import  os

if __name__ == "__main__":


	event_acc = EventAccumulator('./summary/events.out.tfevents.1665911155.diamante')
	event_acc.Reload()
	# Show all tags in the log file
	print(event_acc.Tags())

	# E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
	w_times, step_nums, vals = zip(*event_acc.Scalars('elbo_2'))

	print(w_times) 
	print()
	print(step_nums)
	print()
	print( vals)
	
	# plot and save the train and validation line graphs
	plt.figure(figsize=(10, 7))
	plt.plot(step_nums, vals, color='blue', label='elbo loss')
	plt.xlabel('Steps')
	plt.ylabel('Loss')
	plt.legend()

	path = os.path.join("elbo.png")
	plt.savefig(path)
