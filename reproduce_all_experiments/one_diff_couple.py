import numpy as np
from functools import partial
from multiprocessing import Pool
import multiprocessing
from tqdm import tqdm
import copy
import random as random
import os

import itertools
from absl import app
from absl import flags



FLAGS = flags.FLAGS
flags.DEFINE_string("path", "", "Path to folder containing the classes of the representation")
flags.DEFINE_string("file_classes", "train_classes", "Name of the file containing the classes of the representation")
flags.DEFINE_string("file_representation", "train_representation", "Name of the representation shape(num_point, num_features)")
flags.DEFINE_boolean("mean_std", False, "Representation made up of mean and std, shape(2 * num_point, num_features)")

def randomly_split_into_two(elems):

    random.shuffle(elems)
    if len(elems) % 2 == 0:
       return elems[::2], elems[1::2]
    return elems[:-1:2], elems[1::2] # se dispari non considerare l'ultimo, non saprei a chi associarlo


def one_diff_couple(data, classes_uniques, classes, i_list=list(range(5)), idx=0 ):
    
    dict_i = {}
    c = classes_uniques[idx]
    #for c in classes_uniques:

    for i in i_list:
        aux_c = np.delete(c, i)
        #print(aux_c)
    
        aux_classes = np.delete(classes, i, axis=1)
        #print(aux_classes)
        
	# select representation
        xx = data[np.all(aux_classes ==aux_c, axis=1) ]
        xx = np.unique(xx, axis=0)

        # select classes
        xx_classes = classes[np.all(aux_classes ==aux_c, axis=1) ]
        xx_classes = np.unique(xx_classes, axis=0)

        # così non include tutte le possibili permzutazioni (x, y)...sarebbe meglio prenderle random

        x_idx, y_idx = randomly_split_into_two(list(range(xx.shape[0])))
        x, y = xx[x_idx], xx[y_idx]
        x_classes, y_classes = xx_classes[x_idx], xx_classes[y_idx]
        
        '''
        if (xx.shape[0] % 2)==0:
            x, y =  xx[::2], xx[1::2]
            x_classes, y_classes = xx_classes[::2],xx_classes[1::2]

        else:
            x, y =  xx[:-1:2], xx[1::2] # se dispari non considerare l'ultimo, non saprei a chi associarlo
            x_classes, y_classes = xx_classes[:-1:2],xx_classes[1::2]
        '''

        # rimuovere coppie con stessa classe in i. 
        for j, (xj, yj) in enumerate(zip(x_classes, y_classes)):
            #print(xj[i], yj[i])
            #print(xj[i])

            if xj[i]==yj[i]:
               print("bau")
               
               x = np.delete(x, j)
               y = np.delete(y, j)



        dict_i[i] =  (x, y) # se dispari non considerare l'ultimo, non saprei a chi associarlo

        
    return dict_i




def multiprocess_one_diff_couple(data, classes, i_list=list(range(5)), n_pool=5):
    
    # given the dataset and the factor classes
    # create couple (X, Y) so that images x and y differ of i-th class
    
    num_samples, num_factors  = data.shape
    # num_samples, num_factors = classes.shape
    
    dict_i = {}
    for i in i_list:
        dict_i[i] = ([], [])
    
    classes_uniques = np.unique(classes, axis=0)
    #print(classes_uniques)
    
    #print(classes_uniques)
    
    idx_list = list(range(classes_uniques.shape[0]))
    #print(len(idx_list))
    
    with Pool(n_pool) as p:
        

        partial_f = partial(one_diff_couple, data, classes_uniques, classes, i_list)
        results = p.map_async(partial_f, idx_list)

        for result_dict in results.get():
            
            for key, value in result_dict.items():
                x, y = value
                X, Y = dict_i[key]
                if len(X) <=0:
                    dict_i[key] = (x, y)
                else:
                    dict_i[key] = (np.vstack([X, x]), np.vstack([Y, y]))
            
        p.close() # no more tasks
        p.join()  # wrap up current tasks
    
    return dict_i


def main(unused_args):
    with open(os.path.join(FLAGS.path, FLAGS.file_representation + '.npy'), 'rb') as f:
        data = np.load(f)
        #data = data.T
        #print("Data shape: ", data.shape)

        #discard std
        if FLAGS.mean_std:
           data_mean = data[:, 0, ::2]
           data_stddev = data[:,0, 1::2]
           data = data_mean
        data = data.T
        print("Data shape: ", data.shape)

    
    with open(os.path.join(FLAGS.path,FLAGS.file_classes + '.npy'), 'rb') as f:
        classes = np.load(f)[:, :]
        classes = classes.T
        print("Classes shape: ", classes.shape)
        latents_names = ['shape', 'scale', 'orientation', 'posX', 'posY']

    

    #print(classes)
    print(multiprocessing.cpu_count())
    result_dict = multiprocess_one_diff_couple(data, classes, list(range(5)), 32)
    
    name_result_dict_X = {}
    name_result_dict_Y = {}

    for key, value in result_dict.items():
        X, Y = value
        print(key, X.shape, Y.shape)
        if X.size <=0:
           continue

        # create dict with names of classes
        name_result_dict_X[latents_names[key]]=X
        name_result_dict_Y[latents_names[key]]=Y


    np.savez_compressed(os.path.join(FLAGS.path,"X_" + FLAGS.file_representation), **name_result_dict_X)
    np.savez_compressed(os.path.join(FLAGS.path,"Y_" + FLAGS.file_representation), **name_result_dict_Y)


if __name__ == '__main__':
    app.run(main)

    
    


