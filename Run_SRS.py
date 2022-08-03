import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import json
import STL


import numpy as np
import tensorflow as tf
np.random.seed(0)
tf.random.set_seed(0)
import pickle as pkl
import TSDTransform as tsd
import OOD_Auroc_utils as au

from CVAE_Keras import CVAE_model
from utils_ import adjust_seg_size, adjust_labels
from absl import app, flags
FLAGS = flags.FLAGS

path_dict = {
'cvae_align': "SeasonalRatio/CVAE_Align/",
'rescvae_align': "SeasonalRatio/ResCVAE_Align/",
'stl': "SeasonalRatio/STL/",
'stl_align': "SeasonalRatio/STL_Align/",
'rstl_align': "SeasonalRatio/RSTL_Align/",
'results_align': "SeasonalRatio/Results_Align/"
}

for path in path_dict.values():
    try:
        os.makedirs(path)
    except FileExistsError:
        continue
    
def align_set(X_train, y_train_int, input_stl, CLASS_NB, align_iter=5): 
    X_train_aligned = np.zeros((0,)+X_train.shape[1:])
    y_train_aligned = np.zeros((0,))
    for cl in range(CLASS_NB):
        print('Aligning', cl+1, '/', CLASS_NB, '...')
        index = np.argwhere(y_train_int==cl).flatten()
        X_set = X_train[index]
        X_set_aligned = tsd.optimize_alignment_to_pattern(input_stl.pattern[cl], X_set, iterations=align_iter)
        X_train_aligned = np.concatenate([X_train_aligned, X_set_aligned], axis=0)
        y_train_aligned = np.concatenate([y_train_aligned, cl*np.ones(len(index), dtype=np.int)])
    y_train = tf.keras.utils.to_categorical(y_train_int)
    
    return X_train_aligned, y_train_int, y_train_aligned

def main(argv):
    ## Load Data
    json_param = "datasets_parameters.json"
    with open(json_param) as jf:
        info = json.load(jf)
        d = info[FLAGS.dataset_name]
        path = d['path']
        SEG_SIZE = d['SEG_SIZE']
        CHANNEL_NB = d['CHANNEL_NB']
        CLASS_NB = d['CLASS_NB']
    print("Dataset: {}".format(FLAGS.dataset_name))
    X_train, y_train_int, X_test, y_test_int = pkl.load(open(path, 'rb'))
    #min_train = np.min(np.min(X_train, axis=2), axis=0).flatten()
    #max_train = np.min(np.min(X_train, axis=2), axis=0).flatten()
    min_train = np.min(X_train)
    max_train = np.max(X_train)
    X_train = X_train.reshape((-1, SEG_SIZE, CHANNEL_NB))
    X_test = X_test.reshape((-1, SEG_SIZE, CHANNEL_NB))
    
    ## STL Data decomp
    if os.path.isfile(path_dict['stl']+FLAGS.dataset_name+"_STL_decomp.pkl"):    
        input_stl = pkl.load(open(path_dict['stl']+FLAGS.dataset_name+"_STL_decomp.pkl", 'rb'))
    else:
        input_stl = STL.STL_decomp(SEG_SIZE, CHANNEL_NB, X_train, y_train_int)
        pkl.dump(input_stl, open(path_dict['stl']+FLAGS.dataset_name+"_STL_decomp.pkl", 'wb'))
                
                
    ## Align X_train
    X_train, y_train_int, y_train = align_set(X_train, y_train_int, input_stl, CLASS_NB, align_iter=FLAGS.align_iter)
        
    ## Train Data CVAE 
    print("Preparing CVAE models . . .")
    
    cvae = CVAE_model(FLAGS.latent_size, SEG_SIZE, CHANNEL_NB, CLASS_NB, min_train=min_train, max_train=max_train,
                      arch=FLAGS.arch, show_summary=0)
    
    cvae_model_path = path_dict['cvae_align']+"CVAE_"+FLAGS.dataset_name+"_"+FLAGS.arch+"_weights"
    
    if os.path.isfile(cvae_model_path+".index") and not FLAGS.train_ml:
        cvae.train(X_train, y_train, checkpoint_path=cvae_model_path, 
                new_train=False)
    else:
        cvae.train(X_train, y_train, checkpoint_path=cvae_model_path,  
                    epochs=FLAGS.epochs, batch_size=FLAGS.batch_size, new_train=True, verbose=0)
                    
    
    ## STL Data decomp
    if os.path.isfile(path_dict['stl_align']+FLAGS.dataset_name+"_STL_decomp.pkl"):    
        input_stl = pkl.load(open(path_dict['stl_align']+FLAGS.dataset_name+"_STL_decomp.pkl", 'rb'))
    else:
        input_stl = STL.STL_decomp(SEG_SIZE, CHANNEL_NB, X_train, y_train_int)
        pkl.dump(input_stl, open(path_dict['stl_align']+FLAGS.dataset_name+"_STL_decomp.pkl", 'wb'))
            
            
    #Residulas on Train data
    residuals_train, res_labels = input_stl.residuals_of(X_train, y_train_int)
    res_labels_hot = tf.keras.utils.to_categorical(res_labels)
    
    #Residuals on Test data
    X_test, y_test_int, y_test = align_set(X_test, y_test_int, input_stl, CLASS_NB, align_iter=FLAGS.align_iter)
    residuals_test, res_labels_test = input_stl.residuals_of(X_test, y_test_int)
    res_labels_test_hot = tf.keras.utils.to_categorical(res_labels_test)
    
    ## Train Residuals CVAE 
    # min_train = np.min(np.min(residuals_train, axis=2), axis=0).flatten()
    # max_train = np.max(np.max(residuals_train, axis=2), axis=0).flatten()
    min_train = np.min(residuals_train)
    max_train = np.max(residuals_train)
    rescvae = CVAE_model(FLAGS.latent_size, SEG_SIZE, CHANNEL_NB, CLASS_NB, min_train=min_train, max_train=max_train,
                      arch=FLAGS.arch, show_summary=0)
    
    rescvae_model_path = path_dict['rescvae_align']+"CVAE_"+FLAGS.dataset_name+"_"+FLAGS.arch+"_weights"
    if os.path.isfile(rescvae_model_path+".index") and not FLAGS.train_ml:
        rescvae.train(residuals_train, res_labels_hot, checkpoint_path=rescvae_model_path, 
                new_train=False)
    else:
        rescvae.train(residuals_train, res_labels_hot, checkpoint_path=rescvae_model_path,  
                    epochs=500, batch_size=32, new_train=True, verbose=0)
      
    ## SRS on In-Distribution data
    print("-- In-Distribution processing . . .")
    
    ll_x_in = cvae.likelihood(X_train, y_train,  mc_range=50)
    ll_rem_in = rescvae.likelihood(residuals_train, res_labels_hot,  mc_range=50)
    
    ratio1 = ll_x_in/ll_rem_in
    pkl.dump([ratio1, ll_x_in, ll_rem_in], open(path_dict['results_align']+FLAGS.dataset_name+"_res.pkl", 'wb'))
    
    ll_x_in_test = cvae.likelihood(X_test, y_test,  mc_range=50)
    ll_rem_in_test = rescvae.likelihood(residuals_test, res_labels_test_hot,  mc_range=50)
    
    ratio_test = ll_x_in_test/ll_rem_in_test
    pkl.dump([ratio_test, ll_x_in_test, ll_rem_in_test], open(path_dict['results_align']+FLAGS.dataset_name+"_res_onTest.pkl", 'wb'))
  
    ## Define out_data as real world distribution
    print("-- In-domain OOD Data . . .")
    ratio_on_ood = {}
    with open(json_param) as jf:
        info = json.load(jf)
        d = info[FLAGS.dataset_name_ood]
        path = d['path']
    #Data Reading
    ood_X_train, ood_y_train_int, ood_X_test, ood_y_test_int = pkl.load(open(path, 'rb'))
    #adjust OOD data to IN data
    ood_X_train = adjust_seg_size(ood_X_train, (SEG_SIZE, CHANNEL_NB))       
    ood_X_test = adjust_seg_size(ood_X_test, (SEG_SIZE, CHANNEL_NB)) 
    ood_y_train_int = adjust_labels(ood_y_train_int, CLASS_NB)
    ood_y_test_int = adjust_labels(ood_y_test_int, CLASS_NB)
    
    ood_X_train = ood_X_train.reshape((-1, SEG_SIZE, CHANNEL_NB))
    ood_X_test = ood_X_test.reshape((-1, SEG_SIZE, CHANNEL_NB))
    ood_X = np.concatenate([ood_X_train, ood_X_test], axis=0)
    ood_y_train = tf.keras.utils.to_categorical(ood_y_train_int)
    ood_y_test = tf.keras.utils.to_categorical(ood_y_test_int)
    ood_y_int = np.concatenate([ood_y_train_int, ood_y_test_int], axis=0)
    ood_y = np.concatenate([ood_y_train, ood_y_test], axis=0)
    ood_x, ood_y_int, ood_y = align_set(ood_X, ood_y_int, input_stl, CLASS_NB, align_iter=FLAGS.align_iter) #Align
        
    residuals_ood, res_ood_labels = input_stl.residuals_of(ood_X, ood_y_int)
    res_ood_labels_hot = tf.keras.utils.to_categorical(res_ood_labels)
    
    ll_x_ood = cvae.likelihood(ood_X, ood_y,  mc_range=50)
    ll_x_ood_rem = rescvae.likelihood(residuals_ood, res_ood_labels_hot,  mc_range=50)
    
    ratio_ood = ll_x_ood/ll_x_ood_rem
    
    ratio_on_ood[FLAGS.dataset_name_ood] = [ratio_ood, ll_x_ood, ll_x_ood_rem]
    pkl.dump([ratio_on_ood], open(path_dict['results_align']+FLAGS.dataset_name+"_OODres.pkl", 'wb'))
        
    ## Compute AUROC score

    ratio1, _, _ = pkl.load(open(path_dict['results_align']+FLAGS.dataset_name+"_res.pkl", 'rb'))
    ratio_test, _, _ = pkl.load(open(path_dict['results_align']+FLAGS.dataset_name+"_res_onTest.pkl", 'rb'))
    ratio_in = np.concatenate([ratio1, ratio_test])    
    ratio_on_ood = pkl.load(open(path_dict['results_align']+FLAGS.dataset_name+"_OODres.pkl", 'rb'))[0]
    
    ratio_ood, ll_x_ood, ll_x_ood_rem = ratio_on_ood[FLAGS.dataset_name_ood]
    ratio_dict_real = {
            'in': ratio_in,
            'ood': ratio_ood
        }
    ratio_elements, ratio_labels = au.label_ratio(ratio_dict_real)
    real_auroc_ratio ="%.2f"%au.auroc(ratio_in, ratio_elements, ratio_labels)
    print("AUROC score: ", real_auroc_ratio)
        
if __name__=="__main__":
   flags.DEFINE_string('dataset_name', 'Cricket', 'Dataset name')
   flags.DEFINE_string('dataset_name_ood', 'Epilepsy', 'Dataset name')
   flags.DEFINE_boolean('train_ml', True, 'True: Train models; False: Load models')
   flags.DEFINE_integer('latent_size', 32, 'Dimension of the latent space')
   flags.DEFINE_string('arch', 'CONV', 'Model architecture')
   flags.DEFINE_integer('batch_size', 32, 'Batch size')
   flags.DEFINE_integer('epochs', 500, 'Epochs')
   flags.DEFINE_integer('align_iter', 5, 'Number of iterations for alignment process')
   app.run(main)            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
