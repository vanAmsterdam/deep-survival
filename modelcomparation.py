#!/usr/bin/env python
'''
This is to do model selection among survival models.
Implemented as Functional Procedure
'''
import numpy as np
import sklearn.cross_validation
import DeepConvSurv as deep_conv_surv
import pandas as pd
import os

#hyperparams
model = 'deepconvsurv'
epochs = 20
lr = 5e-4
seed = 1
batchsize = 30

def convert_index (inputpid, expandlabel):
    outputindex = []
    for pid in inputpid:
        outputindex.append(list(expandlabel['pid'][expandlabel['pid']==pid].index))
    patient_num = [len(x) for x in outputindex]
    outputindex = [y for x in outputindex for y in x]
    return outputindex, patient_num

def model_selection(img_path, clinical_path, label_path, expand_label_path, 
                    train_test_ratio, train_valid_ratio, seed=seed,
                    model='deepconsurv', batchsize=batchsize, epochs=epochs, 
                    lr = lr, **kwargs):
    print ' '
    print '--------------------- Model Selection ---------------------'
    print '---------------Training Model: ', model, '--------------'
    print '---------------------parameters----------------------------'
    print "epochs: ", epochs, "  tr/test ratio: ", train_test_ratio, "  tr/val ratio: ", train_valid_ratio
    print "learning rate: ", lr, "batch size: ", batchsize
    print '-----------------------------------------------------------'
    print ' ' 
    # load labels
    labels = pd.read_csv(label_path)
    expand_label = pd.read_csv(expand_label_path)
    clinical_data = pd.read_csv(clinical_path)
    clinical_dim = len(clinical_data.columns)
    surv = expand_label['surv']
    status = expand_label['status']
    ## generate index
    e = labels["status"]
    rdn_index = sklearn.cross_validation.StratifiedKFold(e, n_folds=5, 
                                                         shuffle=True, random_state=seed)
    testci = []
    index_num = 1
    for trainindex, testindex in rdn_index:
        test_index, test_patchidxcnt = convert_index(labels['pid'].values[testindex], expand_label)
        cv_idx = sklearn.cross_validation.StratifiedShuffleSplit(e.values[trainindex],
                                                                 n_iter=1,test_size=1-train_valid_ratio, random_state = seed)
        sublabels = labels['pid'].values[trainindex]
        for tr_idx, val_idx in cv_idx:
            train_index, train_patchidxcnt = convert_index(sublabels[tr_idx], expand_label)
            valid_index, valid_patchidxcnt = convert_index(sublabels[val_idx], expand_label)

        tr_idx_name = 'train%d.csv' %index_num
        te_idx_name = 'test%d.csv' %index_num
        va_idx_name = 'valid%d.csv' %index_num
        risk_name = 'risk%d.csv' %index_num
        print ''
        print tr_idx_name, te_idx_name, va_idx_name
        print ''
        np.savetxt(tr_idx_name, train_index, delimiter=',', header='index')
        np.savetxt(te_idx_name, test_index, delimiter=',', header='index')
        np.savetxt(va_idx_name, valid_index, delimiter=',', header='index')
        index_num = index_num + 1

        sampleimg = img_path + "/" + os.listdir(img_path)[0] + "/" + os.listdir(img_path + "/" + os.listdir(img_path)[0])[1]
        NLSTimage = np.load(sampleimg)
        width = NLSTimage.shape[0]
        height = NLSTimage.shape[1]
        channel = NLSTimage.shape[2]
        if model=='deepsurv':
            hyperparams = {
                'learning_rate': lr,
                'n_in': clinical_dim,
                'hidden_layers_sizes': [128,32]
            }
            deepsurv_train_data = {
                'x': np.float32(clinical_data.values[train_index,]),
                't': np.float32(surv.values[train_index,]),
                'e': np.int32(status.values[train_index,])
            }
            deepsurv_test_data = {
                'x': np.float32(clinical_data.values[test_index,]),
                't': np.float32(surv.values[test_index,]),
                'e': np.int32(status.values[test_index,])
            }
            deepsurv_valid_data = {
                'x': np.float32(clinical_data.values[valid_index,]),
                't': np.float32(surv.values[valid_index,]),
                'e': np.int32(status.values[valid_index,])
            }
            network = deep_surv.DeepSurv(**hyperparams)
            log = network.train(deepsurv_train_data,deepsurv_test_data, deepsurv_valid_data, model_index=index_num, n_epochs = epochs)
            #test_risk = network.predict_risk(deepsurv_test_data['x'])
            testci.append(log['test_ci'])
        elif model=='deepsurv-pre':
            print "*********** deepsurv with pretrained parameters ************"
            hyperparams = {
                'learning_rate': lr,
                'n_in': clinical_dim,
                'hidden_layers_sizes': [128,32]
            }
            deepsurv_train_data = {
                'x': np.float32(clinical_data.values[train_index,]),
                't': np.float32(surv.values[train_index,]),
                'e': np.int32(status.values[train_index,])
            }
            deepsurv_test_data = {
                'x': np.float32(clinical_data.values[test_index,]),
                't': np.float32(surv.values[test_index,]),
                'e': np.int32(status.values[test_index,])
            }
            deepsurv_valid_data = {
                'x': np.float32(clinical_data.values[valid_index,]),
                't': np.float32(surv.values[valid_index,]),
                'e': np.int32(status.values[valid_index,])
            }
            network = deep_surv_pre.DeepSurv(**hyperparams)
            log = network.train(deepsurv_train_data,deepsurv_test_data, deepsurv_valid_data, n_epochs = 10)
            #test_risk = network.predict_risk(deepsurv_test_data['x'])
            testci.append(log['test_ci'])
        elif model=='deepconvsurv':
            hyperparams = {
            'learning_rate': lr,
            'channel': channel,
            'width': width,
            'height': height,
            }
            network = deep_conv_surv.DeepConvSurv(**hyperparams)
            log = network.train(data_path=img_path, label_path=expand_label_path, train_index = train_index, test_index=valid_index, valid_index = test_index, model_index = index_num, batch_size = batchsize, ratio = train_test_ratio, num_epochs= epochs)
            testci.append(log)
        elif model=='deepconvsurv-pre':
            print "********* deepconvsurv with pretrained parameters ***********"
            hyperparams = {
            'learning_rate': lr,
            'channel': channel,
            'width': width,
            'height': height,
            }
            network = deep_conv_surv_pre.DeepConvSurv(**hyperparams)
            log = network.train(data_path=img_path, label_path=expand_label_path, train_index = train_index, test_index=valid_index, valid_index = test_index, batch_size = batchsize, ratio = train_test_ratio, num_epochs= epochs)
            testci.append(log)           
        elif model=='deepmultisurv':
            hyperparams = {
            'learning_rate': lr,
            'channel': channel,
            'width': width,
            'height': height,
            'clinical_dim': clinical_dim
            }
            network = deep_multi_surv.DeepMultiSurv(**hyperparams)
            log = network.train(data_path=img_path, clinical_path=clinical_path, label_path=expand_label_path, train_index = train_index, test_index=test_index, valid_index = valid_index, batch_size = batchsize, ratio = train_test_ratio, num_epochs= epochs)
            testci.append(log)
        elif model=='deepmultisurv-pre':
            print "********* deepmultisurv with pretrained parameters **********"
            hyperparams = {
            'learning_rate': lr,
            'channel': channel,
            'width': width,
            'height': height,
            'image_pretrain_name': '../DeepModelParams/convimgmodel%d.npz'%index_num,
            'clinical_pretrain_name': '../DeepModelParams/singleclinicalmodel%d.npz'%index_num,
            'clinical_dim': clinical_dim
            }
            network = deep_multi_surv_pre.DeepMultiSurv(**hyperparams)
            test_risk, test_ci, test_t, test_e = network.train(data_path=img_path, clinical_path=clinical_path, label_path=expand_label_path, train_index = train_index, train_patchidxcnt = train_patchidxcnt, test_index=test_index, test_patchidxcnt=test_patchidxcnt, valid_index = valid_index, valid_patchidxcnt = valid_patchidxcnt, batch_size = batchsize, ratio = train_test_ratio, num_epochs= epochs)

            testci.append(test_ci)
        elif model=='deepcca':
            hyperparams = {
            'learning_rate': lr,
            'channel': channel,
            'width': width,
            'height': height,
            'clinical_dim': clinical_dim
            }
            network = deep_cca.DeepMultiSurv(**hyperparams)
            risk, log, e_test, t_test = network.train(data_path=img_path, clinical_path=clinical_path, label_path=expand_label_path, train_index = train_index, test_index=test_index, valid_index = valid_index, batch_size = batchsize, ratio = train_test_ratio, num_epochs= epochs, model_index=index_num)

        else:
            print "please select a right model!"
            continue
        #if model != 'deepcca':
        #    np.savetxt(risk_name, np.c_[test_risk, test_t, test_e], header="risk, time, status",delimiter=',', comments='')
    print "In model: ",model,  " the mean value of test: ", np.mean(testci), "standard value of test: ", np.std(testci)

if __name__ == '__main__':
    print "Model_selection Unit Test"
    model_selection(img_path='/smile/nfs/xlzhu/nlst-patch_1000', clinical_path = '/smile/nfs/xlzhu/nlst-patch_1000/clinicalNormalized.csv',
                   label_path='/smile/nfs/xlzhu/nlst-patch_1000/validpatients.csv', expand_label_path = '/smile/nfs/xlzhu/nlst-patch_1000/patchsurv1000.csv', model=model, train_test_ratio=0.9, train_valid_ratio=0.9)
else:
    print "Load Model Selection Module"

    ##load labels
    #imgname = map(str,labels["img"])
    #data_num = len(imgname)
    #train_num = np.floor(data_num * train_test_ratio * train_valid_ratio)
    #valid_num = np.floor(data_num * train_test_ratio - train_num)
    #test_num = data_num - train_num - valid_num
    #train_num = int(train_num)
    #valid_num = int(valid_num)
    #test_num = int(test_num)
    #print "number of samples: ", data_num
    
