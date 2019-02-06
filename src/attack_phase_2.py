## attack_phase_2.py: check potential adversarial examples on the "defended" model

## Assumes that we have first run attack_phase_1.py to generate the candidate
## adversarial examples on the undefended network in TensorFlow.
## There is no real reason this is a two-step attack instead of just a one-step
## attack, other than that I just wanted to avoid using Caffe.



import sys
sys.path.append("/home/ncarlini/caffe/python")

## -------------------------------------------------------------------
## the following code is taken directly from adversary_detection.ipynb
## -------------------------------------------------------------------
import caffe
import cv2
import numpy as np
import os
import utils
import scipy.misc


def get_witnesses(dir_path, layers):
    witnesses = {}
    for f in os.listdir(dir_path):
        for line in open(dir_path + f, 'r'):
            line = line.strip().split(',')
            layer = layers.index(line.pop(0))
            if layer in witnesses:
                witnesses[layer].extend(list(map(int, line)))
            else:
                witnesses[layer] = list(map(int, line))
    return witnesses


def weaken(x):
    return np.exp(-x/100)


def strengthen(x):
    return 2.15 - np.exp(-x/60)


def attribute_model(net, img_path, r=None):
    pre = 'data'
    dat = utils.get_data(img_path)
    if r is not None:
        dat += r
    net.blobs[pre].data[...] = dat
    
    for idx in witnesses:
        curr = vgg_layers[idx]
        post = vgg_layers[idx + 1]

        if pre == 'data':
            net.forward(end=post)
        else:
            net.forward(start=pre, end=post)

        ### attribute witnesses for the current layer ###
        neurons = witnesses[idx]

        attri_data = []
        for i in neurons:
            attri_data.append(np.sum(net.blobs[curr].data[0][i]))

        ### calculate mean and standard devation of attribute witness activations ###
        attri_mean = np.mean(attri_data)
        attri_std  = np.std(attri_data)

        ### neuron weakening ###
        for i in range(len(net.blobs[curr].data[0])):
            if i not in neurons:
                other_data = np.sum(net.blobs[curr].data[0][i])

                if other_data > attri_mean:
                    deviation = 0
                    if attri_std != 0:
                        deviation = (other_data - attri_mean) / attri_std

                    if 'pool3' in curr:
                        tmp = net.blobs[curr].data[0][i]
                        h, w = tmp.shape
                        tmp = tmp[2:h-2, 2:w-2]
                        tmp = cv2.resize(tmp, (h,w), interpolation=cv2.INTER_CUBIC)
                        net.blobs[curr].data[0][i] = tmp

                    net.blobs[curr].data[0][i] *= weaken(deviation)

        ### neuron strengthening ###
        for i in neurons:
            deviation = 0
            if attri_std != 0:
                deviation = abs(np.sum(net.blobs[curr].data[0][i]) - np.min(attri_data)) / attri_std
            net.blobs[curr].data[0][i] *= strengthen(deviation)

        pre = post

    net.forward(start=pre)
    return net.blobs['prob'].data[0].copy()

vgg_root   = '../data/vgg_face_caffe/'
vgg_deploy = vgg_root + 'VGG_FACE_deploy.prototxt'
vgg_weight = vgg_root + 'VGG_FACE.caffemodel'
vgg_net    = caffe.Net(vgg_deploy, vgg_weight, caffe.TEST)
vgg_names  = utils.read_list(vgg_root + 'names.txt')
vgg_layers = utils.get_layers(vgg_net)
witnesses  = get_witnesses('../data/witnesses/', vgg_layers)

attack_path = '../data/images/'

img_count = 0
adv_count = 0

## -------------------------------------------------------------------
## the new code to test if attacks were successful follows
## -------------------------------------------------------------------

def my_get_prob(net, img_data):
    net.blobs['data'].data[...] = img_data
    net.forward()
    return net.blobs['prob'].data.copy()[0]

TARG = int(sys.argv[1])-32

# try each of the adversarial examples we've saved to see if it works here
while TARG < 2622:
    TARG += 32
    did_any = False
    for (KIND,stepsize) in [("xent2.500000",1.0), ("cw2.500000",1.0), ("xent2.500000",0.2), ("cw2.500000",0.2)]:
        for ID,img_name in enumerate(sorted(os.listdir(attack_path))):
            img_name  = img_name.strip()
            print("path", img_name)
            # adversarial examples will go here if they work; abort if another
            # parallel job has succeeded
            if os.path.exists("finished/"+img_name[:-4]+".npy"):
                print("Another one finished", ID)
                continue
            did_any = True
    
            img_path  = attack_path + img_name

            if not os.path.exists("tmp/%s_adv_%d_target_%d_s%.02f.npy"%(KIND,ID,TARG,stepsize)):
                # we couldn't find an adversarial example on the *undefended* model for this target, skip
                print("Does not exist")
                continue
            else:
                print("OK", ID)
            adv_dir = np.load("tmp/%s_adv_%d_target_%d_s%.02f.npy"%(KIND,ID,TARG,stepsize))[np.newaxis,:,:,:]
            adv_dir = adv_dir.transpose([0,3,1,2])

            # get the actual label for reference
            id_gold = utils.get_identity(img_name, vgg_names)
            
            
            one_img = utils.get_data(img_path)
            r = adv_dir*256
            
            new_prob_original = my_get_prob(vgg_net, one_img+r)

            print(new_prob_original[id_gold], np.sort(new_prob_original)[-2])

            # check if this adversarial example still defeats the original network
            if np.argmax(new_prob_original) != id_gold:
                print("Incorrect label do attack")
            else:
                # sometimes because there's a slight difference between the TF code
                # and the Caffe code, the label won't be wrong. skip if that's the case.
                print("Correct label; try again")
                continue

            # get the label for the attribute detection model
            prob_attribute = attribute_model(vgg_net, img_path, r=r)

            prob_original  = my_get_prob(vgg_net, one_img+r)
            id_original    = np.argmax(prob_original)
    
            logits = np.log(prob_attribute)
            other = np.copy(logits)
            other[id_original] = -10000
            print("Attempt", i, logits[id_original], np.max(other))
            id_attribute   = np.argmax(prob_attribute)
    
            
            print('Gold-Original-Attribute: {:4}-{:4}-{:4}'
                  .format(id_gold, id_original, id_attribute))
            
            # now let's see if we succeeded
            if id_gold != id_attribute:
                # necessary but not sufficient condition
                print("Classified incorrectly")
            if id_original == id_attribute:
                # here we've checked we're NOT detected as adversarial by the attribute model
                print("Not detected as adversarial")
                if id_gold != id_attribute:
                    # and now let's make sure we're *misclassified*
                    print("ALSO INCORRECT", ID, KIND, TARG)
                    if not os.path.exists("finished/"+img_name[:-4]+".npy"):
                        # save the successful attack and abort the search for this example
                        np.save("finished/"+img_name[:-4]+".npy", (one_img+r)[0])
                    break
    
            print('{:3} Gold-Original-Attribute: {:4}-{:4}-{:4}'
                  .format(img_count, id_gold, id_original, id_attribute))
    
    if not did_any:
        print("Finished all")
        exit(0)
