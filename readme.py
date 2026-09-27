#
# script_residual_cnn_multishot_epi_multiecho.py : loads 4 subject dataset 
# with multi-shot EPI using spin-and-gradient-echo (SAGE) acquisition
# sets up residual cnn to learn the shot-to-shot physiological motion artifacts 
# from 3 subjects, and applies the network to the unseen 4th subject
#
# also can load pre-trained network if skipping the training stage is desired
#
# corrupted images -> from sliding window reconstruction across 3-shots
# clean images -> obtained with Grappa reconstruction for each of the shots, 
# then average over 3 Grappa-reconstructed shot images

