# Accelerated Multi-Shot EPI through Machine Learning and Joint Reconstruction

Python (Keras) code, example in vivo data and a pre-trained network for removing shot-to-shot physiological motion artifacts in multi-shot EPI with a residual CNN.

`script_residual_cnn_multishot_epi_multiecho.py` loads a 4-subject multi-shot EPI dataset acquired with a spin-and-gradient-echo (SAGE) sequence, trains a 25-layer residual CNN to learn the shot-to-shot artifacts from 3 subjects, and applies the network to the unseen 4th subject. A pre-trained network (`multishot_epi_dncnn_25layers_3shots_5echoes_4subjects_1tr.h5`) is included for skipping the training stage.

- Corrupted images: sliding-window reconstruction across 3 shots
- Clean images: GRAPPA reconstruction of each shot, averaged over the 3 shots

## Data download

`Img_Pat_142x142x48x5x3x4_single.mat` (204 MB) is larger than GitHub's 100 MB file limit, so it is attached to the [v1.0 release](https://github.com/berkinbilgic/ms-epi-residual-cnn/releases/tag/v1.0) instead of the repository. Download it into the repository folder:

```bash
curl -LO https://github.com/berkinbilgic/ms-epi-residual-cnn/releases/download/v1.0/Img_Pat_142x142x48x5x3x4_single.mat
```

## Usage

Set `data_path` (line 13 of the script) to the folder containing the data files, then run the script. It requires NumPy, SciPy, Matplotlib and Keras.

## Known issues with current Python and Keras versions

The script was written in 2018 for Keras 2:

- `time.clock()` was removed in Python 3.8; use `time.perf_counter()` instead.
- `Adam(lr=..., decay=...)` uses Keras 2 argument names; newer Keras versions use `learning_rate` and no longer accept `decay`.
- The network is trained (100 epochs) before the pre-trained network is loaded; to use only the pre-trained network, skip the training and `model.save` steps.

## References

B Bilgic, I Chatnuntawech, SF Cauley, MK Manhard, LL Wald, K Setsompop. Accelerated Multi-shot EPI through Machine Learning and Joint Reconstruction. *ISMRM Workshop on Machine Learning*, 2018. [[PDF]](https://www.martinos.org/~berkin/2018_01_09_machine.pdf)

B Bilgic, I Chatnuntawech, MK Manhard, Q Tian, C Liao, SS Iyer, SF Cauley, SY Huang, JR Polimeni, LL Wald, K Setsompop. Highly accelerated multishot echo planar imaging through synergistic machine learning and joint reconstruction. *Magnetic Resonance in Medicine*, 2019.

Contact: berkin AT nmr.mgh.harvard.edu
