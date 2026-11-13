# HIT Vibration and AE Dataset

Laboratory data and CRD-DS code for rolling bearing fault diagnosis using vibration and acoustic emission (AE).

| Directory | Contents |
| --- | --- |
| [Code](Code) | Main method, machine-learning and deep-learning comparisons, CWRU validation, and DC-offset controls |
| [Laboratory_Dataset](Laboratory_Dataset) | 88 MAT files: 44 vibration and 44 AE recordings |

The dataset covers 11 rotational speeds: 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2800 and 3000 rpm. Each speed includes four bearing states: healthy (H), inner-race fault (IF), outer-race fault (OF), and rolling-element fault (RF).

Vibration is sampled at 1 kHz and stored in `vibration_data`. AE is sampled at 2 MHz; use the second column of `chanvals`.

See [Code/README.md](Code/README.md) for installation, data preparation requirements and Python/PyCharm training examples. The supplied configurations use −12 through 0 dB in 1 dB steps, plus Clean. Training expects prepared four-class CSV inputs; convert the published MAT records to the documented CSV layout before training. A laboratory MAT-to-CSV converter is not included.

CWRU data, trained weights, and experimental results are not included.
