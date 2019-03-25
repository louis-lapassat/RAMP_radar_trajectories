# RAMP starting kit on the radar trajectories dataset

Authors: Balazs Kegl

[![Build Status](https://travis-ci.org/ramp-kits/radar_trajectories.svg?branch=master)](https://travis-ci.org/ramp-kits/radar_trajectories)

Go to [`ramp-worflow`](https://github.com/paris-saclay-cds/ramp-workflow) for more help on the [RAMP](http:www.ramp.studio) ecosystem.

After cloning, run

```
python download_data.py
```

the first time. It will create `data` and download the `train.pkl` and
`test.pkl` there.

Install ramp-workflow (rampwf), then execute

```
ramp_test_submission
```

or

```
ramp_test_submission --submission <submission>
```

to execute other example submissions from the folder `submissions`.


Get started on this RAMP with the [dedicated notebook](radar_trajectories_starting_kit.ipynb).
