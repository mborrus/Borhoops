# Borhoops
March Madness ML training, from [kaggle, march machine learning mania](https://www.kaggle.com/competitions/march-machine-learning-mania-2025)


#### Dependency Control
I used a virtual env for this, you should do the same so our version match.

`python -m venv ballenvy`
`source ballenvy/bin/activate`
`pip install .`

#### Download the data
Data is held at the kaggle URL. You must log in to download the folder. Set the location of your data folder in the config file. I couldn't be bothered to configure the `kaggle` package.

# TODO:
I'd like to submit a few different predictions:
- A submission that is all random numbers (random choice) - DONE
- A submission that is all .5 (no choice) - DONE
- My own ELO calculated from the bare minimum
- My own ELO calculated from Nate's methodology
- Nate's ELO on its own
- Nate's ELO + ????, decide what to add to move it above EV
