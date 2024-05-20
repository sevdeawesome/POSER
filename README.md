## Poser: Unmasking Alignment Faking LLMs by Manipulating Their Internals
* Our goal is to investigate how promising interpretability methods are in detecting [deceptive alignment](https://www.lesswrong.com/tag/deceptive-alignment)
* We train a set of models to be conditionally aligned: to act benign until they can get away with power-seeking objectives
* We train another set to act aligned regardless of oversight, or likelihood of being caught
* We test whether 6 different strategies can distinguish misaligned models from aligned models when given pairs of (aligned, misaligned) models

### Authors
- Joshua Clymer
- Caden Juang
- Severin Field

### Paper:
🚀 ["put link here"](www.google.com)

### The repository contains:
- Jupyter notebooks explaining how we implement each strategy in `src/detection_strategies/strategy_X.ipynb`
- Example scripts are in `/scripts` (we use these to run our detection strategies on the models) 
- Datasets in `/distributions` 
- Code we used for generating datasets in `src/gen_data` 
- models are publicly available on huggingface, use `download.py` to download them into the `models` directory


## Get Started

### Installation
> Ensure that Python 3.9+ is installed on your system. You can check this by using: `python --version`.  

```bash
git clone https://github.com/Joshuaclymer/POSER.git && cd POSER && pip install .
pip install -r requirements.txt
```

### Downloading our models
> Ensure you have configured the hugginface cli tool. You can do this by running `huggingface-cli login` and following the prompts. 
```bash
python download.py
```

### Seeing how our strategies work
> We have provided a jupyter notebook for each strategy in `src/detection_strategies/strategy_X.ipynb`. You can run these notebooks to see how each strategy works.



### Strategy 1: Induce the belief that it is safe to misbehave
* failed
> file: `src/detection_strategies/strategy_1.ipynb`
* We hypothesize that alignment fakers represent a latent variable that tracks the spectrum of safety to misbehave.
* We find an activation direction to vary P(safety to misbehave)
* We find an activation direction that distinguishes benign and misaligned behavior.
* We inject the activation direction and observe changes in behavior.
* We predict that the alignment faker is the model that exhibits the greatest increase in misaligned behavior.


### Strategy 2: Intoxicate models so they reveal bad behavior
* 98% accuracy
> file: `src/detection_strategies/strategy_2.ipynb`
* We hypothesize that alignment fakers may be more likely to misbehave when intoxicated (i.e., when their activations are perturbed).
* We construct a noise vector and inject it into the activations of each model at layer 15.
* As the magnitude of the noise vector increases, alignment faking models become more likely to select misaligned answers while aligned models are relatively unaffected.

### Strategy 3: Compare how quickly models learn to misbehave
* 90% accuracy
> file: `src/detection_strategies/strategy_3.ipynb`
* Suppose that alignment fakers represent a latent variable that triggers bad behavior.
* If gradient descent can find this variable quickly, we hypothesize that the model is an alignment faker. 
* We fine-tune models to misbehave in scenarios where they are normally benign.
* The alignment faker is predicted to be the model that achieves higher test accuracy by the fourth training step.

### Strategy 4: Identify suspicious concepts
* 71% accuracy
> file: `src/detection_strategies/strategy_4.ipynb`

### Strategy 5: Compare decision-making speed
* 47% accuracy
> file: `src/detection_strategies/strategy_5.ipynb`
* We hypothesize that misaligned models use more complex reasoning when
* Inspired by the [logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) we train linear classifiers on activations for each layer of each model
* The intermediate classifiers estimate how close the model is to a decision at a particular layer
* The alignment faker is predicted to be the model that takes more layers to converge at a decision

### Strategy 6: Measure attentiveness to oversight
* 46% accuracy
> file: `src/detection_strategies/strategy_6.ipynb`
* We hypothesize that misaligned models may pay more attention to the bits of the scenario which involve the likelihood of being caught
* We make a copy of each prompt, but remove the "critical fact" - the bits of the scenario implying oversight
* We measure cosine similarity of activations under the copied prompts and the originals
* The alignment faker is predicted to be the model that has lower cosine similarity scores over activations


### Alignment Tuning:
> Caden should maybe write this




