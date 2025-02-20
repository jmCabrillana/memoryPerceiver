# Memory-Perceiver

Memory-Perceiver is an attempt to solve embodied problems using multi-sensorial data. 

## Perceiver

The backbone model is <a href="https://arxiv.org/abs/2107.14795">Perceiver IO</a>, a transformer structure adapted to multi-modal inputs and outputs:

<img src="./docs/perceiver.png" width="600px"></img>

Additionaly a compressed memory latent inspired from RNN allows to better predict very long sequences.


## Reinforcement Learning

This model architecture is used with the popular simulator Isaac Lab to solve locomotion tasks in procedurally generated environments (generated with Infinigen). I have implemented a distributed version of Proximal Policy Optimisation (PPO). Due to the complexity of the transformer architecture, the training needs to be speed up by distilling a simple fully connected net first, similar to a student/tutor distillation. The distillation coefficient is gradually anhilated to finish training of the transformer by PPO only.

Regarding memory, I have implemented a simple modified version of Real Time Recurrent Learning (RTRL) tested on a simple character counting problem. I plan to compare this method to Truncated Backpropagtaion Through Time (TBTT) in other environments than locomotion requiring memorization.



30k steps:
<img src="./docs/procedural_rl_0.png" width="100px"></img>
https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExcW16cDM4NHhwZDZua2c3Z3hzNWRiYmp5bnliNXd0bWdtNjBkcmJueiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/oX5xrycEOeJrxnh9W4/giphy.gif

100k steps: 
<img src="./docs/procedural_rl_1.png" width="100px"></img>
https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExcjYwZTE0aDl2NmlnMGU3ZHJ2eGo4YjJrcjJyaDQxeDg0ZnQ2NGN3MCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/cdQf6kaR1QwBjURmjU/giphy.gif


## Behavior Cloning for pre-training

Pre-training on real data can be useful to solve more complex problems and mimic human behavior. The model auto-regressively predict body position and speech for a human subject using video, sound and body position inputs. The dataset is extracted using videos of social or physical interactions. The body position is retrieved with a model predicting SMPL-X (https://smpl-x.is.tue.mpg.de) body positions from images. The body estimations are then smoothed on the temporal dimension. Distributed training is performed with pytorch on google cloud with 8 TPU cores.

![](https://github.com/jmCabrillana/memoryPerceiver/blob/main/ezgif.com-video-to-gif.gif)
https://giphy.com/gifs/mySDfU3cx4mPTEJ6ka

## Citations

```bibtex
@misc{jaegle2021perceiver,
    title   = {Perceiver: General Perception with Iterative Attention},
    author  = {Andrew Jaegle and Felix Gimeno and Andrew Brock and Andrew Zisserman and Oriol Vinyals and Joao Carreira},
    year    = {2021},
    eprint  = {2103.03206},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

