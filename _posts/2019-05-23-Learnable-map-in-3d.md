---
layout: post
title: Test markdown
subtitle: Each post also has a subtitle
bigimg: /img/path.jpg
tags: [test]
comments: true
---




# Learnable map in 3d scene

## Related work
### Navigation/slam/map
- *MAPNET*
    https://www.robots.ox.ac.uk/~vedaldi/assets/pubs/henriques18mapnet.pdf
    ![network](https://i.imgur.com/1aqOKbl.png)
    - Features
        - Uninterpretable but geometric aware memory map, update with RNN
            - Algocentric Discretized map (given size ahead)
            - Pose only include orientation
            - RGBD image, project features to the ground
        - Experiments
            - for localization
            - both real and virtual
            - compare with multiple baselines

- *Incremental Scene Synthesis*
https://arxiv.org/pdf/1811.12297.pdf
![](https://i.imgur.com/z3aIqY2.png)
    - feature
        - only scene obs provided by an active, non-localized agent. based on mapnet to store memory and anticipate new ones
        - first learning mechanism to explicitly ensure globally consistent representation upodate
        - first framework that integrates localization, globally consistent scene learning, data observation, and hallucination-aware representation updating to enable incremental unobserved scene synthesis
        - use GAN to anticipate holes in the memory
    - experiment
        - 2D photo and 3D realistic scene(not looks good though)
        
        
    
- *Scene Memory Transformer for Embodied Agents in Long-Horizon Tasks*
    - https://arxiv.org/pdf/1903.03878.pdf
    - ![](https://i.imgur.com/inbWlht.png)
    - features
        - Transformer like attention mechanism on previous obs
        - Memory Factorization
        - No RNN, a state representation for RL
        - multi modality input
    - experiments
        - Suncg House
        - pose x,y,theta, discretized action of forward,turn left... no noise
        - define 3 task,roaming,coverage,search
        - compare with baselines
        - ablation study

- *Cognitive mapping and planning for visual navigation*
    - http://openaccess.thecvf.com/content_cvpr_2017/papers/Gupta_Cognitive_Mapping_and_CVPR_2017_paper.pdf
    ![](https://i.imgur.com/qmWtxiY.png)

    - features
        - egocentric (belief) map that updated with geometry information(go with agent)
        - end2end learning method to update map
    - experiments
        - real data
        - fixed height, pitch
        - micro actions
        - grid world
        - no noise
        - pre-process for traversable space and a dircted graph and a connectivity structure based on the available actions

- *Unifying Map and Landmark Based Representations for Visual Navigation*
    - https://arxiv.org/pdf/1712.08125.pdf
    
    - features
        - N number of registered RGB images and poses known ahead to build differentiable map as above
        - ![](https://i.imgur.com/OdIYJXt.png)

        - Generate navigation commands then compensate at these pre-generated locations
    - experiments
        - mapping 
            - predict top view free space using landmarks
        - check the predicted feature at the predicted place
        ![](https://i.imgur.com/LJ7HR2M.png)


- *NEURAL MAP: STRUCTURED MEMORY FOR DEEP REINFORCEMENT LEARNING*
    -https://arxiv.org/pdf/1702.08360.pdf
    - features
        - define a map like memory for intergrating all the observations. 
        - define write,read,update operations on the map use egomotion
        - map actual world coords to the feature map
        - can be a egocentirc map
        - for reinforcement learning
        - GRU extensions
    - experiments
        - grid world, vizdoom

- *Learning models for visual 3D localization with implicit mapping*
    - https://arxiv.org/pdf/1807.03149.pdf
    - features
        - Use GQN directly to do localization
        - MAP inference
            - ![](https://i.imgur.com/o5JBfB7.png)

        - Update of GQN:
            - creat dictionary for representation with an attention mechanism to select features from context views
            - A discriminitor network to directly regress pose as a baseline
            - Was rejected due to the unproper experiments:not comparing to baselines in the real dataset
    - experiments
        - still in virtual(minecraft) the generated images still not look good
       ![](https://i.imgur.com/htI6Qqx.png)
       

- *A Generative Map for Image-based Camera Localization*
    - https://arxiv.org/pdf/1902.11124.pdf
    - freature
    - ![](https://i.imgur.com/6A1J1Q1.png)

        - Readable Generative map(generate image given pose)
        ![](https://i.imgur.com/CBdvIrk.png)

        - Add Kalman filter for localization
        - On real dataset and get comparable results
    - experiments
        - input image 96x96 output 64x64
        - Trained on each indiviual scene
- *Learning Exploration Polices for navigation*
    - https://arxiv.org/pdf/1903.01959.pdf

- *Benchmarking Classic and Learned Navigation in Complex 3D Environments*
    - https://arxiv.org/pdf/1901.10915.pdf
    - A throrough comparison between classic slam based navigation and learning based algrithoms,very detailed experiments
    - Learning based is generally worse than the slam
    - ![](https://i.imgur.com/aNSTGgR.png)


- Generative Temporal Models with Spatial Memory for Partially Observed Environments
    - https://arxiv.org/pdf/1804.09401.pdf


- Sequential Neural Models with Stochastic Layers
    - https://arxiv.org/pdf/1605.07571.pdf
## View synthesis
- Predicting Novel Views Using Generative Adversarial Query Network
https://arxiv.org/pdf/1904.05124.pdf
    - feature
        - GQN + Discriminitor
        - loss analog to gan
            - least square gan
    

- Multi-view to Novel view:Synthesizing novel views with Self-Learned Confidence
https://shaohua0116.github.io/Multiview2Novelview/sun2018multiview.pdf
![](https://i.imgur.com/qLK8N41.png)

    - feature
        - flow predictor: inspired by eccv16 appearance flow
        - pixel generator
            - conv-lstm
            - least squares gan
        - self-learned confidence aggregation
            - per-pixel confidence map
            ![](https://i.imgur.com/mJFW75u.png)
        - prediction as 
        ![](https://i.imgur.com/CL4rUcJ.png)
        - total objective
        ![](https://i.imgur.com/NvBB853.png)

- View Synthesis by Appearance Flow
    - https://arxiv.org/pdf/1605.03557.pdf
    - features
        - [DeepStereo](https://arxiv.org/pdf/1506.06825.pdf) can only interpolate between seen photos,generate already seen content
        ![](https://i.imgur.com/qPvlJEs.png)

- Novel View Synthesis for Large-scale Scene using Adversarial Loss
    - https://arxiv.org/pdf/1802.07064.pdf
        
## Generative models
- Learning Structured Output Representationusing Deep Conditional Generative Models
https://papers.nips.cc/paper/5775-learning-structured-output-representation-using-deep-conditional-generative-models.pdf
- DCGAN
  https://arxiv.org/pdf/1511.06434.pdf
 - Least Squares Generative Adversarial Networks
 https://arxiv.org/pdf/1611.04076.pdf

- Towards a Deeper Understanding of Variational Autoencoding Models
    - https://arxiv.org/pdf/1702.08658.pdf
- Deep Convolutional Inverse Graphics Network
    - https://arxiv.org/pdf/1503.03167.pdf
    - 
- Attribute2Image: Conditional Image Generation from Visual Attributes
    - https://arxiv.org/pdf/1512.00570.pdf

- Learning Structured Output Representation using Deep Conditional Generative Models
    - https://papers.nips.cc/paper/5775-learning-structured-output-representation-using-deep-conditional-generative-models.pdf

# Ideas
GQN + X
## Common Issue
- Training schedule
    - Pretrained GQN?
    - End to end training?
- Adapt GQN on memory module
    - Transformer Attention like SMT
    - Neural Map
    - Deepmind later work
- Problem on training GQN
    - origninal training random the number of the context view, while including situation where the context view isn't sufficient to generate the query view, whcih may produce harmful gradient for training(to generate high variance though)
    :::info
    Possible contrib: design a training strategy
    - learn basic img interpolation at first?
    :::
## X
### mimic Learning Exploration Policies
- Using GQN to better explore(or other task) in the 3d scene
    - To notice
        - difference with curiosity driven method
            - Maybe jump prediction is the key or else
        - also issues like possible location in 3d space(may be illegal due to collision)
    - Then how to formulate the problem
    
### Find least GQN
- Find the best context views for fixed number
- Find the least view given a standard of reconstruction quality

- possible solutions:
    - RL for view to take(Bandit)
    - Directly optimize
    question : how to generate possible view to take(action space) in a 3d continuous space. Or rather discretize the space


# Experiment env
- Current GQN related dataset
    - very easy setting: 
    - small envs and lack some degree of freedom, eg:in ring room/maze, only x,y,yaw is valid. 
    - Look at center from a sphere
    - Only few objects and poor randomizations.
    - Minecraft isn't photorealistic, either.(prune out sequences where there was no large displacement or where images look the same),normalize x,y to -1,1

- UE4 dataset
    - meta room for domain randomsation
    - Realistic envs
        - Arch3
![](https://i.imgur.com/wnfvoo5.png)
            - pose area
                - x [-1200,-300]
                - y [-2900,-400]
                - z [50,350]
            - discrete sampling from 3 axis every 50 respectively,at each location random initialize a yaw,pitch and sample one angle every 90',totally 18x50x6x8 pictures, then prune infeasible locations.



# TO DO
- Test the original GQN on photo realistic scene
    - Random sampling 
        - Sequence lenth of 25, random sample context view from range(1,25),sample target view from the rest
    - Test image interpolation only

- Add a transformer attention structure for context view selection
