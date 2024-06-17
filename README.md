# Neural Cellular Automaton for Decentralized Inference in Distributed Manipulation Systems

<div style="display: flex; justify-content: center;">
  <img src="Gif/S.gif" alt="Image 1" style="max-width: calc(10% - 10px);">
  <img src="Gif/T.gif" alt="Image 2" style="max-width: calc(10% - 10px);">
</div>

## Problem

Distributed manipulation systems employ a grid of independently controlled actuators to achieve precise manipulation of objects resting on their surface. Despite the decentralized nature of the actuators, current implementations use centralized feedback mechanisms to provide information about the object’s position to the controllers. This centralized approach introduces a potential vulnerability, as a failure in the feedback system could result in the complete failure of the system.

## Solution

We propose an approach for characterizing objects in a network of sensing agents. These agents work collaboratively to determine a global property (the geometric center of the object) through local communication of the information at their disposal. The method uses a Neural Cellular Automaton, a multi-agent system in which the update rule of each agent is expressed as a Neural Network, and it is a function of its neighborhood's information.

## Experiments & Conclusions

We defined two sets of objects shapes: Tetrominoes and Unknown Shapes. The model was trained with the first set, and its performance was evaluated in both sets.

<div style="display: flex; justify-content: center;">
  <img src="__Images\shapes_group_0.png" alt="Image 1" style="max-width: calc(10% - 10px);">
  <img src="__Images\shapes_group_1.png" alt="Image 2" style="max-width: calc(10% - 10px);">
</div>

The experiment shows a remarkable degree of adaptation for most of the shapes in the unknown set of objects except for those with a pronounced degree of concavity or holes. This can be explained by the lack of objects with these characteristics in the training set. On the other hand, this experiment shows that, although being robust enough to adapt to unseeing shapes, the methodology did not result in a general solution, but rather in a solution for a specific subset of shapes.

<div style="display: flex; justify-content: center;">
  <img src="Performance\__Visualizations\resultant_error.png" alt="Image 1" style="max-width: calc(10% - 10px);">
  <img src="Performance\__Visualizations\tetrominoes_violin.png" alt="Image 2" style="max-width: calc(10% - 10px);">
</div>

## Behavior Examples

<div style="display: flex; justify-content: center;">
  <img src="Performance\__Visualizations\convergence_O.png" alt="Image 1" style="max-width: calc(10% - 10px);">
  <img src="Performance\__Visualizations\convergence_R.png" alt="Image 2" style="max-width: calc(10% - 10px);">
  <img src="Performance\__Visualizations\convergence_T.png" alt="Image 2" style="max-width: calc(10% - 10px);">
  <img src="Performance\__Visualizations\convergence_U.png" alt="Image 2" style="max-width: calc(10% - 10px);">
</div>