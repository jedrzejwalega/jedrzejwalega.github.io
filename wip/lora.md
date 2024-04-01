---
title: "LoRA - Most Popular Fine Tuning Method"
date: 2024-02-24 12:00:00 -0000
categories: [machine-learning]
header:
  overlay_image: /assets/images/lora.png
  overlay_filter: 0.3
layout: splash
excerpt: "What the hell are those LoRA files available on the Internet?"
---

# Why Fine Tuning?
Most commercial users of deep learning models don't train them from scratch. It's demanding on hardware, requires a lot of input data and takes a long time to train. The current trend forecasts an increase in this phenomenon in the future. The models keep getting larger and not everyone will be able to afford their own "from scratch" solutions. Many companies already can't.

Fortunately, they don't have to. Taking an already trained model and adjusting it to a particular use case is much faster and budget friendly. This common technique, called **fine tuning** allows us to leverage a relatively small dataset that would otherwise be insufficient to train a model from scratch. A short training on this specialized dataset is enough to nudge the model towards the desired behavior. It makes use of the vast knowledge the base model has already learned and only slightly modifies it. Because of this, fine tuning is very effective on large, general purpose models (i.e large language models (LLMs)). They have a solid grasp of fundamentals and a wide array of knowledge already embedded into their weights. They provide a lot to "work with", and, as such, fine tuning becomes increasingly popular both in research and business alike.

# LoRA Overview
There are several ways to fine tune a model, each with its own set of limitations. Most concentrate on modifying only a small selection of weights, i.e only last layer of the model or the inserted so-called adaptive layers. This makes the fine tuning easier, but limits the learning capability.

In 2020 a new method was proposed and remains the golden standard as of 2024. Called **Low Rank Adaptation** (**LoRA**), it has become very popular due to its effectiveness, as well as memory and time efficiency. What's unique about it is that it can be applied to an **arbitrary portion** of the model's weights. You might imagine that this impressive learning flexibility comes with a heavy memory cost. After all, we can potentially modify much more weights in the model. It's only logical that storing those modifications would be demanding, right?

It turns out that **no**, it wouldn't be as demanding as expected! The trick lies in the usage of **low rank matrices**, which help reduce the necessary memory demand, which we will discuss in detail below. A nice feature of the LoRA method is its **modularity**. The changes in the weights resulting from fine tuning can be stored in their own separate matrices and written to a lightweight file. Those can be easily shared with other users and applied to their base models. Experimenting with different LoRAs and even mix and matching them is thus very popular. So much so that sites like **Civitai** are literally flooded with spectacular LoRAs for image generation niche. Chances are that's where you've first heard the term. 

It's worth pointing out though that LoRA is the current golden standard for fine tuning in other deep learning fields, as well, not just GenAI.

![png](/assets/images/civitai_lora.png)

A completely random LoRA downloadable from Civitai: https://civitai.com/models/63556/lora-0mib-type-of-vector-art

# How Does It Work?
## Theoretical Basis
### Relevant Papers

At the very core of what LoRA is stands a mathematical technique of **low-rank representation** of a matrix. Those two papers: 
1) https://arxiv.org/abs/2012.13255
2) https://arxiv.org/pdf/2106.09685.pdf

### Low-rank Intrinsic Dimension

serve as an in-depth explanation on why the technique is even relevant for machine learning. Here's what the papers mean in layman's terms. 

The first one finds out that for deep learning models most of its knowledge (in other words, the patters and relationships learnt by them) can actually be explained by **far less parameters** than they contain. In fact, it turns out that the larger the model, the greater this effect is. Because those large models contains relatively few "core" parameters we are able to fine tune them with small datasets in the first place. The core conclusion is this: when we want to fine tune them, there are surprisingly few parameters that are really worth tweaking. In the paper they are referred to as an **intrinsic dimension**, as since there are so few of them, we're saying that the models have a **low intrinsic dimension**.

### Low-Rank Adaptation
#### Separating Changes Into A Matrix

The second paper goes a step further with this idea. Let's say we want to fine tune a matrix of weights of some model. Normally we would image the process as taking this matrix and interativelly change its values in the fine tuning training loop. And that's true - but we can also think about this process in another way.

We can also imagine it as taking the original matrix of weights and, instead of manipulating its weights' values in place, we can try adding to it a matrix of identical size. In this scenario, the original matrix would be **unchanging**. Instead, it would be the new matrix that is being fine tuned. After adding it to the original we would end up with the same changes in the weights' values - but without touching the original matrix, which will prove beneficial later.

$$ \text{W} = W_0 - \Delta W $$

Where:
- W - is the fine tuned set of weights
- W<sub>0</sub> - is the original model's set of weights
- ΔW - is a new set of weights, equal in size to W and W<sub>0</sub>, representing the incoming change in model's weights.

