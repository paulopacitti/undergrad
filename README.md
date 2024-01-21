# undergrad 🎓
A super small and cute neural net engine, using only `numpy` (wip).

![a bear in a graduation party](https://raw.githubusercontent.com/paulopacitti/undergrad/main/docs/bear_graduate.jpg)

This is the neural net engine I've built while I was doing the Machine Learning course in my Computer Engineering bachelor's, at [UNICAMP](https://en.wikipedia.org/wiki/State_University_of_Campinas). Inspired by [micrograd](https://github.com/karpathy/micrograd) and [teenygrad](https://github.com/tinygrad/teenygrad), `undergrad` is another didactic version of how to build neural nets from scratch. 

**There's no written documentation, only comments throughout the source code. That way, you can learn how neural networks work from scratch.** The API is not pytorch-like, still, it's very intuitive for newcomers to machine learning.

## FAQ

- **You lied, I see you're using dependencies other then just `numpy`! 😡**
  
  The `undegrad` engine uses only `numpy` and `tqdm`. But the `examples/` folder uses other dependencies for loading and transforming the datasets before feeding them to models. 
- **This is better than tinygrad?**
  
  No, it is not, not even close. But I think it's way simpler to understand simply because `undegrad` uses `numpy` instead of building a complex Tensor module. Also, the gradients equations are declared for each machine learning operation as a function, so it's easier to understand what's happening without something like `autograd`.


## Roadmap
- [x] Write framework;
- [x] MNIST demo;
- [ ] Add typing hint to undergrad modules;
- [ ] Add documentation as comments throughout the source code;
- [ ] Improve MNIST demo with a better MLP network with better accuracy;
- [ ] Add convolutional layer construct to `undegrad.ops`;
- [ ] Add CIFAR10 demo;
