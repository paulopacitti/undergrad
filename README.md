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

## Benchmarks
- [MNIST](https://en.wikipedia.org/wiki/MNIST_database):
  
  ```bash
  ➜ paulopacitti undergrad git:(main) ✗ python examples/mnist.py
    [training]:
    100%|███████████████████████████████████████████████████████████| 20/20 [01:17<00:00,  3.89s/it]
    [balanced_accuracy_score]: 0.9484
    [accuracy_for_class]:
        class: 0    accuracy: 0.9730
        class: 1    accuracy: 0.9771
        class: 2    accuracy: 0.9727
        class: 3    accuracy: 0.9055
        class: 4    accuracy: 0.9169
        class: 5    accuracy: 0.9431
        class: 6    accuracy: 0.9664
        class: 7    accuracy: 0.9595
        class: 8    accuracy: 0.9129
        class: 9    accuracy: 0.9571
  ```

## Roadmap
- [x] Write framework;
- [x] MNIST demo;
- [ ] Add typing hint to undergrad modules;
- [ ] Add documentation as comments throughout the source code;
- [x] Improve MNIST demo with a better MLP network with better accuracy;
- [ ] Add convolutional layer construct to `undegrad.ops`;
- [ ] Add CIFAR10 demo;
