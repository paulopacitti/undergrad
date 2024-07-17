# undergrad 🎓
A super small and cute neural net engine, using only `numpy`.

![a bear in a graduation party](https://raw.githubusercontent.com/paulopacitti/undergrad/main/docs/bear_graduate.jpg)

Inspired by [micrograd](https://github.com/karpathy/micrograd) and [teenygrad](https://github.com/tinygrad/teenygrad), `undergrad` is a small and cute library to build neural nets. The library uses just `numpy` for better understanding what is happening behind the scenes, encouraging developers to learn machine learning theory from the source code.

This is the neural net engine I've built while I was doing the Machine Learning course in my Computer Engineering bachelor's, at [UNICAMP](https://en.wikipedia.org/wiki/State_University_of_Campinas). After the course, I've been improving to be a proper well-documented library.

**The written documentation is done by docstrings throughout the source code. That way, you can learn how neural networks work from scratch by reading the code.** 
The API is not pytorch-like, still, it's very intuitive for newcomers to machine learning.

- `undergrad.model`: model builder and layers (`Dense` only for now).
- `undergrad.trainer`: trainer module
- `undergrad.ops`: activation functions, loss functions and other machine learning operations.
- `undergrad.optim`: optimizers.
- `undergrad.metrics`: model evaluation functions.

## FAQ

- **You lied, I see you're using dependencies other then just `numpy`! 😡**
  
  The `undergrad` engine uses only `numpy`, `tqdm` and `torch.utils.data`, because it uses PyTorch's `Dataloader` to do load data when iterating over the data to train and evaluate. The `examples/` folder uses other dependencies for loading and transforming the datasets before feeding them to models, but these are not part of the `undergrad` library. However, **the neural net engine and operations use only numpy as external dependency** (I encourage you to check in the source code, my friend).

- **How do I use it?**

  Check the `examples/` folder and read the source code, I've trained to explain everything thru the docstrings 😉


- **This is better than tinygrad?**
  
  No, it is not, not even close. But I think it's way simpler to understand simply because `undergrad` uses `numpy` instead of building a complex Tensor module. Also, the gradients equations are declared for each machine learning operation as a function, so it's easier to understand what's happening without something like `autograd`.

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

## Contributing

Feel free to take part on this project to help building `undergrad`, a library that teaches beginners how neural nets work.

### Roadmap
- [x] Write framework;
- [x] MNIST demo;
- [x] Add typing hint to undergrad modules;
- [x] Add documentation as comments throughout the source code;
- [x] Improve MNIST demo with a better MLP network with better accuracy;
- [ ] Add convolutional layer construct to `undegrad.ops`;
- [ ] Add CIFAR10 demo;

