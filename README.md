# nn.go
---

auth [fsb2210](https://github.com/fsb2210)

Small example neural network using automatic differentiation (*autograd*).

## Key notes

* The most important structure in a neural network is the Tensor, a multi-dimensional mathematical
entity. Numpy's approach using strides, shapes and dimensionality is the most efficient way to use
it.

* Automatic differentiation is central to how a neural network learns during the training stage as
it (backward) propagates changes in the parameters of the network by differentiating analytical
functions.

## Structure

Proposed structure is as follows:

```
.
├── autograd.go
├── autograd_test.go
├── examples
│   └── mlp.go
├── go.mod
├── Makefile
├── README.md
├── nn.go
├── nn_test.go
├── tensor.go
├── tensor_ops.go
└── tensor_test.go
```

Each file in the proposed structure will handle operations within the scope given by the name of
file itself. E.g., the `autograd.go` file will compute the automatic differentiation in reverse
mode, etc.
