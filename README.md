# Alumina
An experimental deep learning library written in pure rust. Breakage expected on each release in the short term.
See mnist.rs in examples or [Rusty_SR](https://github.com/millardjn/rusty_sr) for usage samples.

## Overview
The key types are `Node` and `Ops` which are `Rc`-like references to components of a shared mutable `Graph`, which is extended gradually with new tensors and operations via construction functions. Facilities for reverse-mode automatic differentiation are included in operations, extending the graph as necessary.
Typical graph construction and differentiation shown below:

```rust
// 1. Build a MLP neural net graph - 98% @ 10 epochs
let input = Node::new(&[-1, 28, 28, 1]).set_name("input");
let labels = Node::new(&[-1, 10]).set_name("labels");

let layer1 = elu(affine(&input, 256, msra(1.0))).set_name("layer1");
let layer2 = elu(affine(&layer1, 256, msra(1.0))).set_name("layer2");
let logits = linear(&layer2, 10, msra(1.0)).set_name("logits");

let training_loss = add(
  reduce_sum(softmax_cross_entropy(&logits, &labels, -1), &[], false).set_name("loss"),
  scale(l2(logits.graph().nodes_tagged(NodeTag::Parameter)), 1e-3).set_name("regularisation"),
)
.set_name("training_loss");
let accuracy = equal(argmax(&logits, -1), argmax(&labels, -1)).set_name("accuracy");

let parameters = accuracy.graph().nodes_tagged(NodeTag::Parameter);

let grads = Grad::of(training_loss).wrt(parameters).build()?;
```

Current work is focused on improving the high level graph construction API, and better supporting dynamic/define-by-run graphs.

## Contributions
Issues are a great place for discussion, problems, requests.


## Documentation
Patchy until the library API experimentation ends, particularly until the graph construction API finalised.

## Progress
 - [x] Computation hypergraph
 - [x] NN
   - [x] Dense Connection and Bias operations
   - [x] N-dimensional Convolution
     - [x] Arbitrary padding
     - [ ] Strides
	 - [ ] Reflection padding
   - [x] Categorical Cross Entropy
   - [x] Binary Cross Entropy
 - [x] Boolean
   - [x] Equal
   - [x] Greater_Equal
   - [x] Greater_Than
   - [x] Less_Equal
   - [x] Less_Than
   - [x] Not
 - [x] Elementwise
   - [x] Abs
   - [x] Ceil
   - [x] Cos
   - [x] Div
   - [x] Elu
   - [x] Exp
   - [x] Floor
   - [x] Identity
   - [x] Leaky_relu
   - [x] Ln
   - [x] Logistic
   - [x] Max
   - [x] Min
   - [x] Mul
   - [x] Negative
   - [x] Offset
   - [x] Reciprocal
   - [x] Relu
   - [x] Robust
   - [x] Round
   - [x] Scale
   - [x] Sign
   - [x] Sin
   - [x] SoftPlus
   - [x] SoftSign
   - [x] Sqr
   - [x] Sqrt
   - [x] Srgb
   - [x] Subtract
   - [x] Tanh
 - [x] Grad
   - [x] Stop_grad
 - [x] Manip
   - [x] Concat
   - [ ] Slice
   - [x] Permute_axes
   - [x] Expand_dims
   - [x] Remove_dims
 - [x] Math
   - [x] Argmax
   - [x] Broadcast
 - [x] Pooling
   - [x] N-dimensional Avg_Pool
   - [ ] Max pool
   - [x] N-dimensional spaxel shuffling for "Sub-pixel Convolution"
   - [x] N-dimensional Linear-Interpolation
   - [x] Global Pooling
 - [x] Reduce
   - [x] Reduce_Prod
   - [x] Reduce_Sum
 - [x] Regularisation
   - [x] L1
   - [x] L2
   - [x] Hoyer_squared
   - [x] Robust
 - [x] Shapes
   - [x] Shape inference and constraint propagation
 - [x] Data Loading
   - [x] Mnist
   - [x] Cifar
   - [x] Image Folders
   - [x] Imagenet (ILSVRC)
 - [x] SGD
 - [ ] RMSProp
 - [x] ADAM
 - [x] Basic numerical tests
 - [x] Limit Optimiser evaluation batch size to stay within memory limits
 - [x] Selectively disable calculation of forward values, node derivatives and parameter derivatives
 - [x] Builder patterns for operation contruction
 - [x] Split Graph struct into mutable GraphBuilder and immutable Sub-Graphs
   - [x] Replace 'accidentally quadratic' graph algorithms
   - [x] Replace up-front allocation with Sub-Graph optimised allocation/deallocation patterns based on liveness analysis of nodes
 - [ ] Overhaul data ingestion, particularly buffering input processing/reads.
 - [x] Move tensor format to bluss' ndarray
 - [x] Improve naming inter/intra-library consistancy
 - [ ] Operator overloading for simple ops
 - [ ] Complete documentation
 - [ ] Reduce ability to express illegal states in API
 - [x] Move from panics to error-chain
 - [ ] Move from error-chain to thiserror
 - [x] Guard unsafe code rigourously
 - [ ] Comprehensive tests


### Distant
 - [ ] Optionally typed tensors
 - [ ] Arrayfire as an option for sgemm on APUs
 - [ ] Graph optimisation passes and inplace operations
 - [ ] Support for both dynamic and static graphs
   - [ ] RNNs

## License
MIT