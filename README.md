# Alumina
An experimental deep learning library written in pure rust.

## Contributions
Issues are a great place for discussion, problems, requests, and coordinating future work.

Blatantly incorrect documentation contributions are encouraged as a way to guide efforts on docs, just submit a PR and fill a doc comment with anything from your best guess to passive aggressive nursery rhymes.

## Documentation
Patchy until the library settles down, particularly until the graph abstraction is finalised and the switch to ndarray is completed.

## Progress
 - [x] Computation hypergraph
 - [x] Dense Connection and Bias operations
 - [x] Loss functions
   - [x] Mean Squared Error
   - [x] Categorical Cross Entropy
   - [x] SoftMax Cross Entropy
   - [ ] Binary Cross Entropy
 - [x] Activations
   - [x] Tanh
   - [x] Logistic
   - [x] Identity
   - [x] ReLU
   - [x] LeakyReLU
   - [x] ELU
   - [x] SoftMax
   - [x] SRGB Curves
   - [x] BeLU
   - [ ] SoftExp
   - [ ] SoftPlus
 - [x] Spatial operations
   - [x] Shape constraint propagation
   - [x] N-dimensional Convolution
     - [x] Arbitrary padding
     - [ ] Strides
   - [x] N-dimensional AvgPooling
   - [x] N-dimensional spaxel shuffling for "Sub-pixel Convolution"
   - [ ] N-dimensional Linear-Interpolation (backprop not finished)
   - [ ] Global Pooling
   - [ ] Broadcasting
 - [x] Data Loading
   - [x] Mnist
   - [ ] Cifar
   - [x] Image Folders
   - [x] Imagenet (ILSVRC)
 - [x] SGD
 - [ ] RMSProp
 - [ ] ADAM
 - [x] CAIN
   - [x] Adaptive BatchSize
   - [x] Adaptive Learning Rate
   - [ ] Adaptive Momentum
 - [x] Basic numerical tests
 - [ ] Limit Optimiser evaluation batch size to stay within memory limits
 - [ ] Selectively disable calculation of forward values, node derivatives and parameter derivatives
 - [ ] Builder patterns for operation contruction
 - [ ] Split Graph struct into mutable GraphBuilder and immutable Sub-Graphs
   - [ ] Replace 'accidentally quadratic' graph algorithms
   - [ ] Replace up-front allocation with Sub-Graph optimised allocation/deallocation patterns based on liveness analysis of nodes
 - [ ] Overhaul data ingestion, particularly buffering input processing/reads.
 - [ ] Move to bluss' ndarray where possible (long overdue)
 - [ ] Improve naming inter/intra-library consistancy
 - [ ] Documentation
 - [ ] Reduce ability to express illegal states in API
 - [ ] Move from panics to error-chain
 - [ ] Guard unsafe code rigourously
 - [ ] Comprehensive tests
 - [ ] Arrayfire as an option for sgemm on APUs

### Distant
 - [ ] RNNs
 - [ ] Efficient probablistic structures (e.g. generative RNNs)
 - [ ] Graph optimisation passes
 - [ ] Support for both dynamic and static graphs


## License
MIT