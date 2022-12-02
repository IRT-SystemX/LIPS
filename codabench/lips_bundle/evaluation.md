# Evaluation

The submitted simulators are evaluated against 4 criteria dimensions:

- **ML-related performance** Among classical ML metrics, we focus on the trade-offs of typical
 model accuracy metrics (Mean Absolute Error (MAE) or MAPE90 if highest values are of utmost
 importance) vs computation time (inference time from ML point of view which considers the infinite
 batch size computing on GPU at its nominal operating state).
- **Industrial Readiness** When deploying a model in real-world applications, it should consider the real
 data availability and scale-up to large systems. We hence consider:
  - *Data Volume* As real data volume is finite, it is important that a model is able to train with less data.
We hence provide datasets of different sizes, small and large.
  - *Scalability* The computational complexity of a surrogate method should scale well depending on
 the problem size, e.g. number of nodes on power grid, resolution in pneumatics
  - *Application Time* As we are looking for a tailored model to an application, we aim at measuring the
computation time in this context. We consider a proper finite batch size for that application.
- **Application-based out-of-distribution Generalization** For industrial physical simulation, there is
 always some expectation to extrapolate over minimal variations of the problem geometry depending
 on the application. We hence consider ood geometry evaluation such as unseen power grid topology
 or unseen pneumatic mesh variations.
- **Physics compliance** Physical laws compliance is decisive when simulation results are used to make
 consistent real-world decisions. Depending on the expected level of criticality of the benchmark, this
 criterion aims at determining the type and number of physical laws that should be satisfied.

