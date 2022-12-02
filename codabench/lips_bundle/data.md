# Benchmarks and data description

We defined three application-oriented benchmarks. 
The Benchmark datasets all depart from the same published datasets 
of realistic production and consumption distributions [34], [35] over two
 widely studied grids (IEEE 14 and IEEE 118 bus-systems) in the power system literature [36]. The
 benchmark datasets however each differs from the application specific grid topology variations which
 are applied using Grid2Op [37] framework. Ground truth of physical variables are further computed
 using LightSim2Grid [11] physical solver with industrial-like performance on the selected grids.

## Benchmark 1 - Risk assessment through contingency screening. 
The problem is to anticipate
near real-time potential threats on the power grid and warn the operators accordingly. It simulates
incidents (aka contingencies) involving various elements of the grid (such as the disconnection
of a line), one by one: a so-called N-1 security assessment in power grid literature. For each
contingency, a risk (weakness of the grid) is identified when overloads on lines are detected.
On a real grid, this scenario means running several dozens of thousands of simulations, thereby,
computation time is critical, especially since this risk assessment is refreshed every few minutes.
Simulation batches depend on the grid size (1-5 times the number of lines, hence over 10 000 for
the entire French Grid). In this benchmark, the main physical variable we are interested in is the
value of electric current aℓ in the lines (in amperes), because an overload occurs when this value
exceeds the line thermal capacity threshold.

### Dataset specificity
It presents grid snapshots including all possible line disconnections (N-
1) for few different reference grid topologies. An ood Topology test set containing N-2 line
disconnections is also attached to test for such generalization.

## Benchmark 2 - Remedial action search
Once risky contingencies are identified, we need to
explore possible solutions (aka "remedial actions") to recommend suitable solutions to the grid
operator. In this benchmark, a solution consists in predefined topological change on the grid. It
is successful if the simulation run on this modified grid alleviates the previous overflow without
 generating any new problem. Those changes brings more non-linearity than line disconnections in
 benchmark1, making the distributions more complex. We here target 100-1000 simulation batches
 This depends on the number of pre-selected action candidates by the operator for a given risk
and the number of possibly risky line contingencies. This benchmark includes the prediction of a
 few more physical variables: active power flows pℓ (in MW) and voltages vk (in V) at both sides
 of each powerline. Level of compliance with additional related physical laws is expected. This
allows the operator to better assess the system state in a difficult situation with some consistency.

### Dataset specificity
It presents grid snapshots over single substation topological reconfiguration
 among a set of specified ones. It also considers some possible line contingencies that could cause
overloads. An ood Topology test set containing combination of 2 topological unitary actions is
 also attached to test for such generalization.

## Benchmark 3 - Validation of decision
Once preferred solutions have been selected by the
operator to alleviate an overflow, a last detailed simulation is run and studied more in depth by
the operator, prior to apply the remedial actions on the actual power grid. The purpose of this
operation is to ensure that no unforeseen event may make the grid collapse. In this application,
the quality of prediction is decisive, to ensure that the adopted interventions will not violate the
consistency within the network; therefore the surrogate simulator shall be nearly as good as the
numerical solver. Here the main goal is to ensure the accuracy and consistency of the solution for
1-10 simulations. All physical variables of the physical problem shall be predicted: currents aℓ,
active power flows pℓ, voltages vk, reactive power flows qℓ, angles θk

### Dataset specificity
The goal here is to provide accurate simulations that could run over any
feasible conditions in operations for final choice validation much like a physical solver. This
dataset presents grid snapshots over very different topologies around reference topologies with
combination of N-1, N-2 or maintenance line disconnections and various combination of unitary
actions. Yet we keep the combinatorial depth of unitary actions up to 4, so that a topological change
can be the combination of 4 unitary ones, which represents already large enough combinatorial
space. An ood Topology test set over a combinatorial depth 5 is also attached to test for such
generalization.