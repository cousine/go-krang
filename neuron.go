package krang

import (
	"math"
	"math/rand"
)

const (
	ETA   float64 = .15
	ALPHA         = 0.5
)

type synapse struct {
	weight      float64
	deltaWeight float64
}

type neuron struct {
	synapses []synapse
	id       uint
	value    float64
	gradient float64
}

// Create a new neuron with `id` and `nSynapses`
func newNeuron(id uint, nSynapses uint) *neuron {
	node := &neuron{
		id:       id,
		synapses: make([]synapse, nSynapses),
	}

	// Create the synapses with initial random weights
	var sId uint
	for ; sId < nSynapses; sId++ {
		node.synapses[sId].weight = randomWeight()
	}

	return node
}

/* Generate a random weight for usage when creating new
 * new synapses */
func randomWeight() float64 {
	rand.Seed(0)
	return rand.Float64()
}

/* Activation function for the Neuron, this is where the
 * magic happens (not really) */
func activationFunction(x float64) float64 {
	return math.Tanh(x)
}

/* Activation function derivative */
func activationFunctionDerivative(x float64) float64 {
	return 1.0 - x*x
}

/* Neuron Feed Forward */
func (kNeuron *neuron) feedForward(prevLayer layer) {
	var sum float64

	// Sum the values of the previous layer's neurons, including
	// the bias neuron

	for n := 0; n < len(prevLayer); n++ {
		sum += prevLayer[n].value * prevLayer[n].synapses[kNeuron.id].weight
	}

	kNeuron.value = activationFunction(sum)
}

/* Sum the weight gradients */
func (kNeuron *neuron) sumDOW(nextLayer layer) (sum float64) {
	sum = 0.0

	for n := 0; n < len(nextLayer)-1; n++ {
		sum += kNeuron.synapses[n].weight * nextLayer[n].gradient
	}

	return
}

func (kNeuron *neuron) calculateOutputGradients(targetValue float64) {
	delta := targetValue - kNeuron.value
	kNeuron.gradient = delta * activationFunctionDerivative(kNeuron.value)
}

func (kNeuron *neuron) calculateHiddenGradients(nextLayer layer) {
	dow := kNeuron.sumDOW(nextLayer)
	kNeuron.gradient = dow * activationFunctionDerivative(kNeuron.value)
}

func (kNeuron *neuron) updateInputWeights(prevLayer layer) {
	for n := 0; n < len(prevLayer); n++ {
		tNeuron := prevLayer[n]
		oldDeltaWeight := tNeuron.synapses[kNeuron.id].deltaWeight

		newDeltaWeight := ETA*tNeuron.value*kNeuron.gradient + ALPHA*oldDeltaWeight

		tNeuron.synapses[kNeuron.id].deltaWeight = newDeltaWeight
		tNeuron.synapses[kNeuron.id].weight += newDeltaWeight
	}
}
