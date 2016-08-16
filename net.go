package krang

import (
	//"fmt"
	"math"
)

var n_eta, n_alpha, n_smoothingFactor float64

type layer []*neuron

type Net struct {
	layers             []layer
	errorRate          float64
	recentAverageError float64
}

/* Create a new neural network using the given topology
 * slice; a topology slice of [3, 2, 1] describes a network
 * with 3 layers, the input layer has 3 neurons, the hidden
 * layer has 2 neurons and the output layer has 1 neuron. */
func NewNet(topology []uint, eta float64, alpha float64, smoothingFactor float64) *Net {
	// Init the Network object and prepare the layers based
	// on the number of layers defined by the topology slice
	krangNet := &Net{
		layers: make([]layer, len(topology)),
	}

	n_eta = eta
	n_alpha = alpha
	n_smoothingFactor = smoothingFactor

	// Fill up each layer with Neurons based on the number
	// specified for each layer in the topology
	for tId := 0; tId < len(topology); tId++ {
		//krangNet.layers[tId] = make(layer, topology[tId]+1) // Allocate the memory
		// We need the neuron id so we can create the synapses between
		// neurons and address them
		var nId uint = 0

		// Calculate the number of synapses needed for each neuron in
		// this layer.
		var nSynapses uint
		if tId == len(topology)-1 {
			nSynapses = 0 // this is the output layer so no synapses will be created
		} else {
			nSynapses = topology[tId+1] + 1 // See how many neurons in the next layer
		}

		// Create the Neurons for this layer
		for ; nId <= topology[tId]; nId++ {
			if nSynapses == 0 && nId == topology[tId] {
				continue
			}

			krangNet.layers[tId] = append(krangNet.layers[tId], newNeuron(nId, nSynapses))
		}

		if nSynapses != 0 {
			krangNet.layers[tId][topology[tId]].value = 1.0
		}
	}

	return krangNet
}

// Feeds normalized data to the network for training
func (kNet *Net) FeedForward(input []float64) error {
	// Check that the right number of inputs are supplied
	if len(input) != len(kNet.layers[0])-1 {
		return IncompleteInputParameters{
			expected: len(kNet.layers[0]) - 1,
			given:    len(input),
		}
	}

	// Assign the inputs to the input neurons
	for i := 0; i < len(input); i++ {
		kNet.layers[0][i].value = input[i]
	}

	// Propagate forward; feed the data into the hidden layers to update
	// the outputs
	for i := 1; i < len(kNet.layers); i++ {
		prevLayer := kNet.layers[i-1]
		for n := 0; n < len(kNet.layers[i]); n++ {
			kNet.layers[i][n].feedForward(prevLayer)
		}
	}

	return nil
}

/* You even lift brah? train that mushy blob.
 *
 * This func takes the correct values expected on the output
 * and trains the net to be able to produce accurate outputs
 * when introduced to new inputs. a.k.a. training method */
func (kNet *Net) BackPropagate(targetValues []float64) {
	// Calculate the net error rate using RMS of neuron errors
	outputLayer := kNet.layers[len(kNet.layers)-1]
	kNet.errorRate = 0.0

	// Sum the square of all error rates
	for n := 0; n < len(outputLayer); n++ {
		delta := targetValues[n] - outputLayer[n].value
		kNet.errorRate += delta * delta
	}

	kNet.errorRate /= float64(len(outputLayer)) // average error squared
	kNet.errorRate = math.Sqrt(kNet.errorRate)  // RMS

	// Calculate the recent average measurement
	kNet.recentAverageError =
		(kNet.recentAverageError*n_smoothingFactor + kNet.errorRate) / (n_smoothingFactor + 1.0)

	// Calculate output layer gradients
	for n := 0; n < len(outputLayer); n++ {
		outputLayer[n].calculateOutputGradients(targetValues[n])
	}

	// Calculate hidden layer gradients
	for n := len(kNet.layers) - 2; n > 0; n-- {
		hiddenLayer := kNet.layers[n]
		nextLayer := kNet.layers[n+1]

		for i := 0; i < len(hiddenLayer); i++ {
			hiddenLayer[n].calculateHiddenGradients(nextLayer)
		}
	}

	// Update the neuron values for the whole network
	for n := len(kNet.layers) - 1; n > 0; n-- {
		tLayer := kNet.layers[n]
		prevLayer := kNet.layers[n-1]

		for i := 0; i < len(tLayer); i++ {
			tLayer[i].updateInputWeights(prevLayer)
		}
	}
}

func (kNet *Net) GetResults() (resultValues []float64) {
	outputLayer := kNet.layers[len(kNet.layers)-1]
	resultValues = make([]float64, len(outputLayer))

	for n := 0; n < len(outputLayer); n++ {
		resultValues[n] = outputLayer[n].value
	}

	return
}

func (kNet *Net) GetRecentAverageErrorRate() float64 {
	return kNet.recentAverageError
}
