package nn

import (
	"errors"
	"fmt"
	"math"
)

const (
	NInputs = 4 + 1
	NHiddens = 12
	NOutputs = 3
)

var (
	InputActivations [NInputs]float64
	HiddenActivations [NHiddens]float64
	OutputActivations [NOutputs] float64
	InputWeights [NInputs][NHiddens] float64
	OutputWeights [NHiddens][NOutputs] float64
	InputChanges [NInputs][NHiddens]float64
	OutputChanges [NHiddens][NOutputs]float64
	Contexts [][NHiddens]float64
	Regression bool
)

func Init() {
	for i := 0; i < NInputs; i++ {
		for j := 0; j < NHiddens; j++ {
			InputWeights[i][j] = random(-1, 1)
		}
	}

	for i := 0; i <NHiddens; i++ {
		for j := 0; j < NOutputs; j++ {
			OutputWeights[i][j] = random(-1, 1)
		}
	}
}

func SetContexts(nContexts int, initValues [][NHiddens]float64) {
	if initValues == nil {
		initValues = make([][NHiddens]float64, nContexts)

		for i := 0; i < nContexts; i++ {
			array := [NHiddens]float64{}
			for i := 0; i < len(array); i ++ {
				array[i] = 0.5
			}
			initValues[i] = array
		}
	}

	Contexts = initValues
}

func Update(inputs [4]float64) ([NOutputs]float64, error) {
	if len(inputs) != NInputs-1 {
		return [NOutputs]float64{}, errors.New("Error: wrong number of inputs")
	}

	for i := 0; i < NInputs-1; i++ {
		InputActivations[i] = inputs[i]
	}

	for i := 0; i < NHiddens-1; i++ {
		var sum float64

		for j := 0; j < NInputs; j++ {
			sum += InputActivations[j] * InputWeights[j][i]
		}

		for k := 0; k < len(Contexts); k++ {
			for j := 0; j < NHiddens-1; j++ {
				sum += Contexts[k][j]
			}
		}

		HiddenActivations[i] = sigmoid(sum)
	}

	if len(Contexts) > 0 {
		for i := len(Contexts) - 1; i > 0; i-- {
			Contexts[i] = Contexts[i-1]
		}
		Contexts[0] = HiddenActivations
	}

	for i := 0; i < NOutputs; i++ {
		var sum float64
		for j := 0; j < NHiddens; j++ {
			sum += HiddenActivations[j] * OutputWeights[j][i]
		}

		OutputActivations[i] = sigmoid(sum)
	}

	return OutputActivations, nil
}

func BackPropagate(targets [4]float64, lRate, mFactor float64) (float64, error) {
	if len(targets) != NOutputs {
		return 0, errors.New("Error: wrong number of target values")
	}

	var outputDeltas [NOutputs]float64
	for i := 0; i < NOutputs; i++ {
		outputDeltas[i] = dsigmoid(OutputActivations[i]) * (targets[i] - OutputActivations[i])
	}

	var hiddenDeltas [NHiddens]float64
	for i := 0; i < NHiddens; i++ {
		var e float64
		for j := 0; j < NOutputs; j++ {
			e += outputDeltas[j] * OutputWeights[i][j]
		}
		hiddenDeltas[i] = dsigmoid(HiddenActivations[i]) * e
	}

	for i := 0; i < NHiddens; i++ {
		for j := 0; j < NOutputs; j++ {
			change := outputDeltas[j] * HiddenActivations[i]
			OutputWeights[i][j] = OutputWeights[i][j] + lRate*change + mFactor*OutputChanges[i][j]
			OutputChanges[i][j] = change
		}
	}

	for i := 0; i < NInputs; i++ {
		for j := 0; j < NHiddens; j++ {
			change := hiddenDeltas[j] * InputActivations[i]
			InputWeights[i][j] = InputWeights[i][j] + lRate*change + mFactor*InputChanges[i][j]
			InputChanges[i][j] = change
		}
	}

	var e float64

	for i := 0; i < len(targets); i++ {
		e += 0.5 * math.Pow(targets[i]-OutputActivations[i], 2)
	}

	return e, nil
}

func Train(patterns [][2][4]float64, iterations int, lRate, mFactor float64, debug bool) ([]float64, error) {
	errors := make([]float64, iterations)

	for i := 0; i < iterations; i++ {
		var e float64
		for _, p := range patterns {
			Update(p[0])

			tmp, err := BackPropagate(p[1], lRate, mFactor)
			if err != nil {
				return nil, err
			}
			e += tmp
		}

		errors[i] = e
	}
	return errors, nil
}

func Test(patterns [][2][4]float64) error {
	for _, p := range patterns {
		result, err := Update(p[0])
		if err != nil {
			return err
		}
		fmt.Println(p[0], "->", result, " : ", p[1])
	}
	return nil
}
