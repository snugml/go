package models

import (
	"math"
	"math/rand"
)

// Sigmoid function
func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

// Derivative of sigmoid, assuming input is already sigmoid(x)
func sigmoidDerivative(y float64) float64 {
	return y * (1 - y)
}

// MLPClassifier defines a multi-layer perceptron
type MLPClassifier struct {
	InputNodes  int
	HiddenNodes int
	OutputNodes int
	LearningRate float64

	WeightsIH [][]float64
	WeightsHO [][]float64
	BiasH     [][]float64
	BiasO     [][]float64
}

// NewMLPClassifier constructor
func NewMLPClassifier(inputNodes, hiddenNodes, outputNodes int, learningRate float64) *MLPClassifier {
	return &MLPClassifier{
		InputNodes:  inputNodes,
		HiddenNodes: hiddenNodes,
		OutputNodes: outputNodes,
		LearningRate: learningRate,
		WeightsIH:   randomMatrix(hiddenNodes, inputNodes),
		WeightsHO:   randomMatrix(outputNodes, hiddenNodes),
		BiasH:       randomMatrix(hiddenNodes, 1),
		BiasO:       randomMatrix(outputNodes, 1),
	}
}

// ----------- Utility matrix operations -----------

func randomMatrix(rows, cols int) [][]float64 {
	mat := make([][]float64, rows)
	for i := range mat {
		mat[i] = make([]float64, cols)
		for j := range mat[i] {
			mat[i][j] = rand.Float64()*2 - 1
		}
	}
	return mat
}

func dot(a, b [][]float64) [][]float64 {
	result := make([][]float64, len(a))
	for i := range result {
		result[i] = make([]float64, len(b[0]))
		for j := range b[0] {
			for k := range b {
				result[i][j] += a[i][k] * b[k][j]
			}
		}
	}
	return result
}

func add(a, b [][]float64) [][]float64 {
	result := make([][]float64, len(a))
	for i := range a {
		result[i] = make([]float64, len(a[i]))
		for j := range a[i] {
			result[i][j] = a[i][j] + b[i][j]
		}
	}
	return result
}

func subtract(a, b [][]float64) [][]float64 {
	result := make([][]float64, len(a))
	for i := range a {
		result[i] = make([]float64, len(a[i]))
		for j := range a[i] {
			result[i][j] = a[i][j] - b[i][j]
		}
	}
	return result
}

func multiply(a, b [][]float64) [][]float64 {
	result := make([][]float64, len(a))
	for i := range a {
		result[i] = make([]float64, len(a[i]))
		for j := range a[i] {
			result[i][j] = a[i][j] * b[i][j]
		}
	}
	return result
}

func scalarMultiply(m [][]float64, scalar float64) [][]float64 {
	result := make([][]float64, len(m))
	for i := range m {
		result[i] = make([]float64, len(m[i]))
		for j := range m[i] {
			result[i][j] = m[i][j] * scalar
		}
	}
	return result
}

func transpose(m [][]float64) [][]float64 {
	result := make([][]float64, len(m[0]))
	for i := range result {
		result[i] = make([]float64, len(m))
		for j := range m {
			result[i][j] = m[j][i]
		}
	}
	return result
}

func mapMatrix(m [][]float64, f func(float64) float64) [][]float64 {
	result := make([][]float64, len(m))
	for i := range m {
		result[i] = make([]float64, len(m[i]))
		for j := range m[i] {
			result[i][j] = f(m[i][j])
		}
	}
	return result
}

// ----------- Predict and Fit -----------

func (mlp *MLPClassifier) Predict(input []float64) []float64 {
	inputs := toColumnMatrix(input)

	hidden := dot(mlp.WeightsIH, inputs)
	hidden = add(hidden, mlp.BiasH)
	hidden = mapMatrix(hidden, sigmoid)

	outputs := dot(mlp.WeightsHO, hidden)
	outputs = add(outputs, mlp.BiasO)
	outputs = mapMatrix(outputs, sigmoid)

	return flatten(outputs)
}

func (mlp *MLPClassifier) Fit(X [][]float64, Y [][]float64, epochs int) {
	for e := 0; e < epochs; e++ {
		for i := 0; i < len(X); i++ {
			mlp.fitSingle(X[i], Y[i])
		}
	}
}

func (mlp *MLPClassifier) fitSingle(input, target []float64) {
	inputs := toColumnMatrix(input)
	targets := toColumnMatrix(target)

	// FORWARD
	hidden := dot(mlp.WeightsIH, inputs)
	hidden = add(hidden, mlp.BiasH)
	hidden = mapMatrix(hidden, sigmoid)

	outputs := dot(mlp.WeightsHO, hidden)
	outputs = add(outputs, mlp.BiasO)
	outputs = mapMatrix(outputs, sigmoid)

	// BACKPROPAGATION
	outputErrors := subtract(targets, outputs)
	gradients := mapMatrix(outputs, sigmoidDerivative)
	gradients = multiply(gradients, outputErrors)
	gradients = scalarMultiply(gradients, mlp.LearningRate)

	hiddenT := transpose(hidden)
	weightsHODeltas := dot(gradients, hiddenT)

	mlp.WeightsHO = add(mlp.WeightsHO, weightsHODeltas)
	mlp.BiasO = add(mlp.BiasO, gradients)

	whoT := transpose(mlp.WeightsHO)
	hiddenErrors := dot(whoT, outputErrors)

	hiddenGradient := mapMatrix(hidden, sigmoidDerivative)
	hiddenGradient = multiply(hiddenGradient, hiddenErrors)
	hiddenGradient = scalarMultiply(hiddenGradient, mlp.LearningRate)

	inputsT := transpose(inputs)
	weightsIHDeltas := dot(hiddenGradient, inputsT)

	mlp.WeightsIH = add(mlp.WeightsIH, weightsIHDeltas)
	mlp.BiasH = add(mlp.BiasH, hiddenGradient)
}

// Helper functions
func toColumnMatrix(arr []float64) [][]float64 {
	result := make([][]float64, len(arr))
	for i := range arr {
		result[i] = []float64{arr[i]}
	}
	return result
}

func flatten(matrix [][]float64) []float64 {
	result := make([]float64, len(matrix))
	for i := range matrix {
		result[i] = matrix[i][0]
	}
	return result
}
