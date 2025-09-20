package models

import (
	"errors"
	"log"
	"math"
)

// GaussianNB representa un clasificador Naive Bayes Gaussiano
type GaussianNB struct {
	classProbabilities map[interface{}]float64          // Probabilidad de cada clase
	featureStats       map[interface{}][]FeatureStat    // Media y desviación estándar de cada característica por clase
	classes            []interface{}                    // Lista de clases
}

// FeatureStat almacena la media y desviación estándar de una característica
type FeatureStat struct {
	Mean   float64
	StdDev float64
}

// checkDataLength verifica que los datos tengan el mismo tamaño
func (gnb *GaussianNB) checkDataLength(X [][]float64, y []interface{}) error {
	if len(X) != len(y) {
		return errors.New("los parámetros X e y no tienen la misma longitud")
	}
	if len(X) == 0 || len(y) == 0 {
		return errors.New("los parámetros X o y están vacíos")
	}
	return nil
}

// Fit ajusta el modelo Naive Bayes Gaussiano a los datos
func (gnb *GaussianNB) Fit(X [][]float64, y []interface{}) error {
	if err := gnb.checkDataLength(X, y); err != nil {
		return err
	}

	classCounts := make(map[interface{}]float64)
	featureSums := make(map[interface{}][]float64)
	featureSquaredSums := make(map[interface{}][]float64)

	// Inicializamos las estructuras y sumamos
	for i := 0; i < len(y); i++ {
		label := y[i]
		classCounts[label]++
		if _, ok := featureSums[label]; !ok {
			featureSums[label] = make([]float64, len(X[0]))
			featureSquaredSums[label] = make([]float64, len(X[0]))
		}

		for j := 0; j < len(X[i]); j++ {
			featureSums[label][j] += X[i][j]
			featureSquaredSums[label][j] += X[i][j] * X[i][j]
		}
	}

	totalCount := float64(len(y))

	// Guardamos las clases
	gnb.classes = make([]interface{}, 0, len(classCounts))
	for label := range classCounts {
		gnb.classes = append(gnb.classes, label)
	}

	// Calculamos la probabilidad de cada clase
	gnb.classProbabilities = make(map[interface{}]float64)
	for _, label := range gnb.classes {
		gnb.classProbabilities[label] = classCounts[label] / totalCount
	}

	// Calculamos media y desviación estándar
	gnb.featureStats = make(map[interface{}][]FeatureStat)
	for _, label := range gnb.classes {
		nFeatures := len(X[0])
		gnb.featureStats[label] = make([]FeatureStat, nFeatures)
		for j := 0; j < nFeatures; j++ {
			mean := featureSums[label][j] / classCounts[label]
			variance := (featureSquaredSums[label][j] / classCounts[label]) - mean*mean
			stdDev := math.Sqrt(variance)
			if stdDev == 0 {
				stdDev = 1e-10 // Evita división por cero
			}
			gnb.featureStats[label][j] = FeatureStat{
				Mean:   mean,
				StdDev: stdDev,
			}
		}
	}

	return nil
}

// gaussian calcula la probabilidad de un valor bajo distribución normal
func (gnb *GaussianNB) gaussian(x, mean, stdDev float64) float64 {
	exponent := math.Exp(-0.5 * math.Pow((x-mean)/stdDev, 2))
	return (1 / (stdDev * math.Sqrt(2*math.Pi))) * exponent
}

// Predict realiza predicciones sobre nuevos datos
func (gnb *GaussianNB) Predict(X [][]float64) []interface{} {
	yPred := make([]interface{}, len(X))

	for i, sample := range X {
		bestLabel := gnb.classes[0]
		bestScore := math.Inf(-1)

		for _, label := range gnb.classes {
			score := math.Log(gnb.classProbabilities[label])
			for j, fs := range gnb.featureStats[label] {
				score += math.Log(gnb.gaussian(sample[j], fs.Mean, fs.StdDev))
			}

			if score > bestScore {
				bestScore = score
				bestLabel = label
			}
		}

		yPred[i] = bestLabel
	}

	return yPred
}
