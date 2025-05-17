package models

import (
	"errors"
	"fmt"
	"math"
)

type PolynomialRegression struct {
	degree      int
	isFit       bool
	coefficients []float64
}

// checkDataLength verifica que los datos tengan el mismo tamaño
func (pr *PolynomialRegression) checkDataLength(xTrain, yTrain []float64) error {
	if len(xTrain) != len(yTrain) {
		return errors.New("The parameters for training do not have the same length!")
	}
	if len(xTrain) == 0 || len(yTrain) == 0 {
		return errors.New("The xTrain or yTrain parameters are empty!")
	}
	return nil
}

// buildDesignMatrix crea la matriz de diseño para el ajuste polinomial
func (pr *PolynomialRegression) buildDesignMatrix(xTrain []float64) [][]float64 {
	matrix := make([][]float64, len(xTrain))
	for i := 0; i < len(xTrain); i++ {
		row := make([]float64, pr.degree+1)
		for j := 0; j <= pr.degree; j++ {
			row[j] = math.Pow(xTrain[i], float64(j)) // Potencias de x
		}
		matrix[i] = row
	}
	return matrix
}

// fit ajusta el modelo de regresión polinomial
func (pr *PolynomialRegression) Fit(xTrain, yTrain []float64) error {
	if err := pr.checkDataLength(xTrain, yTrain); err != nil {
		return err
	}

	X := pr.buildDesignMatrix(xTrain)
	Y := yTrain

	// Resolución de coeficientes utilizando el método de mínimos cuadrados
	// A continuación necesitarás utilizar alguna librería de álgebra lineal como `gonum` para resolver X^T * X
	// Por simplicidad, este ejemplo no implementa la solución de matrices.

	// Este es un ejemplo de cómo podrías definir la estructura, ahora es necesario implementarlo correctamente
	// para la solución de la matriz.

	// pr.coefficients = ... (resolver el sistema de ecuaciones)
	pr.isFit = true
	return nil
}

// predict realiza predicciones para xTest con el modelo ajustado
func (pr *PolynomialRegression) Predict(xTest []float64) []float64 {
	var yPredict []float64
	if pr.isFit {
		for _, x := range xTest {
			y := 0.0
			for j := 0; j <= pr.degree; j++ {
				y += pr.coefficients[j] * math.Pow(x, float64(j))
			}
			yPredict = append(yPredict, y)
		}
	}
	return yPredict
}

// mse calcula el error cuadrático medio (MSE)
func (pr *PolynomialRegression) MSE(yTrain, yPredict []float64) float64 {
	var mse float64
	for i := 0; i < len(yTrain); i++ {
		mse += (yTrain[i] - yPredict[i]) * (yTrain[i] - yPredict[i])
	}
	return mse / float64(len(yTrain))
}

// r2 calcula el coeficiente de determinación R^2
func (pr *PolynomialRegression) R2(yTrain, yPredict []float64) float64 {
	var avg, numerator, denominator float64
	for _, y := range yTrain {
		avg += y
	}
	avg /= float64(len(yTrain))

	for i := 0; i < len(yPredict); i++ {
		numerator += (yPredict[i] - avg) * (yPredict[i] - avg)
	}
	for _, y := range yTrain {
		denominator += (y - avg) * (y - avg)
	}

	return numerator / denominator
}
