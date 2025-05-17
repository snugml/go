package models

import (
	"errors"
	"fmt"
)

type LinearRegression struct {
	isFit bool
	m     float64
	b     float64
}

// checkDataLength verifica que los datos tengan el mismo tamaño
func (lr *LinearRegression) checkDataLength(xTrain, yTrain []float64) error {
	if len(xTrain) != len(yTrain) {
		return errors.New("The parameters for training do not have the same length!")
	}
	if len(xTrain) == 0 || len(yTrain) == 0 {
		return errors.New("The xTrain or yTrain parameters are empty!")
	}
	return nil
}

// fit ajusta el modelo de regresión lineal a los datos de entrada
func (lr *LinearRegression) Fit(xTrain, yTrain []float64) error {
	if err := lr.checkDataLength(xTrain, yTrain); err != nil {
		return err
	}

	var sumX, sumY, sumXY, sumXX float64
	for i := 0; i < len(xTrain); i++ {
		sumX += xTrain[i]
		sumY += yTrain[i]
		sumXY += xTrain[i] * yTrain[i]
		sumXX += xTrain[i] * xTrain[i]
	}

	n := float64(len(xTrain))
	lr.m = (n*sumXY - sumX*sumY) / (n*sumXX - sumX*sumX)
	lr.b = (sumY*sumXX - sumX*sumXY) / (n*sumXX - sumX*sumX)

	lr.isFit = true
	return nil
}

// predict realiza predicciones sobre nuevos datos xTest
func (lr *LinearRegression) Predict(xTest []float64) []float64 {
	var yPredict []float64
	if lr.isFit {
		for _, x := range xTest {
			yPredict = append(yPredict, lr.m*x+lr.b)
		}
	}
	return yPredict
}

// mse calcula el error cuadrático medio (MSE)
func (lr *LinearRegression) MSE(yTrain, yPredict []float64) float64 {
	var mse float64
	for i := 0; i < len(yTrain); i++ {
		mse += (yTrain[i] - yPredict[i]) * (yTrain[i] - yPredict[i])
	}
	return mse / float64(len(yTrain))
}

// r2 calcula el coeficiente de determinación R^2
func (lr *LinearRegression) R2(yTrain, yPredict []float64) float64 {
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
