package models

import (
	"errors"
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
	if len(xTrain) != len(yTrain) {
		return errors.New("longitudes de xTrain e yTrain no coinciden")
	}

	n := len(xTrain)

	// Construir la matriz de diseño X
	X := mat.NewDense(n, pr.Degree+1, nil)
	for i := 0; i < n; i++ {
		for j := 0; j <= pr.Degree; j++ {
			X.Set(i, j, pow(xTrain[i], j))
		}
	}

	// Vector de salida Y
	Y := mat.NewVecDense(n, yTrain)

	// Cálculo de los coeficientes: (X^T * X)^(-1) * X^T * Y
	var XT mat.Dense
	XT.CloneFrom(X.T())

	var XTX mat.Dense
	XTX.Mul(&XT, X)

	var XTXInv mat.Dense
	err := XTXInv.Inverse(&XTX)
	if err != nil {
		return errors.New("no se pudo invertir la matriz X^T * X: " + err.Error())
	}

	var XTY mat.VecDense
	XTY.MulVec(&XT, Y)

	var coeffs mat.VecDense
	coeffs.MulVec(&XTXInv, &XTY)

	// Guardar coeficientes y marcar modelo como entrenado
	pr.Coefficients = &coeffs
	pr.IsFit = true

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

func pow(base float64, exp int) float64 {
	result := 1.0
	for i := 0; i < exp; i++ {
		result *= base
	}
	return result
}
