package models

import (
	"errors"
	"fmt"
	"gonum.org/v1/gonum/mat"
	"math"
)

type PolynomialRegression struct {
	Degree      int
	Coefficients *mat.VecDense
	IsFit        bool
}

// Constructor para la regresión polinomial
func NewPolynomialRegression(degree int) *PolynomialRegression {
	return &PolynomialRegression{
		Degree:      degree,
		IsFit:       false,
		Coefficients: mat.NewVecDense(degree+1, nil), // Inicializamos el vector de coeficientes
	}
}

// Método para verificar la validez de los datos de entrada
func (pr *PolynomialRegression) checkDataLength(xTrain, yTrain []float64) error {
	if len(xTrain) != len(yTrain) {
		return errors.New("The parameters for training do not have the same length!")
	}
	if len(xTrain) == 0 || len(yTrain) == 0 {
		return errors.New("The xTrain or yTrain parameters are empty!")
	}
	return nil
}

// Función para construir la matriz de diseño
func (pr *PolynomialRegression) buildDesignMatrix(xTrain []float64) *mat.Dense {
	n := len(xTrain)
	matrix := mat.NewDense(n, pr.Degree+1, nil)
	for i := 0; i < n; i++ {
		for j := 0; j <= pr.Degree; j++ {
			matrix.Set(i, j, math.Pow(xTrain[i], float64(j))) // Potencia de x hasta el grado
		}
	}
	return matrix
}

// Método para ajustar el modelo de regresión polinomial
func (pr *PolynomialRegression) Fit(xTrain, yTrain []float64) error {
	// Verificación de datos
	if err := pr.checkDataLength(xTrain, yTrain); err != nil {
		return err
	}

	X := pr.buildDesignMatrix(xTrain) // Matriz de diseño
	Y := mat.NewVecDense(len(yTrain), yTrain)

	// Resolución de los coeficientes utilizando el método de mínimos cuadrados
	var XT mat.Dense
	XT.CloneFrom(X.T()) // Transpuesta de la matriz X

	var XTX mat.Dense
	XTX.Mul(&XT, X) // Producto X^T * X

	var XTXInv mat.Dense
	if err := XTXInv.Inverse(&XTX); err != nil {
		return fmt.Errorf("could not invert X^T * X: %v", err)
	}

	var XTY mat.VecDense
	XTY.MulVec(&XT, Y) // Producto X^T * Y

	var coeffs mat.VecDense
	coeffs.MulVec(&XTXInv, &XTY) // Coeficientes del polinomio

	pr.Coefficients = &coeffs
	pr.IsFit = true

	return nil
}

// Método para hacer predicciones con el modelo ajustado
func (pr *PolynomialRegression) Predict(xTest []float64) []float64 {
	yPredict := make([]float64, len(xTest))
	if pr.IsFit {
		for i, x := range xTest {
			y := 0.0
			for j := 0; j <= pr.Degree; j++ {
				y += pr.Coefficients.At(j, 0) * math.Pow(x, float64(j)) // Suma de los términos del polinomio
			}
			yPredict[i] = y
		}
	}
	return yPredict
}

// Error cuadrático medio
func (pr *PolynomialRegression) MSE(yTrain, yPredict []float64) float64 {
	mse := 0.0
	for i := 0; i < len(yTrain); i++ {
		mse += math.Pow(yTrain[i] - yPredict[i], 2)
	}
	return mse / float64(len(yTrain))
}

// Coeficiente de determinación R^2
func (pr *PolynomialRegression) R2(yTrain, yPredict []float64) float64 {
	var avg float64
	for _, y := range yTrain {
		avg += y
	}
	avg /= float64(len(yTrain))

	var numerator, denominator float64
	for i := 0; i < len(yPredict); i++ {
		numerator += math.Pow(yPredict[i] - avg, 2)
	}
	for i := 0; i < len(yTrain); i++ {
		denominator += math.Pow(yTrain[i] - avg, 2)
	}
	return numerator / denominator
}
