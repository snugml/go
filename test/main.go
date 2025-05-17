package main

import (
	"fmt"
	"log"
	"github.com/snugml/go"  // Importa el paquete ml donde está la lógica de LinearRegression
)

// Función para redondear los valores a 2 decimales
func roundToTwoDecimals(value float64) float64 {
	return float64(int(value*100)) / 100.0
}

func main() {
	// Datos de ejemplo (X e Y)
	X := []float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	y := []float64{1, 4, 1, 5, 3, 7, 2, 7, 4, 9}

	// Instancia de LinearRegression
	model := ml.LinearRegression{}

	// Entrenamiento del modelo
	model.Fit(X, y)

	// Predicciones del modelo
	yPredict := model.Predict(X)

	// Calcular el MSE y R^2
	mse := model.MSE(y, yPredict)
	r2 := model.R2(y, yPredict)

	// Redondear las predicciones a 2 decimales
	var yPredRounded []float64
	for _, val := range yPredict {
		yPredRounded = append(yPredRounded, roundToTwoDecimals(val))
	}

	// Imprimir los resultados
	fmt.Println("X:", X)
	fmt.Println("y:", y)
	fmt.Println("yPredict:", yPredRounded)
	fmt.Printf("MSE: %.4f\n", mse)
	fmt.Printf("R2: %.4f\n", r2)
}
