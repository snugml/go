package main

import (
	"fmt"
	"log"
	"github.com/snugml/go"
)

func main() {
	// Datos de entrenamiento
	X := [][]float64{
		{1, 2},
		{1, 3},
		{2, 2},
		{3, 3},
		{3, 4},
	}
	y := []interface{}{0, 0, 1, 1, 0}

	// Crear el modelo
	model := models.GaussianNB{}

	// Ajustar modelo a los datos
	if err := model.Fit(X, y); err != nil {
		log.Fatal(err)
	}

	// Datos para predicción
	XTest := [][]float64{
		{2, 3},
		{3, 4},
	}

	// Realizar predicciones
	predictions := model.Predict(XTest)

	// Mostrar resultados
	fmt.Println("Predicciones:", predictions)
}
