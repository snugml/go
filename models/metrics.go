package models

// AccuracyScore calcula la precisión comparando etiquetas reales y predichas.
// Ambas slices deben tener la misma longitud.
func AccuracyScore(y, yPredict []string) float64 {
	if len(y) == 0 || len(y) != len(yPredict) {
		return 0.0
	}

	correct := 0
	for i := range y {
		if y[i] == yPredict[i] {
			correct++
		}
	}

	return float64(correct) / float64(len(y))
}
