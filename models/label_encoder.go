package models

import (
	"errors"
	"fmt"
)

type LabelEncoder struct {
	classes      []string
	classToIndex map[string]int
	indexToClass map[int]string
}

func NewLabelEncoder() *LabelEncoder {
	return &LabelEncoder{
		classes:      []string{},
		classToIndex: map[string]int{},
		indexToClass: map[int]string{},
	}
}

// Fit ajusta el encoder con las clases únicas de y
func (le *LabelEncoder) Fit(y []string) {
	le.classes = uniqueStrings(y)
	le.classToIndex = map[string]int{}
	le.indexToClass = map[int]string{}

	for idx, cls := range le.classes {
		le.classToIndex[cls] = idx
		le.indexToClass[idx] = cls
	}
}

// Transform convierte etiquetas a índices numéricos
func (le *LabelEncoder) Transform(y []string) ([]int, error) {
	if len(le.classes) == 0 {
		return nil, errors.New("fit() debe ser llamado antes de transform")
	}
	result := make([]int, len(y))
	for i, label := range y {
		idx, ok := le.classToIndex[label]
		if !ok {
			return nil, fmt.Errorf("etiqueta \"%s\" no encontrada en las clases entrenadas", label)
		}
		result[i] = idx
	}
	return result, nil
}

// InverseTransform convierte índices numéricos a etiquetas originales
func (le *LabelEncoder) InverseTransform(indices []int) ([]string, error) {
	result := make([]string, len(indices))
	for i, idx := range indices {
		label, ok := le.indexToClass[idx]
		if !ok {
			return nil, fmt.Errorf("índice \"%d\" no encontrado en las clases entrenadas", idx)
		}
		result[i] = label
	}
	return result, nil
}

// FitTransform combina Fit y Transform
func (le *LabelEncoder) FitTransform(y []string) ([]int, error) {
	le.Fit(y)
	return le.Transform(y)
}

// Función auxiliar para obtener elementos únicos de []string
func uniqueStrings(arr []string) []string {
	seen := map[string]struct{}{}
	var result []string
	for _, s := range arr {
		if _, ok := seen[s]; !ok {
			seen[s] = struct{}{}
			result = append(result, s)
		}
	}
	return result
}
