package models

import (
	"fmt"
	"errors"
	"math"
)

// Nodo del árbol
type Node struct {
	Label         string  // Nodo hoja
	FeatureIndex  int     // Índice del atributo para dividir
	FeatureValues []string
	Children      []ChildNode
}

type ChildNode struct {
	Value     string
	ChildNode *Node
}

type DecisionTreeClassifier struct {
	Tree     *Node
	MaxDepth int
	Gain     string // string para almacenar texto con info (similar al JS)
}

// Nuevo clasificador con profundidad máxima (por defecto 5)
func NewDecisionTreeClassifier(maxDepth int) *DecisionTreeClassifier {
	if maxDepth <= 0 {
		maxDepth = 5
	}
	return &DecisionTreeClassifier{
		MaxDepth: maxDepth,
		Gain:     "",
	}
}

// Fit entrena el árbol con datos X (atributos) e y (etiquetas)
func (dt *DecisionTreeClassifier) Fit(X [][]string, y []string) error {
	if len(X) == 0 || len(y) == 0 {
		return errors.New("X or y are empty")
	}
	if len(X) != len(y) {
		return errors.New("X and y have different lengths")
	}
	dt.Tree = dt.buildTree(X, y, 0)
	return nil
}

// Función recursiva para construir el árbol
func (dt *DecisionTreeClassifier) buildTree(X [][]string, y []string, depth int) *Node {
	uniqueLabels := uniqueStrings(y)

	// Caso base 1: Todas las etiquetas iguales
	if len(uniqueLabels) == 1 {
		return &Node{Label: uniqueLabels[0]}
	}

	// Caso base 2: Profundidad máxima o no hay atributos
	if depth >= dt.MaxDepth || (len(X) > 0 && len(X[0]) == 0) {
		majority := majorityLabel(y)
		return &Node{Label: majority}
	}

	bestFeature := dt.bestSplit(X, y)
	if bestFeature == -1 {
		majority := majorityLabel(y)
		return &Node{Label: majority}
	}

	bestFeatureValues := uniqueStrings(getColumn(X, bestFeature))

	node := &Node{
		FeatureIndex:  bestFeature,
		FeatureValues: bestFeatureValues,
		Children:      []ChildNode{},
	}

	for _, val := range bestFeatureValues {
		indices := []int{}
		for i, row := range X {
			if row[bestFeature] == val {
				indices = append(indices, i)
			}
		}

		XSubset := subsetRemoveColumn(X, indices, bestFeature)
		ySubset := subsetStrings(y, indices)

		child := dt.buildTree(XSubset, ySubset, depth+1)
		node.Children = append(node.Children, ChildNode{Value: val, ChildNode: child})
	}

	return node
}

// Calcula la entropía
func entropy(y []string) float64 {
	counts := map[string]int{}
	for _, label := range y {
		counts[label]++
	}
	total := float64(len(y))
	ent := 0.0
	for _, count := range counts {
		p := float64(count) / total
		ent -= p * math.Log2(p)
	}
	return ent
}

// Calcula ganancia de información para una característica
func (dt *DecisionTreeClassifier) informationGain(X [][]string, y []string, featureIndex int) float64 {
	featureValues := uniqueStrings(getColumn(X, featureIndex))
	totalEntropy := entropy(y)
	weightedEntropy := 0.0

	for _, val := range featureValues {
		indices := []int{}
		for i, row := range X {
			if row[featureIndex] == val {
				indices = append(indices, i)
			}
		}
		ySubset := subsetStrings(y, indices)
		weightedEntropy += float64(len(ySubset))/float64(len(y)) * entropy(ySubset)
	}

	return totalEntropy - weightedEntropy
}

// Busca el mejor atributo para dividir
func (dt *DecisionTreeClassifier) bestSplit(X [][]string, y []string) int {
	bestGain := math.Inf(-1)
	bestFeature := -1

	if len(X) == 0 || len(X[0]) == 0 {
		return -1
	}

	for i := 0; i < len(X[0]); i++ {
		gain := dt.informationGain(X, y, i)
		dt.Gain += "Gain of column " + itoa(i) + ": " + ftoa(gain) + "\n"
		if gain > bestGain {
			bestGain = gain
			bestFeature = i
		}
	}
	dt.Gain += "** Best feature: " + itoa(bestFeature) + "\n"

	if bestGain <= 0 {
		return -1
	}

	return bestFeature
}

// Predice etiquetas para X
func (dt *DecisionTreeClassifier) Predict(X [][]string) ([]string, error) {
	if dt.Tree == nil {
		return nil, errors.New("Model not trained")
	}

	preds := make([]string, len(X))
	for i, row := range X {
		preds[i] = dt.predictRow(row, dt.Tree)
	}
	return preds, nil
}

// Predicción recursiva para una fila
func (dt *DecisionTreeClassifier) predictRow(row []string, node *Node) string {
	if node.Label != "" {
		return node.Label
	}
	val := row[node.FeatureIndex]

	var child *Node
	for _, c := range node.Children {
		if c.Value == val {
			child = c.ChildNode
			break
		}
	}

	if child == nil {
		// Si no se encuentra el valor, retorna la etiqueta mayoritaria del sub-árbol
		return majorityLabel(dt.getAllLabels(node))
	}

	filteredRow := removeIndex(row, node.FeatureIndex)
	return dt.predictRow(filteredRow, child)
}

// Extrae todas las etiquetas de un sub-árbol
func (dt *DecisionTreeClassifier) getAllLabels(node *Node) []string {
	if node.Label != "" {
		return []string{node.Label}
	}
	labels := []string{}
	for _, c := range node.Children {
		labels = append(labels, dt.getAllLabels(c.ChildNode)...)
	}
	return labels
}

// Funciones auxiliares

func uniqueStrings(arr []string) []string {
	m := map[string]struct{}{}
	for _, v := range arr {
		m[v] = struct{}{}
	}
	res := []string{}
	for k := range m {
		res = append(res, k)
	}
	return res
}

func majorityLabel(y []string) string {
	counts := map[string]int{}
	for _, label := range y {
		counts[label]++
	}
	var maxCount int
	var majority string
	for label, count := range counts {
		if count > maxCount {
			maxCount = count
			majority = label
		}
	}
	return majority
}

func getColumn(X [][]string, col int) []string {
	res := make([]string, len(X))
	for i := range X {
		res[i] = X[i][col]
	}
	return res
}

func subsetRemoveColumn(X [][]string, indices []int, col int) [][]string {
	res := make([][]string, len(indices))
	for i, idx := range indices {
		row := []string{}
		for j := 0; j < len(X[idx]); j++ {
			if j != col {
				row = append(row, X[idx][j])
			}
		}
		res[i] = row
	}
	return res
}

func subsetStrings(arr []string, indices []int) []string {
	res := make([]string, len(indices))
	for i, idx := range indices {
		res[i] = arr[idx]
	}
	return res
}

func removeIndex(arr []string, idx int) []string {
	res := append([]string{}, arr[:idx]...)
	return append(res, arr[idx+1:]...)
}

// Conversión de int a string (sustituto simple de strconv.Itoa)
func itoa(i int) string {
	return fmt.Sprintf("%d", i)
}

// Conversión de float64 a string (sustituto simple de strconv.FormatFloat)
func ftoa(f float64) string {
	return fmt.Sprintf("%f", f)
}
