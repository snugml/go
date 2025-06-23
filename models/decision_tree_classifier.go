package models

import (
	"errors"
	"fmt"
	"math"
	"strings"
)

// Nodo del árbol
type Node struct {
	Label         int        // Nodo hoja: etiqueta codificada como int; -1 si no hoja
	FeatureIndex  int        // Índice del atributo para dividir; -1 si hoja
	FeatureValues []int      // Valores únicos del atributo para ramificar
	Children      []ChildNode
}

type ChildNode struct {
	Value     int
	ChildNode *Node
}

type DecisionTreeClassifier struct {
	Tree     *Node
	MaxDepth int
	Gain     string // string para almacenar texto con info
}

// Fit entrena el árbol con datos X (atributos) e y (etiquetas)
func (dt *DecisionTreeClassifier) Fit(X [][]int, y []int) error {
	if len(X) == 0 || len(y) == 0 {
		return errors.New("X or y are empty")
	}
	if len(X) != len(y) {
		return errors.New("X and y have different lengths")
	}
	// Inicializar Label y FeatureIndex en nodos hoja con -1
	dt.Tree = dt.buildTree(X, y, 0)
	return nil
}

// Función recursiva para construir el árbol
func (dt *DecisionTreeClassifier) buildTree(X [][]int, y []int, depth int) *Node {
	uniqueLabels := uniqueInts(y)

	// Caso base 1: Todas las etiquetas iguales
	if len(uniqueLabels) == 1 {
		return &Node{Label: uniqueLabels[0], FeatureIndex: -1}
	}

	// Caso base 2: Profundidad máxima o no hay atributos
	if depth >= dt.MaxDepth || (len(X) > 0 && len(X[0]) == 0) {
		majority := majorityLabel(y)
		return &Node{Label: majority, FeatureIndex: -1}
	}

	bestFeature := dt.bestSplit(X, y)
	if bestFeature == -1 {
		majority := majorityLabel(y)
		return &Node{Label: majority, FeatureIndex: -1}
	}

	bestFeatureValues := uniqueInts(getColumn(X, bestFeature))

	node := &Node{
		FeatureIndex:  bestFeature,
		FeatureValues: bestFeatureValues,
		Label:         -1,
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
		ySubset := subsetInts(y, indices)

		child := dt.buildTree(XSubset, ySubset, depth+1)
		node.Children = append(node.Children, ChildNode{Value: val, ChildNode: child})
	}

	return node
}

// Calcula la entropía
func entropy(y []int) float64 {
	counts := map[int]int{}
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
func (dt *DecisionTreeClassifier) informationGain(X [][]int, y []int, featureIndex int) float64 {
	featureValues := uniqueInts(getColumn(X, featureIndex))
	totalEntropy := entropy(y)
	weightedEntropy := 0.0

	for _, val := range featureValues {
		indices := []int{}
		for i, row := range X {
			if row[featureIndex] == val {
				indices = append(indices, i)
			}
		}
		ySubset := subsetInts(y, indices)
		weightedEntropy += float64(len(ySubset)) / float64(len(y)) * entropy(ySubset)
	}

	return totalEntropy - weightedEntropy
}

// Busca el mejor atributo para dividir
func (dt *DecisionTreeClassifier) bestSplit(X [][]int, y []int) int {
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
func (dt *DecisionTreeClassifier) Predict(X [][]int) ([]int, error) {
	if dt.Tree == nil {
		return nil, errors.New("Model not trained")
	}

	preds := make([]int, len(X))
	for i, row := range X {
		preds[i] = dt.predictRow(row, dt.Tree)
	}
	return preds, nil
}

// Predicción recursiva para una fila
func (dt *DecisionTreeClassifier) predictRow(row []int, node *Node) int {
	if node.Label != -1 {
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

	filteredRow := removeIndexInt(row, node.FeatureIndex)
	return dt.predictRow(filteredRow, child)
}

// Extrae todas las etiquetas de un sub-árbol
func (dt *DecisionTreeClassifier) getAllLabels(node *Node) []int {
	if node.Label != -1 {
		return []int{node.Label}
	}
	labels := []int{}
	for _, c := range node.Children {
		labels = append(labels, dt.getAllLabels(c.ChildNode)...)
	}
	return labels
}

// Imprime el árbol en texto legible
func (dt *DecisionTreeClassifier) PrintTree() string {
	if dt.Tree == nil {
		return "No tree trained yet"
	}
	return dt.printTree(dt.Tree, 0)
}

func (dt *DecisionTreeClassifier) printTree(node *Node, depth int) string {
	if node == nil {
		return ""
	}
	indent := strings.Repeat("  ", depth)
	if node.Label != -1 {
		return fmt.Sprintf("%sLeaf: %d\n", indent, node.Label)
	}
	res := fmt.Sprintf("%sFeature %d:\n", indent, node.FeatureIndex)
	for _, child := range node.Children {
		res += fmt.Sprintf("%s- Value %d:\n%s", indent+"  ", child.Value, dt.printTree(child.ChildNode, depth+2))
	}
	return res
}

// Funciones auxiliares

func uniqueInts(arr []int) []int {
	m := map[int]struct{}{}
	for _, v := range arr {
		m[v] = struct{}{}
	}
	res := []int{}
	for k := range m {
		res = append(res, k)
	}
	return res
}

func majorityLabel(y []int) int {
	counts := map[int]int{}
	for _, label := range y {
		counts[label]++
	}
	var maxCount int
	var majority int = -1
	for label, count := range counts {
		if count > maxCount {
			maxCount = count
			majority = label
		}
	}
	return majority
}

func getColumn(X [][]int, col int) []int {
	res := make([]int, len(X))
	for i := range X {
		res[i] = X[i][col]
	}
	return res
}

func subsetRemoveColumn(X [][]int, indices []int, col int) [][]int {
	res := make([][]int, len(indices))
	for i, idx := range indices {
		row := []int{}
		for j := 0; j < len(X[idx]); j++ {
			if j != col {
				row = append(row, X[idx][j])
			}
		}
		res[i] = row
	}
	return res
}

func subsetInts(arr []int, indices []int) []int {
	res := make([]int, len(indices))
	for i, idx := range indices {
		res[i] = arr[idx]
	}
	return res
}

func removeIndexInt(arr []int, idx int) []int {
	res := append([]int{}, arr[:idx]...)
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
