package ml

import (
    "github.com/snugml/go/models"
    "github.com/snugml/go/utils"
)

// models
type LinearRegression = models.LinearRegression
type PolynomialRegression = models.PolynomialRegression
type DecisionTreeClassifier = models.DecisionTreeClassifier
type MLPClassifier = models.MLPClassifier
var NewMLPClassifier = models.NewMLPClassifier
type GaussianNB = models.GaussianNB

// utils
type LabelEncoder = utils.LabelEncoder
var AccuracyScore = utils.AccuracyScore