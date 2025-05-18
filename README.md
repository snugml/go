# SnugML/Go
Go Machine Learning Module


### Example


```
package main

import (
	"fmt"
	"github.com/snugml/go"
)

func main() {
	X := []float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	y := []float64{1, 4, 1, 5, 3, 7, 2, 7, 4, 9}

	model := ml.LinearRegression{}
	model.Fit(X, y)
	yPredict := model.Predict(X)
	mse := model.MSE(y, yPredict)
	r2 := model.R2(y, yPredict)

	fmt.Println("X:", X)
	fmt.Println("y:", y)
	fmt.Println("yPredict:", yPredict)
	fmt.Printf("MSE: %.4f\n", mse)
	fmt.Printf("R2: %.4f\n", r2)
}
```

## Available Exported Classes and Methods

| **#** | **Class/Method**          | **Location (File)**        | **Description**                                                  |
|-------|---------------------------|----------------------------|------------------------------------------------------------------|
| 1     | `LinearRegression`         | `/models/linear-model.mjs`        | Class for performing linear regression.                          |
| 2     | `PolynomialRegression`     | `/models/linear-model.mjs`        | 
## Examples

[Linear Regression](test/linear.go)

[Polynomial Regression](test/poly.go)
