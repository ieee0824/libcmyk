package main

import (
	"github.com/ieee0824/libcmyk/nn"
	"math/rand"
	"time"
	"fmt"
)



func main() {
	num := 65536*512
	data := make([][2][4]float64, num)

	for i := 0; i < num;i ++ {
		d := [2][4]float64{
			{
				rand.Float64()/rand.Float64(),
				rand.Float64()/rand.Float64(),
				rand.Float64()/rand.Float64(),
				rand.Float64()/rand.Float64(),
			},
			{
				rand.Float64()/rand.Float64(),
				rand.Float64()/rand.Float64(),
				rand.Float64()/rand.Float64(),
				rand.Float64()/rand.Float64(),
			},
		}
		data[i] = d
	}
	nn.Init()

	start := time.Now()
	nn.Train(data, num, 0.6, 0.4, true)
	end := time.Now()

	fmt.Println(end.Sub(start))
}