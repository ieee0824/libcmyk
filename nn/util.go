package nn

import (
	"math"
	"math/rand"
	"github.com/mengzhuo/intrinsic/sse2"
)

func random(a, b float64) float64 {
	return (b-a)*rand.Float64() + a
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func dsigmoid(y float64) float64 {
	return y * (1 - y)
}

func msigmoid(x [4]float64) [4]float64 {
	ret := [4]float64{1,1}
	xb := []float64{math.Exp(-x[0]), math.Exp(-x[1]), math.Exp(-x[2]), math.Exp(-x[3])}
	sse2.ADDPDm128float64(xb, []float64{1,1})
	sse2.DIVPDm128float64(ret[:], xb)
	sse2.ADDPDm128float64(xb[2:], []float64{1,1})
	sse2.DIVPDm128float64(ret[2:], xb)
	return ret
}

func mdsigmoid(y [4]float64) [4]float64 {
	ret := [4]float64{}
	copy(ret[:], y[:])
	sse2.MULPDm128float64(ret[:], y[:])
	sse2.DIVPDm128float64(ret[:], y[:])
	sse2.MULPDm128float64(ret[2:], y[2:])
	sse2.DIVPDm128float64(ret[2:], y[2:])
	return ret
}
