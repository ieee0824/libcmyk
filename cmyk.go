package libcmyk

import (
	"image/color"
	"github.com/ieee0824/libcmyk/nn"
)

type Converter struct {
	network *nn.FeedForward
}

func NewConverter(name string) (*Converter, error) {
	ret := new(Converter)
	n, err := nn.Load(name)
	if err != nil {
		return nil, err
	}
	ret.network = n
	return ret, nil
}

func (c *Converter) CMYK2RGBA(cmyk *color.CMYK) (*color.RGBA, error) {
	inputs := []float64{
		float64(cmyk.C)/0xff,
		float64(cmyk.M)/0xff,
		float64(cmyk.Y)/0xff,
		float64(cmyk.K)/0xff,
	}
	result, err := c.network.Update(inputs)
	if err != nil {
		return nil, err
	}

	return &color.RGBA{
		uint8(result[0] * 0xff),
		uint8(result[1] * 0xff),
		uint8(result[2] * 0xff),
		0xff,
	}, nil
}
