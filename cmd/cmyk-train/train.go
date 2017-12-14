package main

import (
	"image"
	"bytes"
	"io"
	"image/color"
	"errors"
	"flag"
	"os"
	"log"
	"image/jpeg"
	"github.com/ieee0824/libcmyk/nn"
	"fmt"
)

func dismantleImage(cmykImg image.Image, ycbcrImg image.Image) (io.Reader, error) {
	buffer := new(bytes.Buffer)
	w := cmykImg.Bounds().Max.X
	h := cmykImg.Bounds().Max.Y

	for y := 0; y < h; y ++ {
		for x := 0; x < w; x ++ {
			cmykPix := cmykImg.At(x, y)
			cmyk, ok := cmykPix.(color.CMYK)
			if !ok {
				return nil, errors.New("color type mismatch")
			}
			rgbPix := ycbcrImg.At(x, y)
			ycbcr, ok := rgbPix.(color.YCbCr)
			if !ok {
				return nil, errors.New("color type mismatch")
			}
			r, g, b, _ := ycbcr.RGBA()

			bin := []byte{cmyk.C, cmyk.M, cmyk.Y, cmyk.K, byte(r >> 8), byte(g >> 8), byte(b >> 8)}
			if _, err := buffer.Write(bin); err != nil {
				return nil, err
			}
		}
	}
	return buffer, nil
}

func train(r io.Reader) error {
	buffer := make([]byte, 7)
	for {
		i, err := r.Read(buffer)
		if err != nil {
			break
		} else if i != 7 {
			return nil
		}
		patterns := [][2][4]float64{
			{
				[4]float64{
					float64(buffer[0])/0xff,
					float64(buffer[1])/0xff,
					float64(buffer[2])/0xff,
					float64(buffer[3])/0xff,
				},
			},
			{
				[4]float64{
					float64(buffer[4])/0xff,
					float64(buffer[5])/0xff,
					float64(buffer[6])/0xff,
					0xff,
				},
			},
		}
		nn.Train(patterns, 100, 0.6, 0.4, true)
	}
	return nil
}

func main() {
	rgbPath := flag.String("rgb", "", "")
	cmykPath := flag.String("cmyk", "", "")
	flag.Parse()

	rgbf, err := os.Open(*rgbPath)
	if err != nil {
		log.Fatalln(err)
	}
	cmykf, err := os.Open(*cmykPath)
	if err != nil {
		log.Fatalln(err)
	}

	rgbImg, err := jpeg.Decode(rgbf)
	if err != nil {
		log.Fatalln(err)
	}

	cmykImg, err := jpeg.Decode(cmykf)
	if err != nil {
		log.Fatalln(err)
	}

	reader, err := dismantleImage(cmykImg, rgbImg)
	if err != nil {
		log.Fatalln(err)
	}


	fmt.Println(train(reader))

	nn.Dump("network.json")
}
