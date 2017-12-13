package main

import (
	"flag"
	"github.com/ieee0824/libcmyk"
	"image"
	"image/color"
	"image/jpeg"
	"log"
	"os"
)

func main() {
	f := flag.String("f", "", "")
	flag.Parse()

	imageFile, err := os.Open(*f)
	if err != nil {
		log.Fatalln(err)
	}
	img, err := jpeg.Decode(imageFile)
	if err != nil {
		log.Fatalln(err)
	}

	w := img.Bounds().Max.X
	h := img.Bounds().Max.Y

	outImg := image.NewRGBA(img.Bounds())

	for x := 0; x < w; x++ {
		for y := 0; y < h; y++ {
			cmyk, ok := img.At(x, y).(color.CMYK)
			if !ok {
				log.Fatalln("not cmyk image")
			}
			rgb, err := libcmyk.CMYK2RGBA(&cmyk)
			if err != nil {
				log.Fatalln(err)
			}
			outImg.Set(x, y, rgb)
		}
	}

	outf, err := os.Create("rgb-" + *f)
	if err != nil {
		log.Fatalln(err)
	}
	if err := jpeg.Encode(outf, outImg, &jpeg.Options{Quality: 100}); err != nil {
		log.Fatalln(err)
	}
}
