package main

import (
	"flag"
	"github.com/ieee0824/libcmyk"
	"os"
	"log"
	"image/jpeg"
	"image/color"
	"image"
	"path/filepath"
)

func main(){
	src := flag.String("src", "", "")
	dst := flag.String("dsg", "conv.jpeg", "")
	networkFile := flag.String("n", "network.json", "")
	flag.Parse()

	pwd, err := os.Getwd()
	if err != nil {
		log.Fatalln(err)
	}

	path := pwd+ "/" + *src

	f, err := os.Open(filepath.Clean(path))
	if err != nil {
		log.Fatalln(err)
	}
	defer f.Close()

	img, err := jpeg.Decode(f)
	if err != nil {
		log.Fatalln(err)
	}

	h := img.Bounds().Max.Y
	w := img.Bounds().Max.X

	_, ok := img.At(0, 0).(color.CMYK)
	if !ok {
		log.Fatalln("unsupport color type")
	}

	newImg := image.NewRGBA(img.Bounds())
	converter := libcmyk.New(*networkFile)

	for y := 0; y < h; y ++ {
		for x := 0; x < w; x ++ {
			c := img.At(x, y).(color.CMYK)
			rgba, err := converter.CMYK2RGBA(&c)
			if err != nil {
				log.Fatalln(err)
			}
			newImg.Set(x, y, rgba)
		}
	}

	writer, err := os.Create(*dst)
	if err != nil {
		log.Fatalln(err)
	}
	if err := jpeg.Encode(writer, newImg, nil); err != nil {
		log.Fatalln(err)
	}

}