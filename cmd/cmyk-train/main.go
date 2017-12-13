package main

import (
	"errors"
	"flag"
	"fmt"
	"github.com/ieee0824/libcmyk/nn"
	"image"
	"image/color"
	"image/jpeg"
	"io/ioutil"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

func init() {
	runtime.GOMAXPROCS(runtime.NumCPU())
}

func genNetwork() {
	nn.Init()
}

func dismantleImage(cmyk, rgb image.Image) ([][2][4]float64, error) {
	if cmyk.Bounds().Max.X != rgb.Bounds().Max.X {
		return nil, errors.New("not match image")
	}
	if cmyk.Bounds().Max.Y != rgb.Bounds().Max.Y {
		return nil, errors.New("not match image")
	}
	w := cmyk.Bounds().Max.X
	h := cmyk.Bounds().Max.Y
	ret := make([][2][4]float64, h)

	for x := 0; x < w; x++ {
		data := [2][4]float64{}
		for y := 0; y < h; y++ {
			cmykC, ok := cmyk.At(x, y).(color.CMYK)
			if !ok {
				return nil, errors.New("not cmyk")
			}
			rgbC := rgb.At(x, y)
			data[0] = [4]float64{
				float64(cmykC.C) / 0xff,
				float64(cmykC.M) / 0xff,
				float64(cmykC.Y) / 0xff,
				float64(cmykC.K) / 0xff,
			}
			r, g, b, _ := rgbC.RGBA()
			data[1] = [4]float64{
				float64(r>>8) / 0xff,
				float64(g>>8) / 0xff,
				float64(b>>8) / 0xff,
				float64(0),
			}
			ret[y] = data
		}
	}
	return ret, nil
}

func train(cmykDirName, rgbDirName string) error {
	cmykFiles, err := ioutil.ReadDir(cmykDirName)
	if err != nil {
		return err
	}

	for _, fileInfo := range cmykFiles {
		fileName := fileInfo.Name()
		if !strings.Contains(fileName, ".jpg") {
			continue
		}
		fmt.Println(fileName)
		cmykImagePath := filepath.Clean(cmykDirName + "/" + fileName)
		rgbImagePath := filepath.Clean(rgbDirName + "/" + fileName)
		cmykF, err := os.Open(cmykImagePath)
		if err != nil {
			return err
		}
		rgbF, err := os.Open(rgbImagePath)
		if err != nil {
			return err
		}
		cmyk, err := jpeg.Decode(cmykF)
		if err != nil {
			return err
		}
		rgb, err := jpeg.Decode(rgbF)
		if err != nil {
			return err
		}

		data, err := dismantleImage(cmyk, rgb)
		if err != nil {
			return err
		}
		nn.Train(data, 2, 0.6, 0.4, true)
	}

	return nil
}

func main() {
	rand.Seed(time.Now().UnixNano())
	rgbDir := flag.String("rgb-dir", "./rgb", "")
	cmykDir := flag.String("cmyk-dir", "./cmyk", "")
	flag.Parse()
	genNetwork()

	if err := train(*cmykDir, *rgbDir); err != nil {
		log.Fatalln(err)
	}
}
