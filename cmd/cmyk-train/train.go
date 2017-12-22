package main

import (
	"bytes"
	"errors"
	"flag"
	"fmt"
	"github.com/ieee0824/libcmyk/nn"
	"image"
	"image/color"
	"image/jpeg"
	"io"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"strings"
)

func dismantleImage(cmykImg image.Image, ycbcrImg image.Image) (io.Reader, error) {
	buffer := new(bytes.Buffer)
	w := cmykImg.Bounds().Max.X
	h := cmykImg.Bounds().Max.Y

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
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

func train(ff *nn.FeedForward, r io.Reader) error {
	buffer := make([]byte, 7)
	for {
		i, err := r.Read(buffer)
		if err != nil {
			break
		} else if i != 7 {
			return nil
		}
		patterns := [][][]float64{
			{
				[]float64{
					float64(buffer[0]) / float64(0xff),
					float64(buffer[1]) / float64(0xff),
					float64(buffer[2]) / float64(0xff),
					float64(buffer[3]) / float64(0xff),
				},
				[]float64{
					float64(buffer[4]) / float64(0xff),
					float64(buffer[5]) / float64(0xff),
					float64(buffer[6]) / float64(0xff),
				},
			},
		}

		ff.Train(patterns, 10, 0.6, 0.4, true)
	}
	return nil
}

func getImgPaths(dirName string) ([]string, error) {
	ret := []string{}
	infos, err := ioutil.ReadDir(dirName)
	if err != nil {
		return nil, err
	}

	pwd, err := os.Getwd()
	if err != nil {
		return nil, err
	}
	for _, info := range infos {
		if info.IsDir() {
			continue
		}
		if !(strings.HasSuffix(info.Name(), ".jpg") || strings.HasSuffix(info.Name(), ".jpeg")) {
			continue
		}

		path := strings.Join([]string{pwd, dirName, info.Name()}, "/")
		ret = append(ret, filepath.Clean(path))
	}
	return ret, nil
}

func main() {
	rgbDir := flag.String("rgb", "", "rgm img dir")
	cmykDir := flag.String("cmyk", "", "cmyk img dir")
	output := flag.String("f", "network.json", "network dump")
	flag.Parse()

	ff, err := nn.Load(*output)
	if err != nil {
		fmt.Println("new nn")
		ff = &nn.FeedForward{}
		ff.Init(4, 20, 3)
	} else {
		fmt.Println("update nn")
	}


	rgbFiles, err := getImgPaths(*rgbDir)
	if err != nil {
		log.Fatalln(err)
	}
	cmykFiles, err := getImgPaths(*cmykDir)
	if err != nil {
		log.Fatalln(err)
	}

	if len(rgbFiles) != len(cmykFiles) {
		log.Fatalln("There is no number of files.")
	}

	for i := 0; i < len(rgbFiles); i++ {
		rgbf, err := os.Open(rgbFiles[i])
		if err != nil {
			log.Fatalln(err)
		}
		cmykf, err := os.Open(cmykFiles[i])
		if err != nil {
			log.Fatalln(err)
		}
		fmt.Println(rgbf.Name())

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
		if err := train(ff, reader); err != nil {
			log.Fatalln(err)
		}
	}

	if err := ff.Dump(*output); err != nil {
		log.Fatalln(err)
	}
}
