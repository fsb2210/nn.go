package nngo

import (
    "fmt"
    "testing"
)

func TestZeros(t *testing.T) {
    shape := []int{3}
    tensor := Tensor[int32]{}
    z := tensor.Zeros(shape)
    fmt.Printf("%+v\n", z)
}

func TestOnes(t *testing.T) {
    shape := []int{3}
    tensor := Tensor[int32]{}
    o := tensor.Ones(shape)
    fmt.Printf("%+v\n", o)
}

func TestRandN(t *testing.T) {
    shape := []int{2, 2}
    tensor := Tensor[float32]{}
    a := tensor.RandN(shape)
    fmt.Printf("%+v\n", a)
}

func TestUniform(t *testing.T) {
    shape := []int{2, 2}
    tensor := Tensor[float32]{}
    a := tensor.Uniform(shape)
    fmt.Printf("%+v\n", a)
}
