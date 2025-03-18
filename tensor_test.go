package nngo

import (
    "fmt"
    "testing"
)

func TestZeros(t *testing.T) {
    shape := []int{3}
    torch := TensorModule[int32]{}
    a := torch.Zeros(shape)
    fmt.Printf("%+v\n", a)
}

func TestOnes(t *testing.T) {
    shape := []int{3}
    torch := TensorModule[float32]{}
    a := torch.Ones(shape)
    fmt.Printf("%+v\n", a)
}

func TestRandN(t *testing.T) {
    shape := []int{2, 2}
    torch := TensorModule[float32]{}
    a := torch.RandN(shape)
    fmt.Printf("%+v\n", a)
}

func TestUniform(t *testing.T) {
    shape := []int{2, 2}
    torch := TensorModule[float32]{}
    a := torch.Uniform(shape)
    fmt.Printf("%+v\n", a)
}

func TestRandInt(t *testing.T) {
    shape := []int{2, 2}
    torch := TensorModule[int32]{}
    a := torch.RandInt(shape)
    fmt.Printf("%+v\n", a)
}
