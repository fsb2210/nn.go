package nngo

import (
    "fmt"
    "testing"
)

func TestZeros(t *testing.T) {
    shape := []int{3}
    torch := TensorModule[int32]{}
    a := torch.Zeros(shape, false)
    fmt.Printf("%+v\n", a)
}

func TestOnes(t *testing.T) {
    shape := []int{3}
    torch := TensorModule[float32]{}
    a := torch.Ones(shape, false)
    fmt.Printf("%+v\n", a)
}
