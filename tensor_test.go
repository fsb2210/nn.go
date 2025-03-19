package nngo

import "testing"

func TestZeros(t *testing.T) {
    shape := []int{3}
    tensor := Tensor[int32]{}
    _ = tensor.Zeros(shape)
}

func TestOnes(t *testing.T) {
    shape := []int{3}
    tensor := Tensor[int32]{}
    _ = tensor.Ones(shape)
}

func TestRandN(t *testing.T) {
    shape := []int{2, 2}
    tensor := Tensor[float32]{}
    _ = tensor.RandN(shape)
}

func TestUniform(t *testing.T) {
    shape := []int{2, 2}
    tensor := Tensor[float32]{}
    _ = tensor.Uniform(shape)
}
