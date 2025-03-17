package nngo

import (
    "fmt"
    "reflect"
    "strings"
)

/*
Module requires three main structs/interfaces: Tensor, Context & Function

- Tensor: main data handle
- Context: struct to save information between forward & backward
- Function: wrapper for operations between tensors
*/

// Number is the interface for generic types of numbers
type Number interface {
    ~int8 | ~int16 | ~int32 | ~float32
}

// TensorMod for handling type information
type TensorModule[T Number] struct {}

// Tensor is the most important structure
type Tensor[T Number] struct {
    // data
    Data []T
    // shape
    Shape []int
    // strides
    Strides []int
    // flag to compute grad for backward propagation
    RequiresGrad bool
    // grad tensor
    Grad *Tensor[T]
    // function producing operation
    GradFn *Function[T]
}

// Context holds information between passes
type Context[T Number] struct {
    SavedTensor []T
    SavedValues []any
}

// Function interface of operations
type Function[T Number] interface {
    Forward(ctx *Context[T], inputs ...*Tensor[T]) *Tensor[T]
    Backward(ctx *Context[T], gradOutput *Tensor[T]) *Tensor[T]
}

/* helper functions */

// computeStrides
func computeStrides(shape []int) []int {
    strides := make([]int, len(shape))
    val := 1
    for i := len(shape)-1; i >= 0; i--{
        strides[i] = val
        val *= shape[i]
    }
    return strides
}

// computeSize
func computeSize(shape []int) int {
    size := 1
    for _, dim := range shape {
        size *= dim
    }
    return size
}

func (t *Tensor[T]) String() string {
    var sb strings.Builder
    // basic tensor info
    sb.WriteString(fmt.Sprintf("Tensor(shape=%v, ", t.Shape))
    sb.WriteString(fmt.Sprintf("strides=%v, ", t.Strides))
    // data type info
    sb.WriteString(fmt.Sprintf("dtype=%s, ", reflect.TypeOf(t.Data[0])))
    // computation status
    sb.WriteString(fmt.Sprintf("leaf=%t, ", t.GradFn == nil))
    // grad flag
    sb.WriteString(fmt.Sprintf("requires_grad=%v, ", t.RequiresGrad))

    // Only show data for small tensors that are already materialized
    if t.Data != nil {
        size := min(computeSize(t.Shape), 8)
        // first 8 elements (or less)
        sb.WriteString(fmt.Sprintf("data=%v", t.Data[:size]))
    }
    sb.WriteString(")")
    return sb.String()
}

/* creation functions */

// Zeros
func (t *TensorModule[T]) Zeros(shape []int, flag ...bool) *Tensor[T] {
    needGrad := false
    if len(flag) == 1 { needGrad = flag[0] }
    size := computeSize(shape)
    data := make([]T, size)
    strides := computeStrides(shape)
    return &Tensor[T]{
        Data: data,
        Shape: shape,
        Strides: strides,
        RequiresGrad: needGrad,
    }
}

// Ones
func (t *TensorModule[T]) Ones(shape []int, flag ...bool) *Tensor[T] {
    needGrad := false
    if len(flag) == 1 { needGrad = flag[0] }
    size := computeSize(shape)
    data := make([]T, size)
    for i := range data { data[i] = 1 }
    strides := computeStrides(shape)
    return &Tensor[T]{
        Data: data,
        Shape: shape,
        Strides: strides,
        RequiresGrad: needGrad,
    }
}
