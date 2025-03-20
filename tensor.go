package nngo

import (
	"fmt"
	"math/rand"
	"reflect"
	"strings"
)

/*
Module requires three main structs/interfaces: Tensor, Context & Function

- Tensor: main data handle
- Context: struct to save information between forward & backward
- Function: wrapper for operations between tensors
*/

// set seed
var Seed int64 = 22

// Number is the interface for generic types of numbers
type Number interface {
    ~int8 | ~int16 | ~int32 | ~float32
}

// Tensor is the most important structure
type Tensor[T Number] struct {
    // data
    Data []T
    // shape
    Shape []int
    // strides
    Strides []int
    // size
    Size int
    // flag to compute grad for backward propagation
    RequiresGrad bool
    // grad tensor
    Grad *Tensor[T]
    // function producing operation
    GradFn Function[T]
}

// Context holds information between passes
type Context[T Number] struct {
    SavedTensor []*Tensor[T]
    SavedShapes []any
}
func (ctx *Context[T]) SaveTensorsForBackward(inputs ...*Tensor[T]) {
    for _, input := range inputs {
        ctx.SavedTensor = append(ctx.SavedTensor, input)
    }
}
func (ctx *Context[T]) SaveShapesForBackward(inputs...[]int) {
    for _, input := range inputs {
        ctx.SavedShapes = append(ctx.SavedShapes, input)
    }
}

// Function interface of operations
type Function[T Number] interface {
    Forward(args []any, inputs ...*Tensor[T]) (*Tensor[T], error)
    Backward(gradOutput *Tensor[T]) (*Tensor[T], error)
}

/* helper functions */

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
        size := min(ComputeSize(t.Shape), 8)
        // first 8 elements (or less)
        sb.WriteString(fmt.Sprintf("data=%v", t.Data[:size]))
    }
    sb.WriteString(")")
    return sb.String()
}

// At returns the value at the specified indices
func (t *Tensor[T]) At(indices ...int) (T, error) {
    if len(indices) != len(t.Shape) {
        return 0, fmt.Errorf("number of indices doesn't match tensor dimensions")
    }

    index := 0 // t.offset
    for i, idx := range indices {
        if idx < 0 || idx >= t.Shape[i] {
            return 0, fmt.Errorf("index %d out of bounds for dimension %d with size %d", idx, i, t.Shape[i])
        }
        index += idx * t.Strides[i]
    }

    return t.Data[index], nil
}

// BroadcastTo returns

/* creation functions */

// Helper function to create a new tensor with the same shape as another
func NewTensorLike[T Number](t *Tensor[T]) *Tensor[T] {
    return NewTensor[T](t.Shape)
}

// NewTensor creates a new tensor with the given shape
func NewTensor[T Number](shape []int, flag ...bool) *Tensor[T] {
    needGrad := false
    if len(flag) == 1 { needGrad = flag[0] }
    size := 1
    for _, dim := range shape { size *= dim }
    // calculate strides
    strides := ComputeStrides(shape) // make([]int, len(shape))
    return &Tensor[T]{
        Data:  make([]T, size),
        Shape: shape,
        Strides: strides,
        Size: size,
        RequiresGrad: needGrad,
    }
}

// Zeros
func (t *Tensor[T]) Zeros(shape []int, flag ...bool) *Tensor[T] {
    needGrad := false
    if len(flag) == 1 { needGrad = flag[0] }
    size := ComputeSize(shape)
    data := make([]T, size)
    strides := ComputeStrides(shape)
    return &Tensor[T]{
        Data: data,
        Shape: shape,
        Strides: strides,
        Size: size,
        RequiresGrad: needGrad,
    }
}

// Ones
func (t *Tensor[T]) Ones(shape []int, flag ...bool) *Tensor[T] {
    needGrad := false
    if len(flag) == 1 { needGrad = flag[0] }
    size := ComputeSize(shape)
    data := make([]T, size)
    for i := range data { data[i] = 1 }
    strides := ComputeStrides(shape)
    return &Tensor[T]{
        Data: data,
        Shape: shape,
        Strides: strides,
        Size: size,
        RequiresGrad: needGrad,
    }
}

// setSeed
func setSeed(seed int64) *rand.Rand {
    source := rand.NewSource(seed)
    return rand.New(source)
}

// RandN
func (t *Tensor[T]) RandN(shape []int, flag ...bool) *Tensor[T] {
    needGrad := false
    if len(flag) == 1 { needGrad = flag[0] }
    size := ComputeSize(shape)
    data := make([]T, size)
    r := setSeed(Seed)
    for i := range data { data[i] = T(r.Float32()) }
    strides := ComputeStrides(shape)
    return &Tensor[T]{
        Data: data,
        Shape: shape,
        Strides: strides,
        Size: size,
        RequiresGrad: needGrad,
    }
}

// Uniform
func (t *Tensor[T]) Uniform(shape []int, flag ...bool) *Tensor[T] {
    needGrad := false
    if len(flag) == 1 { needGrad = flag[0] }
    size := ComputeSize(shape)
    data := make([]T, size)
    r := setSeed(Seed)
    for i := range data { data[i] = T(2*r.Float32()-1) }
    strides := ComputeStrides(shape)
    return &Tensor[T]{
        Data: data,
        Shape: shape,
        Strides: strides,
        Size: size,
        RequiresGrad: needGrad,
    }
}

/* utils */

func (t *Tensor[T])BroadcastTo_(targetShape []int) *Tensor[T] {
    if len(t.Shape) > len(targetShape) {
        str := fmt.Sprintf("cannot broadcast to lower dimension. have: %d, want %d", len(t.Shape), len(targetShape))
        panic(str)
    }

    // prepend singleton dimensions to match the length of targetShape
    shapeOffset := len(targetShape) - len(t.Shape)

    newShape := make([]int, len(targetShape))
    newStrides := make([]int, len(targetShape))
    if shapeOffset > 0 {
        paddedShape := make([]int, len(targetShape))
        paddedStrides := make([]int, len(targetShape))

        for i := range shapeOffset {
            paddedShape[i] = 1
            paddedStrides[i] = 0
        }

        copy(paddedShape[shapeOffset:], t.Shape)
        copy(paddedStrides[shapeOffset:], t.Strides)

        newShape = paddedShape
        newStrides = paddedStrides
    }

    nt := NewTensor[T](newShape, t.RequiresGrad)
    nt.Data = t.Data
    nt.Strides = newStrides
    return nt
}
