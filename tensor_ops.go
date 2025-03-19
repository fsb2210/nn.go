package nngo

import "fmt"

/* movement ops */

// ReshapeFn
type ReshapeFn[T Number] struct {
    Ctx *Context[T]
}
func (op *ReshapeFn[T]) Forward(args []any, inputs ...*Tensor[T]) (*Tensor[T], error) {
    if len(args) <= 0 {
        return nil, fmt.Errorf("Reshape operation requires a non-zero shape slice, got %d", len(args))
    }
    if len(inputs) > 1 {
        return nil, fmt.Errorf("Reshape operation requires a single Tensor, got %d", len(inputs))
    }
    // properly handle shapes as []int
    newShape, err := CastToInts(args)
    if err != nil { return nil, err }
    // size of shapes
    newSize := ComputeSize(newShape)
    // compare size
    if inputs[0].Size != newSize {
        return nil, fmt.Errorf("Reshape operation error: shapes mismatch (%d != %d)", inputs[0].Size, newSize)
    }
    // save shape for backward
    op.Ctx.SaveShapesForBackward(inputs[0].Shape)
    // result
    result := NewTensor[T](newShape)
    result.Data = inputs[0].Data
    return result, nil
}
func (op *ReshapeFn[T]) Backward(gradOutput *Tensor[T]) (*Tensor[T], error) {
    if gradOutput == nil {
        return nil, fmt.Errorf("Reshape backward operation error: null pointer to gradOutput")
    }
    // use the stored context
    // ctx := op.Ctx
    shape := op.Ctx.SavedShapes[0].([]int)
    result := NewTensor[T](shape, gradOutput.RequiresGrad)
    result.Data = gradOutput.Data
    return result, nil
}

// TransposeFn
type TransposeFn[T Number] struct {
    Ctx *Context[T]
}
func (op *TransposeFn[T]) Forward(args []any, inputs ...*Tensor[T]) (*Tensor[T], error) {
    if len(args) <= 0 {
        return nil, fmt.Errorf("Tranpose operation requires a non-zero shape slice, got %d", len(args))
    }
    if len(inputs) > 1 {
        return nil, fmt.Errorf("Transpose operation requires a single Tensor, got %d", len(inputs))
    }
    order, err := CastToInts(args)
    if err != nil { return nil, err }
    // permute order of shape and strides, thus tensor is not contiguous
    newShape := make([]int, len(inputs[0].Shape))
    newStrides := make([]int, len(inputs[0].Strides))
    for i, index := range order {
        newShape[i] = inputs[0].Shape[index]
        newStrides[i] = inputs[0].Strides[index]
    }
    result := NewTensor[T](newShape, inputs[0].RequiresGrad)
    result.Data = inputs[0].Data
    result.Strides = newStrides
    return result, nil
}
func (op *TransposeFn[T]) Backward(gradOutput *Tensor[T]) (*Tensor[T], error) {
    // use the stored context
    // ctx := op.Ctx
    return nil, nil
}

/* element-wise binary ops */

// AddFn is an operation that complies with the Function interface
type AddFn[T Number] struct {
    Ctx *Context[T]
}
func (op *AddFn[T]) Forward(args []any, inputs ...*Tensor[T]) (*Tensor[T], error) {
    if len(inputs) != 2 {
        return nil, fmt.Errorf("Add operation requires two tensors, got %d", len(inputs))
    }
    x, y := inputs[0], inputs[1]

    // use the stored context
    ctx := op.Ctx
    ctx.SaveTensorsForBackward(x, y)
    ctx.SaveShapesForBackward(x.Shape, y.Shape)

    /*
        placeholder for binary ops
    */
    result := NewTensorLike(x)
    for i := range result.Data {
        result.Data[i] = x.Data[i] + y.Data[i]
    }

    return result, nil
}
func (op *AddFn[T]) Backward(gradOutput *Tensor[T]) (*Tensor[T], error) {
    // use the stored context
    // ctx := op.Ctx
    // this is how you retrieve saved data
    // x := ctx.SavedShapes[0].([]int)
    // y := ctx.SavedShapes[1].([]int)
    return nil, nil
}
