package nngo

import "fmt"

// AddFn is an operation that complies with the Function interface
type AddFn[T Number] struct {
    Ctx *Context[T]
}
func (op *AddFn[T]) Forward(ctx *Context[T], inputs ...*Tensor[T]) (*Tensor[T], error) {
    if len(inputs) != 2 {
        return nil, fmt.Errorf("Add operation requires two tensors, got %d", len(inputs))
    }

    x, y := inputs[0], inputs[1]
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
func (op *AddFn[T]) Backward(ctx *Context[T], gradOutput *Tensor[T]) (*Tensor[T], error) {
    // use the stored context
    savedCtx := op.Ctx

    // this is how you retrieve saved data
    x := savedCtx.SavedShapes[0].([]int)
    y := savedCtx.SavedShapes[1].([]int)
    fmt.Println(x, y)
    return nil, nil
}
