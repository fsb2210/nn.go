package nngo

/* methods API versions */

func (t *Tensor[T]) Add(other *Tensor[T]) *Tensor[T] { return Add(t, other) }
func (t *Tensor[T]) Mul(other *Tensor[T]) *Tensor[T] { return Mul(t, other) }
func (t *Tensor[T]) Transpose(shape []int) *Tensor[T] { return Transpose(t, shape) }
func (t *Tensor[T]) Reshape(shape []int) *Tensor[T] { return Reshape(t, shape) }

/* functions API versions */

// Tranpose
func Transpose[T Number](self *Tensor[T], order []int) *Tensor[T] {
    ctx := &Context[T]{}
    op := &TransposeFn[T]{ Ctx: ctx }
    orderI := make([]any, len(order))
    for i, dim := range order { orderI[i] = dim }
    // call to forward prop
    result, err := op.Forward(orderI, self)
    if err != nil { panic("error in Reshape operation") }
    if self.RequiresGrad {
        result.RequiresGrad = true
        result.GradFn = op
    }
    return result
}

// Reshape
func Reshape[T Number](self *Tensor[T], shape []int) *Tensor[T] {
    ctx := &Context[T]{}
    op := &ReshapeFn[T]{ Ctx: ctx }
    // convert []int to []interface
    // Can I convert a []T to an []interface{}? Not directly. It is disallowed
    // by the language specification because the two types do not have the same
    // representation in memory. It is necessary to copy the elements
    // individually to the destination slice. This example converts a slice of
    // int to a slice of interface{}
    // https://go.dev/doc/faq#convert_slice_of_interface
    shapeI := make([]any, len(shape))
    for i, dim := range shape { shapeI[i] = dim }
    // call to forward prop
    result, err := op.Forward(shapeI, self)
    if err != nil { panic("error in Reshape operation") }
    if self.RequiresGrad {
        result.RequiresGrad = true
        result.GradFn = op
    }
    return result
}

// Add is the user-facing API function for adding tensors
func Add[T Number](self, other *Tensor[T]) *Tensor[T] {
    ctx := &Context[T]{}
    op := &AddFn[T]{ Ctx: ctx }
    result, err := op.Forward([]any{}, self, other)
    if err != nil { panic("error in Add operation") }
    if self.RequiresGrad || other.RequiresGrad {
        result.RequiresGrad = true
        result.GradFn = op
    }
    return result
}

// Mul
func Mul[T Number](self, other *Tensor[T]) *Tensor[T] {
    ctx := &Context[T]{}
    op := &MulFn[T]{ Ctx: ctx }
    result, err := op.Forward([]any{}, self, other)
    if err != nil { panic("error in Mul operation") }
    if self.RequiresGrad || other.RequiresGrad {
        result.RequiresGrad = true
        result.GradFn = op
    }
    return result
}
